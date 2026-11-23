"""Decision-point training, following ReasonMaxxer ("Rethinking RL for LLM Reasoning", arXiv 2605.06241) and its
reference code (github.com/farukakgul/ReasonMaxxer, scripts/{select_mid_pool,prepare_training_data,
train_reasonmaxxer}.py, example recipe examples/qwen25_1p5b/): advantage-weighted log-likelihood at the high-entropy
decision points of the model's OWN solutions (correct AND incorrect), with a KL anchor to the frozen starting model at
all other positions.

    A_i   = clip((r_i - mean_q r) / pstd_q r, -2.5, 2.5)            per problem q, over its solutions i
    D_i   = positions where the frozen model's next-token entropy > tau (absolute, nats; recipe: 1.4, sweep 1.0-2.2)
    L     = -(1/|D|) sum_{t in D} A_i log p_theta(x_t | x_<t)  +  0.2 * mean_{t not in D} KL(p_ref || p_theta)

Recipe defaults reproduced here (their example: Qwen2.5-1.5B base, 300 problems x 20 rollouts at T 0.6 / top-p 0.95):
  - problems: mixed outcomes only, the --n_problems (50) closest to a 0.5 pass rate (ties: fewer truncations, shorter)
  - trim: keep the shortest --trim (80%) of the selected rollouts
  - KL anchor over every non-decision position INCLUDING the prompt (their default; --kl_scope response to restrict)
  - LoRA r32 / alpha 64 on q,k,v,o; lr 1e-4; warmup 50; grad accum 8; 2 epochs capped at --max_steps 149
  - fixed curriculum order (no shuffle): rollouts sorted by signal clarity = |A| x (decision tokens / length)
  - adapter checkpoints every --save_every optimizer steps (they select the best on 60 held-out problems; here use
    train/merge_adapter.py + a dev eval to pick one)
Deviation: the reference model is the policy with its adapter disabled (identical to their separate frozen copy).

Rewards r_i: --label gold (answer correct) or --label majority (answer equals the majority answer of the problem's
label samples if present, else of its solutions: label-free; not in the paper).

    python -m train.decision_lora --gens results/explore/sft_train300_n20 --model olmo2-1b-sft --label gold \
        --out checkpoints/explore/rmx_gold
"""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import prompts
from common.io import read_jsonl
from common.models import resolve


def solutions(r):
    if "paths" in r:
        return [p["output"] for p in r["paths"]], [p["score"] for p in r["paths"]]
    return r["outputs"], r["scores"]


def build(rows, tok, fmt, label, max_len, prefix, n_problems, trim, rft=False):
    """Select mid-difficulty problems, compute advantages, tokenize, trim long rollouts (select_mid_pool + prepare)."""
    stats, cands = Counter(), []
    for qi, r in enumerate(rows):
        outs, scores = solutions(r)
        if label == "gold":
            rew = [float(s["lenient_correct"]) for s in scores]
        else:
            pool = r.get("label_answers") or [s["lenient_answer"] for s in scores]
            v = Counter(a for a in pool if a is not None)
            if not v:
                continue
            maj = v.most_common(1)[0][0]
            stats["majority_correct"] += int(maj == r["answer"]); stats["labelled"] += 1
            rew = [float(s["lenient_answer"] == maj) for s in scores]
        mu = float(np.mean(rew))
        if rft and mu > 0:      # RFT-matched data: every problem with a correct solution, incl. all-correct ones
            pass
        elif mu in (0.0, 1.0):
            stats["skipped_uniform"] += 1
            continue
        n_trunc = sum(bool(s.get("truncated")) for s in scores)
        avg_len = float(np.mean([len(o) for o in outs]))
        cands.append((abs(mu - 0.5), n_trunc, avg_len, qi, rew))
    cands.sort(key=lambda c: c[:4])
    if n_problems > 0:
        cands = cands[:n_problems]
    ex = []
    for _, _, _, qi, rew in cands:
        r = rows[qi]; outs, scores = solutions(r)
        mu, sd = float(np.mean(rew)), max(float(np.std(rew)), 1e-8)
        prompt = prompts.build_prompt(fmt, r["question"], [], tok)
        p_ids = tok(prompt, add_special_tokens=(fmt == "plain")).input_ids
        seen = set()
        for o, rw in zip(outs, rew):
            if rft and (rw < 0.5 or o in seen):   # RFT: correct solutions only, deduplicated, weight 1
                continue
            seen.add(o)
            body = o[len(prefix.strip()):].lstrip() if prefix and o.startswith(prefix.strip()) else o
            text = ((prefix + " " + body) if prefix else (" " + o)) + "\n\n" if fmt == "plain" else o + tok.eos_token
            t_ids = tok(text, add_special_tokens=False).input_ids
            ex.append({"ids": (p_ids + t_ids)[:max_len], "P": len(p_ids), "A": 1.0 if rft else (rw - mu) / sd, "q": qi,
                       "pos": rw > 0.5})
    stats["problems"] = len(cands)
    stats["rollouts_selected"] = len(ex)
    ex.sort(key=lambda e: len(e["ids"]))
    ex = ex[: int(round(trim * len(ex)))]
    stats["rollouts_after_trim"] = len(ex)
    stats["pos_rollouts"] = sum(e["pos"] for e in ex)
    return ex, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="olmo2-1b-sft")
    ap.add_argument("--format", choices=["plain", "chat"], default=None)
    ap.add_argument("--prefix", default="", help="e.g. \" Let's think step by step.\" for base models")
    ap.add_argument("--label", choices=["gold", "majority"], default="gold")
    ap.add_argument("--n_problems", type=int, default=50, help="mixed-outcome problems closest to 0.5 pass rate (0 = all)")
    ap.add_argument("--trim", type=float, default=0.8, help="keep this fraction of the shortest selected rollouts")
    ap.add_argument("--tau", type=float, default=1.4, help="absolute entropy threshold (nats) for decision points")
    ap.add_argument("--tau_pct", type=float, default=None, help="if set, use this percentile of entropies instead")
    ap.add_argument("--gate", choices=["entropy", "margin", "all", "weight"], default="entropy",
                    help="decision points by reference entropy > tau (ReasonMaxxer) or by reference top-2 probability "
                         "margin < --margin_thr (our fork criterion)")
    ap.add_argument("--margin_thr", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=2.0,
                    help="--gate weight: soft fork weighting, per-token advantage A * w_t with w_t = (1 - m_t)^gamma "
                         "(m_t = reference top-2 probability margin); KL anchor weighted by 1 - w_t")
    ap.add_argument("--share", type=float, default=None,
                    help="instead of an absolute threshold, gate this fraction of response tokens (e.g. 0.05)")
    ap.add_argument("--kl", type=float, default=0.2)
    ap.add_argument("--kl_scope", choices=["all", "response"], default="all")
    ap.add_argument("--adv_clip", type=float, default=2.5)
    ap.add_argument("--beta_neg", type=float, default=1.0,
                    help="scale for negative advantages (ReasonMaxxer code's --beta_neg; < 1 tames the push-down)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=149)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--shuffle", action="store_true", help="shuffle instead of the clarity curriculum")
    ap.add_argument("--positive_only", action="store_true", help="ablation: drop negative rollouts (= gated RFT)")
    ap.add_argument("--rft_weights", action="store_true",
                    help="RFT-matched data: all problems with >=1 correct solution, correct solutions only "
                         "(deduplicated), advantage 1; use with --n_problems 0 --trim 1 --max_steps 0")
    ap.add_argument("--save_every", type=int, default=0, help="save the LoRA adapter every N optimizer steps")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    hf_id, fmt = resolve(args.model, args.format)
    tok = AutoTokenizer.from_pretrained(hf_id)
    rows = read_jsonl(f"{args.gens}/{args.split}.generations.jsonl")[: args.limit]
    ex, stats = build(rows, tok, fmt, args.label, args.max_len, args.prefix, args.n_problems, args.trim,
                      args.rft_weights)
    if args.positive_only:
        ex = [e for e in ex if e["A"] > 0]
    print(f"{len(ex)} rollouts; stats {dict(stats)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    def ref_logp(ids):
        with torch.no_grad(), model.disable_adapter(), torch.autocast("cuda", dtype=torch.bfloat16):
            return torch.log_softmax(model(ids).logits[0, :-1].float(), -1)

    # teacher-forced reference statistics at response positions (score_rollouts.py); score: higher = more decision-like
    for e in tqdm(ex, desc="score reference", mininterval=30):
        lr = ref_logp(torch.tensor([e["ids"]], device="cuda"))[e["P"] - 1:]
        if args.gate == "all":
            e["S"] = torch.ones(lr.shape[0])
        elif args.gate == "weight":
            top = lr.exp().topk(2, -1).values
            e["S"] = (1 - (top[:, 0] - top[:, 1])).clamp(0, 1).pow(args.gamma).cpu()   # fork weight w_t
        elif args.gate == "entropy":
            e["S"] = (-(lr.exp() * lr).sum(-1)).cpu()
        else:
            top = lr.exp().topk(2, -1).values
            e["S"] = -(top[:, 0] - top[:, 1]).cpu()          # negative top-2 margin
    allS = torch.cat([e["S"] for e in ex]).numpy()
    if args.gate in ("all", "weight"):
        tau = -1.0       # every response token takes part; "weight" scales each by w_t
    elif args.share is not None:
        tau = float(np.percentile(allS, 100 * (1 - args.share)))
    elif args.gate == "entropy":
        tau = args.tau if args.tau_pct is None else float(np.percentile(allS, args.tau_pct))
    else:
        tau = -args.margin_thr
    for e in ex:
        e["D"] = e["S"] > tau
        e["clarity"] = abs(e["A"]) * float((e["S"] if args.gate == "weight" else e["D"].float()).mean())
    dshare = float(np.mean([float(e["D"].float().mean()) for e in ex]))
    if args.gate == "weight":
        w = torch.cat([e["S"] for e in ex])
        print(f"fork weights (gamma {args.gamma}): mean {float(w.mean()):.3f}; share > 0.5: {float((w > 0.5).float().mean()):.3f}; "
              f"effective token share (sum w / n) {float(w.sum() / len(w)):.3f}", flush=True)
    print(f"gate {args.gate}: threshold {tau:.3f}; decision share of response tokens = {dshare:.4f}", flush=True)
    if not args.shuffle:
        ex.sort(key=lambda e: -e["clarity"])

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = min(math.ceil(len(ex) / args.accum) * args.epochs, args.max_steps or 10**9)
    warm = max(0, min(args.warmup, total - 1))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model.train(); step = 0; log = []; n_micro = 0; done = False
    bar = tqdm(total=total, desc="optimizer steps", mininterval=30)
    for ep in range(args.epochs):
        order = list(range(len(ex)))
        if args.shuffle:
            random.shuffle(order)
        for i in order:
            e = ex[i]
            ids = torch.tensor([e["ids"]], device="cuda")
            lr_ = ref_logp(ids)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lq = torch.log_softmax(model(ids).logits[0, :-1].float(), -1)
            y = ids[0, 1:]
            D = torch.zeros(len(y), dtype=torch.bool, device="cuda"); D[e["P"] - 1:] = e["D"].cuda()
            a = max(-args.adv_clip, min(args.adv_clip, e["A"]))
            if a < 0:
                a *= args.beta_neg
            if args.gate == "weight":   # soft: A_t = A * w_t over all response tokens; KL weighted by 1 - w_t
                W = torch.zeros(len(y), device="cuda"); W[e["P"] - 1:] = e["S"].cuda()
                R = torch.zeros(len(y), dtype=torch.bool, device="cuda"); R[e["P"] - 1:] = True
                lp = lq.gather(-1, y[:, None]).squeeze(-1)
                l_dec = -(a * (W * lp)[R].sum() / W[R].sum().clamp_min(1e-6))
                V = 1 - W
                if args.kl_scope == "response":
                    V[: e["P"] - 1] = 0
                kl_t = (lr_.exp() * (lr_ - lq)).sum(-1)
                l_kl = (V * kl_t).sum() / V.sum().clamp_min(1e-6)
            else:
                l_dec = -(a * lq.gather(-1, y[:, None]).squeeze(-1)[D]).mean() if D.any() else torch.zeros((), device="cuda")
                keep = ~D
                if args.kl_scope == "response":
                    keep[: e["P"] - 1] = False
                l_kl = (lr_.exp() * (lr_ - lq)).sum(-1)[keep].mean() if keep.any() else torch.zeros((), device="cuda")
            ((l_dec + args.kl * l_kl) / args.accum).backward()
            n_micro += 1
            if n_micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1; bar.update(1)
                if step % 10 == 0:
                    rec = {"step": step, "epoch": ep, "l_dec": float(l_dec), "l_kl": float(l_kl), "lr": sched.get_last_lr()[0]}
                    log.append(rec); print(rec, flush=True)
                if args.save_every and step % args.save_every == 0:
                    model.save_pretrained(out / f"adapter_step{step}")
                if step >= total:
                    done = True
                    break
        if done:
            break
    bar.close()
    model.save_pretrained(out / f"adapter_step{step}")
    model.merge_and_unload().to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    json.dump({"args": vars(args), "stats": dict(stats), "threshold": tau, "decision_share": dshare, "steps": step,
               "log": log}, open(out / "decision_config.json", "w"), indent=2)
    print(f"saved merged model ({step} optimizer steps) to {out}", flush=True)


if __name__ == "__main__":
    main()
