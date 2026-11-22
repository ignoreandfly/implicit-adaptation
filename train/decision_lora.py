"""Decision-point training, following ReasonMaxxer ("Rethinking RL for LLM Reasoning", arXiv 2605.06241, Sec. 5):
advantage-weighted log-likelihood at the high-entropy decision points of the model's OWN full solutions (correct AND
incorrect), with a KL anchor to the frozen starting model at all other response positions.

    A_i   = clip((r_i - mean_q r) / (std_q r + eps), -2.5, 2.5)        per question q, over its solutions i
    D_i   = response positions where the frozen model's next-token entropy > tau  (~top few % of positions)
    L     = -(1/|D_i|) sum_{t in D_i} A_i log p_theta(x_t | x_<t)  +  lambda * mean_{t not in D_i} KL(p_ref || p_theta)

Rewards r_i: --label gold (answer correct) or --label majority (answer equals the majority answer of the question's
label samples if present, else of the solutions themselves: label-free). Following the paper, only questions with
mixed outcomes (0 < mean r < 1) are used. tau is the --tau_pct percentile of reference entropies over a sample of
response tokens. LoRA (rank 32, q/k/v/o) on the starting model; merged and saved as a full model.

Input: generations with several solutions per question, in evaluation.generate format (outputs + scores) or
methods.run format (paths), e.g. results/judge/olmo2-1b-sft_train1500_n8 or train/fork_branch_data.py output.

    python -m train.decision_lora --gens results/judge/olmo2-1b-sft_train1500_n8 --model olmo2-1b-sft \
        --label majority --out /data/.../decision_sft_major
"""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import prompts
from common.io import read_jsonl
from common.models import resolve


def solutions(r):
    if "paths" in r:
        return [p["output"] for p in r["paths"]], [p["score"] for p in r["paths"]]
    return r["outputs"], r["scores"]


def build(rows, tok, fmt, label, max_len, prefix):
    ex, stats = [], Counter()
    for r in rows:
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
        if mu in (0.0, 1.0):
            stats["skipped_uniform"] += 1
            continue
        sd = float(np.std(rew))
        prompt = prompts.build_prompt(fmt, r["question"], [], tok)
        p_ids = tok(prompt, add_special_tokens=(fmt == "plain")).input_ids
        for o, s, rw in zip(outs, scores, rew):
            if s.get("truncated"):
                continue
            body = o[len(prefix.strip()):].lstrip() if prefix and o.startswith(prefix.strip()) else o
            text = ((prefix + " " + body) if prefix else (" " + o)) + "\n\n" if fmt == "plain" else o + tok.eos_token
            t_ids = tok(text, add_special_tokens=False).input_ids
            a = max(-2.5, min(2.5, (rw - mu) / (sd + 1e-6)))
            ex.append({"ids": (p_ids + t_ids)[:max_len], "P": len(p_ids), "A": a})
        stats["questions"] += 1
    stats["sequences"] = len(ex)
    return ex, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="olmo2-1b-sft")
    ap.add_argument("--format", choices=["plain", "chat"], default=None)
    ap.add_argument("--prefix", default="", help="e.g. \" Let's think step by step.\" for base models")
    ap.add_argument("--label", choices=["gold", "majority"], default="gold")
    ap.add_argument("--tau_pct", type=float, default=95.0, help="decision points = entropy above this percentile")
    ap.add_argument("--kl", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    hf_id, fmt = resolve(args.model, args.format)
    tok = AutoTokenizer.from_pretrained(hf_id)
    rows = read_jsonl(f"{args.gens}/{args.split}.generations.jsonl")[: args.limit]
    ex, stats = build(rows, tok, fmt, args.label, args.max_len, args.prefix)
    print(f"{len(ex)} sequences; stats {dict(stats)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    def ref_logp(ids):
        with torch.no_grad(), model.disable_adapter(), torch.autocast("cuda", dtype=torch.bfloat16):
            return torch.log_softmax(model(ids).logits[0].float(), -1)

    # entropy threshold from a sample of reference response positions
    hs = []
    for e in random.sample(ex, min(200, len(ex))):
        lr = ref_logp(torch.tensor([e["ids"]], device="cuda"))[e["P"] - 1:-1]
        hs.append((-(lr.exp() * lr).sum(-1)).cpu().numpy())
    tau = float(np.percentile(np.concatenate(hs), args.tau_pct))
    print(f"decision threshold tau = {tau:.3f} nats ({args.tau_pct}th percentile)", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = math.ceil(len(ex) * args.epochs / args.accum); warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / max(1, total)))
    model.train(); step = 0; log = []; dshare = []
    for ep in range(args.epochs):
        order = list(range(len(ex))); random.shuffle(order)
        for n, i in enumerate(order):
            e = ex[i]
            ids = torch.tensor([e["ids"]], device="cuda")
            lr_ = ref_logp(ids)[e["P"] - 1:-1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lq = torch.log_softmax(model(ids).logits[0].float(), -1)[e["P"] - 1:-1]
            y = ids[0, e["P"]:]
            H = -(lr_.exp() * lr_).sum(-1)
            D = H > tau
            dshare.append(float(D.float().mean()))
            l_dec = -(e["A"] * lq.gather(-1, y[:, None]).squeeze(-1)[D]).mean() if D.any() else torch.zeros((), device="cuda")
            nd = ~D
            l_kl = (lr_.exp() * (lr_ - lq)).sum(-1)[nd].mean() if nd.any() else torch.zeros((), device="cuda")
            ((l_dec + args.kl * l_kl) / args.accum).backward()
            if (n + 1) % args.accum == 0 or n == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    rec = {"step": step, "l_dec": float(l_dec), "l_kl": float(l_kl), "decision_share": float(np.mean(dshare[-200:]))}
                    log.append(rec); print(rec, flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model.merge_and_unload().to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    json.dump({"args": vars(args), "stats": dict(stats), "tau": tau, "log": log}, open(out / "decision_config.json", "w"), indent=2)
    print(f"saved merged model to {out}", flush=True)


if __name__ == "__main__":
    main()
