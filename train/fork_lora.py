"""Fork-contrastive self-distillation: train a LoRA only at fork tokens (ICCV env: torch + peft).

Each fork from train/fork_pairs.py gives two candidate tokens (greedy top-1, 2nd-ranked) at the same
prefix, each with m sampled continuations. With a label answer y* per question:
    --label majority : y* = majority of the separate label samples (no gold answers used)
    --label gold     : y* = gold answer
    v_k = fraction of token k's continuations whose answer == y*;  skip forks with v_1 == v_2
    advantages A_1 = v_1 - (v_1 + v_2)/2,  A_2 = -A_1        (--shuffle: random sign = control)
Loss per fork (ReasonMaxxer-style, applied only at the fork position):
    L = -(A_1 log p(tok_1 | prefix) + A_2 log p(tok_2 | prefix))
      + lambda * mean_t KL(p_base || p_lora) over the response prefix positions (anchor)
--loss dpo (bounded alternative; the unbounded advantage loss above destroyed the model in the first
overnight run, even with shuffled labels): token-level DPO at the fork, winner w = higher-value token,
    L = -|A| * log sigmoid(beta * [(log p(w) - log p_ref(w)) - (log p(l) - log p_ref(l))])
with p_ref = the frozen SFT model (adapter disabled), plus the same KL anchor on the prefix.
LoRA rank 32, alpha 64, on q/k/v/o. The adapter is merged and saved as a full model for vLLM eval.

    python -m train.fork_lora --pairs data/fork_pairs/olmo2-1b-sft_train1000.jsonl --label majority \
        --out /data/azfarm/siddhant_fork_lora/olmo2-1b-sft_major
"""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.io import read_jsonl
from common.models import resolve


def build(rows, label, shuffle, seed):
    rng = random.Random(seed)
    ex, stats = [], Counter()
    for r in rows:
        if label == "gold":
            y = r["gold"]
        else:
            v = Counter(a for a in r["label_answers"] if a is not None)
            if not v:
                stats["no_label"] += 1
                continue
            y = v.most_common(1)[0][0]
            stats["label_correct"] += int(y == r["gold"])
            stats["labelled"] += 1
        for f in r["forks"]:
            v1 = sum(a == y for a in f["ans1"]) / len(f["ans1"])
            v2 = sum(a == y for a in f["ans2"]) / len(f["ans2"])
            if v1 == v2:
                stats["tie"] += 1
                continue
            a1 = v1 - (v1 + v2) / 2
            if shuffle and rng.random() < 0.5:
                a1 = -a1
            ex.append({"ids": r["prompt_ids"] + r["greedy_ids"][:f["t"]], "P": len(r["prompt_ids"]),
                       "tok1": f["tok1"], "tok2": f["tok2"], "a1": a1})
            stats["pairs"] += 1
            stats["prefer_2nd"] += int(a1 < 0)
    return ex, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", default="olmo2-1b-sft")
    ap.add_argument("--label", choices=["majority", "gold"], default="majority")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--kl", type=float, default=0.2)
    ap.add_argument("--loss", choices=["adv", "dpo"], default="adv")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    hf_id, fmt = resolve(args.model)
    tok = AutoTokenizer.from_pretrained(hf_id)
    rows = read_jsonl(args.pairs)
    ex, stats = build(rows, args.label, args.shuffle, args.seed)
    print(f"{len(ex)} training forks; stats {dict(stats)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                             task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = math.ceil(len(ex) * args.epochs / args.accum)
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / max(1, total)))

    log = []
    step = 0
    model.train()
    for ep in range(args.epochs):
        order = list(range(len(ex)))
        random.shuffle(order)
        for n, i in enumerate(order):
            e = ex[i]
            ids = torch.tensor([e["ids"]], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(ids)
            lp = torch.log_softmax(out.logits[0].float(), -1)
            last = lp[-1]
            need_ref = args.loss == "dpo" or (args.kl > 0 and len(e["ids"]) - e["P"] > 0)
            if need_ref:
                with torch.no_grad(), model.disable_adapter(), torch.autocast("cuda", dtype=torch.bfloat16):
                    lref = torch.log_softmax(model(ids).logits[0].float(), -1)
            if args.loss == "dpo":
                w, l = (e["tok1"], e["tok2"]) if e["a1"] > 0 else (e["tok2"], e["tok1"])
                z = (last[w] - lref[-1, w]) - (last[l] - lref[-1, l])
                l_dec = -abs(e["a1"]) * 2 * torch.nn.functional.logsigmoid(args.beta * z)
            else:
                l_dec = -(e["a1"] * last[e["tok1"]] - e["a1"] * last[e["tok2"]])
            l_kl = torch.zeros((), device="cuda")
            if args.kl > 0 and len(e["ids"]) - e["P"] > 0:
                lb = lref[e["P"] - 1:-1]
                lq = lp[e["P"] - 1:-1]
                l_kl = (lb.exp() * (lb - lq)).sum(-1).mean()
            loss = (l_dec + args.kl * l_kl) / args.accum
            loss.backward()
            if (n + 1) % args.accum == 0 or n == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    rec = {"step": step, "epoch": ep, "l_dec": float(l_dec), "l_kl": float(l_kl)}
                    log.append(rec); print(rec, flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged = merged.to(torch.bfloat16)
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    json.dump({"args": vars(args), "stats": dict(stats), "train_forks": len(ex), "log": log},
              open(out / "fork_lora_config.json", "w"), indent=2)
    print(f"saved merged model to {out}", flush=True)


if __name__ == "__main__":
    main()
