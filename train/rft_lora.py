"""Baseline for fork-contrastive self-distillation: rejection-sampling fine-tuning (RFT / STaR-style).

LoRA fine-tuning with the ordinary next-token loss on the model's OWN sampled solutions that are kept as
"good": --label gold (answer correct) or --label majority (answer equals the majority of the question's
samples; no gold labels). Duplicate solutions are removed. Same LoRA size / lr / epochs as train/fork_lora.py
so the two are comparable; the adapter is merged and saved as a full model.

Input: a methods.run --method sample run (generations jsonl with outputs + scores), e.g. Base + zero-shot CoT,
n = 8, on the same 1,000 training questions as the fork pairs.

    python -m train.rft_lora --gens results/screen/olmo2-1b-base/train/sample_prefix-..._T0.7_n8 --split train \
        --model olmo2-1b-base --prefix " Let's think step by step." --label gold --out /data/.../rft_gold
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

from common import prompts
from common.io import read_jsonl
from common.models import resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="olmo2-1b-base")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--label", choices=["gold", "majority"], default="gold")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    hf_id, fmt = resolve(args.model)
    tok = AutoTokenizer.from_pretrained(hf_id)
    rows = read_jsonl(f"{args.gens}/{args.split}.generations.jsonl")
    ex, kept_q, label_ok = [], 0, 0
    for r in rows:
        if "paths" in r:   # methods.run format
            outs = [p["output"] for p in r["paths"]]; scores = [p["score"] for p in r["paths"]]
        else:              # evaluation.generate format
            outs, scores = r["outputs"], r["scores"]
        ans = [s["lenient_answer"] for s in scores]
        if args.label == "gold":
            keep = [o for o, s in zip(outs, scores) if s["lenient_correct"]]
        else:
            v = Counter(a for a in (r.get("label_answers") or ans) if a is not None)
            if not v:
                continue
            maj = v.most_common(1)[0][0]
            label_ok += int(maj == r["answer"])
            keep = [o for o, a in zip(outs, ans) if a == maj]
        keep = list(dict.fromkeys(keep))
        if keep:
            kept_q += 1
        prompt = prompts.build_prompt(fmt, r["question"], [], tok)
        p_ids = tok(prompt, add_special_tokens=(fmt == "plain")).input_ids
        for o in keep:
            # the sampled output already starts with the forced prefix text; base answers end at a blank line
            target = (" " + o + "\n\n") if fmt == "plain" else (o + tok.eos_token)
            t_ids = tok(target, add_special_tokens=False).input_ids
            ids = (p_ids + t_ids)[: args.max_len]
            ex.append({"ids": ids, "P": len(p_ids)})
    print(f"{len(ex)} training solutions from {kept_q}/{len(rows)} questions; label={args.label}"
          + (f"; majority correct on {label_ok}" if args.label == "majority" else ""), flush=True)

    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = math.ceil(len(ex) * args.epochs / args.accum); warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / max(1, total)))
    model.train(); step = 0; log = []
    for ep in range(args.epochs):
        order = list(range(len(ex))); random.shuffle(order)
        for n, i in enumerate(order):
            e = ex[i]
            ids = torch.tensor([e["ids"]], device="cuda")
            labels = ids.clone(); labels[0, : e["P"]] = -100
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, labels=labels).loss / args.accum
            loss.backward()
            if (n + 1) % args.accum == 0 or n == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    rec = {"step": step, "epoch": ep, "loss": float(loss) * args.accum}; log.append(rec); print(rec, flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model.merge_and_unload().to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    json.dump({"args": vars(args), "train_solutions": len(ex), "log": log}, open(out / "rft_config.json", "w"), indent=2)
    print(f"saved merged model to {out}", flush=True)


if __name__ == "__main__":
    main()
