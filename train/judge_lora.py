"""Pairwise self-judge: train a LoRA so the model picks which of two of ITS OWN solutions is correct.

Data: a generation run on GSM8K TRAINING questions with several samples per question (evaluation.generate
--split train --n_samples 8). For each question, solutions are grouped by final answer; a training pair is
(correct solution, incorrect solution with a different, non-empty answer), up to --pairs_per_q per question,
each shown in BOTH orders. Prompt (chat format for post-trained models, plain otherwise):
    Question: ... / Solution A: ... / Solution B: ... / Which solution reaches the correct final answer?
    Reply with A or B.
Loss: cross-entropy over the two answer tokens {"A", "B"} at the reply position. 10% of questions are held
out to report pairwise judge accuracy. The adapter is merged and saved as a full model.

    python -m train.judge_lora --gens results/judge/olmo2-1b-sft_train1500_n8 --model olmo2-1b-sft \
        --out /data/azfarm/siddhant_fork_lora/judge_sft
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.io import read_jsonl
from common.models import resolve

JUDGE_Q = ("Which solution reaches the correct final answer? Reply with A or B.")


def judge_text(fmt, tok, question, sol_a, sol_b):
    body = f"Question: {question}\n\nSolution A:\n{sol_a.strip()}\n\nSolution B:\n{sol_b.strip()}\n\n{JUDGE_Q}"
    if fmt == "chat":
        return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
    return body + "\nAnswer:"


def ab_ids(tok, fmt):
    lead = "" if fmt == "chat" else " "
    a, b = tok.encode(lead + "A", add_special_tokens=False), tok.encode(lead + "B", add_special_tokens=False)
    assert len(a) == 1 and len(b) == 1, (a, b)
    return a[0], b[0]


def build_pairs(rows, per_q, rng, max_chars):
    pairs = []
    for qi, r in enumerate(rows):
        good, bad = [], []
        for o, s in zip(r["outputs"], r["scores"]):
            if len(o) > max_chars or s.get("truncated"):
                continue
            if s["lenient_correct"]:
                good.append(o)
            elif s["lenient_answer"] is not None:
                bad.append((o, s["lenient_answer"]))
        good = list(dict.fromkeys(good))
        if not good or not bad:
            continue
        cand = [(g, b) for g in good for b, _ in bad]
        rng.shuffle(cand)
        for g, b in cand[:per_q]:
            pairs.append({"q": qi, "question": r["question"], "good": g, "bad": b})
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="olmo2-1b-sft")
    ap.add_argument("--pairs_per_q", type=int, default=4)
    ap.add_argument("--max_chars", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed); torch.manual_seed(args.seed)
    hf_id, fmt = resolve(args.model)
    tok = AutoTokenizer.from_pretrained(hf_id)
    tid_a, tid_b = ab_ids(tok, fmt)
    rows = read_jsonl(f"{args.gens}/{args.split}.generations.jsonl")
    pairs = build_pairs(rows, args.pairs_per_q, rng, args.max_chars)
    qs = sorted({p["q"] for p in pairs}); rng.shuffle(qs)
    held = set(qs[: max(1, len(qs) // 10)])
    train = [p for p in pairs if p["q"] not in held]; val = [p for p in pairs if p["q"] in held]
    ex = []
    for p in train:  # both orders
        ex.append((judge_text(fmt, tok, p["question"], p["good"], p["bad"]), 0))
        ex.append((judge_text(fmt, tok, p["question"], p["bad"], p["good"]), 1))
    rng.shuffle(ex)
    print(f"{len(pairs)} pairs from {len(qs)} questions; train examples {len(ex)}, held-out pairs {len(val)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()

    def judge_acc(m):
        m.eval(); ok = 0
        with torch.no_grad():
            for p in val:
                s = 0.0
                for (a, b), good_is_a in (((p["good"], p["bad"]), True), ((p["bad"], p["good"]), False)):
                    ids = tok(judge_text(fmt, tok, p["question"], a, b), add_special_tokens=(fmt == "plain"),
                              return_tensors="pt").input_ids.cuda()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg = m(ids).logits[0, -1].float()
                    d = (lg[tid_a] - lg[tid_b]).item()
                    s += d if good_is_a else -d
                ok += s > 0
        m.train()
        return ok / max(1, len(val))

    base_acc = judge_acc(model)
    print(f"held-out pairwise accuracy BEFORE training (zero-shot judge): {base_acc:.3f}", flush=True)
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
        for n, (text, label) in enumerate(ex):
            ids = tok(text, add_special_tokens=(fmt == "plain"), return_tensors="pt").input_ids.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(ids).logits[0, -1].float()
            loss = torch.nn.functional.cross_entropy(lg[[tid_a, tid_b]][None], torch.tensor([label], device="cuda")) / args.accum
            loss.backward()
            if (n + 1) % args.accum == 0 or n == len(ex) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    rec = {"step": step, "loss": float(loss) * args.accum}; log.append(rec); print(rec, flush=True)
    acc = judge_acc(model)
    print(f"held-out pairwise accuracy AFTER training: {acc:.3f}", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model.merge_and_unload().to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    json.dump({"args": vars(args), "pairs": len(pairs), "train_examples": len(ex), "heldout_pairs": len(val),
               "heldout_acc_before": base_acc, "heldout_acc_after": acc, "log": log},
              open(out / "judge_config.json", "w"), indent=2)
    print(f"saved judge to {out}", flush=True)


if __name__ == "__main__":
    main()
