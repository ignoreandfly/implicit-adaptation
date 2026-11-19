"""LoRA fine-tuning. Conditions differ only in which (prompt, completion) pairs are built.

    full           question -> solution                 (standard SFT reference)
    response_only  "Answer:" -> solution, no question   (Hewitt-style response-only tuning)
    corrupted      question -> corrupted_solution       (format kept, content wrong)

Loss is computed on completion tokens only (+ EOS). Rank, lr, epochs come from the config
so they stay fixed across conditions.

    python -m train.lora_sft --config train/configs/qwen2.5-0.5b.yaml --task arithmetic \
        --condition response_only --seed 0
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import prompts
from common.io import git_hash, read_jsonl, write_json

CONDITIONS = ("full", "response_only", "corrupted")


def build_pairs(rows, condition):
    pairs = []
    for r in rows:
        if condition == "full":
            p, c = prompts.qa_prompt(r["question"]), r["solution"]
        elif condition == "response_only":
            p, c = prompts.RESPONSE_ONLY_PROMPT, r["solution"]
        elif condition == "corrupted":
            p, c = prompts.qa_prompt(r["question"]), r["corrupted_solution"]
        else:
            raise ValueError(condition)
        pairs.append((p, prompts.completion(c)))
    return pairs


def tokenize(tok, prompt, completion, max_len):
    p = tok(prompt).input_ids  # same call the evaluator uses, so prompt tokens match
    c = tok(completion, add_special_tokens=False).input_ids + [tok.eos_token_id]
    return (p + c)[:max_len], ([-100] * len(p) + c)[:max_len]


def collate(batch, pad_id):
    L = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), L), pad_id)
    lab = torch.full((len(batch), L), -100)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        lab[i, :len(y)] = torch.tensor(y)
        att[i, :len(x)] = 1
    return ids, att, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", default="arithmetic")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_name = f"{cfg['name']}_{args.task}_{args.condition}_s{args.seed}"
    out = Path(args.out or f"runs/{run_name}")
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = read_jsonl(f"{args.data_dir or 'data/' + args.task}/train.jsonl")
    data = [tokenize(tok, p, c, cfg["max_len"]) for p, c in build_pairs(rows, args.condition)]
    n_tok = sum(sum(t != -100 for t in y) for _, y in data)
    print(f"{len(data)} examples, {n_tok} loss tokens; example:\n{tok.decode(data[0][0])!r}")

    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.bfloat16).cuda()
    if cfg.get("grad_checkpointing"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
    for p in model.parameters():  # trainable LoRA weights in fp32, base in bf16
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    bs, ga = cfg["batch_size"], cfg.get("grad_accum", 1)
    total = math.ceil(len(data) / (bs * ga)) * cfg["epochs"]
    warm = max(1, int(cfg.get("warmup_ratio", 0.03) * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    log = open(out / "train_log.jsonl", "w")
    model.train()
    step, t0 = 0, time.time()
    for epoch in range(cfg["epochs"]):
        order = list(range(len(data)))
        random.shuffle(order)
        micro = [order[i:i + bs] for i in range(0, len(order), bs)]
        for mi, idx in enumerate(micro):
            ids, att, lab = (t.cuda() for t in collate([data[i] for i in idx], tok.pad_token_id))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            (loss / ga).backward()
            if (mi + 1) % ga == 0 or mi == len(micro) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or step == total:
                    rec = {"step": step, "epoch": epoch, "loss": loss.item(),
                           "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    print(rec, flush=True)

    model.save_pretrained(out / "adapter")
    write_json(out / "run_config.json", {**cfg, **vars(args), "run_name": run_name, "train_steps": step,
                                         "loss_tokens": n_tok, "seconds": round(time.time() - t0, 1),
                                         "git": git_hash()})
    print(f"saved {out}")


if __name__ == "__main__":
    main()
