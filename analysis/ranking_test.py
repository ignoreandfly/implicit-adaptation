"""Reasoning analogue of Hewitt et al.'s (2024, Sec. 4.2, Table 2) response-ranking test.

Does the model already assign higher probability to the correct worked solution of a
question than to a wrong one?  For each GSM8K question q with reference solution s:

  random     p(s | q) > p(s' | q), s' = the solution of a different random question
             (Hewitt's test: the "mapping" from question to solution exists)
  corrupted  p(s | q) > p(s~ | q), s~ = s with ONE intermediate number changed (harder: the
             model must prefer correct arithmetic / reasoning, not just on-topic text)
  final      p(s | q) > p(s^ | q), s^ = s with only the final answer number changed

p(. | q) is the summed log-probability of the solution tokens (Hewitt's comparison); for the
random negative we also report the per-token mean, since lengths differ. Base models use the
plain "Question: q\\nAnswer: s" format, post-trained models their chat template.

    python -m analysis.ranking_test --models olmo2-1b-base olmo2-1b-sft olmo2-1b-dpo olmo2-1b-instruct
"""

import argparse
import random
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import prompts
from common.io import read_jsonl, write_json
from common.models import resolve

NUM_RE = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?!\.?\d)")


def corrupt(sol, rng, final=False):
    """Change one number: an intermediate one (final=False) or the final answer (final=True)."""
    body, sep, tail = sol.rpartition("Therefore, the answer is")
    if final:
        m = list(NUM_RE.finditer(tail))
        if not m:
            return None
        x = m[-1]
        new = str(int(float(x.group())) + rng.choice([-3, -2, -1, 1, 2, 3]))
        return body + sep + tail[:x.start()] + new + tail[x.end():]
    m = [x for x in NUM_RE.finditer(body) if body[max(0, x.start() - 3):x.start()].strip().endswith("=")]
    m = m or list(NUM_RE.finditer(body))
    if not m:
        return None
    x = rng.choice(m)
    v = float(x.group())
    new = str(int(v) + rng.choice([-3, -2, -1, 1, 2, 3])) if v == int(v) else f"{v + rng.choice([-1, 1]):g}"
    return body[:x.start()] + new + body[x.end():] + sep + tail


def score(model, tok, prompt, resp, fmt):
    p = tok(prompt, add_special_tokens=(fmt == "plain")).input_ids
    r = tok((" " + resp) if fmt == "plain" else resp, add_special_tokens=False).input_ids
    ids = torch.tensor([p + r], device="cuda")
    with torch.no_grad():
        lp = torch.log_softmax(model(ids).logits[0, len(p) - 1:-1].float(), -1)
    tl = lp.gather(-1, torch.tensor(r, device="cuda")[:, None]).squeeze(-1)
    return float(tl.sum()), float(tl.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["olmo2-1b-base", "olmo2-1b-sft", "olmo2-1b-dpo", "olmo2-1b-instruct"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ranking_test.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = rng.sample(read_jsonl(f"data/gsm8k/{args.split}.jsonl"), args.n)
    items = []
    for i, r in enumerate(rows):
        j = rng.choice([k for k in range(len(rows)) if k != i])
        items.append({"q": r["question"], "s": r["solution"], "rand": rows[j]["solution"],
                      "corr": corrupt(r["solution"], rng), "final": corrupt(r["solution"], rng, final=True)})

    out = {}
    for name in args.models:
        hf_id, fmt = resolve(name)
        tok = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16).cuda().eval()
        wins = {"random_sum": [], "random_mean": [], "corrupted_sum": [], "final_sum": []}
        for it in items:
            prompt = prompts.build_prompt(fmt, it["q"], [], tok)
            s_sum, s_mean = score(model, tok, prompt, it["s"], fmt)
            r_sum, r_mean = score(model, tok, prompt, it["rand"], fmt)
            wins["random_sum"].append(s_sum > r_sum)
            wins["random_mean"].append(s_mean > r_mean)
            if it["corr"]:
                wins["corrupted_sum"].append(s_sum > score(model, tok, prompt, it["corr"], fmt)[0])
            if it["final"]:
                wins["final_sum"].append(s_sum > score(model, tok, prompt, it["final"], fmt)[0])
        out[name] = {k: {"rate": float(np.mean(v)), "n": len(v)} for k, v in wins.items()}
        print(name, {k: round(v["rate"], 3) for k, v in out[name].items()}, flush=True)
        del model
        torch.cuda.empty_cache()
    write_json(args.out, {"split": args.split, "n": args.n, "seed": args.seed, "results": out,
                          "example": items[0]})


if __name__ == "__main__":
    main()
