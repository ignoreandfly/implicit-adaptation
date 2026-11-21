"""GSM8K: loader that writes our jsonl format, plus a verifier with the same interface as
tasks.arithmetic.verify.

GSM8K rationales look like "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n...\n#### 72".
We drop the <<calc>> annotations, join lines into one paragraph (generation stops at a
blank line, see common.prompts.STOP_STRINGS), and end with our answer template so every
task shares the same "Therefore, the answer is N." ending:

    "Natalia sold 48/2 = 24 clips in May. ... Therefore, the answer is 72."

Usage (needs `datasets` + network):
    python -m tasks.gsm8k --out data/gsm8k
Writes train (7,473 minus dev), dev (200 from train), test (1,319).
"""

import argparse
import random
import re

from common.io import write_jsonl

_CALC_RE = re.compile(r"<<[^>]*>>")


def clean_rationale(raw: str):
    """Return (solution, answer, num_lines) from a GSM8K answer field."""
    body, final = raw.split("####")
    answer = normalize_number(final.strip())
    lines = [_CALC_RE.sub("", ln).strip() for ln in body.strip().split("\n") if ln.strip()]
    lines = [ln if ln[-1] in ".!?" else ln + "." for ln in lines]
    solution = " ".join(lines + [f"Therefore, the answer is {answer}."])
    return solution, answer, len(lines)


def make_row(ex):
    solution, answer, n = clean_rationale(ex["answer"])
    return {"task": "gsm8k", "question": ex["question"].strip(), "answer": answer,
            "num_steps": n, "solution": solution, "raw_answer": ex["answer"]}


# ---------------------------------------------------------------- verifier

_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"
_STRICT_RE = re.compile(r"answer is\s*:?\s*(" + _NUM + ")", re.I)
_LOOSE_ANS_RE = re.compile(r"(?:answer\s*(?:is|:|=)?|####)\s*:?\s*(" + _NUM + ")", re.I)
_INT_RE = re.compile(_NUM)
# "a op b = c", allowing $, commas, decimals and the GSM8K operators (incl. / and x).
_STEP_RE = re.compile(r"(" + _NUM + r")\s*([+\-*/x×÷])\s*(" + _NUM + r")\s*=\s*(" + _NUM + ")")


def normalize_number(s):
    """'$1,200.00' -> '1200', '3.50' -> '3.5', junk -> None."""
    if s is None:
        return None
    s = s.replace("$", "").replace(",", "").strip().rstrip(".")
    try:
        x = float(s)
    except ValueError:
        return None
    return str(int(x)) if x == int(x) else repr(x)


def _apply(a, op, b):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op in "*x×":
        return a * b
    return a / b if b else None


def verify(output: str, answer: str) -> dict:
    """Same keys and semantics as tasks.arithmetic.verify, with numbers normalized
    (commas, $, trailing .0) before comparing."""
    strict = _STRICT_RE.findall(output)
    loose = _LOOSE_ANS_RE.findall(output)
    ints = _INT_RE.findall(output)
    strict_ans = normalize_number(strict[0]) if strict else None
    lenient_ans = normalize_number(loose[0] if loose else (ints[-1] if ints else None))

    eqs = _STEP_RE.findall(output)
    n_ok = 0
    for a, op, b, c in eqs:
        a, b, c = (normalize_number(v) for v in (a, b, c))
        if None in (a, b, c):
            continue
        r = _apply(float(a), op, float(b))
        n_ok += int(r is not None and abs(r - float(c)) < 1e-4 * max(1.0, abs(r)))

    return {
        "strict_answer": strict_ans,
        "lenient_answer": lenient_ans,
        "strict_correct": strict_ans == answer,
        "lenient_correct": lenient_ans == answer,
        "format_ok": len(eqs) >= 2 and bool(_LOOSE_ANS_RE.search(output)),
        "n_steps": len(eqs),
        "n_steps_correct": n_ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/gsm8k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_dev", type=int, default=200)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")
    train = [make_row(e) for e in ds["train"]]
    test = [make_row(e) for e in ds["test"]]
    random.Random(args.seed).shuffle(train)
    splits = {"train": train[args.n_dev:], "dev": train[:args.n_dev], "test": test}
    for name, rows in splits.items():
        write_jsonl(f"{args.out}/{name}.jsonl", rows)
        print(f"{name}: {len(rows)}")
    print("example:", splits["train"][0]["solution"])


if __name__ == "__main__":
    main()
