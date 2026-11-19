"""Multi-step arithmetic: generator, corrupted solutions, and verifier.

Problems are left-associative chains, e.g. "Compute ((4 + 7) * 3) - 5." with solution
"First, 4 + 7 = 11. Then, 11 * 3 = 33. Finally, 33 - 5 = 28. Therefore, the answer is 28."

Usage:
    python -m tasks.arithmetic --out data/arithmetic --seed 0
"""

import argparse
import random
import re

from common.io import write_jsonl

OPS = ["+", "-", "*"]


def _apply(a, op, b):
    return a + b if op == "+" else a - b if op == "-" else a * b


def sample_problem(rng, n_steps, max_val=999):
    """Return (expression, steps) where steps = [(lhs, op, rhs, result), ...].
    All intermediate results lie in [0, max_val]."""
    while True:
        val = rng.randint(1, 20)
        expr = str(val)
        steps = []
        for i in range(n_steps):
            op = rng.choice(OPS)
            b = rng.randint(2, 9) if op == "*" else rng.randint(1, 20)
            new = _apply(val, op, b)
            steps.append((val, op, b, new))
            expr = f"({expr}) {op} {b}" if i > 0 else f"{expr} {op} {b}"
            val = new
        if all(0 <= s[3] <= max_val for s in steps):
            return expr, steps


def corrupt_steps(rng, steps):
    """Same operators and right operands, but every step's result is wrong by 1-9 and the
    error propagates (the next step starts from the wrong value). Keeps the structure,
    destroys the content."""
    out, val = [], steps[0][0]
    for _, op, b, _ in steps:
        wrong = _apply(val, op, b) + rng.choice([-1, 1]) * rng.randint(1, 9)
        out.append((val, op, b, wrong))
        val = wrong
    return out


def render_solution(steps):
    parts = []
    for i, (a, op, b, r) in enumerate(steps):
        lead = "First" if i == 0 else "Finally" if i == len(steps) - 1 else "Then"
        parts.append(f"{lead}, {a} {op} {b} = {r}.")
    parts.append(f"Therefore, the answer is {steps[-1][3]}.")
    return " ".join(parts)


def make_example(rng, n_steps):
    expr, steps = sample_problem(rng, n_steps)
    return {
        "task": "arithmetic",
        "question": f"Compute {expr}.",
        "expression": expr,
        "num_steps": n_steps,
        "answer": str(steps[-1][3]),
        "solution": render_solution(steps),
        "corrupted_solution": render_solution(corrupt_steps(rng, steps)),
    }


def make_split(rng, n, step_range, exclude):
    rows = []
    while len(rows) < n:
        ex = make_example(rng, rng.randint(*step_range))
        if ex["expression"] in exclude:
            continue
        exclude.add(ex["expression"])
        rows.append(ex)
    return rows


# ---------------------------------------------------------------- verifier

_STEP_RE = re.compile(r"(-?\d+)\s*([+\-*x×])\s*(-?\d+)\s*=\s*(-?\d+)")
_STRICT_RE = re.compile(r"answer is\s*:?\s*(-?\d+)", re.I)
_LOOSE_ANS_RE = re.compile(r"answer\s*(?:is|:|=)?\s*:?\s*(-?\d+)", re.I)
_INT_RE = re.compile(r"-?\d+")


def verify(output: str, answer: str) -> dict:
    """Score a (truncated) model output.

    strict:  first number after "answer is" (our template; any casing).
    lenient: first number after any "answer[ is|:|=]" phrase, else the last integer.
             Uses the first committed answer because base models keep generating
             unrelated text with numbers in it after answering.
    format_ok: >=2 equations of the form "a op b = c" and some "answer ..." phrase.
    steps:   local correctness of each equation (a op b really equals c).
    """
    strict = _STRICT_RE.findall(output)
    loose = _LOOSE_ANS_RE.findall(output)
    ints = _INT_RE.findall(output)
    strict_ans = strict[0] if strict else None
    lenient_ans = loose[0] if loose else (ints[-1] if ints else None)

    eqs = _STEP_RE.findall(output)
    n_ok = 0
    for a, op, b, c in eqs:
        op = "*" if op in "x×" else op
        n_ok += int(_apply(int(a), op, int(b)) == int(c))

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
    ap.add_argument("--out", default="data/arithmetic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=5000)
    ap.add_argument("--n_dev", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=500)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen = set()
    # Train/dev/test_id: 2-4 steps. test_ood: 5-6 steps (length generalization).
    splits = {
        "train": make_split(rng, args.n_train, (2, 4), seen),
        "dev": make_split(rng, args.n_dev, (2, 4), seen),
        "test_id": make_split(rng, args.n_test, (2, 4), seen),
        "test_ood": make_split(rng, args.n_test, (5, 6), seen),
    }
    for name, rows in splits.items():
        write_jsonl(f"{args.out}/{name}.jsonl", rows)
        print(f"{name}: {len(rows)}")
    print("example:", splits["train"][0])


if __name__ == "__main__":
    main()
