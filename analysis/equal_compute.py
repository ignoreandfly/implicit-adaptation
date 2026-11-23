"""Equal-compute comparison of multi-path methods: majority vote over the FIRST k paths of each question.

Works on any generations jsonl with {"paths": [{"score": {...}, "gen_tokens": ...}, ...]} (methods.run sample /
fork_mbr / cot_decoding, analysis/ssft_infer.py). For fork_mbr files, path 0 is greedy and branches are ordered by
margin, so "first k" = greedy + the k-1 closest forks (ties go to the first path, i.e. greedy, as in fork_mbr).
Reports acc@k, mean generated tokens@k, and pass@k (any of the first k correct) for k = 1..K.

    python -m analysis.equal_compute results/a/test500.generations.jsonl results/b/test500.generations.jsonl --ks 1 2 4 6 8
"""

import argparse
import json
from collections import Counter


def vote(paths):
    v = Counter(p["score"]["lenient_answer"] for p in paths if p["score"]["lenient_answer"] is not None)
    if not v:
        return False
    best_n = v.most_common(1)[0][1]
    tied = {a for a, c in v.items() if c == best_n}
    first = next(p["score"]["lenient_answer"] for p in paths if p["score"]["lenient_answer"] in tied)
    return any(p["score"]["lenient_correct"] for p in paths if p["score"]["lenient_answer"] == first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    table = {}
    for f in a.files:
        rows = [json.loads(l) for l in open(f)]
        n = len(rows); K = min(len(r["paths"]) for r in rows)
        res = {}
        for k in a.ks:
            if k > K:
                continue
            res[k] = {"acc_vote": sum(vote(r["paths"][:k]) for r in rows) / n,
                      "pass": sum(any(p["score"]["lenient_correct"] for p in r["paths"][:k]) for r in rows) / n,
                      "tokens": sum(sum(p.get("gen_tokens", 0) for p in r["paths"][:k]) for r in rows) / n}
        table[f] = res
        print(f"\n{f}  ({n} questions, {K} paths)")
        for k, v in res.items():
            print(f"  k={k:2d}  vote {v['acc_vote']:.3f}  pass@k {v['pass']:.3f}  tokens/question {v['tokens']:.0f}")
    if a.out:
        json.dump(table, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
