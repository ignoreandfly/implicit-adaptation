"""Equal-compute comparison for Self-SSFT: one path per tag (Cons@k) vs k temperature samples (maj@k).

Each system is a list of paths per question:
  tags:NAME=dir1,dir2,...   one generation dir per tag (path i = first output of dir i), e.g. greedy per <think_i>
  samples:NAME=dir          one dir with n sampled outputs (path i = output i)
For k = 1..K: vote accuracy (majority over the first k paths, ties -> earliest path), pass@k (any correct),
mean number of distinct answers. Paired McNemar (exact binomial) of every system vs the first one at k = K.

    python -m analysis.ssft_cons --split test \
        tags:ssft_gold=r/eval/ssft_gold_t1,r/eval/ssft_gold_t2,r/eval/ssft_gold_t3,r/eval/ssft_gold_t4 \
        samples:rft_gold=r/eval/rft_gold_m4_T0.7_n4 --out r/cons.json
"""

import argparse
import json
from collections import Counter
from math import comb


def load(dirpath, split):
    return {r["idx"]: r for r in map(json.loads, open(f"{dirpath}/{split}.generations.jsonl"))}


def paths_of(spec, split):
    kind, rest = spec.split(":", 1)
    name, dirs = rest.split("=", 1)
    if kind == "tags":
        runs = [load(d, split) for d in dirs.split(",")]
        keys = set.intersection(*(set(r) for r in runs))
        P = {i: [(r[i]["scores"][0]["lenient_answer"], r[i]["scores"][0]["lenient_correct"]) for r in runs] for i in keys}
    else:
        run = load(dirs, split)
        P = {i: [(s["lenient_answer"], s["lenient_correct"]) for s in r["scores"]] for i, r in run.items()}
    return name, P


def vote(paths):
    c = Counter(a for a, _ in paths if a is not None)
    if not c:
        return False
    top = max(c.values())
    win = next(a for a, _ in paths if a is not None and c[a] == top)
    return next(ok for a, ok in paths if a == win)


def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if not x and y); n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n) if n else 1.0
    return {"b_wins": n01, "a_wins": n10, "p": round(p, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    systems = [paths_of(s, args.split) for s in args.systems]
    keys = sorted(set.intersection(*(set(P) for _, P in systems)))
    K = min(min(len(P[i]) for i in keys) for _, P in systems)
    res = {"questions": len(keys), "K": K, "systems": {}}
    for name, P in systems:
        rows = {}
        for k in range(1, K + 1):
            sub = [P[i][:k] for i in keys]
            rows[k] = {"vote": round(sum(map(vote, sub)) / len(keys), 4),
                       "pass": round(sum(any(ok for _, ok in p) for p in sub) / len(keys), 4),
                       "distinct_answers": round(sum(len({a for a, _ in p}) for p in sub) / len(keys), 3)}
        rows["per_path_acc"] = [round(sum(P[i][j][1] for i in keys) / len(keys), 4) for j in range(K)]
        res["systems"][name] = rows
        print(name, {k: v for k, v in rows.items() if k in (1, K)}, "per path", rows["per_path_acc"])
    ref_name, ref = systems[0]
    a = [vote(ref[i][:K]) for i in keys]
    res["mcnemar_vs_" + ref_name] = {n: mcnemar(a, [vote(P[i][:K]) for i in keys]) for n, P in systems[1:]}
    print(res["mcnemar_vs_" + ref_name])
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
