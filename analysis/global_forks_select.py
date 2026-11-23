"""Choose a set of K openings from a global_forks pool run (greedy forward selection of majority-vote accuracy),
cross-validated over random halves of the questions. Targets: gold, or label-free (majority of the saved 8 plain
samples). Every condition uses K = 6 paths. Ties are split evenly (expected accuracy), so path order does not matter."""
import argparse, json, random
from collections import Counter
import numpy as np
from common.io import read_jsonl
from analysis.global_forks import POOL

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="results/global_forks/pool/train.generations.jsonl")
ap.add_argument("--ref", default="results/screen/olmo2-1b-sft/train/vllm_T0.7_n8/train.generations.jsonl")
ap.add_argument("--splits", type=int, default=20)
ap.add_argument("--out", default="results/global_forks/selection.json")
a = ap.parse_args()
K = 6
ref = read_jsonl(a.ref); train = read_jsonl("data/gsm8k/train.jsonl")
recs = read_jsonl(a.pool)
A = {}; gold = {}
for r in recs:
    A.setdefault(r["idx"], {})[r["opening"]] = [p["answer"] for p in r["paths"]]; gold[r["idx"]] = r["gold"]
Q = sorted(A); free = {}
for i in Q:
    assert ref[i]["question"] == train[i]["question"]
    v = Counter(s["lenient_answer"] for s in ref[i]["scores"] if s["lenient_answer"] is not None)
    free[i] = v.most_common(1)[0][0] if v else None
print("questions", len(Q), "label-free target correct", round(np.mean([free[i] == gold[i] for i in Q]), 3))

def vote(ans, t):
    v = Counter(x for x in ans if x is not None)
    if not v: return 0.0
    m = max(v.values()); tied = [x for x, c in v.items() if c == m]
    return (t in tied) / len(tied)
def set_acc(S, qs, tgt):   # S: list of openings; repeated openings use different samples; 6 rotations averaged
    tot = 0
    for i in qs:
        for rot in range(6):
            used = Counter(); ans = []
            for k in S:
                ans.append(A[i][k][(rot + used[k]) % 6]); used[k] += 1
            tot += vote(ans, tgt[i])
    return tot / (6 * len(qs))
def forward(qs, tgt):
    S = []
    while len(S) < K:
        S.append(max((k for k in range(len(POOL)) if k not in S), key=lambda k: set_acc(S + [k], qs, tgt)))
    return S
def single(k, qs, tgt): return np.mean([np.mean([x == tgt[i] for x in A[i][k]]) for i in qs])

res = {c: [] for c in ("plain x6", "best single x6 (gold)", "best single x6 (free)", "top-6 singles (gold)",
                       "set (gold)", "set (free)", "random 6")}
freq = Counter(); rng = random.Random(0)
for s in range(a.splits):
    q = Q[:]; rng.shuffle(q); tr, te = q[:250], q[250:]
    bg = max(range(len(POOL)), key=lambda k: single(k, tr, gold)); bf = max(range(len(POOL)), key=lambda k: single(k, tr, free))
    top = sorted(range(len(POOL)), key=lambda k: -single(k, tr, gold))[:K]
    sg, sf = forward(tr, gold), forward(tr, free); freq.update(sg)
    for c, S in (("plain x6", [0] * 6), ("best single x6 (gold)", [bg] * 6), ("best single x6 (free)", [bf] * 6),
                 ("top-6 singles (gold)", top), ("set (gold)", sg), ("set (free)", sf),
                 ("random 6", rng.sample(range(len(POOL)), K))):
        res[c].append(set_acc(S, te, gold))
base = np.array(res["plain x6"])
for c, v in res.items():
    print(f"{c:24s} held-out maj@6 {np.mean(v):.3f}   vs plain {np.mean(np.array(v) - base):+.3f}")
fg, ff = forward(Q, gold), forward(Q, free); bgs = max(range(len(POOL)), key=lambda k: single(k, Q, gold)); bfs = max(range(len(POOL)), key=lambda k: single(k, Q, free))
print("set gold:", [POOL[k] for k in fg]); print("set free:", [POOL[k] for k in ff]); print("best single gold/free:", POOL[bgs], "|", POOL[bfs])
print("selection frequency over splits:", [(POOL[k], n) for k, n in freq.most_common(10)])
json.dump({"cv": {c: float(np.mean(v)) for c, v in res.items()}, "set_gold": fg, "set_free": ff, "best_gold": bgs, "best_free": bfs,
           "pool": POOL}, open(a.out, "w"), indent=2)
