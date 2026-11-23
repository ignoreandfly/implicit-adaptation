"""Training sets for contrastive weight extrapolation, from SFT samples (evaluation.generate or methods.run format).
  gold_all : every correct solution (= RFT)
  correct / wrong : mixed-outcome questions only, the same number k = min(#correct, #wrong) of each per question
  rand_a / rand_b : the SAME solutions as correct + wrong, shuffled per question into two halves (control)
Truncated solutions and solutions without an answer are dropped; exact duplicates are removed."""
import argparse, random
from pathlib import Path
from common.io import read_jsonl, write_jsonl
ap = argparse.ArgumentParser(); ap.add_argument("--gens", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
rng = random.Random(0); S = {k: [] for k in ("gold_all", "correct", "wrong", "rand_a", "rand_b")}; n = {k: 0 for k in S}
for r in read_jsonl(f"{a.gens}/train.generations.jsonl"):
    P = [(p["output"], p["score"]) for p in r["paths"]] if "paths" in r else list(zip(r["outputs"], r["scores"]))
    seen, C, W = set(), [], []
    for o, s in P:
        if o in seen or s.get("truncated") or s["lenient_answer"] is None: continue
        seen.add(o); (C if s["lenient_correct"] else W).append((o, s))
    k = min(len(C), len(W)); rng.shuffle(C); rng.shuffle(W)
    mix = C[:k] + W[:k]; rng.shuffle(mix)
    for name, x in (("gold_all", C), ("correct", C[:k]), ("wrong", W[:k]), ("rand_a", mix[:k]), ("rand_b", mix[k:])):
        if x:
            S[name].append({"question": r["question"], "answer": r["answer"], "outputs": [o for o, _ in x], "scores": [s for _, s in x]}); n[name] += len(x)
for k, v in S.items(): write_jsonl(Path(a.out) / k / "train.generations.jsonl", v)
print({k: (len(S[k]), n[k]) for k in S})
