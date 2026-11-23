"""Merge several generations files over the same questions into one evaluation.generate-style file
({question, answer, outputs, scores[, label_answers]}), e.g. RFT samples + fork branches.

    python -m train.merge_gens --inputs dirA dirB --split train --out results/fork_branch/union
"""
import argparse
from pathlib import Path

from common.io import read_jsonl, write_jsonl


def sols(r):
    if "paths" in r:
        return [p["output"] for p in r["paths"]], [p["score"] for p in r["paths"]]
    return r["outputs"], r["scores"]


ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True)
ap.add_argument("--split", default="train")
ap.add_argument("--out", required=True)
a = ap.parse_args()
merged = {}
for d in a.inputs:
    for r in read_jsonl(f"{d}/{a.split}.generations.jsonl"):
        m = merged.setdefault(r["question"], {"question": r["question"], "answer": r["answer"], "outputs": [], "scores": []})
        o, s = sols(r); m["outputs"] += o; m["scores"] += s
        if r.get("label_answers"):
            m.setdefault("label_answers", []).extend(r["label_answers"])
write_jsonl(Path(a.out) / f"{a.split}.generations.jsonl", list(merged.values()))
print(f"merged {len(merged)} questions from {len(a.inputs)} files -> {a.out}")
