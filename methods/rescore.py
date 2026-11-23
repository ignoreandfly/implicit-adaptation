"""Score every saved path of a multi-path run with token signals, then compare selection rules.

Signals per path (teacher-forced pass over prompt + output, output tokens only):
    neg_entropy     -mean next-token entropy (nats)
    self_certainty  mean KL(Uniform || p) (Kang et al. 2025)
    answer_conf     mean p(top1) - p(top2) over the answer's tokens (CoT-decoding's Delta)
    dtr / dtr50     deep-thinking ratio over all / first 50 tokens (Chen et al. 2026; g=0.5 nats, rho=0.85)
    dc_lowest / dc_bottom10 / dc_tail
                    DeepConf (Fu et al. 2025) token confidence C_t = -mean log-prob of the top-20 tokens,
                    averaged over sliding windows of DC_WINDOW tokens: lowest window, mean of the bottom
                    10% of windows, last window. (Paper uses 2048-token windows; our traces are ~100-150.)
    neg_length      -number of tokens (control)
Rules (all use lenient answers):
    best:<s>    path with the highest s
    top50:<s>   majority vote among the top half of paths by s (Think@n's rule)
    wvote:<s>   vote weighted by s, min-max scaled to [0,1] within the problem
    eba / eba_mean / centroid  Embedding-based agreement (Ontalvilla et al. 2026), native-hidden-state
                              variant: cosine distance between final-layer hidden states (last output
                              token, or mean over output tokens), agglomerative clustering (average
                              linkage) with k chosen by silhouette, answer of the sample closest to the
                              dominant cluster's centroid; "centroid" = closest to the global mean
    borda<p>:self_certainty   Kang et al. 2025: vote weight (N - rank + 1)^p by self-certainty rank
    dc<eta>:<dc signal>       DeepConf offline: keep the top eta% of paths by confidence, then
                              confidence-weighted vote (eta = 10 -> "DeepConf-low", 90 -> "-high")
    majority / oracle / single (mean single-path accuracy)

    python -m methods.rescore results/screen/olmo2-1b-base/dev/cot_decoding_k10
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import prompts
from common.io import read_jsonl, write_json, write_jsonl
from common.models import resolve
from tasks import get_task

SIGNALS = ["neg_entropy", "self_certainty", "answer_conf", "dtr", "dtr50", "dc_lowest", "dc_bottom10", "dc_tail", "neg_length"]
DC_SIGNALS = ["dc_lowest", "dc_bottom10", "dc_tail"]
DC_WINDOW = 32


def path_signals(model, tok, prompt, out, answer_norm, normalize, g=0.5, rho=0.85, fmt="plain"):
    L = model.config.num_hidden_layers
    # chat templates carry their own special tokens; plain outputs follow "Answer:" after a space
    p_ids = tok(prompt, add_special_tokens=(fmt == "plain")).input_ids
    o_ids = tok((" " + out) if fmt == "plain" else out, add_special_tokens=False).input_ids[:512]
    T = len(o_ids)
    if T == 0:
        return {**{s: -1e9 for s in SIGNALS}, "_emb_last": None, "_emb_mean": None}
    ids = torch.tensor([p_ids + o_ids], device="cuda")
    with torch.no_grad():
        res = model(ids, output_hidden_states=True)
        pos = slice(len(p_ids) - 1, len(p_ids) - 1 + T)
        lf = torch.log_softmax(res.logits[0, pos].float(), -1)
        pf = lf.exp()
        H = -(pf * lf).sum(-1)
        V = lf.shape[-1]
        sc = (-lf.mean(-1) - math.log(V))  # KL(U||p) = -1/V sum log p - log V
        top20 = lf.topk(20, -1).values
        tokconf = (-top20.mean(-1)).cpu().numpy()
        top2 = pf.topk(2, -1).values
        margin = (top2[:, 0] - top2[:, 1]).cpu().tolist()
        run_min = torch.full((T,), float("inf"), device="cuda")
        settle = torch.full((T,), L, device="cuda")
        for l in range(1, L):
            lp = torch.log_softmax(model.lm_head(model.model.norm(res.hidden_states[l][0, pos])).float(), -1)
            m = (0.5 * (pf + lp.exp())).clamp_min(1e-30).log()
            d = 0.5 * (pf * (lf - m)).sum(-1) + 0.5 * (lp.exp() * (lp - m)).sum(-1)
            run_min = torch.minimum(run_min, d)
            newly = (run_min <= g) & (settle == L)
            settle[newly] = l
        deep = (settle >= math.ceil(rho * L)).float()
    # answer-token confidence: tokens overlapping the last occurrence of the answer number
    conf = 0.0
    if answer_norm is not None:
        import re
        text = " " + out
        span = None
        for mm in re.finditer(r"-?\$?\d[\d,]*(?:\.\d+)?", text):
            if normalize(mm.group()) == answer_norm:
                span = (mm.start(), mm.end())
        if span:
            vals, prev = [], 0
            for i in range(T):
                end = len(tok.decode(o_ids[: i + 1]))
                if end > span[0] and prev < span[1]:
                    vals.append(margin[i])
                prev = end
                if prev >= span[1]:
                    break
            conf = float(np.mean(vals)) if vals else 0.0
    w = min(DC_WINDOW, T)
    groups = np.convolve(tokconf, np.ones(w) / w, mode="valid")
    bottom = np.sort(groups)[: max(1, int(np.ceil(0.1 * len(groups))))]
    return {"neg_entropy": -float(H.mean()), "self_certainty": float(sc.mean()), "answer_conf": conf,
            "dtr": float(deep.mean()), "dtr50": float(deep[:50].mean()),
            "dc_lowest": float(groups.min()), "dc_bottom10": float(bottom.mean()), "dc_tail": float(tokconf[-w:].mean()),
            "neg_length": -float(T),
            "_emb_last": res.hidden_states[-1][0, len(p_ids) + T - 1].float().cpu().numpy(),
            "_emb_mean": res.hidden_states[-1][0, len(p_ids):len(p_ids) + T].float().mean(0).cpu().numpy()}


def eba_pick(E, centroid_only=False):
    """Index of the sample closest to the dominant cluster's centroid (cosine)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.metrics import silhouette_score
    X = E / np.linalg.norm(E, axis=1, keepdims=True).clip(1e-8)
    N = len(X)
    labels = np.zeros(N, int)
    if not centroid_only and N >= 3:
        Z = linkage(X, method="average", metric="cosine")
        best = -1.0
        for k in range(2, N):
            lab = fcluster(Z, k, criterion="maxclust")
            if len(set(lab)) < 2 or len(set(lab)) >= N:
                continue
            sc = silhouette_score(X, lab, metric="cosine")
            if sc > best:
                best, labels = sc, lab
    dom = np.bincount(labels).argmax()
    members = np.where(labels == dom)[0]
    c = X[members].mean(0)
    return int(members[np.argmax(X[members] @ c)])


def vote(paths, weights=None):
    tally = defaultdict(float)
    for j, x in enumerate(paths):
        a = x["score"]["lenient_answer"]
        if a is not None:
            tally[a] += 1.0 if weights is None else weights[j]
    return max(tally, key=tally.get) if tally else None


def evaluate(rows):
    n = len(rows)
    acc = defaultdict(float)
    for r in rows:
        paths, gold = r["paths"], r["answer"]
        ok = [x["score"]["lenient_correct"] for x in paths]
        acc["single"] += np.mean(ok)
        acc["oracle"] += any(ok)
        acc["majority"] += vote(paths) == gold
        for s in SIGNALS:
            v = np.array([x["signals"][s] for x in paths])
            acc[f"best:{s}"] += ok[int(v.argmax())]
            top = np.argsort(-v, kind="stable")[: max(1, len(paths) // 2)]
            acc[f"top50:{s}"] += vote([paths[j] for j in top]) == gold
            w = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else np.ones_like(v)
            acc[f"wvote:{s}"] += vote(paths, w) == gold
        for name, key, conly in (("eba", "_emb_last", False), ("eba_mean", "_emb_mean", False),
                                 ("centroid", "_emb_last", True)):
            if all(x.get(key) is not None for x in paths):
                j = eba_pick(np.stack([x[key] for x in paths]), conly)
                acc[name] += ok[j]
        N = len(paths)
        sc = np.array([x["signals"]["self_certainty"] for x in paths])
        rank = np.empty(N)
        rank[np.argsort(-sc, kind="stable")] = np.arange(1, N + 1)
        for pw in (0.5, 1.0):
            acc[f"borda{pw}:self_certainty"] += vote(paths, (N - rank + 1) ** pw) == gold
        for s in DC_SIGNALS:
            v = np.array([x["signals"][s] for x in paths])
            for eta in (10, 90):
                keep = np.argsort(-v, kind="stable")[: max(1, int(round(N * eta / 100)))]
                acc[f"dc{eta}:{s}"] += vote([paths[j] for j in keep], [v[j] for j in keep]) == gold
    return {k: v / n for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    run = Path(args.run)
    meta = json.loads((run / f"{args.split}.metrics.json").read_text())
    rows = read_jsonl(run / f"{args.split}.generations.jsonl")
    hf_id, fmt = resolve(meta["model"])
    task = get_task(meta["args"]["task"])
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16).cuda().eval()
    few = meta["args"].get("few_shot", 0)
    ex = (random.Random(meta["args"]["seed"]).sample(read_jsonl(f"data/{meta['args']['task']}/train.jsonl"), few)
          if few else [])
    for i, r in enumerate(rows):
        prompt = prompts.build_prompt(fmt, r["question"], ex, tok)
        for x in r["paths"]:
            sig = path_signals(model, tok, prompt, x["output"], x["score"]["lenient_answer"],
                               task.normalize_number, fmt=fmt)
            x["_emb_last"], x["_emb_mean"] = sig.pop("_emb_last"), sig.pop("_emb_mean")
            x["signals"] = sig
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)
    res = evaluate(rows)
    for r in rows:
        for x in r["paths"]:
            x.pop("_emb_last", None), x.pop("_emb_mean", None)
    write_jsonl(run / f"{args.split}.rescored.jsonl", rows)
    write_json(run / f"{args.split}.selection.json", res)
    keys = (["single", "majority", "oracle", "eba", "eba_mean", "centroid", "borda0.5:self_certainty", "borda1.0:self_certainty"]
            + [f"dc{e}:{s}" for s in DC_SIGNALS for e in (10, 90)]
            + [f"{r}:{s}" for r in ("best", "top50", "wvote") for s in SIGNALS])
    for k in keys:
        print(f"{k:28s} {res[k]:.3f}")


if __name__ == "__main__":
    main()
