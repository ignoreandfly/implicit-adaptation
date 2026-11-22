"""Where do correct samples leave the greedy path, and is that point a fork token?

For each dev problem: tokens of the greedy zero-shot-CoT output g and of each of the 10
self-consistency samples s (same prompt and prefix). The divergence point of s is the first
position t where s_t != g_t. We score g once with the base model to get, at every position,
the next-token entropy H_t and the probability / rank of the token s_t that the sample took.

Reported for CORRECT and (control) INCORRECT samples, separately for problems where greedy is
wrong but the self-consistency majority is right ("rescued") and for all problems:
    entropy at divergence, its percentile among all greedy-path token entropies,
    share of divergences at fork tokens (entropy above the 80th percentile, Wang et al. 2025),
    rank of the sample's token under the greedy distribution (1 = would have been greedy),
    divergence position (tokens into the answer).
Runs on CPU by default (the base model only scores the greedy paths).

    python -m analysis.fork_divergence
"""

import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import prompts
from common.io import read_jsonl, write_json
from common.models import resolve

R = "results/screen/olmo2-1b-base/dev"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--greedy", default=f"{R}/greedy_prefix-let-s-think-step-by-step")
    ap.add_argument("--samples", default=f"{R}/sample_prefix-let-s-think-step-by-step_T0.7_n10")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/fork_divergence.json")
    args = ap.parse_args()

    hf_id, fmt = resolve("olmo2-1b-base")
    tok = AutoTokenizer.from_pretrained(hf_id)
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype).to(args.device).eval()
    torch.set_num_threads(32)

    G = read_jsonl(f"{args.greedy}/dev.generations.jsonl")
    S = read_jsonl(f"{args.samples}/dev.generations.jsonl")
    all_H, recs = [], []
    for i, (g, s) in enumerate(zip(G, S)):
        prompt = prompts.build_prompt(fmt, g["question"], [], tok)
        p_ids = tok(prompt).input_ids
        enc = lambda text: tok(" " + text, add_special_tokens=False).input_ids
        g_ids = enc(g["paths"][0]["output"])
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([p_ids + g_ids], device=args.device)).logits[0, len(p_ids) - 1:].float(), -1)
        H = -(lp.exp() * lp).sum(-1).cpu().numpy()          # H[t] = entropy before greedy token t
        all_H.extend(H[:len(g_ids)].tolist())
        greedy_ok = g["paths"][0]["score"]["lenient_correct"]
        sc_ok = s["paths"][s["chosen"]]["score"]["lenient_correct"]
        for x in s["paths"]:
            s_ids = enc(x["output"])
            t = next((k for k in range(min(len(s_ids), len(g_ids))) if s_ids[k] != g_ids[k]), None)
            if t is None:
                continue  # identical prefix up to the shorter length (no divergence inside the greedy path)
            row = lp[t]
            recs.append({"problem": i, "correct": bool(x["score"]["lenient_correct"]), "greedy_ok": bool(greedy_ok),
                         "rescued": bool(sc_ok and not greedy_ok), "t": t, "H": float(H[t]),
                         "p_tok": float(row[s_ids[t]].exp()), "rank": int((row > row[s_ids[t]]).sum()) + 1,
                         "greedy_tok": tok.decode([g_ids[t]]), "sample_tok": tok.decode([s_ids[t]])})
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(G)}", flush=True)

    all_H = np.array(all_H)
    fork_thr = float(np.quantile(all_H, 0.8))

    def summ(rs):
        if not rs:
            return {"n": 0}
        h = np.array([r["H"] for r in rs])
        return {"n": len(rs), "entropy_mean": float(h.mean()), "entropy_median": float(np.median(h)),
                "entropy_percentile_median": float(np.median([(all_H < v).mean() for v in h])),
                "fork_share": float((h > fork_thr).mean()),
                "rank_dist": {k: float(np.mean([r["rank"] == k for r in rs])) for k in (2, 3)} |
                             {"4+": float(np.mean([r["rank"] >= 4 for r in rs]))},
                "position_median": float(np.median([r["t"] for r in rs])),
                "position_first5_share": float(np.mean([r["t"] < 5 for r in rs]))}

    groups = {
        "rescued_correct": [r for r in recs if r["rescued"] and r["correct"]],
        "rescued_incorrect": [r for r in recs if r["rescued"] and not r["correct"]],
        "greedywrong_correct": [r for r in recs if not r["greedy_ok"] and r["correct"]],
        "greedywrong_incorrect": [r for r in recs if not r["greedy_ok"] and not r["correct"]],
        "all_correct": [r for r in recs if r["correct"]],
        "all_incorrect": [r for r in recs if not r["correct"]],
    }
    res = {"fork_threshold_nats": fork_thr, "greedy_token_entropy_mean": float(all_H.mean()),
           "groups": {k: summ(v) for k, v in groups.items()},
           "top_divergence_tokens_rescued_correct": _top(groups["rescued_correct"])}
    write_json(args.out, res)
    print(json.dumps(res, indent=1))


def _top(rs, k=15):
    from collections import Counter
    return Counter(f'{r["greedy_tok"]!r} -> {r["sample_tok"]!r}' for r in rs).most_common(k)


if __name__ == "__main__":
    main()
