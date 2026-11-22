"""Route 2: derive single-pass decoding rules from the model's OWN samples (no RL model).

On GSM8K TRAINING problems, take N samples per problem from the model (e.g. SFT, T=0.7) and label
each sample "good" or "bad":
    --label majority   good = agrees with the majority answer of its N samples (no gold labels)
    --label gold       good = correct against the GSM8K training answer
Teacher-force every sample through the same model to find its fork positions (next-token entropy
above the 80th percentile over all positions). For each token w chosen at a fork, count how often it
appears in good vs bad samples and compute a smoothed log-odds ratio

    LOR(w) = log((g_w + a) / (G + a)) - log((b_w + a) / (B + a))

(G, B = total fork tokens in good / bad samples). Tokens seen fewer than --min_count times are
dropped. The first response token gets its own table (decisions there differ most).
Output JSON: {fork_thr, bias: {token_id: LOR}, first_bias: {...}, ...}, consumed by
methods/rl_rules.py --rl_bias_json (applied at forks as greedy on log p + beta * clip(LOR)).

    python -m analysis.self_rules --gens results/rl_delta/olmo2-1b-sft_train300_n16 --label majority
"""

import argparse
import math
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import prompts
from common.io import read_jsonl, write_json
from common.models import resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="olmo2-1b-sft")
    ap.add_argument("--label", choices=["majority", "gold"], default="majority")
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--smooth", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    hf_id, fmt = resolve(args.model)
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16).cuda().eval()
    rows = read_jsonl(f"{args.gens}/{args.split}.generations.jsonl")

    recs = []  # (good, position, entropy, token)
    for i, r in enumerate(rows):
        prompt = prompts.build_prompt(fmt, r["question"], [], tok)
        p_ids = tok(prompt, add_special_tokens=(fmt == "plain")).input_ids
        answers = [s["lenient_answer"] for s in r["scores"]]
        maj = Counter(a for a in answers if a is not None).most_common(1)
        maj = maj[0][0] if maj else None
        for out, s in zip(r["outputs"], r["scores"]):
            good = s["lenient_correct"] if args.label == "gold" else (s["lenient_answer"] is not None and s["lenient_answer"] == maj)
            o_ids = tok(out, add_special_tokens=False).input_ids[:512]
            if not o_ids:
                continue
            with torch.no_grad():
                lp = torch.log_softmax(model(torch.tensor([p_ids + o_ids], device="cuda")).logits[0, len(p_ids) - 1:-1].float(), -1)
            H = (-(lp.exp() * lp).sum(-1)).cpu().numpy()
            for t, (h, w) in enumerate(zip(H, o_ids)):
                recs.append((bool(good), t, float(h), w))
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)

    H_all = np.array([x[2] for x in recs])
    thr = float(np.quantile(H_all, 0.8))

    def table(sel):
        g, b = Counter(), Counter()
        for good, t, h, w in sel:
            (g if good else b)[w] += 1
        G, B, a = sum(g.values()), sum(b.values()), args.smooth
        out = {}
        for w in set(g) | set(b):
            if g[w] + b[w] >= args.min_count:
                out[int(w)] = math.log((g[w] + a) / (G + a)) - math.log((b[w] + a) / (B + a))
        return out, G, B

    bias, G, B = table([x for x in recs if x[1] > 0 and x[2] > thr])
    first, G1, B1 = table([x for x in recs if x[1] == 0])
    show = lambda d: [(tok.decode([w]), round(v, 2)) for w, v in sorted(d.items(), key=lambda kv: -kv[1])]
    res = {"model": args.model, "label": args.label, "gens": args.gens, "fork_thr": thr,
           "fork_tokens_good": G, "fork_tokens_bad": B, "first_tokens_good": G1, "first_tokens_bad": B1,
           "bias": bias, "first_bias": first,
           "top_good": show(bias)[:25], "top_bad": show(bias)[-25:], "first_ranked": show(first)}
    out = args.out or f"{args.gens}/self_rules_{args.label}.json"
    write_json(out, res)
    print(f"fork thr {thr:.3f}; fork tokens good/bad {G}/{B}; first good/bad {G1}/{B1}")
    print("favoured at forks:", res["top_good"][:15])
    print("disfavoured at forks:", res["top_bad"][-15:])
    print("first token:", res["first_ranked"])


if __name__ == "__main__":
    main()
