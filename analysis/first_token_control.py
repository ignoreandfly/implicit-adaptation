"""Control for first-token branching vs temperature voting (1B SFT, chat, vLLM): is the gain the forks or just greedy paths?

Held-out TRAIN questions (disjoint from the global_forks pool). Same pipeline for all conditions, majority vote over the
first k paths (ties go to the earliest path):
  samp_T{t}_s{seed}   10 samples at temperature t
  g+samp_T{t}         greedy path first, then 9 samples (seed 0): the path order fork methods use
  ft_greedy           top-10 first tokens of the question, each continued greedily
  ft_T{t}             top-10 first tokens, each continued at temperature t
Differences vs ft_greedy: sampling is averaged over seeds, 95% bootstrap CI over questions.

    CUDA_VISIBLE_DEVICES=6 $VPY -m analysis.first_token_control --n_questions 1000 --out results/first_token_control
"""
import argparse, json, random
from collections import Counter
from pathlib import Path
import numpy as np
from common import prompts
from common.io import read_jsonl, write_json, write_jsonl
from common.models import resolve
from tasks import get_task

KS = [1, 2, 4, 6, 8, 10]; TEMPS = [0.3, 0.5, 0.7]; SEEDS = [0, 1, 2]; K = 10


def vote(paths):
    v = Counter(p["answer"] for p in paths if p["answer"] is not None)
    if not v:
        return None
    m = max(v.values())
    return next(p["answer"] for p in paths if v.get(p["answer"]) == m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="olmo2-1b-sft"); ap.add_argument("--n_questions", type=int, default=1000)
    ap.add_argument("--max_tokens", type=int, default=512); ap.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    task = get_task("gsm8k"); rows = read_jsonl("data/gsm8k/train.jsonl")
    pool = set(random.Random(0).sample(range(len(rows)), 500))          # used by analysis.global_forks
    idx = sorted(random.Random(1).sample([i for i in range(len(rows)) if i not in pool], a.n_questions))
    hf_id, fmt = resolve(a.model); assert fmt == "chat"
    llm = LLM(model=hf_id, dtype="bfloat16", gpu_memory_utilization=a.gpu_memory_utilization, max_model_len=4096, seed=0,
              enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    P = [tok(prompts.build_prompt(fmt, rows[i]["question"], [], tok), add_special_tokens=False).input_ids for i in idx]

    def rec(i, pre, c):
        text = prompts.truncate(tok.decode(pre + list(c.token_ids), skip_special_tokens=True), fmt)
        s = task.verify(text, rows[i]["answer"])
        return {"answer": s["lenient_answer"], "tokens": len(c.token_ids) + len(pre), "output": text}

    def gen(prefixes, n, T, seed):   # prefixes: per question list of forced token lists -> per question list of paths
        reqs = [(q, pre) for q in range(len(P)) for pre in prefixes[q]]
        sp = SamplingParams(n=n, temperature=T, top_p=1.0, max_tokens=a.max_tokens, seed=seed)
        outs = llm.generate([{"prompt_token_ids": P[q] + pre} for q, pre in reqs], sp)
        res = [[] for _ in P]
        for (q, pre), o in zip(reqs, outs):
            res[q] += [rec(idx[q], pre, c) for c in o.outputs]
        return res

    none = [[[]] for _ in P]
    first = llm.generate([{"prompt_token_ids": p} for p in P], SamplingParams(temperature=0, max_tokens=1, logprobs=K))
    ft = [[[t] for t, _ in sorted(o.outputs[0].logprobs[0].items(), key=lambda x: -x[1].logprob)[:K]] for o in first]
    C = {"greedy": gen(none, 1, 0.0, 0), "ft_greedy": gen(ft, 1, 0.0, 0)}
    for T in TEMPS:
        C[f"ft_T{T}"] = gen(ft, 1, T, 0)
        for s in SEEDS:
            C[f"samp_T{T}_s{s}"] = gen(none, K, T, s)
        C[f"g+samp_T{T}"] = [g + r[:K - 1] for g, r in zip(C["greedy"], C[f"samp_T{T}_s0"])]
    out = Path(a.out)
    write_jsonl(out / "train.generations.jsonl", [{"idx": i, "gold": rows[i]["answer"], "paths": {c: C[c][q] for c in C if not c.startswith("g+")}}
                                                  for q, i in enumerate(idx)])
    G = [rows[i]["answer"] for i in idx]; ks = KS
    ok = {c: {k: np.array([vote(r[:k]) == g for r, g in zip(C[c], G)], float) for k in ks if k <= len(C[c][0])} for c in C}
    cov = {c: {k: float(np.mean([any(p["answer"] == g for p in r[:k]) for r, g in zip(C[c], G)])) for k in ok[c]} for c in C}
    tk = {c: {k: float(np.mean([sum(p["tokens"] for p in r[:k]) for r in C[c]])) for k in ok[c]} for c in C}
    S = {}
    for c in C:
        if not c.endswith(("_s0", "_s1", "_s2")):
            S[c] = {k: ok[c][k] for k in ok[c]}
    for T in TEMPS:
        S[f"samp_T{T}"] = {k: np.mean([ok[f"samp_T{T}_s{s}"][k] for s in SEEDS], 0) for k in ks}
    rng = np.random.default_rng(0); n = len(G); B = rng.integers(0, n, (4000, n))
    table = {}
    print("condition        " + "".join(f"   k={k:<4d}" for k in ks))
    for c in S:
        table[c] = {k: {"vote": float(S[c][k].mean())} for k in S[c]}
        print(f"{c:16s} " + "".join(f"   {S[c][k].mean():.3f} " if k in S[c] else "         " for k in ks))
    print("ft_greedy minus condition [95% bootstrap CI]")
    for c in S:
        if c in ("ft_greedy", "greedy"):
            continue
        line = ""
        for k in ks:
            d = S["ft_greedy"][k] - S[c][k]; lo, hi = np.percentile(d[B].mean(1), [2.5, 97.5])
            table[c][k]["ft_greedy_minus"] = [float(d.mean()), float(lo), float(hi)]
            line += f" {d.mean():+.3f}[{lo:+.3f},{hi:+.3f}]"
        print(f"{c:16s}{line}")
    write_json(out / "train.metrics.json", {"table": table, "coverage": cov, "tokens": tk, "args": vars(a)})
    print("coverage@6", {c: round(v[6], 3) for c, v in cov.items() if 6 in v}); print("tokens@6", {c: round(v[6]) for c, v in tk.items() if 6 in v})
    print("CONTROL_DONE")


if __name__ == "__main__":
    main()
