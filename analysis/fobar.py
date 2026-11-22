"""Backward verification for label-free answer selection (FOBAR, Jiang et al. 2023, arXiv 2308.07758;
self-verification, Weng et al. 2022, arXiv 2212.09561), with the SAME small model.

For each question and each distinct candidate answer A in a pool (e.g. the first k of the model's
samples, or fork-MBR branches): mask one number of the question at a time as "X", tell the model
"If we know the answer to the above question is A, what is the value of unknown variable X?",
sample a few backward solutions, and check whether they recover the masked number.

    backward(A) = fraction of backward samples (over masks) that recover the masked number
    FOBAR score(A) = forward_share(A)^alpha * (backward(A) + eps)^(1 - alpha)      (alpha = 0.5)

Rules reported: majority (forward only), backward-only among pool answers, FOBAR; plus rescue/harm
vs. majority and the oracle. Only questions whose pool has >= 2 distinct answers need backward
calls; the others keep the unanimous answer. Uses vLLM (implicit-vllm env).

    python -m analysis.fobar --pool sampling --k 8
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict

from common import prompts
from common.io import read_jsonl, write_json
from common.models import resolve
from tasks import get_task

NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])")
BACK = "If we know the answer to the above question is {a}, what is the value of unknown variable X?"


def masks(question, rng, m):
    spans = [x for x in NUM.finditer(question)]
    rng.shuffle(spans)
    out = []
    for x in spans[:m]:
        out.append((question[:x.start()] + "X" + question[x.end():], x.group().replace(",", "")))
    return out


def load_pool(args):
    if args.pool == "sampling":
        rows = read_jsonl(args.sampling_gens)
        return [(r["question"], r["answer"], [s["lenient_answer"] for s in r["scores"][:args.k]],
                 [s["lenient_correct"] for s in r["scores"][:args.k]]) for r in rows]
    rows = read_jsonl(args.fork_gens)
    return [(r["question"], r["answer"], [p["score"]["lenient_answer"] for p in r["paths"][:1 + args.k]],
             [p["score"]["lenient_correct"] for p in r["paths"][:1 + args.k]]) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="olmo2-1b-sft")
    ap.add_argument("--pool", choices=["sampling", "fork"], default="sampling")
    ap.add_argument("--sampling_gens", default="results/olmo2-1b-sft/gsm8k/0shot_sub500_n64_t0.7/test.generations.jsonl")
    ap.add_argument("--fork_gens", default="results/screen/olmo2-1b-sft/test500/fork_mbr_F8_minpos0/test500.generations.jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--masks", type=int, default=3)
    ap.add_argument("--n_back", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_answers", type=int, default=4, help="backward-check the top-N pool answers")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    task = get_task("gsm8k")
    hf_id, fmt = resolve(args.model)
    tok = AutoTokenizer.from_pretrained(hf_id)
    rng = random.Random(0)
    pool = load_pool(args)

    reqs, keys = [], []   # one request per (question, answer, mask)
    for qi, (q, gold, answers, _) in enumerate(pool):
        votes = Counter(a for a in answers if a is not None)
        if len(votes) < 2:
            continue
        ms = masks(q, rng, args.masks)
        for a, _ in votes.most_common(args.max_answers):
            for mq, hidden in ms:
                text = prompts.build_prompt(fmt, mq + " " + BACK.format(a=a), [], tok)
                ids = tok(text, add_special_tokens=(fmt == "plain")).input_ids
                reqs.append(TokensPrompt(prompt_token_ids=ids))
                keys.append((qi, a, task.normalize_number(hidden)))
    print(f"{len(reqs)} backward prompts x {args.n_back} samples", flush=True)

    llm = LLM(model=hf_id, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=4096,
              seed=0)
    sp = SamplingParams(n=args.n_back, temperature=args.temperature, top_p=1.0, max_tokens=384, seed=0)
    outs = llm.generate(reqs, sp)

    hit = defaultdict(lambda: [0, 0])  # (qi, a) -> [recovered, total]
    for (qi, a, hidden), o in zip(keys, outs):
        for c in o.outputs:
            pred = task.verify(c.text, hidden)["lenient_answer"]
            hit[(qi, a)][0] += int(pred == hidden)
            hit[(qi, a)][1] += 1

    res = Counter()
    n = len(pool)
    per_q = []
    for qi, (q, gold, answers, correct) in enumerate(pool):
        votes = Counter(a for a in answers if a is not None)
        maj = votes.most_common(1)[0][0] if votes else None
        if len(votes) < 2:
            picks = {"majority": maj, "backward": maj, "fobar": maj}
        else:
            N = sum(votes.values())
            cand = [a for a, _ in votes.most_common(args.max_answers)]
            bwd = {a: hit[(qi, a)][0] / max(1, hit[(qi, a)][1]) for a in cand}
            fob = {a: (votes[a] / N) ** args.alpha * (bwd[a] + 1e-3) ** (1 - args.alpha) for a in cand}
            picks = {"majority": maj,
                     "backward": max(cand, key=lambda a: (bwd[a], votes[a])),
                     "fobar": max(cand, key=lambda a: (fob[a], votes[a]))}
        for rname, p in picks.items():
            res[rname] += p == gold
        res["oracle"] += any(correct)
        for rname in ("backward", "fobar"):
            res[f"rescue_{rname}"] += (picks[rname] == gold) and (picks["majority"] != gold)
            res[f"harm_{rname}"] += (picks[rname] != gold) and (picks["majority"] == gold)
        per_q.append({"question": q, "answer": gold, "picks": picks})
    summary = {k: (v / n if not k.startswith(("rescue", "harm")) else v) for k, v in res.items()}
    summary.update({"pool": args.pool, "k": args.k, "masks": args.masks, "n_back": args.n_back,
                    "problems": n, "backward_prompts": len(reqs)})
    out = args.out or f"results/fobar/{args.model}_{args.pool}_k{args.k}.json"
    write_json(out, {"summary": summary, "per_question": per_q})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
