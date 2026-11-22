"""Data for fork-contrastive self-distillation (vLLM; implicit-vllm env).

For each GSM8K TRAINING question (never dev/test):
  1. label samples: n_label samples at T -> their answers (majority label is computed later, so the
     label samples are separate from the fork continuations that get scored).
  2. greedy decode with top-2 logprobs at every step; the F positions with the smallest top-2
     probability margin are the forks.
  3. at each fork, for both the greedy token (top-1) and the 2nd-ranked token, sample m continuations
     from (prompt + greedy prefix + token) -> their final answers.
Output jsonl, one row per question: question, gold, prompt_ids, greedy_ids, label_answers,
forks: [{t, tok1, tok2, margin, ans1: [...], ans2: [...]}].

    python -m train.fork_pairs --model olmo2-1b-sft --n 1000 --out data/fork_pairs/olmo2-1b-sft_train1000.jsonl
"""

import argparse
import math

from common import prompts
from common.io import read_jsonl, write_jsonl
from common.models import resolve
from tasks import get_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="olmo2-1b-sft")
    ap.add_argument("--split", default="train")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n_label", type=int, default=8)
    ap.add_argument("--forks", type=int, default=4)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    task = get_task("gsm8k")
    hf_id, fmt = resolve(args.model)
    tok = AutoTokenizer.from_pretrained(hf_id)
    rows = read_jsonl(f"data/gsm8k/{args.split}.jsonl")[args.offset:args.offset + args.n]
    llm = LLM(model=hf_id, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=4096, enable_prefix_caching=True, seed=0)
    p_ids = [tok(prompts.build_prompt(fmt, r["question"], [], tok), add_special_tokens=(fmt == "plain")).input_ids
             for r in rows]
    ans = lambda text, gold: task.verify(text, gold)["lenient_answer"]

    # 1. label samples
    lab = llm.generate([TokensPrompt(prompt_token_ids=p) for p in p_ids],
                       SamplingParams(n=args.n_label, temperature=args.temperature, top_p=1.0,
                                      max_tokens=args.max_tokens, seed=1))
    label_answers = [[ans(c.text, r["answer"]) for c in o.outputs] for o, r in zip(lab, rows)]
    print("label samples done", flush=True)

    # 2. greedy with top-2 logprobs
    gr = llm.generate([TokensPrompt(prompt_token_ids=p) for p in p_ids],
                      SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=2))
    greedy, forks = [], []
    for o in gr:
        c = o.outputs[0]
        ids = list(c.token_ids)
        cand = []
        for t, lp in enumerate(c.logprobs or []):
            top = sorted(lp.items(), key=lambda kv: -kv[1].logprob)[:2]
            if len(top) < 2 or top[0][0] != ids[t]:
                continue
            margin = math.exp(top[0][1].logprob) - math.exp(top[1][1].logprob)
            cand.append((margin, t, top[0][0], top[1][0]))
        greedy.append((ids, c.text))
        forks.append(sorted(cand)[:args.forks])
    print("greedy done", flush=True)

    # 3. continuations from both tokens at each fork
    reqs, keys = [], []
    for qi, (p, (ids, _), fk) in enumerate(zip(p_ids, greedy, forks)):
        for fi, (_, t, t1, t2) in enumerate(fk):
            for which, tk in ((1, t1), (2, t2)):
                reqs.append(TokensPrompt(prompt_token_ids=p + ids[:t] + [tk]))
                keys.append((qi, fi, which, t))
    outs = llm.generate(reqs, SamplingParams(n=args.m, temperature=args.temperature, top_p=1.0,
                                             max_tokens=args.max_tokens, seed=2))
    cont = {}
    for (qi, fi, which, t), o in zip(keys, outs):
        ids = greedy[qi][0]
        tk = forks[qi][fi][1 + which]
        head = tok.decode(ids[:t] + [tk], skip_special_tokens=True)
        cont[(qi, fi, which)] = [ans(head + c.text, rows[qi]["answer"]) for c in o.outputs]

    out = []
    for qi, r in enumerate(rows):
        out.append({"question": r["question"], "gold": r["answer"], "prompt_ids": p_ids[qi],
                    "greedy_ids": greedy[qi][0], "greedy_answer": ans(greedy[qi][1], r["answer"]),
                    "label_answers": label_answers[qi],
                    "forks": [{"t": t, "tok1": t1, "tok2": t2, "margin": m,
                               "ans1": cont[(qi, fi, 1)], "ans2": cont[(qi, fi, 2)]}
                              for fi, (m, t, t1, t2) in enumerate(forks[qi])]})
    write_jsonl(args.out, out)
    print(f"wrote {len(out)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
