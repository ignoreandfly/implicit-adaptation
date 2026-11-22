"""Fork-branch self-training data (vLLM): minimal, on-policy corrections of the greedy path at its closest calls.

For each GSM8K TRAINING question:
  1. n_label samples (T = 0.7) -> label answers (for the label-free / majority variant; gold answers are kept too)
  2. greedy decode with top-2 logprobs; the F positions with the smallest top-2 margin are the forks
  3. at each fork, take the 2nd-ranked token and continue: greedily (one branch) and, with --m > 0, also m
     sampled continuations (T = 0.7)
Output (evaluation.generate-style generations, so train/rft_lora.py and train/decision_lora.py read it directly):
  {"question", "answer", "label_answers": [...], "outputs": [greedy, branch_1, ...], "scores": [verify(...)],
   "fork_pos": [null, t_1, ...], "kind": ["greedy", "branch", ...]}
Typical uses:
  - fork-branch RFT: train.rft_lora on these files (keeps the correct / majority-agreeing outputs, i.e. the greedy
    path when right and the corrected branches otherwise)
  - branch-contrastive: train.decision_lora on these files (correct branches positive, failing greedy negative)

    python -m train.fork_branch_data --model olmo2-1b-sft --n 3000 --forks 4 \
        --out results/fork_branch/olmo2-1b-sft_train3000
"""

import argparse
import math
from pathlib import Path

from common import prompts
from common.io import read_jsonl, write_json, write_jsonl
from common.models import resolve
from tasks import get_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="olmo2-1b-sft")
    ap.add_argument("--format", choices=["plain", "chat"], default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--n_label", type=int, default=8)
    ap.add_argument("--forks", type=int, default=4)
    ap.add_argument("--m", type=int, default=0, help="extra sampled continuations per branch")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    task = get_task("gsm8k")
    hf_id, fmt = resolve(args.model, args.format)
    tok = AutoTokenizer.from_pretrained(hf_id)
    rows = read_jsonl(f"data/gsm8k/{args.split}.jsonl")[args.offset:args.offset + args.n]
    stop = prompts.STOP_STRINGS if fmt == "plain" else None
    llm = LLM(model=hf_id, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=4096, enable_prefix_caching=True, seed=0)
    p_ids = [tok(prompts.build_prompt(fmt, r["question"], [], tok) + args.prefix,
                 add_special_tokens=(fmt == "plain")).input_ids for r in rows]
    TP = lambda ids: TokensPrompt(prompt_token_ids=ids)
    ans = lambda text, gold: task.verify(text, gold)
    pre = args.prefix.strip()
    full = lambda t: (pre + " " + t).strip() if pre else t
    trunc = lambda t: prompts.truncate(t, fmt)

    lab = llm.generate([TP(p) for p in p_ids], SamplingParams(n=args.n_label, temperature=args.temperature, top_p=1.0,
                                                              max_tokens=args.max_tokens, seed=1, stop=stop))
    gr = llm.generate([TP(p) for p in p_ids], SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                                                             logprobs=2, stop=stop))
    greedy, forks = [], []
    for o in gr:
        c = o.outputs[0]; ids = list(c.token_ids); cand = []
        for t, lp in enumerate(c.logprobs or []):
            top = sorted(lp.items(), key=lambda kv: -kv[1].logprob)[:2]
            if len(top) == 2 and top[0][0] == ids[t]:
                cand.append((math.exp(top[0][1].logprob) - math.exp(top[1][1].logprob), t, top[1][0]))
        greedy.append((ids, c.text)); forks.append(sorted(cand)[:args.forks])

    reqs, keys = [], []
    for qi, (p, (ids, _), fk) in enumerate(zip(p_ids, greedy, forks)):
        for _, t, t2 in fk:
            reqs.append(TP(p + ids[:t] + [t2])); keys.append((qi, t, t2, "branch"))
    outs = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop=stop)) if reqs else []
    samp = []
    if args.m > 0 and reqs:
        samp = llm.generate(reqs, SamplingParams(n=args.m, temperature=args.temperature, top_p=1.0,
                                                 max_tokens=args.max_tokens, seed=2, stop=stop))
    per_q = {i: [] for i in range(len(rows))}
    for k, (qi, t, t2, _) in enumerate(keys):
        head = tok.decode(greedy[qi][0][:t] + [t2], skip_special_tokens=True)
        per_q[qi].append((t, "branch", trunc(head + outs[k].outputs[0].text)))
        if samp:
            for c in samp[k].outputs:
                per_q[qi].append((t, "branch_sample", trunc(head + c.text)))

    res, n_fix = [], 0
    for qi, r in enumerate(rows):
        g_text = trunc(greedy[qi][1])
        texts = [full(g_text)] + [full(x) for _, _, x in per_q[qi]]
        scores = [ans(x, r["answer"]) for x in texts]
        n_fix += int(not scores[0]["lenient_correct"] and any(s["lenient_correct"] for s in scores[1:]))
        res.append({"question": r["question"], "answer": r["answer"],
                    "label_answers": [ans(trunc(c.text), r["answer"])["lenient_answer"] for c in lab[qi].outputs],
                    "outputs": texts, "scores": scores, "fork_pos": [None] + [t for t, _, _ in per_q[qi]],
                    "kind": ["greedy"] + [k for _, k, _ in per_q[qi]]})
    out = Path(args.out)
    write_jsonl(out / f"{args.split}.generations.jsonl", res)
    g_acc = sum(x["scores"][0]["lenient_correct"] for x in res) / len(res)
    write_json(out / f"{args.split}.metrics.json", {"questions": len(res), "greedy_acc": g_acc,
               "greedy_wrong_but_a_branch_right": n_fix, "args": vars(args)})
    print(f"wrote {len(res)} rows; greedy acc {g_acc:.3f}; questions where greedy is wrong but a branch is right: {n_fix}")


if __name__ == "__main__":
    main()
