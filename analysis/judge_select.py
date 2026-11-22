"""Use a pairwise judge to choose between the top answers of a model's own candidate pool.

For each test question, take the first k samples (from a saved n=64 run), group by answer, and:
    majority      plain vote (ties -> earliest)
    judge_close   if the runner-up answer is within --margin votes of the leader, ask the judge to choose
                  between one representative solution of each (both orders, logit difference averaged);
                  otherwise keep the majority answer
    judge_top2    always let the judge decide between the top-2 answers (when there are >= 2)
Compared for: the trained judge (--judge path) and the untrained model as a zero-shot judge (--judge base).
Reports accuracy, rescue / harm vs majority, how often the judge was consulted, and the wrong-majority subset
(questions where majority is wrong but the correct answer is in the pool).

    python -m analysis.judge_select --model olmo2-1b-sft --judge /data/.../judge_sft --k 4 8 16
"""

import argparse
import json
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.io import read_jsonl, write_json
from common.models import resolve
from train.judge_lora import ab_ids, judge_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="olmo2-1b-sft")
    ap.add_argument("--judge", default="base", help="'base' = untrained model as judge, or a path to a judge")
    ap.add_argument("--pool", default="results/olmo2-1b-sft/gsm8k/0shot_sub500_n64_t0.7/test.generations.jsonl")
    ap.add_argument("--k", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--margin", type=int, default=1)
    ap.add_argument("--max_chars", type=int, default=1500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    hf_id, fmt = resolve(args.model)
    tok = AutoTokenizer.from_pretrained(hf_id)
    tid_a, tid_b = ab_ids(tok, fmt)
    path = hf_id if args.judge == "base" else args.judge
    judge = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
    rows = read_jsonl(args.pool)
    cache = {}

    def prefer_first(question, s1, s2):
        key = (question, s1, s2)
        if key not in cache:
            tot = 0.0
            for a, b, sign in ((s1, s2, 1.0), (s2, s1, -1.0)):
                ids = tok(judge_text(fmt, tok, question, a[: args.max_chars], b[: args.max_chars]),
                          add_special_tokens=(fmt == "plain"), return_tensors="pt").input_ids.cuda()
                with torch.no_grad():
                    lg = judge(ids).logits[0, -1].float()
                tot += sign * (lg[tid_a] - lg[tid_b]).item()
            cache[key] = tot
        return cache[key] > 0

    res = {}
    for k in args.k:
        stats = Counter()
        for r in rows:
            outs, sc = r["outputs"][:k], r["scores"][:k]
            gold = r["answer"]
            votes = Counter(s["lenient_answer"] for s in sc if s["lenient_answer"] is not None)
            if not votes:
                continue
            order = votes.most_common()
            maj = order[0][0]
            rep = {a: next(o for o, s in zip(outs, sc) if s["lenient_answer"] == a) for a, _ in order[:2]}
            in_pool = gold in votes
            picks = {"majority": maj}
            if len(order) >= 2:
                a1, a2 = order[0][0], order[1][0]
                jpick = a1 if prefer_first(r["question"], rep[a1], rep[a2]) else a2
                picks["judge_top2"] = jpick
                picks["judge_close"] = jpick if order[0][1] - order[1][1] <= args.margin else maj
                stats["consulted_close"] += order[0][1] - order[1][1] <= args.margin
            else:
                picks["judge_top2"] = picks["judge_close"] = maj
            for name, p in picks.items():
                stats[f"acc_{name}"] += p == gold
                if name != "majority":
                    stats[f"rescue_{name}"] += (p == gold) and (maj != gold)
                    stats[f"harm_{name}"] += (p != gold) and (maj == gold)
            if maj != gold and in_pool:
                stats["wrong_majority_with_gold_in_pool"] += 1
                for name in ("judge_close", "judge_top2"):
                    stats[f"wm_fixed_{name}"] += picks[name] == gold
            stats["oracle"] += in_pool
        n = len(rows)
        res[k] = {key: (v / n if key.startswith(("acc_", "oracle")) else v) for key, v in stats.items()}
        print(f"k={k}: " + json.dumps({key: (round(v, 3) if isinstance(v, float) else v) for key, v in res[k].items()}), flush=True)
    out = args.out or f"results/judge/select_{args.model}_{'base' if args.judge == 'base' else 'trained'}.json"
    write_json(out, {"judge": args.judge, "pool": args.pool, "margin": args.margin, "results": res})


if __name__ == "__main__":
    main()
