"""Greedy decoding with an optional forced answer prefix and repetition penalty.

--prefix " Let's think step by step."   zero-shot CoT (Kojima et al. 2022)
--prefix " First,"                      forced first token(s): one hand-picked CoT path
--rep_penalty 1.2                       control for DoLa's repetition penalty
"""

from methods.common import prefix_slug, apply_repetition_penalty, make_path, manual_decode, single_path_summary


def tag(args):
    p = prefix_slug(args.prefix)
    return f"prefix-{p}" + (f"_rep{args.rep_penalty}" if args.rep_penalty != 1.0 else "")


def run(model, tok, texts, answers, task, args, ends):
    texts = [t + args.prefix for t in texts]
    res = []
    for b in range(0, len(texts), args.batch_problems * 8):
        batch = texts[b:b + args.batch_problems * 8]
        gen = manual_decode(model, tok, batch, args.max_new_tokens,
                            lambda lg, hs, g: apply_repetition_penalty(lg, g, args.rep_penalty).argmax(-1), ends)
        for i, ids in enumerate(gen):
            res.append({"paths": [make_path(tok, ids, task, answers[b + i], args.max_new_tokens,
                                            prefix=args.prefix.lstrip())], "chosen": 0})
        print(f"{b + len(batch)}/{len(texts)}", flush=True)
    return res


summarize = single_path_summary
