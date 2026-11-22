"""Budget forcing (Muennighoff et al. 2025, s1), adapted to a base model.

Greedy decode (optionally after --prefix). Whenever the model stops (EOS or a stop string)
before max_new_tokens, cut at the stop, append --wait_text and continue, up to --rounds times.
The answer is read from the text after the last forced "Wait" if it states a number there,
otherwise from the whole output.
"""

import torch

from common import prompts
from methods.common import prefix_slug, make_path, manual_decode, single_path_summary


def tag(args):
    p = prefix_slug(args.prefix)
    return f"prefix-{p}_r{args.rounds}"


def run(model, tok, texts, answers, task, args, ends):
    wait = args.wait_text
    res = []
    bs = args.batch_problems * 8
    for b in range(0, len(texts), bs):
        batch = [t + args.prefix for t in texts[b:b + bs]]
        cont = [""] * len(batch)  # generated text so far (after the prompt + prefix)
        used = [0] * len(batch)
        for r in range(args.rounds + 1):
            gen = manual_decode(model, tok, [t + c for t, c in zip(batch, cont)],
                                args.max_new_tokens, lambda lg, hs, g: lg.argmax(-1), ends)
            for i, ids in enumerate(gen):
                piece = prompts.truncate(tok.decode(ids, skip_special_tokens=True), "plain")
                cont[i] = (cont[i] + " " + piece).strip() if cont[i] else piece
                used[i] += len(ids)
                if r < args.rounds:
                    cont[i] = cont[i] + wait
        cont = [c[: -len(wait)] if c.endswith(wait) else c for c in cont]  # drop trailing unused wait

        def last_segment(out):
            """Text after the last forced wait, cut to its last complete sentence (the forced
            round often loops until max_new_tokens); falls back to the whole output."""
            seg = out.split(wait.strip())[-1]
            if "." in seg.rstrip(".") and not seg.rstrip().endswith("."):
                seg = seg[: seg.rstrip().rfind(".") + 1]
            return seg if any(ch.isdigit() for ch in seg) else out

        for i, c in enumerate(cont):
            p = make_path(tok, [], task, answers[b + i], args.max_new_tokens, prefix=(args.prefix.strip() + " " + c).strip(),
                          score_text=last_segment, extra={"total_gen_tokens": used[i]})
            p["gen_tokens"] = used[i]
            res.append({"paths": [p], "chosen": 0})
        print(f"{b + len(batch)}/{len(texts)}", flush=True)
    return res


summarize = single_path_summary
