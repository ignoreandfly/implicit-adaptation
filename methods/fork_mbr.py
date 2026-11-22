"""Fork-gated MBR: majority voting restricted to the model's own decision points.

1. Greedy decode, recording at every step the top-2 tokens and their probability margin.
2. Take the F positions with the smallest margin (the closest calls) along the greedy path.
3. At each, replace the greedy token by the 2nd-ranked token and continue greedily to the end
   (RL's choice at forks is usually SFT's 2nd choice; see analysis/rl_delta.py).
4. Answer = majority vote over {greedy answer} + {F branch answers}; ties go to the greedy answer
   (a single disagreeing branch can never override greedy).
Cost = (1 + F) greedy decodes (branches reuse no computation here; a prefix cache would make them
cheaper). Ordering branches by margin means one run with --fm_forks F gives the results for
1..F (reported in the metrics as acc_F1 .. acc_F<F>, with rescue/harm counts vs greedy).
--fm_where step_start: only consider forks at the start of a reasoning step (first response token
or right after a sentence end / newline), where RL's changes concentrate (analysis/rl_delta.py);
"any" considers every position. Works for plain and chat models; no training, no labels.
"""

from collections import Counter

import torch

from methods.common import make_path


def tag(args):
    return f"F{args.fm_forks}_minpos{args.fm_min_pos}" + ("" if args.fm_where == "any" else f"_{args.fm_where}")


SENT_END = (".", "!", "?", "\n", ":")


def _step_start(tok, ids, t):
    return t == 0 or tok.decode(ids[:t]).rstrip(" ").endswith(SENT_END)


def _greedy(model, tok, id_lists, max_new, ends, fmt):
    """Greedy-complete each id list (left-padded batch). Returns new token ids and per-step
    (top1, top2, margin) for every sequence."""
    pad = tok.pad_token_id
    L = max(len(x) for x in id_lists)
    ids = torch.tensor([[pad] * (L - len(x)) + x for x in id_lists], device="cuda")
    att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in id_lists], device="cuda")
    B = len(id_lists)
    cur, pos, past = ids, (att.cumsum(-1) - 1).clamp_min(0), None
    gen, tops, done = [[] for _ in range(B)], [[] for _ in range(B)], [False] * B
    for _ in range(max_new):
        with torch.no_grad():
            out = model(input_ids=cur, attention_mask=att, position_ids=pos, past_key_values=past, use_cache=True,
                        logits_to_keep=1)  # only the last position's logits (the prompt pass is long)
        past = out.past_key_values
        p = torch.softmax(out.logits[:, -1].float(), -1)
        v, ix = p.topk(2, -1)
        v, ix = v.tolist(), ix.tolist()
        for b in range(B):
            if done[b]:
                continue
            t = ix[b][0]
            if t in ends:
                done[b] = True
                continue
            gen[b].append(t)
            tops[b].append((ix[b][0], ix[b][1], v[b][0] - v[b][1]))
            if fmt == "plain":
                from methods.common import stop_hit
                if stop_hit(tok, gen[b]):
                    done[b] = True
        if all(done):
            break
        nxt = torch.tensor([pad if done[b] else ix[b][0] for b in range(B)], device="cuda")
        cur = nxt[:, None]
        att = torch.cat([att, torch.ones((B, 1), dtype=att.dtype, device="cuda")], 1)
        pos = pos[:, -1:] + 1
    return gen, tops


def run(model, tok, texts, answers, task, args, ends):
    fmt, F = args.fmt, args.fm_forks
    res = []
    bs = args.batch_problems * 4
    for b in range(0, len(texts), bs):
        batch = texts[b:b + bs]
        p_ids = [tok(t, add_special_tokens=(fmt == "plain")).input_ids for t in batch]
        gen, tops = _greedy(model, tok, p_ids, args.max_new_tokens, ends, fmt)
        # choose fork positions: smallest top-2 margin, at least fm_min_pos tokens into the answer
        branch_inputs, owner = [], []
        forks = []
        for i in range(len(batch)):
            cand = [(m, t) for t, (_, _, m) in enumerate(tops[i]) if t >= args.fm_min_pos
                    and (args.fm_where == "any" or _step_start(tok, gen[i], t))]
            chosen = [t for _, t in sorted(cand)[:F]]
            forks.append(chosen)
            for t in chosen:
                branch_inputs.append(p_ids[i] + gen[i][:t] + [tops[i][t][1]])
                owner.append((i, t))
        bgen = []
        for c in range(0, len(branch_inputs), args.fm_chunk):  # branches in chunks to bound memory
            bgen += _greedy(model, tok, branch_inputs[c:c + args.fm_chunk], args.max_new_tokens, ends, fmt)[0]
        per = {i: [] for i in range(len(batch))}
        for (i, t), g in zip(owner, bgen):
            per[i].append((t, gen[i][:t] + [tops[i][t][1]] + g, tops[i][t][2], len(g) + 1))
        for i in range(len(batch)):
            paths = [make_path(tok, gen[i], task, answers[b + i], args.max_new_tokens, fmt=fmt,
                               extra={"fork_pos": None, "margin": None, "new_tokens": len(gen[i])})]
            for t, ids, m, n_new in sorted(per[i], key=lambda x: x[2]):  # by margin, smallest first
                paths.append(make_path(tok, ids, task, answers[b + i], args.max_new_tokens, fmt=fmt,
                                       extra={"fork_pos": t, "margin": m, "new_tokens": n_new,
                                              "swap": [tok.decode([tops[i][t][0]]), tok.decode([tops[i][t][1]])]}))
            res.append({"paths": paths, "chosen": _vote(paths, len(paths) - 1)})
        print(f"{b + len(batch)}/{len(texts)}", flush=True)
    return res


def _vote(paths, f):
    """Index of the path whose answer wins a majority vote over greedy + first f branches;
    ties (including all-different) go to greedy."""
    cand = paths[:1 + f]
    votes = Counter(x["score"]["lenient_answer"] for x in cand if x["score"]["lenient_answer"] is not None)
    if not votes:
        return 0
    g = cand[0]["score"]["lenient_answer"]
    best, n = votes.most_common(1)[0]
    if g is not None and votes[g] == n:
        return 0
    return next(j for j, x in enumerate(cand) if x["score"]["lenient_answer"] == best)


def summarize(results, answers):
    n = len(results)
    out = {}
    greedy_ok = [r["paths"][0]["score"]["lenient_correct"] for r in results]
    out["acc_greedy"] = sum(greedy_ok) / n
    F = max(len(r["paths"]) - 1 for r in results)
    for f in range(1, F + 1):
        ok = [r["paths"][_vote(r["paths"], f)]["score"]["lenient_correct"] for r in results]
        out[f"acc_F{f}"] = sum(ok) / n
        out[f"rescue_F{f}"] = sum(o and not g for o, g in zip(ok, greedy_ok))
        out[f"harm_F{f}"] = sum(g and not o for o, g in zip(ok, greedy_ok))
        out[f"cost_F{f}"] = sum(sum(x["new_tokens"] for x in r["paths"][:1 + f]) for r in results) / \
            max(1, sum(r["paths"][0]["new_tokens"] for r in results))
    out["acc_oracle"] = sum(any(x["score"]["lenient_correct"] for x in r["paths"]) for r in results) / n
    return out
