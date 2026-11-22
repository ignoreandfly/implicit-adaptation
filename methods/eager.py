"""EAGer: entropy-aware generation (arXiv 2510.11170), Algorithm 1.

Per prompt, start with one sequence. At every step, for each active sequence:
    H = entropy of the top-20 next-token probabilities (renormalised), temperature tau
    if H >= theta and the prompt has < M sequences:
        continue with the top-1 token and spawn ONE new branch starting with the top-2 token
    else:
        sample from the temperature-tau, top-p 0.95 distribution
Branches share their prefix (KV cache rows are duplicated, not recomputed). The final answer is
a majority vote over the finished sequences (the paper's Cons@k). Paper defaults: tau = 0.6,
top-p = 0.95, theta in [1.8, 2.7] (tuned per model for long-CoT models), M = 32; we use M = --n.
"""

from collections import Counter

import torch

from methods.common import make_path, prefix_slug, stop_hit


def tag(args):
    return f"theta{args.eager_theta}_M{args.n}" + (f"_prefix-{prefix_slug(args.prefix)}" if args.prefix else "")


def _top_p_sample(logp, top_p):
    probs = logp.exp()
    sp, si = probs.sort(-1, descending=True)
    keep = sp.cumsum(-1) - sp < top_p
    sp = sp * keep
    choice = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1).squeeze(-1)
    return si.gather(-1, choice[:, None]).squeeze(-1)


def run(model, tok, texts, answers, task, args, ends):
    M, theta, tau = args.n, args.eager_theta, args.eager_tau
    res = []
    for b in range(0, len(texts), args.batch_problems):
        batch = [t + args.prefix for t in texts[b:b + args.batch_problems]]
        P = len(batch)
        enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
        att = enc.attention_mask
        cur, curpos, past = enc.input_ids, (att.cumsum(-1) - 1).clamp_min(0), None
        owner, gens = list(range(P)), [[] for _ in range(P)]
        count, finished, generated = [1] * P, [[] for _ in range(P)], [0] * P
        for _ in range(args.max_new_tokens):
            with torch.no_grad():
                out = model(input_ids=cur, attention_mask=att, position_ids=curpos, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logp = torch.log_softmax(out.logits[:, -1].float() / tau, -1)
            top = logp.topk(20, -1)
            q = torch.softmax(top.values, -1)
            H = (-(q * q.clamp_min(1e-30).log()).sum(-1)).tolist()
            sampled = _top_p_sample(logp, 0.95).tolist()
            top2 = top.indices[:, :2].tolist()
            parents, toks, new_owner, new_gens = [], [], [], []
            for r in range(len(owner)):
                p = owner[r]
                cands = [sampled[r]]
                if H[r] >= theta and count[p] < M:
                    cands = top2
                    count[p] += 1
                for t in cands:
                    generated[p] += 1
                    g = gens[r] + [t]
                    if t in ends:
                        finished[p].append(gens[r])
                    elif stop_hit(tok, g) or len(g) >= args.max_new_tokens:
                        finished[p].append(g)
                    else:
                        parents.append(r); toks.append(t); new_owner.append(p); new_gens.append(g)
            if not parents:
                break
            idx = torch.tensor(parents, device="cuda")
            past.reorder_cache(idx)
            att = torch.cat([att[idx], torch.ones((len(parents), 1), dtype=att.dtype, device="cuda")], 1)
            curpos = curpos[idx][:, -1:] + 1
            cur = torch.tensor(toks, device="cuda")[:, None]
            owner, gens = new_owner, new_gens
        for r, p in enumerate(owner):  # anything still active at the step limit
            finished[p].append(gens[r])
        for p in range(P):
            paths = [make_path(tok, g, task, answers[b + p], args.max_new_tokens, prefix=args.prefix.lstrip())
                     for g in finished[p]]
            votes = Counter(x["score"]["lenient_answer"] for x in paths if x["score"]["lenient_answer"] is not None)
            top_ans = votes.most_common(1)[0][0] if votes else None
            chosen = next((j for j, x in enumerate(paths) if x["score"]["lenient_answer"] == top_ans), 0)
            res.append({"paths": paths, "chosen": chosen, "generated_tokens": generated[p]})
        print(f"{b + P}/{len(texts)}", flush=True)
    return res


def summarize(results, answers):
    n = len(results)
    return {"acc_majority": sum(r["paths"][r["chosen"]]["score"]["lenient_correct"] for r in results) / n,
            "acc_oracle": sum(any(x["score"]["lenient_correct"] for x in r["paths"]) for r in results) / n,
            "acc_single_sample": sum(sum(x["score"]["lenient_correct"] for x in r["paths"]) / len(r["paths"])
                                     for r in results) / n,
            "mean_sequences": sum(len(r["paths"]) for r in results) / n,
            "mean_generated_tokens": sum(r["generated_tokens"] for r in results) / n}
