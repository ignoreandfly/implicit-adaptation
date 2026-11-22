"""Power-SMC (Azizi et al. 2026, arXiv 2602.10273, Algorithm 1): sequential Monte Carlo
targeting the sequence-level power distribution pi_alpha(y|x) ∝ p(y|x)^alpha.

Per problem, N particles decode in one batch. At token t:
    proposal  q_t = softmax(alpha_t * log p_t)        (locally optimal, = temperature 1/alpha_t)
    weight    log W += alpha_t * log p_t(y_t) - log q_t(y_t)   (= log sum_v p_t(v)^alpha_t)
    ramp      alpha_t = 1 + (alpha - 1) * min(1, t / T_ramp); when alpha_t rises from the previous
              value, log W += (alpha_t - alpha_{t-1}) * log p(y_{<t})  (stage-boundary correction)
    resample  if ESS = 1 / sum(w_norm^2) < kappa * N: systematic resampling of the particles,
              KV-cache rows reordered by the ancestor indices, weights reset
Finished particles (EOS / stop string / max length) are absorbing and keep their weight.
Output: one particle drawn in proportion to its final weight (the paper's rule). We also
report the max-weight particle and a majority vote over the N final particles.
Paper defaults: alpha = 4, N = 64, kappa = 0.5, T_ramp = 100.
"""

from collections import Counter

import numpy as np
import torch

from methods.common import make_path, prefix_slug, stop_hit


def tag(args):
    return f"a{args.smc_alpha}_N{args.n}_ramp{args.smc_ramp}" + \
        (f"_prefix-{prefix_slug(args.prefix)}" if args.prefix else "")


def systematic(w, rng):
    N = len(w)
    pos = (rng.random() + np.arange(N)) / N
    return np.minimum(np.searchsorted(np.cumsum(w), pos), N - 1)


def run(model, tok, texts, answers, task, args, ends):
    N, alpha, kappa, T_ramp = args.n, args.smc_alpha, args.smc_kappa, args.smc_ramp
    rng = np.random.default_rng(args.seed)
    res, resample_events = [], 0
    for b in range(0, len(texts), args.batch_problems):
        batch = [t + args.prefix for t in texts[b:b + args.batch_problems]]
        P = len(batch)
        enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
        rows = torch.arange(P, device="cuda").repeat_interleave(N)  # row r belongs to problem r // N
        att = enc.attention_mask[rows]
        cur, curpos, past = enc.input_ids[rows], (att.cumsum(-1) - 1).clamp_min(0), None
        R = P * N
        gens = [[] for _ in range(R)]
        done = np.zeros(R, bool)
        logW = np.zeros(R)        # log importance weight
        cum_logp = np.zeros(R)    # log p(y_{<t}) under the base model
        prev_alpha = 1.0
        generated = np.zeros(P, int)
        for t in range(args.max_new_tokens):
            a_t = 1.0 + (alpha - 1.0) * min(1.0, t / T_ramp) if T_ramp > 0 else alpha
            if a_t != prev_alpha:  # stage boundary: reweight the prefix to the new exponent
                logW += (a_t - prev_alpha) * cum_logp  # finished particles too: target is p(y)^alpha_t
                prev_alpha = a_t
            with torch.no_grad():
                out = model(input_ids=cur, attention_mask=att, position_ids=curpos, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logp = torch.log_softmax(out.logits[:, -1].float(), -1)
            logq = torch.log_softmax(a_t * logp, -1)
            y = torch.multinomial(logq.exp(), 1).squeeze(-1)
            lp_y = logp.gather(-1, y[:, None]).squeeze(-1).double().cpu().numpy()
            lq_y = logq.gather(-1, y[:, None]).squeeze(-1).double().cpu().numpy()
            y = y.tolist()
            for r in range(R):
                if done[r]:
                    continue
                generated[r // N] += 1
                logW[r] += a_t * lp_y[r] - lq_y[r]
                cum_logp[r] += lp_y[r]
                if y[r] in ends:
                    done[r] = True
                    continue
                gens[r].append(y[r])
                if stop_hit(tok, gens[r]) or len(gens[r]) >= args.max_new_tokens:
                    done[r] = True
            if done.all():
                break
            # resample within each problem when its ESS is low
            anc = np.arange(R)
            for p in range(P):
                sl = slice(p * N, (p + 1) * N)
                lw = logW[sl] - logW[sl].max()
                w = np.exp(lw) / np.exp(lw).sum()
                if 1.0 / (w ** 2).sum() < kappa * N:
                    resample_events += 1
                    anc[sl] = p * N + systematic(w, rng)
            if (anc != np.arange(R)).any():
                gens = [list(gens[a]) for a in anc]
                done, cum_logp = done[anc].copy(), cum_logp[anc].copy()
                for p in range(P):
                    sl = slice(p * N, (p + 1) * N)
                    if (anc[sl] != np.arange(p * N, (p + 1) * N)).any():
                        logW[sl] = 0.0
                idx = torch.tensor(anc, device="cuda")
                past.reorder_cache(idx)
                att = att[idx]
                curpos = curpos[idx]
                y = [y[a] for a in anc]
            nxt = torch.tensor([tok.pad_token_id if done[r] else y[r] for r in range(R)], device="cuda")
            cur = nxt[:, None]
            att = torch.cat([att, torch.ones((R, 1), dtype=att.dtype, device="cuda")], 1)
            curpos = curpos[:, -1:] + 1
        for p in range(P):
            sl = range(p * N, (p + 1) * N)
            paths = [make_path(tok, gens[r], task, answers[b + p], args.max_new_tokens, prefix=args.prefix.lstrip())
                     for r in sl]
            lw = np.array([logW[r] for r in sl])
            w = np.exp(lw - lw.max())
            w /= w.sum()
            chosen = int(rng.choice(N, p=w))
            votes = Counter(x["score"]["lenient_answer"] for x in paths if x["score"]["lenient_answer"] is not None)
            top = votes.most_common(1)[0][0] if votes else None
            res.append({"paths": paths, "chosen": chosen, "max_weight": int(w.argmax()),
                        "majority": next((j for j, x in enumerate(paths) if x["score"]["lenient_answer"] == top), 0),
                        "weights": w.round(4).tolist(), "generated_tokens": int(generated[p])})
        print(f"{b + P}/{len(texts)}  resample events so far: {resample_events}", flush=True)
    return res


def summarize(results, answers):
    n = len(results)
    ok = lambda r, j: r["paths"][j]["score"]["lenient_correct"]
    return {"acc_weight_sample": sum(ok(r, r["chosen"]) for r in results) / n,
            "acc_max_weight": sum(ok(r, r["max_weight"]) for r in results) / n,
            "acc_majority": sum(ok(r, r["majority"]) for r in results) / n,
            "acc_oracle": sum(any(x["score"]["lenient_correct"] for x in r["paths"]) for r in results) / n,
            "distinct_outputs": sum(len({x["output"] for x in r["paths"]}) for r in results) / n,
            "mean_generated_tokens": sum(r["generated_tokens"] for r in results) / n}
