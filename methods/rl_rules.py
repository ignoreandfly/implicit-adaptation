"""Rule-based "RL" for an SFT model: a fixed token-bias table derived from where RLVR changed the
model, applied on top of OLMo 2 1B SFT at decoding time (Hewitt-style product of experts):

    greedy on  log p_SFT(w | x) + gate(x) * r(w),
    r(w) = +beta for the K tokens Instruct promotes most, -beta for the K it demotes most

The table comes from analysis/rl_delta.py (summed probability-mass shift p_Instruct - p_SFT on SFT's
own greedy solutions to 300 GSM8K TRAINING problems). gate(x) = 1 at fork positions (SFT next-token
entropy above the 80th-percentile threshold measured on those trajectories) with --rl_scope fork,
or always with --rl_scope all. No training, no prompt change.

--rl_swaps M: instead of a bias table, use the M most frequent argmax swaps (a -> b) observed between
SFT and Instruct at those training positions: at a fork, if SFT's greedy token is a and b is in
SFT's top-5, emit b. (RL's choice is SFT's 2nd choice ~72% of the time, so swaps are local.)

--rl_bias_json PATH: route 2 (no RL model): a bias table derived from the model's OWN samples by
analysis/self_rules.py (log-odds of fork tokens in majority-agreeing vs disagreeing samples on
training problems). Greedy on log p + beta * clip(LOR, -2, 2) at forks, and on the first response
token with the separate first-token table.

--rl_oracle: upper bound for any fork-gated rule. Decode with SFT, but at fork positions take the
Instruct model's greedy token instead (two models in lock-step).
beta = 0 is the control (plain SFT greedy through the same decoding loop).
"""

import json

import torch
from transformers import AutoModelForCausalLM

from common.models import resolve
from methods.common import make_path, manual_decode, single_path_summary


def tag(args):
    if args.rl_bias_json:
        import os
        return f"self_{os.path.basename(args.rl_bias_json).replace('.json', '')}_b{args.rl_beta}_{args.rl_scope}"
    if args.rl_swaps:
        return f"swaps{args.rl_swaps}_{args.rl_scope}"
    if args.rl_oracle:
        return f"oracle_{args.rl_scope}_thr{args.rl_fork_thr or 'auto'}"
    return f"K{args.rl_k}_b{args.rl_beta}_{args.rl_scope}"


def load_table(tok, path, k):
    s = json.load(open(path))

    def ids(entries):
        out = []
        for t, _, _ in entries:
            e = tok.encode(t, add_special_tokens=False)
            if len(e) == 1:
                out.append(e[0])
            if len(out) == k:
                break
        return out

    return ids(s["top_promoted_tokens"]), ids(s["top_demoted_tokens"]), s["fork_threshold_H_A"]


def load_swaps(tok, path, m):
    import ast
    s = json.load(open(path))
    pairs = {}
    for k, _ in s["top_argmax_swaps_a_to_b"]:
        a, b = (ast.literal_eval(x) for x in k.split(" -> "))
        ea, eb = tok.encode(a, add_special_tokens=False), tok.encode(b, add_special_tokens=False)
        if len(ea) == 1 and len(eb) == 1 and ea[0] not in pairs:
            pairs[ea[0]] = eb[0]
        if len(pairs) == m:
            break
    return pairs


def run(model, tok, texts, answers, task, args, ends):
    fmt = args.fmt
    if args.prefix:  # e.g. zero-shot CoT for base models; the prefix is kept in the scored output
        texts = [t + args.prefix for t in texts]
    V = model.get_output_embeddings().weight.shape[0]
    up, down, thr = load_table(tok, args.rl_table, args.rl_k)
    thr = args.rl_fork_thr or thr
    r = torch.zeros(V, device="cuda")
    r[torch.tensor(up, device="cuda")] += args.rl_beta
    r[torch.tensor(down, device="cuda")] -= args.rl_beta
    print(f"table: +{len(up)} / -{len(down)} tokens, beta={args.rl_beta}, scope={args.rl_scope}, fork H>{thr:.3f}", flush=True)
    stats = {"steps": 0, "fork_steps": 0}

    if args.rl_oracle:
        expert = AutoModelForCausalLM.from_pretrained(resolve(args.rl_oracle_model)[0], torch_dtype=torch.bfloat16).cuda().eval()
        return _oracle(model, expert, tok, texts, answers, task, args, ends, thr, stats)

    self_first = None
    if args.rl_bias_json:
        sj = json.load(open(args.rl_bias_json))
        r = torch.zeros(V, device="cuda")
        for w, v in sj["bias"].items():
            r[int(w)] = args.rl_beta * max(-2.0, min(2.0, v))
        self_first = torch.zeros(V, device="cuda")
        for w, v in sj["first_bias"].items():
            self_first[int(w)] = args.rl_beta * max(-2.0, min(2.0, v))
        thr = args.rl_fork_thr or sj["fork_thr"]
        print(f"self rules: {len(sj['bias'])} fork tokens, {len(sj['first_bias'])} first tokens, fork H>{thr:.3f}", flush=True)
    swaps = load_swaps(tok, args.rl_table, args.rl_swaps) if args.rl_swaps else {}
    if swaps:
        print("swap rules:", [(tok.decode([a]), tok.decode([b])) for a, b in swaps.items()], flush=True)

    def step(logits, hs, gen):
        lp = torch.log_softmax(logits, -1)
        if swaps:
            top = lp.topk(5, -1).indices
            H = -(lp.exp() * lp).sum(-1)
            nxt = top[:, 0].clone()
            for i in range(lp.shape[0]):
                a = int(nxt[i])
                if a in swaps and (args.rl_scope == "all" or H[i] > thr) and swaps[a] in top[i].tolist():
                    nxt[i] = swaps[a]
                    stats["fork_steps"] += 1
                stats["steps"] += 1
            return nxt
        if args.rl_beta == 0:
            return lp.argmax(-1)
        if self_first is not None:
            H = -(lp.exp() * lp).sum(-1, keepdim=True)
            gate = (H > thr).float() if args.rl_scope == "fork" else torch.ones_like(H)
            first = torch.tensor([[1.0] if not g else [0.0] for g in gen], device=lp.device)
            return (lp + (1 - first) * gate * r + first * self_first).argmax(-1)
        if args.rl_scope == "all":
            gate = torch.ones(lp.shape[0], 1, device=lp.device)
        else:
            H = -(lp.exp() * lp).sum(-1, keepdim=True)
            gate = (H > thr).float()
            stats["fork_steps"] += int(gate.sum()); stats["steps"] += gate.numel()
        return (lp + gate * r).argmax(-1)

    res = []
    bs = args.batch_problems * 8
    for b in range(0, len(texts), bs):
        gen = manual_decode(model, tok, texts[b:b + bs], args.max_new_tokens, step, ends, fmt=fmt)
        for i, ids in enumerate(gen):
            res.append({"paths": [make_path(tok, ids, task, answers[b + i], args.max_new_tokens, fmt=fmt,
                                            prefix=args.prefix.lstrip())], "chosen": 0})
        print(f"{b + len(gen)}/{len(texts)}  fork share {stats['fork_steps'] / max(1, stats['steps']):.3f}", flush=True)
    return res


def _oracle(model, expert, tok, texts, answers, task, args, ends, thr, stats):
    """SFT and expert decode the same sequence in lock-step; at forks the expert's argmax is taken."""
    fmt, res, bs = args.fmt, [], args.batch_problems * 8
    for b in range(0, len(texts), bs):
        enc = tok(texts[b:b + bs], return_tensors="pt", padding=True, add_special_tokens=(fmt == "plain")).to("cuda")
        att = enc.attention_mask
        B = att.shape[0]
        cur, pos = enc.input_ids, (att.cumsum(-1) - 1).clamp_min(0)
        pa = pb = None
        gen, done = [[] for _ in range(B)], [False] * B
        for _ in range(args.max_new_tokens):
            with torch.no_grad():
                oa = model(input_ids=cur, attention_mask=att, position_ids=pos, past_key_values=pa, use_cache=True)
                ob = expert(input_ids=cur, attention_mask=att, position_ids=pos, past_key_values=pb, use_cache=True)
            pa, pb = oa.past_key_values, ob.past_key_values
            la = torch.log_softmax(oa.logits[:, -1].float(), -1)
            H = -(la.exp() * la).sum(-1)
            fork = (H > thr) if args.rl_scope == "fork" else torch.ones_like(H, dtype=torch.bool)
            nxt = torch.where(fork, ob.logits[:, -1].argmax(-1), la.argmax(-1))
            stats["fork_steps"] += int(fork[[not d for d in done]].sum()) if not all(done) else 0
            stats["steps"] += sum(not d for d in done)
            for i in range(B):
                if done[i]:
                    continue
                t = int(nxt[i])
                if t in ends:
                    done[i] = True
                else:
                    gen[i].append(t)
            if all(done):
                break
            nxt = torch.where(torch.tensor(done, device="cuda"), torch.full_like(nxt, tok.pad_token_id), nxt)
            cur = nxt[:, None]
            att = torch.cat([att, torch.ones((B, 1), dtype=att.dtype, device="cuda")], 1)
            pos = pos[:, -1:] + 1
        for i, ids in enumerate(gen):
            res.append({"paths": [make_path(tok, ids, task, answers[b + i], args.max_new_tokens, fmt=fmt,
                                            prefix=args.prefix.lstrip())], "chosen": 0})
        print(f"{b + B}/{len(texts)}  expert-token share {stats['fork_steps'] / max(1, stats['steps']):.3f}", flush=True)
    return res


summarize = single_path_summary
