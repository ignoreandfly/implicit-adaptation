"""Helpers shared by screening methods: path records, a manual KV-cached decode loop."""

import re

import torch

from common import prompts


def make_path(tok, gen_ids, task, answer, max_new_tokens, prefix="", extra=None, score_text=None, fmt="plain"):
    """Score one generated continuation. prefix = text forced after "Answer:" (kept in the output)."""
    text = prefix + tok.decode(gen_ids, skip_special_tokens=True)
    out = prompts.truncate(text, fmt)
    score = task.verify(score_text(out) if score_text else out, answer)
    hit_stop = fmt == "plain" and any(s in text for s in prompts.STOP_STRINGS)
    return {"output": out, "gen_tokens": len(gen_ids), "truncated": len(gen_ids) >= max_new_tokens and not hit_stop,
            "score": score, **(extra or {})}


def stop_hit(tok, ids, fmt="plain"):
    """Plain-format stop strings; chat models stop at EOS only."""
    return fmt == "plain" and any(s in tok.decode(ids[-12:], skip_special_tokens=True) for s in prompts.STOP_STRINGS)


def encode(tok, texts, fmt="plain"):
    """Batch-encode prompts: chat templates already contain their special tokens, so only plain prompts get a BOS."""
    return tok(texts, return_tensors="pt", padding=True, add_special_tokens=(fmt == "plain")).to("cuda")


def gen_stop_strings(fmt="plain"):
    """stop_strings for model.generate: blank line etc. for plain prompts, none (EOS only) for chat."""
    return prompts.STOP_STRINGS if fmt == "plain" else None


def manual_decode(model, tok, texts, max_new_tokens, next_token_fn, ends, need_hidden=False, fmt="plain"):
    """Greedy-style decoding where next_token_fn(final_logits[B,V] fp32, hidden[list of B x d] or None,
    generated: list[list[int]]) -> LongTensor[B] picks each token. Stops at EOS / stop strings.
    Returns generated token ids per prompt (stop-string text included; truncate() removes it)."""
    # chat templates already contain their special tokens; only plain prompts get a BOS added
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=(fmt == "plain")).to("cuda")
    att = enc.attention_mask
    B = att.shape[0]
    cur_ids, cur_pos = enc.input_ids, (att.cumsum(-1) - 1).clamp_min(0)
    past, gen = None, [[] for _ in range(B)]
    done = [False] * B
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids=cur_ids, attention_mask=att, position_ids=cur_pos, past_key_values=past,
                        use_cache=True, output_hidden_states=need_hidden)
        past = out.past_key_values
        hs = [h[:, -1] for h in out.hidden_states] if need_hidden else None
        nxt = next_token_fn(out.logits[:, -1].float(), hs, gen)
        for b in range(B):
            if done[b]:
                continue
            t = int(nxt[b])
            if t in ends:
                done[b] = True
                continue
            gen[b].append(t)
            if fmt == "plain" and stop_hit(tok, gen[b]):
                done[b] = True
        if all(done):
            break
        nxt = torch.where(torch.tensor(done, device="cuda"), torch.full_like(nxt, tok.pad_token_id), nxt)
        cur_ids = nxt[:, None]
        att = torch.cat([att, torch.ones((B, 1), dtype=att.dtype, device="cuda")], 1)
        cur_pos = cur_pos[:, -1:] + 1
    return gen


def apply_repetition_penalty(scores, gen, penalty):
    """HF-style penalty, but only on tokens generated so far (not the question's numbers)."""
    if penalty == 1.0:
        return scores
    for b, ids in enumerate(gen):
        if ids:
            idx = torch.tensor(sorted(set(ids)), device=scores.device)
            s = scores[b, idx]
            scores[b, idx] = torch.where(s < 0, s * penalty, s / penalty)
    return scores


def single_path_summary(results, answers):
    n = len(results)
    return {"acc_chosen": sum(r["paths"][r["chosen"]]["score"]["lenient_correct"] for r in results) / n}


def prefix_slug(prefix):
    """Filesystem-safe tag for a forced prefix, unique per prefix text."""
    s = re.sub(r"[^a-z0-9]+", "-", prefix.lower()).strip("-")
    return s[:40] or "none"
