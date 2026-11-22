"""Rule-based reasoning adapter, after Hewitt et al. 2024 ("Instruction Following without
Instruction Tuning", Sec. 6): a product of experts with the base model,

    p(w | x)  ∝  p_base(w | x) * exp(r(w, x))      i.e. greedy on  log p_base + r,

where r is a sum of hand-written rule scores. No training, no prompt change (the plain
"Question: ...\\nAnswer:" prompt). Rules mirror Hewitt's three, adapted to reasoning:

  stop   (Hewitt's EOS rule)  until the response contains a finished answer statement
         ("answer is N." with the number closed by a full stop),
         stopping tokens (EOS and any token containing a newline) get -stop_pen; afterwards
         EOS and blank-line tokens get +end_bonus, so the model ends right after answering.
  shift  (Hewitt's uniform token changes)  +step_bonus on step connectives
         (" First", " Then", " So", " Therefore", " Next", " Finally", " Since", " Let");
         -drift_pen on tokens base OLMo forks into when it drifts off task
         ("[", "Explanation", "Question", "Q").
  first  (analogue of Hewitt's -5 on "I"/"We", which start refusals)  -first_digit_pen on
         digit-initial tokens until the response has a non-space character (base OLMo usually
         emits a bare " " first, then blurts a number instead of reasoning), and +first_step_bonus
         on words that open a worked solution (" Let", " First", " Since", " To", " We") there.
  rep    (Hewitt's -1.5 repetition rule)  -rep_pen once for every alphabetic token already in
         the response (digits/operators/punctuation exempt: math legitimately repeats them), and
         -ngram_pen on any token that would repeat a 4-gram already in the response (catches the
         "26/2+6=26/2+6=..." loops that the word-level rule cannot).

--ra_off stop,shift,first,rep switches rules off (leave-one-out ablation, as in Hewitt Table 4).
Rule weights are tuned on the dev split only (Hewitt tuned on a separate validation set).
"""

import re

import torch

from methods.common import make_path, manual_decode, single_path_summary

STEP_WORDS = [" First", " Then", " So", " Therefore", " Next", " Finally", " Since", " Let"]
FIRST_WORDS = [" Let", " First", " Since", " To", " We"]  # words that open a worked solution
DRIFT_WORDS = ["[", "Explanation", " Explanation", "Question", " Question", "Q"]
ANSWERED_RE = re.compile(r"answer is\s*:?\s*\$?\s*-?\d[\d,]*(?:\.\d+)?\s*\.(?!\d)", re.I)  # number closed by a full stop


def tag(args):
    off = ("_off-" + args.ra_off.replace(",", "-")) if args.ra_off else ""
    return (f"s{args.ra_stop_pen}_e{args.ra_end_bonus}_b{args.ra_step_bonus}_d{args.ra_drift_pen}"
            f"_f{args.ra_first_digit_pen}_fs{args.ra_first_step_bonus}_r{args.ra_rep_pen}_n{args.ra_ngram_pen}{off}")


def _single_ids(tok, words):
    ids = []
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        if len(t) == 1:
            ids.append(t[0])
    return ids


def build_rule_tables(tok, V, device):
    strs = [tok.decode([i]) for i in range(V)]
    has_nl = torch.tensor(["\n" in s for s in strs], device=device)
    has_blank = torch.tensor(["\n\n" in s for s in strs], device=device)
    digit_first = torch.tensor([s.strip()[:1].isdigit() for s in strs], device=device)
    alpha = torch.tensor([any(c.isalpha() for c in s) for s in strs], device=device)
    return has_nl, has_blank, digit_first, alpha


def run(model, tok, texts, answers, task, args, ends):
    V = model.get_output_embeddings().weight.shape[0]
    dev = "cuda"
    has_nl, has_blank, digit_first, alpha = build_rule_tables(tok, V, dev)
    off = set(filter(None, args.ra_off.split(",")))
    eos = [i for i in ends if i < V]
    step_ids = torch.tensor(_single_ids(tok, STEP_WORDS), device=dev)
    first_ids = torch.tensor(_single_ids(tok, FIRST_WORDS), device=dev)
    drift_ids = torch.tensor(_single_ids(tok, DRIFT_WORDS), device=dev)
    stop_mask = has_nl.clone()
    stop_mask[eos] = True
    end_mask = has_blank.clone()
    end_mask[eos] = True

    base_r = torch.zeros(V, device=dev)
    if "shift" not in off:
        base_r[step_ids] += args.ra_step_bonus
        base_r[drift_ids] -= args.ra_drift_pen

    def step(logits, hs, gen):
        lp = torch.log_softmax(logits, -1)
        r = base_r.expand_as(lp).clone()
        for b, ids in enumerate(gen):
            text = tok.decode(ids, skip_special_tokens=True)
            if "stop" not in off:
                if ANSWERED_RE.search(text):
                    r[b, end_mask] += args.ra_end_bonus
                else:
                    r[b, stop_mask] -= args.ra_stop_pen
            if "first" not in off and not text.strip():  # until the first non-space token
                r[b, digit_first] -= args.ra_first_digit_pen
                r[b, first_ids] += args.ra_first_step_bonus
            if "rep" not in off and ids:
                seen = torch.tensor(sorted(set(ids)), device=dev)
                seen = seen[alpha[seen]]
                r[b, seen] -= args.ra_rep_pen
                if len(ids) >= 4:
                    ctx = tuple(ids[-3:])
                    nxt = {ids[i + 3] for i in range(len(ids) - 3) if tuple(ids[i:i + 3]) == ctx}
                    if nxt:
                        r[b, torch.tensor(sorted(nxt), device=dev)] -= args.ra_ngram_pen
        return (lp + r).argmax(-1)

    res = []
    bs = args.batch_problems * 8
    for b in range(0, len(texts), bs):
        batch = texts[b:b + bs]
        gen = manual_decode(model, tok, batch, args.max_new_tokens, step, ends)
        for i, ids in enumerate(gen):
            res.append({"paths": [make_path(tok, ids, task, answers[b + i], args.max_new_tokens)], "chosen": 0})
        print(f"{b + len(batch)}/{len(texts)}", flush=True)
    return res


summarize = single_path_summary
