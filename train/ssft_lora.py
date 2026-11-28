"""Self-SSFT: Set Supervised Fine-Tuning (Jia et al., ICLR 2026, arXiv 2510.05132) on the model's OWN traces.

SSFT adds N global forking tokens <think1>..<thinkN>. Each training question has a SET of M distinct reasoning
traces; at every step the traces are matched one-to-one to the tags by minimum total loss (bipartite matching),
and each tag is trained only on its matched trace. The paper reports that plain SFT on several traces per question
(no tags) collapses them into one mode, and that matching keeps the modes apart so <think_i> starts a distinct
reasoning path. The paper used 32B models and teacher traces; here the traces are the model's own RFT samples
(the same data as train/rft_lora.py), with gold or label-free (majority) filtering.

--assign match   SSFT: per-question bipartite matching on the current loss (the method)
--assign random  ablation: tags assigned to traces by a random one-to-one map each step (tags, no matching)
--assign none    control: no tags, plain RFT on the same capped trace sets (= RFT with at most M traces/question)

Each tag is one new token placed right after the prompt (after the chat template's assistant header); the loss is
on the trace only. Trainable: LoRA (same size as rft_lora) + the embedding rows of the new tokens.
The adapter is merged and the model saved with the extended tokenizer, so vLLM loads it directly; generate with
`evaluation.generate --suffix "<think1>"`.

    python -m train.ssft_lora --gens results/ssft_self/sft_train1500_n8 --model olmo2-1b-sft --label gold \
        --n_tags 4 --m 4 --assign match --out checkpoints/ssft_self/ssft_gold
"""

import argparse
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path


def tag_names(n):
    return [f"<think{i}>" for i in range(1, n + 1)]


def jaccard(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / max(1, len(a | b))


def pick_diverse(traces, m):
    """Up to m traces, farthest-first by word-set Jaccard (start from the shortest), so the set covers
    different solutions rather than near-copies."""
    traces = list(dict.fromkeys(traces))
    if len(traces) <= m:
        return traces
    chosen = [min(traces, key=len)]
    rest = [t for t in traces if t != chosen[0]]
    while len(chosen) < m and rest:
        far = max(rest, key=lambda t: -max(jaccard(t, c) for c in chosen))
        chosen.append(far); rest.remove(far)
    return chosen


def assign(cost, how, rng):
    """cost[i][j] = loss of trace j under tag i (N tags x M traces, M <= N). Returns a list of (tag, trace)
    pairs, one per trace, each tag used at most once. match = exact minimum-total-cost assignment
    (brute force over tag permutations; N <= 8 keeps this small)."""
    n, m = len(cost), len(cost[0])
    if how == "random":
        tags = rng.sample(range(n), m)
        return list(zip(tags, range(m)))
    best, best_tags = None, None
    for tags in itertools.permutations(range(n), m):
        c = sum(cost[t][j] for j, t in enumerate(tags))
        if best is None or c < best:
            best, best_tags = c, tags
    return list(zip(best_tags, range(m)))


def build_sets(rows, label, m):
    """Per question: the kept traces (gold-correct or majority-agreeing), deduplicated, diversified to <= m."""
    sets, label_ok, n_major = [], 0, 0
    for r in rows:
        if "paths" in r:
            outs = [p["output"] for p in r["paths"]]; scores = [p["score"] for p in r["paths"]]
        else:
            outs, scores = r["outputs"], r["scores"]
        ans = [s["lenient_answer"] for s in scores]
        if label == "gold":
            keep = [o for o, s in zip(outs, scores) if s["lenient_correct"]]
        else:
            v = Counter(a for a in ans if a is not None)
            if not v:
                continue
            maj = v.most_common(1)[0][0]; n_major += 1
            label_ok += int(maj == r["answer"])
            keep = [o for o, a in zip(outs, ans) if a == maj]
        keep = pick_diverse(keep, m)
        if keep:
            sets.append({"question": r["question"], "traces": keep})
    return sets, label_ok, n_major


def main():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from common import prompts
    from common.io import read_jsonl
    from common.models import resolve

    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="olmo2-1b-sft")
    ap.add_argument("--label", choices=["gold", "majority"], default="gold")
    ap.add_argument("--assign", choices=["match", "random", "none"], default="match")
    ap.add_argument("--n_tags", type=int, default=4)
    ap.add_argument("--m", type=int, default=4, help="max traces per question (<= n_tags)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--emb_lr", type=float, default=1e-3, help="lr of the new tag embeddings")
    ap.add_argument("--tag_noise", type=float, default=0.1, help="init noise of the tag embeddings (x embedding std)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8, help="questions per optimizer step")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=640)
    ap.add_argument("--chunk", type=int, default=4, help="sequences per forward pass (memory)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.m <= args.n_tags or args.assign == "none"

    rng = random.Random(args.seed); torch.manual_seed(args.seed)
    hf_id, fmt = resolve(args.model)
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = read_jsonl(f"{args.gens}/{args.split}.generations.jsonl")[: args.limit]
    sets, label_ok, n_major = build_sets(rows, args.label, args.m)
    n_tr = sum(len(s["traces"]) for s in sets)
    print(f"{len(sets)}/{len(rows)} questions, {n_tr} traces ({n_tr / max(1, len(sets)):.2f}/question); "
          f"label={args.label}" + (f"; majority correct on {label_ok}/{n_major}" if args.label == "majority" else ""),
          flush=True)

    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16).cuda()
    tags = tag_names(args.n_tags) if args.assign != "none" else []
    tag_ids = []
    if tags:
        tok.add_tokens(tags, special_tokens=True)
        tag_ids = tok.convert_tokens_to_ids(tags)
        emb = model.get_input_embeddings()
        if len(tok) > emb.weight.shape[0]:
            model.resize_token_embeddings(len(tok), pad_to_multiple_of=64)
            emb = model.get_input_embeddings()
        with torch.no_grad():   # start each tag as the newline token + small distinct noise (harmless in-place, not interchangeable)
            nl = tok("\n", add_special_tokens=False).input_ids[-1]
            base = emb.weight[nl].float(); sd = emb.weight[: tag_ids[0]].float().std(0)
            g = torch.Generator(device="cpu").manual_seed(args.seed)
            for t in tag_ids:
                emb.weight[t] = (base + args.tag_noise * sd * torch.randn(base.shape, generator=g).to(base.device)
                                 ).to(emb.weight.dtype)

    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                             task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    lora_params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": lora_params, "lr": args.lr}]
    delta = None
    if tags:   # tag embeddings learn through a small (N x d) delta added in a forward hook; merged into the rows at save
        emb_mod = model.get_input_embeddings()
        tag_t = torch.tensor(tag_ids, device="cuda")
        delta = torch.nn.Parameter(torch.zeros(len(tag_ids), emb_mod.weight.shape[1], device="cuda"))

        def add_delta(mod, inp, out):
            hit = (inp[0].unsqueeze(-1) == tag_t).to(delta.dtype)          # [B, L, N]
            return out + (hit @ delta).to(out.dtype)
        hook = emb_mod.register_forward_hook(add_delta)
        groups.append({"params": [delta], "lr": args.emb_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    total = math.ceil(len(sets) * args.epochs / args.accum); warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / max(1, total)))

    def encode(s):
        prompt = prompts.build_prompt(fmt, s["question"], [], tok)
        p_ids = tok(prompt, add_special_tokens=(fmt == "plain")).input_ids
        s["p_ids"] = p_ids
        s["t_ids"] = [tok(((" " + o + "\n\n") if fmt == "plain" else (o + tok.eos_token)),
                          add_special_tokens=False).input_ids for o in s["traces"]]
    for s in sets:
        encode(s)

    def batch(pairs, s):
        """pairs: (tag index or None, trace index). Right-padded ids + labels (loss on the trace only)."""
        seqs, labs = [], []
        for t, j in pairs:
            pre = s["p_ids"] + ([tag_ids[t]] if t is not None else [])
            ids = (pre + s["t_ids"][j])[: args.max_len]
            lab = ([-100] * len(pre) + s["t_ids"][j])[: args.max_len]
            seqs.append(ids); labs.append(lab)
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), tok.pad_token_id, device="cuda")
        lab = torch.full((len(seqs), L), -100, device="cuda")
        att = torch.zeros((len(seqs), L), dtype=torch.long, device="cuda")
        for k, (a, b) in enumerate(zip(seqs, labs)):
            ids[k, : len(a)] = torch.tensor(a); lab[k, : len(b)] = torch.tensor(b); att[k, : len(a)] = 1
        return ids, lab, att

    def per_seq_loss(ids, lab, att):
        """Mean token NLL per sequence, computed --chunk sequences at a time (full-vocab logits are large at 7B)."""
        outs = []
        for i in range(0, ids.shape[0], args.chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids[i:i + args.chunk], attention_mask=att[i:i + args.chunk]).logits[:, :-1]
            tgt = lab[i:i + args.chunk, 1:]
            nll = torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), tgt.clamp(min=0), reduction="none")
            m = (tgt != -100).float()
            outs.append((nll * m).sum(1) / m.sum(1).clamp(min=1))
            del logits, nll
        return torch.cat(outs)

    model.train(); step = 0; log = []; usage = Counter(); agree_prev = []
    prev_assign = {}
    for ep in range(args.epochs):
        order = list(range(len(sets))); rng.shuffle(order)
        for n, qi in enumerate(order):
            s = sets[qi]; M = len(s["traces"])
            if args.assign == "none":
                pairs = [(None, j) for j in range(M)]
            elif args.assign == "random":
                pairs = assign([[0.0] * M for _ in tags], "random", rng)
            else:
                with torch.no_grad():
                    allp = [(t, j) for t in range(len(tags)) for j in range(M)]
                    l = per_seq_loss(*batch(allp, s)).tolist()
                cost = [[l[t * M + j] for j in range(M)] for t in range(len(tags))]
                pairs = assign(cost, "match", rng)
                if qi in prev_assign:
                    agree_prev.append(float(prev_assign[qi] == pairs))
                prev_assign[qi] = pairs
            usage.update(t for t, _ in pairs)
            loss = per_seq_loss(*batch(pairs, s)).mean() / args.accum
            loss.backward()
            if (n + 1) % args.accum == 0 or n == len(order) - 1:
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 20 == 0:
                    rec = {"step": step, "epoch": ep, "loss": float(loss) * args.accum,
                           "tag_usage": dict(sorted(usage.items(), key=lambda x: str(x[0])))}
                    if agree_prev:
                        rec["assignment_stable"] = round(sum(agree_prev) / len(agree_prev), 3)
                    log.append(rec); print(rec, flush=True); usage = Counter(); agree_prev = []

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if delta is not None:
        hook.remove()
        with torch.no_grad():
            w = model.get_input_embeddings().weight
            w[tag_ids] = (w[tag_ids].float() + delta).to(w.dtype)
    model.merge_and_unload().to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    json.dump({"args": vars(args), "tags": tags, "questions": len(sets), "traces": n_tr, "log": log},
              open(out / "ssft_config.json", "w"), indent=2)
    print(f"saved merged model to {out}", flush=True)


if __name__ == "__main__":
    main()
