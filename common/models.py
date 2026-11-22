"""Model registry: short name -> (Hugging Face id, prompt format).

"plain": our `Question: ...\nAnswer:` prompt, cut at stop strings (base models).
"chat":  the model's own chat template, stops at EOS only (post-trained models).

`--model` in evaluation.generate takes a short name from here or any HF id
(then pass --format, default "plain").
"""

MODELS = {
    # OLMo 2 1B (April 2025): every post-training stage is released separately.
    "olmo2-1b-base": ("allenai/OLMo-2-0425-1B", "plain"),
    "olmo2-1b-sft": ("allenai/OLMo-2-0425-1B-SFT", "chat"),
    "olmo2-1b-dpo": ("allenai/OLMo-2-0425-1B-DPO", "chat"),
    "olmo2-1b-instruct": ("allenai/OLMo-2-0425-1B-Instruct", "chat"),  # SFT + DPO + RLVR
    # OLMo 2 7B (Nov 2024), same stage ladder; weights cached under /data/azfarm/siddhant_hf_cache
    "olmo2-7b-base": ("allenai/OLMo-2-1124-7B", "plain"),
    "olmo2-7b-sft": ("allenai/OLMo-2-1124-7B-SFT", "chat"),
    "olmo2-7b-dpo": ("allenai/OLMo-2-1124-7B-DPO", "chat"),
    "olmo2-7b-instruct": ("allenai/OLMo-2-1124-7B-Instruct", "chat"),
}


def resolve(name, fmt=None):
    """Return (hf_id, format). An explicit fmt overrides the registry."""
    if name in MODELS:
        hf_id, default_fmt = MODELS[name]
        return hf_id, fmt or default_fmt
    return name, fmt or "plain"
