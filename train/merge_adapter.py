"""Merge a saved LoRA adapter (train/decision_lora.py --save_every) into its base model and save a full model.

    python -m train.merge_adapter --model olmo2-1b-base --adapter ckpt/adapter_step20 --out /scratch/merged
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.models import resolve

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
hf_id, _ = resolve(a.model)
m = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32)
m = PeftModel.from_pretrained(m, a.adapter).merge_and_unload()
m.to(torch.bfloat16).save_pretrained(a.out, safe_serialization=True)
AutoTokenizer.from_pretrained(hf_id).save_pretrained(a.out)
print("merged", a.adapter, "->", a.out)
