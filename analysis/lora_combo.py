"""theta = SFT + sum_i c_i * (2 B_i A_i) (LoRA alpha/r = 2), built and saved in fp32.
    python -m analysis.lora_combo --model olmo2-1b-sft --terms dir1:1.0 dir2:-0.5 --out scratch"""
import argparse, torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from common.models import resolve
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--terms", nargs="*", default=[]); ap.add_argument("--out", required=True)
a = ap.parse_args(); hf = resolve(a.model)[0]
m = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.float32); P = dict(m.named_parameters())
with torch.no_grad():
    for t in a.terms:
        d, c = t.rsplit(":", 1); L = load_file(f"{d}/adapter_model.safetensors"); ks = [k for k in L if k.endswith("lora_A.weight")]; assert len(ks) == 64, len(ks)
        for k in ks:
            n = k.replace("base_model.model.", "", 1).replace(".lora_A.weight", ".weight")
            P[n] += float(c) * 2.0 * (L[k.replace("lora_A", "lora_B")].float() @ L[k].float())
m.save_pretrained(a.out, safe_serialization=True); AutoTokenizer.from_pretrained(hf).save_pretrained(a.out); print("built", a.terms)
