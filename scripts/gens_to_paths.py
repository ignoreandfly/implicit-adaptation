"""Convert evaluation.generate generations (outputs + scores per row) to the methods.run "paths" layout that
train/rft_lora.py reads, so chat-format samples from the vLLM backend can be used for RFT.

    python scripts/gens_to_paths.py <in_dir>/train.generations.jsonl <out_dir>/train.generations.jsonl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.io import read_jsonl, write_jsonl  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
rows = []
for r in read_jsonl(src):
    paths = [{"output": o, "gen_tokens": s["gen_tokens"], "truncated": s["truncated"],
              "score": {k: v for k, v in s.items() if k not in ("gen_tokens", "truncated")}}
             for o, s in zip(r["outputs"], r["scores"])]
    rows.append({"question": r["question"], "answer": r["answer"], "paths": paths, "chosen": 0})
write_jsonl(Path(dst), rows)
print(f"wrote {len(rows)} rows to {dst}")
