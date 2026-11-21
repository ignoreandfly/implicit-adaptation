"""Collect results/**/<split>.metrics.json into one table.

    python -m analysis.summarize [results_dir]
"""

import json
import sys
from pathlib import Path

import pandas as pd

COLS = ["num_problems", "acc_strict", "acc_lenient", "format_rate", "step_acc", "mean_steps",
        "mean_gen_tokens", "truncation_rate", "seconds"]


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    rows = []
    for f in sorted(root.rglob("*.metrics.json")):
        m = json.loads(f.read_text())
        rows.append({"run": str(f.parent.relative_to(root)), "split": m["split"],
                     **{c: m.get(c) for c in COLS},
                     **{k: v for k, v in m.items() if k.startswith(("pass@", "maj@"))}})
    if not rows:
        print("no results yet")
        return
    df = pd.DataFrame(rows).sort_values(["split", "run"])
    try:  # to_markdown needs the optional `tabulate` package
        print(df.to_markdown(index=False, floatfmt=".3f"))
    except ImportError:
        print(df.to_string(index=False, float_format="%.3f"))
    df.to_csv(root / "summary.csv", index=False)


if __name__ == "__main__":
    main()
