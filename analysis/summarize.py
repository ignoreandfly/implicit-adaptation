"""Collect results/**/<split>.metrics.json into one table.

    python -m analysis.summarize [results_dir]
"""

import json
import sys
from pathlib import Path

import pandas as pd

COLS = ["acc_strict", "acc_lenient", "format_rate", "step_acc", "mean_steps"]


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    rows = []
    for f in sorted(root.rglob("*.metrics.json")):
        m = json.loads(f.read_text())
        rows.append({"run": str(f.parent.relative_to(root)), "split": m["split"],
                     **{c: m.get(c) for c in COLS},
                     **{k: v for k, v in m.items() if k.startswith("pass@")}})
    if not rows:
        print("no results yet")
        return
    df = pd.DataFrame(rows).sort_values(["split", "run"])
    print(df.to_markdown(index=False, floatfmt=".3f") if hasattr(df, "to_markdown") else df.to_string())
    df.to_csv(root / "summary.csv", index=False)


if __name__ == "__main__":
    main()
