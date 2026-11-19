import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021): 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def aggregate(per_problem, ks=(1, 4, 16, 64)):
    """per_problem: list (one per problem) of lists (one per sample) of verify() dicts."""
    n = len(per_problem[0])
    first = [s[0] for s in per_problem]
    all_samples = [x for s in per_problem for x in s]
    m = {
        "num_problems": len(per_problem),
        "samples_per_problem": n,
        "acc_strict": float(np.mean([x["strict_correct"] for x in first])),
        "acc_lenient": float(np.mean([x["lenient_correct"] for x in first])),
        "format_rate": float(np.mean([x["format_ok"] for x in all_samples])),
        "mean_steps": float(np.mean([x["n_steps"] for x in all_samples])),
    }
    tot = sum(x["n_steps"] for x in all_samples)
    m["step_acc"] = sum(x["n_steps_correct"] for x in all_samples) / tot if tot else None
    if n > 1:
        for key in ("strict", "lenient"):
            for k in ks:
                if k <= n:
                    m[f"pass@{k}_{key}"] = float(np.mean(
                        [pass_at_k(n, sum(x[f"{key}_correct"] for x in s), k) for s in per_problem]))
    return m
