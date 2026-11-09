"""Ranking metrics for a one-class-labelled problem.

Because the unlabelled cells are not proven negatives, every metric here is
framed as *ranking against background*: how well does the model lift known
deposits above random ground?  AUC against unlabelled cells is therefore a
conservative (lower-bound) estimate of true discrimination.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

CAPTURE_FRACTIONS = (0.01, 0.05, 0.10, 0.20)


def ranking_metrics(
    pos_scores: np.ndarray,
    unl_scores: np.ndarray,
    fractions: tuple[float, ...] = CAPTURE_FRACTIONS,
) -> dict[str, float]:
    pos = np.asarray(pos_scores, dtype="float64")
    unl = np.asarray(unl_scores, dtype="float64")
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(unl))])
    s = np.concatenate([pos, unl])

    out: dict[str, float] = {
        "n_positive": float(len(pos)),
        "n_unlabelled": float(len(unl)),
        "roc_auc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
        "median_positive_percentile": float(
            100.0 * (1.0 - np.median(np.searchsorted(np.sort(s), pos) / len(s)))
        ),
    }
    order = np.argsort(-s)
    for frac in fractions:
        n = max(1, int(round(frac * len(s))))
        captured = float(y[order[:n]].sum() / len(pos))
        out[f"capture_at_{int(frac * 100)}pct"] = captured
        out[f"lift_at_{int(frac * 100)}pct"] = captured / frac
    return out


def capture_curve(pos_scores: np.ndarray, unl_scores: np.ndarray, n_points: int = 200):
    """Fraction of deposits captured vs fraction of the area ranked."""
    pos = np.asarray(pos_scores, dtype="float64")
    unl = np.asarray(unl_scores, dtype="float64")
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(unl))])
    s = np.concatenate([pos, unl])
    order = np.argsort(-s)
    cum = np.cumsum(y[order]) / max(1, len(pos))
    area = np.arange(1, len(s) + 1) / len(s)
    idx = np.unique(np.linspace(0, len(s) - 1, n_points).astype(int))
    return area[idx], cum[idx]


def bootstrap_auc_ci(
    pos_scores: np.ndarray,
    unl_scores: np.ndarray,
    n_boot: int = 300,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile CI for AUC, resampling positives and unlabelled cells."""
    rng = np.random.default_rng(seed)
    pos = np.asarray(pos_scores, dtype="float64")
    unl = np.asarray(unl_scores, dtype="float64")
    vals = []
    for _ in range(n_boot):
        p = rng.choice(pos, size=len(pos), replace=True)
        u = rng.choice(unl, size=min(len(unl), 5000), replace=True)
        y = np.concatenate([np.ones(len(p)), np.zeros(len(u))])
        vals.append(roc_auc_score(y, np.concatenate([p, u])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
