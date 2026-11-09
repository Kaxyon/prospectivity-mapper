"""SHAP explanations of the prospectivity model.

TreeSHAP gives exact additive contributions for a random forest, so we can
decompose any cell's score into per-feature parts and answer both "what drives
the map overall?" and "why is *this* target ranked highly?".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config


@dataclass
class ShapResult:
    values: np.ndarray            # (n_samples, n_features)
    data: pd.DataFrame            # feature values at those samples
    base_value: float
    feature_names: list[str]
    sample_cells: np.ndarray

    def importance(self) -> pd.DataFrame:
        mean_abs = np.abs(self.values).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )


def _predict_positive(explainer, X):
    out = explainer(X, check_additivity=False)
    values = np.asarray(out.values)
    if values.ndim == 3:  # (n, features, classes)
        values = values[:, :, -1]
        base = np.asarray(out.base_values)
        base = float(base[0, -1]) if base.ndim == 2 else float(base[0])
    else:
        base = float(np.asarray(out.base_values).ravel()[0])
    return values, base


def explain_model(
    model,
    cube,
    cfg: Config,
    n_background: int = 2500,
    positive_cells: np.ndarray | None = None,
) -> ShapResult:
    """SHAP values for a stratified sample of the grid plus all positives."""
    import shap

    rng = np.random.default_rng(cfg.seed)
    all_cells = np.argwhere(np.ones(cube.grid.shape, dtype=bool))
    idx = rng.choice(len(all_cells), size=min(n_background, len(all_cells)), replace=False)
    sample_cells = all_cells[np.sort(idx)]
    if positive_cells is not None and len(positive_cells):
        sample_cells = np.vstack([sample_cells, positive_cells])

    X = cube.sample(sample_cells)
    explainer = shap.TreeExplainer(model)
    values, base = _predict_positive(explainer, X.to_numpy("float32"))
    return ShapResult(
        values=values,
        data=X,
        base_value=base,
        feature_names=list(X.columns),
        sample_cells=sample_cells,
    )


def local_explanation(shap_result: ShapResult, row: int, k: int = 8) -> pd.DataFrame:
    """Largest positive and negative contributions for one sample."""
    vals = shap_result.values[row]
    order = np.argsort(-np.abs(vals))[:k]
    return pd.DataFrame(
        {
            "feature": [shap_result.feature_names[i] for i in order],
            "value": [float(shap_result.data.iloc[row, i]) for i in order],
            "shap": [float(vals[i]) for i in order],
        }
    )
