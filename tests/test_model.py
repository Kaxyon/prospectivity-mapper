from dataclasses import replace

import numpy as np

from prospectivity.grid import grid_from_aoi
from prospectivity.model import (
    fit_naive, fit_pu, score_cells, spatial_blocks,
)


def test_spatial_blocks_are_square(cfg):
    g = grid_from_aoi(cfg.study_area.aoi_wgs84, cfg.study_area.crs, 500)
    blocks = spatial_blocks(g, 8.0)  # 16 cells across
    assert blocks.shape == g.shape
    assert blocks[0, 0] == blocks[15, 15]
    assert blocks[0, 0] != blocks[16, 16]


def _synthetic(n_pos=120, n_unl=1500, seed=0):
    rng = np.random.default_rng(seed)
    Xp = rng.normal(2.0, 1.0, size=(n_pos, 3))
    Xu = rng.normal(0.0, 1.0, size=(n_unl, 3))
    return Xp.astype("float32"), Xu.astype("float32")


def test_score_cells_prior_correction_is_monotone():
    class Fixed:
        def predict_proba(self, X):
            p = np.clip(X[:, 0], 0, 0.9)
            return np.column_stack([1 - p, p])

    X = np.array([[0.1], [0.4], [0.8]])
    raw = score_cells(Fixed(), X)
    corrected = score_cells(Fixed(), X, prior=0.5)
    assert np.all(np.diff(raw) > 0)
    assert np.all(np.diff(corrected) > 0)
    assert corrected.max() <= 1.0


def test_pu_prior_and_separation(cfg):
    cfg = replace(cfg, model=replace(
        cfg.model, n_estimators=40, pu_iterations=2, min_samples_leaf=3
    ))
    Xp, Xu = _synthetic()
    fit = fit_pu(Xp, Xu, cfg, verbose=False)
    assert 0.0 < fit.prior <= 1.0
    assert fit.mode in {"soft", "hard"}

    pos = score_cells(fit.model, Xp, fit.prior).mean()
    unl = score_cells(fit.model, Xu, fit.prior).mean()
    assert pos > unl


def test_naive_model_separates_shifted_classes(cfg):
    cfg = replace(cfg, model=replace(cfg.model, n_estimators=40))
    Xp, Xu = _synthetic()
    model = fit_naive(Xp, Xu, cfg)
    assert score_cells(model, Xp).mean() > score_cells(model, Xu).mean()
