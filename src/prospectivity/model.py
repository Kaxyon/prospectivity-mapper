"""The classifier and the PU-learning procedure.

Two models are trained from the same data:

``naive``
    treats every unlabelled cell as a negative.  This is the usual (biased)
    shortcut and serves as the baseline.

``pu``
    a bagged Elkan-Noto positive-unlabelled learner.  Several classifiers are
    trained on positives vs bootstrap samples of the unlabelled pool; their
    averaged scores identify the cells that are *reliably* negative (the ones
    every bag agrees look barren).  A final classifier is trained on positives
    vs those reliable negatives and its output is divided by the estimated
    label frequency ``c = E[p(y=1|x in P)]``, which restores a probability
    scale from the PU scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from .config import Config
from .grid import Grid
from .labels import Labels


def spatial_blocks(grid: Grid, block_km: float) -> np.ndarray:
    """Integer block id per cell; used to keep train/test spatially disjoint."""
    size = max(1, int(round(block_km * 1000.0 / grid.res[0])))
    row_band = np.arange(grid.height) // size
    col_band = np.arange(grid.width) // size
    n_col_bands = col_band.max() + 1
    return (row_band[:, None] * n_col_bands + col_band[None, :]).astype("int32")


def make_estimator(cfg: Config, n_estimators: int | None = None):
    m = cfg.model
    if m.algorithm == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=n_estimators or m.n_estimators,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=max(5, m.min_samples_leaf * 5),
            l2_regularization=1.0,
            random_state=m.random_state,
        )
    return RandomForestClassifier(
        n_estimators=n_estimators or m.n_estimators,
        max_depth=m.max_depth,
        min_samples_leaf=m.min_samples_leaf,
        max_features=m.max_features,
        class_weight=m.class_weight,
        n_jobs=-1,
        random_state=m.random_state,
        bootstrap=True,
    )


@dataclass
class Split:
    train_pos: np.ndarray
    train_unl: np.ndarray
    test_pos: np.ndarray
    test_unl: np.ndarray


def spatial_split(labels: Labels, blocks: np.ndarray, cfg: Config) -> Split:
    """Hold out whole spatial blocks so test positives are genuinely unseen."""
    rng = np.random.default_rng(cfg.model.random_state)
    pos_block = blocks[labels.positives[:, 0], labels.positives[:, 1]]
    unl_block = blocks[labels.unlabelled[:, 0], labels.unlabelled[:, 1]]

    counts: dict[int, int] = {}
    for b in pos_block:
        counts[int(b)] = counts.get(int(b), 0) + 1
    order = list(counts)
    rng.shuffle(order)

    target = max(1, int(round(cfg.model.test_fraction * len(pos_block))))
    held: list[int] = []
    total = 0
    for b in order:
        if total >= target:
            break
        held.append(b)
        total += counts[b]
    held_set = set(held)

    test_mask = np.isin(pos_block, held)
    unl_test = np.isin(unl_block, held)
    split = Split(
        train_pos=labels.positives[~test_mask],
        train_unl=labels.unlabelled[~unl_test],
        test_pos=labels.positives[test_mask],
        test_unl=labels.unlabelled[unl_test],
    )
    # Subsample the unlabelled training pool: cells are strongly autocorrelated.
    cap = min(len(split.train_unl), cfg.labels.max_unlabelled)
    if cap < len(split.train_unl):
        idx = rng.choice(len(split.train_unl), size=cap, replace=False)
        split.train_unl = split.train_unl[np.sort(idx)]
    if len(split.test_pos) == 0 or len(split.test_unl) == 0:
        raise ValueError("spatial split produced an empty test set; enlarge the AOI")
    return split


@dataclass
class PUFit:
    model: object
    prior: float
    mode: str
    n_negatives: int
    negative_scores: np.ndarray


def fit_naive(X_pos: np.ndarray, X_unl: np.ndarray, cfg: Config):
    X = np.vstack([X_pos, X_unl])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_unl))])
    return make_estimator(cfg).fit(X, y)


def _bagged_unlabelled_scores(
    X_pos: np.ndarray, X_unl: np.ndarray, cfg: Config, verbose: bool
) -> np.ndarray:
    """Average P(s=1|x) over classifiers trained on P vs bootstrap samples of U."""
    rng = np.random.default_rng(cfg.model.random_state)
    n_u = len(X_unl)
    scores = np.zeros(n_u, dtype="float64")
    for i in range(cfg.model.pu_iterations):
        idx = rng.choice(n_u, size=int(0.8 * n_u), replace=False)
        X = np.vstack([X_pos, X_unl[idx]])
        y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(idx))])
        bag = make_estimator(cfg, n_estimators=max(120, cfg.model.n_estimators // 2))
        bag.fit(X, y)
        scores += bag.predict_proba(X_unl)[:, 1]
        if verbose:
            print(f"    PU bag {i + 1}/{cfg.model.pu_iterations} done")
    return scores / cfg.model.pu_iterations


def fit_pu(
    X_pos: np.ndarray, X_unl: np.ndarray, cfg: Config, verbose: bool = True
) -> PUFit:
    """Bagged PU learning with soft negatives and prior correction.

    1. Bagged classifiers estimate P(s=1|x) for every unlabelled cell.
    2. The final classifier sees all unlabelled cells, but weights each by
       ``1 - score``: cells the bags think are probably mineralised count for
       little, likely-barren cells count fully.  Keeping the whole unlabelled
       pool avoids the selection bias of training only on the extreme
       negatives, which in spatial hold-out generalised noticeably worse.
    3. Scores are divided by ``c = mean p`` on the known positives, the
       Elkan-Noto prior correction, to recover a probability scale.
    4. ``pu_mode: hard`` instead keeps only the lowest-scoring fraction of U as
       reliable negatives (classic Elkan-Noto, kept for comparison).
    """
    n_u = len(X_unl)
    scores = _bagged_unlabelled_scores(X_pos, X_unl, cfg, verbose)
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(n_u)])
    X = np.vstack([X_pos, X_unl])

    mode = cfg.model.pu_mode
    if mode == "hard":
        k = max(1, int(round(cfg.model.pu_negative_fraction * n_u)))
        keep = np.argsort(scores)[:k]
        X = np.vstack([X_pos, X_unl[keep]])
        y = np.concatenate([np.ones(len(X_pos)), np.zeros(k)])
        weights = None
    else:
        weights = np.concatenate([np.ones(len(X_pos)), np.clip(1.0 - scores, 0.02, 1.0)])

    final = make_estimator(cfg).fit(X, y, sample_weight=weights)
    prior = float(np.mean(final.predict_proba(X_pos)[:, 1]))
    prior = min(max(prior, 1e-3), 1.0)
    return PUFit(
        model=final,
        prior=prior,
        mode=mode,
        n_negatives=int(len(y) - len(X_pos)),
        negative_scores=scores,
    )


def score_cells(model, X: np.ndarray, prior: float | None = None) -> np.ndarray:
    p = model.predict_proba(X)[:, 1]
    if prior is not None:
        p = np.clip(p / prior, 0.0, 1.0)
    return p.astype("float32")


def score_grid(model, cube, prior: float | None = None) -> np.ndarray:
    """Probability surface for every cell of the analysis grid."""
    X = cube.sample(np.argwhere(np.ones(cube.grid.shape, dtype=bool)))
    return score_cells(model, X.to_numpy("float32"), prior).reshape(cube.grid.shape)
