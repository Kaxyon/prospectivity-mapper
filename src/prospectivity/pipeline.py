"""End-to-end orchestration: build -> train -> evaluate -> explain -> map."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import Config
from .evaluate import bootstrap_auc_ci, capture_curve, ranking_metrics
from .explain import ShapResult, explain_model, local_explanation
from .features import FeatureCube, build_feature_cube, load_cube, save_cube
from .grid import Grid, grid_from_aoi
from .labels import Labels, build_labels
from .model import (
    Split,
    fit_naive,
    fit_pu,
    score_cells,
    score_grid,
    spatial_blocks,
    spatial_split,
)
from .raster import write_geotiff
from .sources import load_deposits
from . import viz


@dataclass
class RunResult:
    grid: Grid
    cube: FeatureCube
    labels: Labels
    split: Split
    naive_model: object
    pu_model: object
    prior: float
    metrics: dict[str, dict]
    prob: np.ndarray
    shap_result: ShapResult | None = None


def _xy_to_lonlat(grid: Grid, rows: np.ndarray, cols: np.ndarray):
    from pyproj import Transformer

    x = grid.bounds.left + (cols + 0.5) * grid.res[0]
    y = grid.bounds.top - (rows + 0.5) * grid.res[1]
    tr = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    return tr.transform(x, y)


def top_targets(
    prob: np.ndarray,
    grid: Grid,
    labels: Labels,
    raw_prob: np.ndarray | None = None,
    n: int = 20,
    min_separation_km: float = 3.0,
    min_distance_to_known_km: float = 5.0,
) -> pd.DataFrame:
    """Greedy non-maximum suppression over a (smoothed) probability surface."""
    raw = prob if raw_prob is None else raw_prob
    flat = prob.ravel()
    order = np.argsort(-flat)
    chosen: list[tuple[int, int]] = []
    sep_cells = max(1, int(round(min_separation_km * 1000 / grid.res[0])))
    for idx in order:
        if not np.isfinite(flat[idx]):
            continue
        r, c = np.unravel_index(idx, prob.shape)
        if labels.distance_km[r, c] < min_distance_to_known_km:
            continue
        if any((abs(r - rr) <= sep_cells and abs(c - cc) <= sep_cells)
               for rr, cc in chosen):
            continue
        chosen.append((int(r), int(c)))
        if len(chosen) >= n:
            break

    rows = np.array([r for r, _ in chosen])
    cols = np.array([c for _, c in chosen])
    lon, lat = _xy_to_lonlat(grid, rows, cols)
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(chosen) + 1),
            "row": rows,
            "col": cols,
            "probability": prob[rows, cols],
            "probability_raw": raw[rows, cols],
            "lon": lon,
            "lat": lat,
            "nearest_deposit_km": labels.distance_km[rows, cols],
        }
    )


def run(cfg: Config, reuse_cache: bool = False) -> RunResult:
    sa = cfg.study_area
    grid = grid_from_aoi(sa.aoi_wgs84, sa.crs, sa.resolution_m)
    print(f"Study area: {sa.name}\n  {grid}")

    print("\n=== Feature engineering ===")
    cache = cfg.paths.processed / "feature_cube.npz"
    if reuse_cache and cache.exists():
        cube = load_cube(cache, grid)
        print(f"  loaded cached cube: {len(cube.names)} features")
    else:
        cube = build_feature_cube(cfg, grid)
        save_cube(cube, cache)
        print(f"  {len(cube.names)} features on a {grid.height} x {grid.width} grid")

    print("\n=== Labelling ===")
    deposits = load_deposits(cfg)
    labels = build_labels(cfg, grid, deposits)

    print("\n=== Spatial split ===")
    blocks = spatial_blocks(grid, cfg.model.spatial_block_km)
    split = spatial_split(labels, blocks, cfg)
    print(
        f"  train: {len(split.train_pos)} pos / {len(split.train_unl)} unlabelled\n"
        f"  test : {len(split.test_pos)} pos / {len(split.test_unl)} unlabelled"
    )

    X_tr_pos = cube.sample(split.train_pos).to_numpy("float32")
    X_tr_unl = cube.sample(split.train_unl).to_numpy("float32")
    X_te_pos = cube.sample(split.test_pos).to_numpy("float32")
    X_te_unl = cube.sample(split.test_unl).to_numpy("float32")

    print("\n=== Training (naive baseline) ===")
    naive = fit_naive(X_tr_pos, X_tr_unl, cfg)

    naive_scores = score_cells(naive, X_te_pos), score_cells(naive, X_te_unl)
    metrics = {"naive": ranking_metrics(*naive_scores)}
    curves = {"naive (P vs background)": capture_curve(*naive_scores)}

    pu = None
    if cfg.model.pu_learning:
        print("\n=== Training (PU learning) ===")
        pu = fit_pu(X_tr_pos, X_tr_unl, cfg)
        print(f"  PU mode: {pu.mode} | unlabelled used: {pu.n_negatives} | "
              f"estimated prior c={pu.prior:.3f}")
        pu_scores = (
            score_cells(pu.model, X_te_pos, pu.prior),
            score_cells(pu.model, X_te_unl, pu.prior),
        )
        metrics["pu"] = ranking_metrics(*pu_scores)
        curves["PU (soft negatives)"] = capture_curve(*pu_scores)

    for name, m in metrics.items():
        lo, hi = bootstrap_auc_ci(
            score_cells(naive if name == "naive" else pu.model, X_te_pos,
                        None if name == "naive" else pu.prior),
            score_cells(naive if name == "naive" else pu.model, X_te_unl,
                        None if name == "naive" else pu.prior),
        )
        m["roc_auc_ci95"] = [lo, hi]
        print(f"  {name:5s} ROC-AUC={m['roc_auc']:.3f} [{lo:.3f}-{hi:.3f}]  "
              f"AP={m['average_precision']:.3f}  "
              f"capture@10%={m['capture_at_10pct']:.2f}")

    # Model selection.  The default is the naive random forest: it performed
    # best in development.  "auto" picks the higher average precision on the
    # spatial hold-out (note: that is also the reporting set).
    selected = cfg.model.selection
    if selected == "auto":
        selected = max(metrics, key=lambda k: metrics[k]["average_precision"])
    if selected == "pu" and pu is None:
        selected = "naive"
    model = pu.model if selected == "pu" else naive
    prior = pu.prior if selected == "pu" else None
    print(f"  selected model: {selected}")

    print("\n=== Scoring the grid ===")
    prob = score_grid(model, cube, prior)
    prob_naive = score_grid(naive, cube)
    # A 500 m random forest surface is speckled by leaf-level noise; a light
    # blur gives the regional picture exploration teams actually rank.
    from scipy import ndimage

    prob_smooth = ndimage.gaussian_filter(prob, sigma=1.0).astype("float32")

    out = cfg.paths.output
    out.mkdir(parents=True, exist_ok=True)
    write_geotiff(prob, grid, out / "prospectivity_probability.tif")
    write_geotiff(prob_smooth, grid, out / "prospectivity_probability_smoothed.tif")
    write_geotiff(prob_naive, grid, out / "prospectivity_probability_naive.tif")
    write_geotiff(labels.distance_km, grid, out / "distance_to_known_deposits_km.tif")

    targets = top_targets(
        prob_smooth, grid, labels, raw_prob=prob,
        n=cfg.targeting.max_targets,
        min_separation_km=cfg.targeting.min_separation_km,
        min_distance_to_known_km=cfg.targeting.min_distance_km,
    )
    targets.to_csv(out / "top_targets.csv", index=False)
    labels.deposits.to_csv(out / "deposits_used.csv", index=False)

    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    viz.plot_study_area(
        cube.layers["elevation_m"], grid, labels.deposits, figures / "01_study_area.png"
    )
    viz.plot_feature_maps(
        cube, grid, figures / "02_feature_layers.png",
        names=["elevation_m", "slope_deg", "lith_sedimentary_carbonate",
               "lith_volcanic_felsic", "dist_fault_km", "mag_nt",
               "mag_hgrad_nt_per_km", "grav_mgal", "geochem_As_log"],
    )
    viz.plot_prospectivity(
        prob_smooth, grid, labels.deposits, figures / "03_prospectivity_map.png",
        targets, model_label=f"{selected} random forest",
    )
    viz.plot_metrics(
        metrics, curves, figures / "04_model_performance.png", final_key=selected
    )

    print("\n=== SHAP ===")
    shap_result = explain_model(model, cube, cfg, positive_cells=labels.positives)
    importance = shap_result.importance()
    importance.to_csv(out / "shap_importance.csv", index=False)
    np.savez_compressed(
        out / "shap_values.npz",
        values=shap_result.values,
        data=shap_result.data.to_numpy("float32"),
        base_value=np.array([shap_result.base_value]),
        feature_names=np.array(shap_result.feature_names),
        cells=shap_result.sample_cells,
    )
    viz.plot_shap_importance(importance, figures / "05_shap_importance.png")
    viz.plot_shap_beeswarm(shap_result, figures / "06_shap_beeswarm.png")
    top3 = importance["feature"].head(3).tolist()
    viz.plot_shap_dependence(shap_result, top3, figures / "07_shap_dependence.png")

    # Explain the highest-ranked target cell (smoothed surface, so not a
    # single-cell speckle).
    best_cell = np.array([int(targets.iloc[0]["row"]), int(targets.iloc[0]["col"])])
    match = np.where((shap_result.sample_cells == best_cell).all(axis=1))[0]
    if not len(match):
        shap_result = explain_model(
            model, cube, cfg, n_background=500, positive_cells=np.array([best_cell])
        )
        match = np.array([len(shap_result.sample_cells) - 1])
    viz.plot_local_explanation(
        shap_result, int(match[0]), figures / "08_shap_local_explanation.png",
        label=f"cell {tuple(int(v) for v in best_cell)}, "
              f"P={prob_smooth[best_cell[0], best_cell[1]]:.3f}",
    )
    local_explanation(shap_result, int(match[0])).to_csv(
        out / "shap_local_top_cell.csv", index=False
    )

    viz.folium_map(
        prob_smooth, grid, labels.deposits, out / "prospectivity_map.html", targets
    )

    (out / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "selected_model": selected,
            "naive_model": naive,
            "pu_model": pu.model if pu is not None else None,
            "prior": prior,
            "feature_names": cube.names,
            "grid": {
                "crs": grid.crs.to_string(),
                "transform": tuple(grid.transform)[:6],
                "width": grid.width,
                "height": grid.height,
            },
            "config": cfg.raw,
        },
        out / "models" / "prospectivity_model.joblib",
    )

    sa_out = {
        "study_area": sa.name,
        "aoi_wgs84": list(sa.aoi_wgs84),
        "crs": grid.crs.to_string(),
        "resolution_m": sa.resolution_m,
        "n_cells": int(grid.width * grid.height),
        "n_positives": int(labels.n_positives),
        "n_unlabelled_train": int(len(split.train_unl)),
        "features": cube.names,
        "metrics": metrics,
        "selected_model": selected,
        "pu_prior": pu.prior if pu is not None else None,
        "pu_mode": pu.mode if pu is not None else None,
        "unlabelled_used": pu.n_negatives if pu is not None else 0,
    }
    (out / "metrics.json").write_text(json.dumps(sa_out, indent=2))
    print(f"\nWrote outputs to {out}")

    return RunResult(
        grid=grid, cube=cube, labels=labels, split=split, naive_model=naive,
        pu_model=pu.model, prior=pu.prior, metrics=metrics, prob=prob,
        shap_result=shap_result,
    )
