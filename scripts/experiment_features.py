"""Feature-set experiment: does adding contact-distance / regional-context
features improve spatial-holdout discrimination?

Run after a pipeline run has cached data/processed/feature_cube.npz:

    PYTHONPATH=src python scripts/experiment_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prospectivity.config import load_config  # noqa: E402
from prospectivity.evaluate import ranking_metrics  # noqa: E402
from prospectivity.features import load_cube  # noqa: E402
from prospectivity.grid import grid_from_aoi  # noqa: E402
from prospectivity.labels import build_labels  # noqa: E402
from prospectivity.model import (  # noqa: E402
    fit_naive, fit_pu, score_cells, spatial_blocks, spatial_split,
)
from prospectivity.sources import load_deposits  # noqa: E402

cfg = load_config("configs/study_area.yaml")
grid = grid_from_aoi(cfg.study_area.aoi_wgs84, cfg.study_area.crs,
                     cfg.study_area.resolution_m)
cube = load_cube(cfg.paths.processed / "feature_cube.npz", grid)
labels = build_labels(cfg, grid, load_deposits(cfg))
split = spatial_split(labels, spatial_blocks(grid, cfg.model.spatial_block_km), cfg)

extra = {}
for group, layer in [
    ("intrusive", "lith_intrusive_felsic"),
    ("carbonate", "lith_sedimentary_carbonate"),
    ("volcanic", "lith_volcanic_felsic"),
]:
    mask = cube.layers[layer] > 0.5
    if mask.any():
        dist = ndimage.distance_transform_edt(
            ~mask, sampling=(grid.res[1], grid.res[0])
        ) / 1000.0
        extra[f"dist_{group}_km"] = dist.astype("float32")

for group in ["intrusive", "volcanic", "carbonate", "clastic", "surficial"]:
    layer = f"lith_{group}"
    if layer in cube.layers:
        extra[f"lith_{group}_prop_10km"] = ndimage.uniform_filter(
            cube.layers[layer], size=21, mode="nearest"
        ).astype("float32")

cube.layers.update(extra)
print("extra features:", list(extra))

onehot = [n for n in cube.names if n.startswith("lith_") and not "prop" in n]
base = list(cube.names)


def run_variant(name, names):
    cube.names = names
    Xp = cube.sample(split.train_pos).to_numpy("float32")
    Xu = cube.sample(split.train_unl).to_numpy("float32")
    Tp = cube.sample(split.test_pos).to_numpy("float32")
    Tu = cube.sample(split.test_unl).to_numpy("float32")
    m = fit_naive(Xp, Xu, cfg)
    mt = ranking_metrics(score_cells(m, Tp), score_cells(m, Tu))
    print(f"{name:34s} n={len(names):2d} AUC={mt['roc_auc']:.3f} "
          f"AP={mt['average_precision']:.3f} cap@10%={mt['capture_at_10pct']:.2f}")
    return mt, (Xp, Xu, Tp, Tu)


print("\n--- naive RF, same spatial split ---")
run_variant("base (43)", base)
run_variant("base + contact distances", base + [k for k in extra if k.startswith("dist_")])
run_variant("base + regional props", base + [k for k in extra if "prop_10km" in k])
run_variant("base + all extra", base + list(extra))
run_variant("base - one-hot lith + all extra",
            [n for n in base if n not in onehot] + list(extra))
all_names = base + list(extra)
_, data = run_variant("best candidate for PU", all_names)
Xp, Xu, Tp, Tu = data
print("\n--- PU on base + all extra ---")
pu = fit_pu(Xp, Xu, cfg)
mt = ranking_metrics(score_cells(pu.model, Tp, pu.prior), score_cells(pu.model, Tu, pu.prior))
print(f"PU AUC={mt['roc_auc']:.3f} AP={mt['average_precision']:.3f} "
      f"cap@10%={mt['capture_at_10pct']:.2f} prior={pu.prior:.3f}")
