"""Geochemistry layers from USGS NGDB stream-sediment samples.

Coverage is sparse (tens of samples per 100 x 100 km window), so each element
is interpolated with inverse-distance weighting and the result is deliberately
smoothed.  These are the least reliable layers in the stack, which is itself a
useful thing for the SHAP analysis to reveal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import ndimage
from scipy.spatial import cKDTree

from .config import Config
from .grid import Grid
from .raster import finite_zscore

PATHFINDER_GROUP = ("As", "Sb", "Ag", "Cu", "Pb", "Zn")


def _project_samples(samples: pd.DataFrame, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    tr = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    x, y = tr.transform(samples["long_wgs84"].to_numpy(), samples["lat_wgs84"].to_numpy())
    return np.asarray(x), np.asarray(y)


def idw_interpolate(
    values: np.ndarray,
    points_xy: np.ndarray,
    grid: Grid,
    k: int = 10,
    power: float = 2.0,
    max_distance_m: float | None = None,
) -> np.ndarray:
    """Inverse-distance-weighted interpolation of scattered points."""
    ok = np.isfinite(values)
    values = values[ok]
    points = points_xy[ok]
    if len(values) == 0:
        return np.full(grid.shape, np.nan, dtype="float32")
    if len(values) == 1:
        return np.full(grid.shape, float(values[0]), dtype="float32")

    xx, yy = grid.centers()
    targets = np.column_stack([xx.ravel(), yy.ravel()])
    k = min(k, len(values))
    dist, idx = cKDTree(points).query(targets, k=k, workers=-1)

    if max_distance_m is not None:
        dist = np.where(dist > max_distance_m, np.inf, dist)

    with np.errstate(divide="ignore", invalid="ignore"):
        weights = 1.0 / np.power(dist, power)
    exact = ~np.isfinite(weights) & np.isfinite(dist)
    weights = np.where(np.isfinite(weights), weights, 0.0)

    vals = values[idx]
    total = weights.sum(axis=1)
    interp = np.divide(
        (weights * vals).sum(axis=1), total,
        out=np.full(len(total), np.nan), where=total > 0,
    )
    # Cells that coincide with a sample take that sample's value exactly.
    hit = exact.any(axis=1)
    interp[hit] = vals[hit, exact[hit].argmax(axis=1)]
    return interp.reshape(grid.shape).astype("float32")


def geochemistry_layers(
    samples: pd.DataFrame, grid: Grid, cfg: Config
) -> dict[str, np.ndarray]:
    """Log-scaled element grids plus a composite pathfinder anomaly score."""
    x, y = _project_samples(samples, grid)
    xy = np.column_stack([x, y])
    elements = [e for e in cfg.geochemistry.elements if e in samples.columns]
    if not elements:
        raise ValueError("none of the configured geochemistry elements are present")

    logs: dict[str, np.ndarray] = {}
    for element in elements:
        values = pd.to_numeric(samples[element], errors="coerce").to_numpy(dtype="float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            log_values = np.where(values > 0, np.log10(values), np.nan)
        grid_values = idw_interpolate(
            log_values, xy, grid,
            k=10,
            power=cfg.geochemistry.idw_power,
            max_distance_m=cfg.geochemistry.idw_radius_km * 1000.0,
        )
        # A light blur removes IDW "bullseyes" without erasing regional trends.
        grid_values = ndimage.gaussian_filter(
            np.nan_to_num(grid_values, nan=float(np.nanmedian(grid_values))), sigma=1.0
        )
        logs[element] = grid_values.astype("float32")

    out = {f"geochem_{el}_log": arr for el, arr in logs.items()}

    present = [el for el in PATHFINDER_GROUP if el in logs]
    if present:
        z = np.stack([finite_zscore(logs[el]) for el in present])
        with np.errstate(invalid="ignore"):
            score = np.nanmean(z, axis=0)
        out["geochem_anomaly_score"] = score.astype("float32")
    return out
