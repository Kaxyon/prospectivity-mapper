"""Terrain derivatives computed from the real Copernicus GLO-30 DEM.

The DEM is resampled to the 500 m analysis grid first, so slopes and
curvatures describe landforms at the scale at which the ore system is being
modelled -- not 30 m micro-relief.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .grid import Grid
from .raster import fill_nan_nearest


def _window_stats(arr: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """NaN-tolerant local mean and standard deviation."""
    a = np.asarray(arr, dtype="float64")
    m = np.isfinite(a)
    filled = np.where(m, a, 0.0)
    count = ndimage.uniform_filter(m.astype("float64"), size, mode="nearest")
    total = ndimage.uniform_filter(filled, size, mode="nearest")
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    sq = ndimage.uniform_filter(filled ** 2, size, mode="nearest")
    var = np.divide(sq, count, out=np.zeros_like(sq), where=count > 0) - mean ** 2
    return mean, np.sqrt(np.clip(var, 0, None))


def terrain_layers(dem: np.ndarray, grid: Grid) -> dict[str, np.ndarray]:
    """Return the terrain feature set on the analysis grid."""
    z = fill_nan_nearest(np.asarray(dem, dtype="float32"))
    dy, dx = grid.res[1], grid.res[0]

    # Gradients; array rows run southwards, so flip the y derivative for a
    # conventional geographic sign.
    gy_row, gx = np.gradient(z.astype("float64"), dy, dx)
    gy = -gy_row
    slope_rad = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)  # clockwise from north, direction of dip

    # Curvature: Laplacian of elevation, expressed per kilometre of distance.
    laplacian = np.gradient(gx, dx, axis=1) + np.gradient(-gy_row, dy, axis=0)
    curvature = laplacian * 1000.0

    mean3, std3 = _window_stats(z, 3)
    tpi_1km = z - mean3

    local_max = ndimage.maximum_filter(z, size=11, mode="nearest")
    local_min = ndimage.minimum_filter(z, size=11, mode="nearest")
    relief_5km = local_max - local_min

    _, std_wide = _window_stats(z, 21)

    return {
        "elevation_m": z.astype("float32"),
        "slope_deg": np.degrees(slope_rad).astype("float32"),
        "aspect_sin": np.sin(aspect).astype("float32"),
        "aspect_cos": np.cos(aspect).astype("float32"),
        "curvature_1e3_per_km": curvature.astype("float32"),
        "roughness_m": (std3 * 2.0).astype("float32"),
        "tpi_1km_m": tpi_1km.astype("float32"),
        "relief_5km_m": relief_5km.astype("float32"),
        "roughness_10km_m": std_wide.astype("float32"),
    }
