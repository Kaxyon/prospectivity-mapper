"""Raster helpers: resampling to the analysis grid, writing GeoTIFFs, and a
handful of small numerical utilities shared by the layer builders.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from .grid import Grid

RESAMPLING = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
    "mode": Resampling.mode,
}


def read_band(path: str | Path, band: int = 1) -> tuple[np.ndarray, dict]:
    """Read one band as float32 with nodata -> NaN."""
    with rasterio.open(path) as src:
        arr = src.read(band).astype("float32")
        nodata = src.nodata
        meta = {
            "transform": src.transform,
            "crs": src.crs,
            "nodata": nodata,
            "width": src.width,
            "height": src.height,
            "count": src.count,
        }
    if nodata is not None and not np.isnan(nodata):
        arr[arr == nodata] = np.nan
    return arr, meta


def warp_to_grid(
    src_path: str | Path,
    grid: Grid,
    method: str = "bilinear",
    src_nodata: float | None = None,
) -> np.ndarray:
    """Resample a raster onto the analysis grid.

    Returns a float32 array of the grid's shape with NaN outside coverage.
    """
    with rasterio.open(src_path) as src:
        arr = np.full(grid.shape, np.nan, dtype="float32")
        nodata = src.nodata if src_nodata is None else src_nodata
        reproject(
            source=rasterio.band(src, 1),
            destination=arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=nodata,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=np.nan,
            resampling=RESAMPLING[method],
        )
    return arr


def write_geotiff(
    array: np.ndarray,
    grid: Grid,
    path: str | Path,
    nodata: float = np.nan,
    dtype: str = "float32",
    compress: str = "deflate",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array, dtype=dtype)
    profile = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": 1,
        "dtype": dtype,
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": nodata,
        "compress": compress,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return path


def gradient_magnitude(arr: np.ndarray, res_x: float, res_y: float) -> np.ndarray:
    """Horizontal gradient magnitude of a potential-field grid.

    For a field in units U, the result is in U per metre; callers usually
    rescale to per-kilometre units for readability.
    """
    gy, gx = np.gradient(np.nan_to_num(arr, nan=np.nanmean(arr)), res_y, res_x)
    return np.hypot(gx, gy)


def finite_zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score that tolerates NaN, leaving NaN in place."""
    out = np.full(arr.shape, np.nan, dtype="float32")
    m = np.isfinite(arr)
    if m.any():
        mu = float(np.mean(arr[m]))
        sd = float(np.std(arr[m])) or 1.0
        out[m] = (arr[m] - mu) / sd
    return out


def fill_nan_nearest(arr: np.ndarray) -> np.ndarray:
    """Fill NaN cells with the nearest finite value (for derivative windows)."""
    from scipy import ndimage

    out = np.asarray(arr, dtype="float32").copy()
    mask = ~np.isfinite(out)
    if not mask.any():
        return out
    if mask.all():
        raise ValueError("cannot fill an all-NaN raster")
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return out[tuple(idx)]


def block_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by an integer factor using NaN-aware block means."""
    if factor <= 1:
        return arr
    h, w = arr.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    crop = arr[:h2, :w2]
    stack = crop.reshape(h2 // factor, factor, w2 // factor, factor)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stack, axis=(1, 3))
