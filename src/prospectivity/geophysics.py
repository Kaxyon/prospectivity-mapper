"""Potential-field layers from real USGS compilations.

* magnetic: residual total-intensity anomaly, 1 km national grid
* gravity:  complete Bouguer anomaly, 2 arc-minute national grid

Both are regional compilations, so we derive the features that exploration
geologists actually read: gradients and structural complexity, which map
buried contacts and fault intersections rather than absolute field strength.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pyproj import Transformer
from scipy import ndimage
from scipy.spatial import cKDTree

from .geochemistry import idw_interpolate
from .grid import Grid
from .raster import fill_nan_nearest, finite_zscore, gradient_magnitude, warp_to_grid
from .sources import load_gravity_xyz


def fft_vertical_derivative(arr: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """First vertical derivative of a potential field, via the Fourier domain.

    dT/dz <=> |k| * F(k) for a field measured on a plane.  A cosine taper above
    60 % of the Nyquist wavenumber suppresses amplified grid noise.
    """
    a = np.nan_to_num(np.asarray(arr, dtype="float64"), nan=float(np.nanmean(arr)))
    ny, nx = a.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k = np.hypot(kx_grid, ky_grid)
    kmax = float(k.max())
    taper = np.ones_like(k)
    if kmax > 0:
        hi = 0.6 * kmax
        mask = k > hi
        taper[mask] = 0.5 * (1.0 + np.cos(np.pi * (k[mask] - hi) / (kmax - hi)))
    return np.fft.ifft2(np.fft.fft2(a) * k * taper).real


def _local_residual(field: np.ndarray, sigma_cells: float, nan_value: float) -> np.ndarray:
    filled = np.nan_to_num(field, nan=nan_value)
    smooth = ndimage.gaussian_filter(filled, sigma=sigma_cells, mode="nearest")
    return field - smooth


def magnetic_layers(magnetic_path: Path, grid: Grid) -> dict[str, np.ndarray]:
    mag = warp_to_grid(magnetic_path, grid, method="bilinear")
    mag = fill_nan_nearest(mag)
    res_x, res_y = grid.res

    hgrad = gradient_magnitude(mag, res_x, res_y) * 1000.0      # nT / km
    vderiv = fft_vertical_derivative(mag, res_y, res_x) * 1000.0  # nT / km
    analytic = np.hypot(hgrad, vderiv)
    residual = _local_residual(mag, sigma_cells=3.0, nan_value=float(np.nanmedian(mag)))

    return {
        "mag_nt": mag.astype("float32"),
        "mag_hgrad_nt_per_km": hgrad.astype("float32"),
        "mag_vderiv_nt_per_km": vderiv.astype("float32"),
        "mag_analytic_signal": analytic.astype("float32"),
        "mag_residual_nt": residual.astype("float32"),
    }


def gravity_layers(gravity_path: Path, grid: Grid) -> dict[str, np.ndarray]:
    lon, lat, mgl = load_gravity_xyz(Path(gravity_path))
    tr = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    x, y = tr.transform(lon, lat)
    xy = np.column_stack([x, y])

    grav = idw_interpolate(mgl.astype("float64"), xy, grid, k=6, power=2.0)
    grav = fill_nan_nearest(grav)
    res_x, res_y = grid.res

    hgrad = gradient_magnitude(grav, res_x, res_y) * 1000.0  # mGal / km
    residual = _local_residual(grav, sigma_cells=3.0, nan_value=float(np.nanmedian(grav)))

    return {
        "grav_mgal": grav.astype("float32"),
        "grav_hgrad_mgal_per_km": hgrad.astype("float32"),
        "grav_residual_mgal": residual.astype("float32"),
    }


def structural_complexity(
    mag_hgrad: np.ndarray, grav_hgrad: np.ndarray
) -> np.ndarray:
    """Combined normalised gradient energy -- a proxy for fault/contact density."""
    a = finite_zscore(mag_hgrad)
    b = finite_zscore(grav_hgrad)
    return np.hypot(np.nan_to_num(a), np.nan_to_num(b)).astype("float32")


def geophysics_layers(
    magnetic_path: Path, gravity_path: Path, grid: Grid
) -> dict[str, np.ndarray]:
    layers = magnetic_layers(magnetic_path, grid)
    layers.update(gravity_layers(gravity_path, grid))
    layers["structural_complexity"] = structural_complexity(
        layers["mag_hgrad_nt_per_km"], layers["grav_hgrad_mgal_per_km"]
    )
    return layers


__all__ = ["geophysics_layers", "fft_vertical_derivative", "structural_complexity"]

# cKDTree is imported for callers that want to interpolate their own grids.
_ = cKDTree
