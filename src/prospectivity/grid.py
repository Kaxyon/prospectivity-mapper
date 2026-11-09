"""The analysis grid: one affine, pixel-aligned raster lattice for everything.

Every layer in this project -- DEM derivatives, rasterised geology, geophysics,
interpolated geochemistry and the final probability surface -- is resampled
onto the same UTM lattice.  That is what lets us stack them into a single
feature cube and sample it with pixel coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.coords import BoundingBox


@dataclass(frozen=True)
class Grid:
    """A north-up raster lattice."""

    crs: CRS
    transform: Affine
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("grid must have positive dimensions")
        if self.transform.b != 0 or self.transform.d != 0:
            raise ValueError("only north-up grids are supported")

    # ------------------------------------------------------------------ basics
    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def res(self) -> tuple[float, float]:
        return (abs(self.transform.a), abs(self.transform.e))

    @property
    def bounds(self) -> BoundingBox:
        left, top = self.transform.c, self.transform.f
        return BoundingBox(
            left=left,
            bottom=top - self.height * self.res[1],
            right=left + self.width * self.res[0],
            top=top,
        )

    @property
    def bounds_tuple(self) -> tuple[float, float, float, float]:
        b = self.bounds
        return (b.left, b.bottom, b.right, b.top)

    @property
    def cell_area_km2(self) -> float:
        return (self.res[0] * self.res[1]) / 1e6

    def xs(self) -> np.ndarray:
        """x coordinate of each column centre."""
        return self.bounds.left + (np.arange(self.width) + 0.5) * self.res[0]

    def ys(self) -> np.ndarray:
        """y coordinate of each row centre (north to south)."""
        return self.bounds.top - (np.arange(self.height) + 0.5) * self.res[1]

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        """2-D arrays of cell-centre x / y coordinates."""
        xx, yy = np.meshgrid(self.xs(), self.ys())
        return xx, yy

    def centers_wgs84(self) -> tuple[np.ndarray, np.ndarray]:
        xx, yy = self.centers()
        tr = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(xx, yy)
        return np.asarray(lon), np.asarray(lat)

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        """(row, col) of a projected point; may fall outside the grid."""
        col = int(math.floor((x - self.bounds.left) / self.res[0]))
        row = int(math.floor((self.bounds.top - y) / self.res[1]))
        return row, col

    def contains(self, x: float, y: float) -> bool:
        b = self.bounds
        return b.left <= x < b.right and b.bottom < y <= b.top

    def window(self, west: float, south: float, east: float, north: float):
        """Rasterio Window covering projected bounds, clipped to the grid."""
        from rasterio.windows import Window

        b = self.bounds
        col0 = max(0, int(math.floor((west - b.left) / self.res[0])))
        col1 = min(self.width, int(math.ceil((east - b.left) / self.res[0])))
        row0 = max(0, int(math.floor((b.top - north) / self.res[1])))
        row1 = min(self.height, int(math.ceil((b.top - south) / self.res[1])))
        return Window(col0, row0, max(0, col1 - col0), max(0, row1 - row0))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        b = self.bounds
        return (
            f"Grid({self.width}x{self.height} @ {self.res[0]:.0f} m, "
            f"{self.crs.to_string()}, bounds={b.left:.0f},{b.bottom:.0f},"
            f"{b.right:.0f},{b.top:.0f})"
        )


def grid_from_aoi(
    aoi_wgs84: tuple[float, float, float, float],
    crs: str,
    resolution_m: float,
) -> Grid:
    """Build a lattice covering a lon/lat AOI, snapped to whole cells.

    The AOI corners are projected to the working CRS and the resulting
    envelope is expanded to the next whole cell, so neighbouring runs on the
    same AOI always produce identical cell centres.
    """
    west, south, east, north = aoi_wgs84
    dst = CRS.from_user_input(crs)
    tr = Transformer.from_crs("EPSG:4326", dst, always_xy=True)

    # Densify the AOI edges: a straight edge in lon/lat is not straight in UTM.
    lons = np.concatenate([
        np.linspace(west, east, 64),
        np.linspace(west, east, 64),
        np.full(64, west),
        np.full(64, east),
    ])
    lats = np.concatenate([
        np.full(64, south),
        np.full(64, north),
        np.linspace(south, north, 64),
        np.linspace(south, north, 64),
    ])
    xs, ys = tr.transform(lons, lats)

    left = math.floor(min(xs) / resolution_m) * resolution_m
    right = math.ceil(max(xs) / resolution_m) * resolution_m
    bottom = math.floor(min(ys) / resolution_m) * resolution_m
    top = math.ceil(max(ys) / resolution_m) * resolution_m

    width = int(round((right - left) / resolution_m))
    height = int(round((top - bottom) / resolution_m))
    transform = Affine(resolution_m, 0.0, left, 0.0, -resolution_m, top)
    return Grid(crs=dst, transform=transform, width=width, height=height)


def grid_from_reference(path) -> Grid:
    """Reconstruct a Grid from an existing GeoTIFF."""
    import rasterio

    with rasterio.open(path) as src:
        return Grid(
            crs=CRS.from_user_input(src.crs),
            transform=src.transform,
            width=src.width,
            height=src.height,
        )
