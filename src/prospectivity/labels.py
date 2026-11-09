"""Turning deposit records into a Positive-Unlabelled learning problem.

Only the locations of *known* deposits are observed.  The ground everywhere
else is unlabelled, not proven barren -- which is exactly why this is a PU
problem rather than a binary classification problem.  We therefore produce:

* **positives** -- grid cells containing a known Au/Ag deposit,
* **unlabelled** -- every other cell that is far enough from a known deposit to
  be a usable unlabelled example (a buffer zone is withheld entirely, because
  the immediate surroundings of a mine are nearly always mineralised).

A raster of distance-to-nearest-known-deposit is kept for display only; it is
deliberately *never* used as a feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import ndimage

from .config import Config
from .grid import Grid

# MRDS development-status codes -> numeric confidence.
CONFIDENCE = {
    "producer": 3,
    "past producer": 3,
    "prospect": 2,
    "occurrence": 1,
    "unknown": 1,
    "plant": 0,   # processing sites, not deposits
    "mill": 0,
}

COMMODITY_ALIASES = {
    "Au": ("gold", "au"),
    "Ag": ("silver", "ag"),
    "Cu": ("copper", "cu"),
    "Pb": ("lead", "pb"),
    "Zn": ("zinc", "zn"),
    "Mo": ("molybdenum", "mo"),
    "W": ("tungsten", "w"),
}


@dataclass
class Labels:
    """Pixel-space labels plus the metadata needed to explain them."""

    positives: np.ndarray           # (N, 2) int row/col of positive cells
    unlabelled: np.ndarray          # (M, 2) int row/col of usable unlabelled cells
    deposits: pd.DataFrame          # filtered deposit records with x/y
    distance_km: np.ndarray         # distance to nearest known deposit
    exclusion_mask: np.ndarray      # cells withheld from both classes

    @property
    def n_positives(self) -> int:
        return int(len(self.positives))


def select_deposits(deposits: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Filter raw MRDS records down to credible deposits of the target metals."""
    df = deposits.copy()
    commodity_text = (
        df[["commod1", "commod2", "commod3"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    aliases: list[str] = []
    for symbol in cfg.labels.commodities:
        aliases.extend(COMMODITY_ALIASES.get(symbol, (symbol.lower(),)))
    keep = pd.Series(False, index=df.index)
    for alias in aliases:
        keep |= commodity_text.str.contains(rf"\b{alias}\b", regex=True)
    df = df[keep].copy()

    df["confidence"] = (
        df["dev_stat"].fillna("unknown").str.strip().str.lower().map(CONFIDENCE).fillna(1)
    )
    df = df[df["confidence"] >= cfg.labels.min_confidence]
    df = df.drop_duplicates(subset=["dep_id"])
    if cfg.labels.max_positives and len(df) > cfg.labels.max_positives:
        # Prefer better-developed records, then thin deterministically.
        df = df.sort_values(["confidence", "dep_id"], ascending=[False, True])
        df = df.head(cfg.labels.max_positives)
    df = df.reset_index(drop=True)
    df["confidence"] = df["confidence"].astype(int)
    return df


def _project(df: pd.DataFrame, grid: Grid) -> pd.DataFrame:
    tr = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    x, y = tr.transform(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    out = df.copy()
    out["x"] = x
    out["y"] = y
    return out


def distance_raster(points_xy: np.ndarray, grid: Grid) -> np.ndarray:
    """Euclidean distance (km) from every cell to the nearest input point."""
    burned = np.zeros(grid.shape, dtype=bool)
    if len(points_xy):
        col = np.floor((points_xy[:, 0] - grid.bounds.left) / grid.res[0]).astype(int)
        row = np.floor((grid.bounds.top - points_xy[:, 1]) / grid.res[1]).astype(int)
        ok = (row >= 0) & (row < grid.height) & (col >= 0) & (col < grid.width)
        burned[row[ok], col[ok]] = True
    if not burned.any():
        return np.full(grid.shape, np.inf, dtype="float32")
    dist_m = ndimage.distance_transform_edt(
        ~burned, sampling=(grid.res[1], grid.res[0])
    )
    return (dist_m / 1000.0).astype("float32")


def build_labels(cfg: Config, grid: Grid, deposits: pd.DataFrame) -> Labels:
    """Construct positives, unlabelled cells and the exclusion buffer."""
    selected = _project(select_deposits(deposits, cfg), grid)
    if selected.empty:
        raise ValueError(
            "no deposits matched the commodity/confidence filter in this AOI"
        )

    dist_km = distance_raster(selected[["x", "y"]].to_numpy(), grid)

    col = np.floor((selected["x"].to_numpy() - grid.bounds.left) / grid.res[0]).astype(int)
    row = np.floor((grid.bounds.top - selected["y"].to_numpy()) / grid.res[1]).astype(int)
    inside = (row >= 0) & (row < grid.height) & (col >= 0) & (col < grid.width)
    selected = selected.loc[inside].reset_index(drop=True)
    cells = np.unique(np.column_stack([row[inside], col[inside]]), axis=0)
    # Keep one representative record per occupied cell.
    first = {}
    for i, (r, c) in enumerate(zip(row[inside], col[inside], strict=True)):
        first.setdefault((int(r), int(c)), i)
    selected = selected.iloc[list(first.values())].reset_index(drop=True)
    selected["row"] = [r for r, _ in first]
    selected["col"] = [c for _, c in first]

    exclusion = dist_km <= (cfg.labels.exclusion_buffer_m / 1000.0)
    print(
        f"  deposits selected: {len(selected)} | positive cells: {len(cells)} | "
        f"excluded buffer cells: {int(exclusion.sum())} "
        f"({exclusion.mean() * 100:.1f}% of grid)"
    )
    if len(cells) < 20:
        raise ValueError(f"only {len(cells)} positive cells; AOI is too small")

    all_cells = np.argwhere(np.ones(grid.shape, dtype=bool))
    unlabelled = all_cells[~exclusion.ravel()]
    return Labels(
        positives=cells,
        unlabelled=unlabelled,
        deposits=selected,
        distance_km=dist_km,
        exclusion_mask=exclusion,
    )
