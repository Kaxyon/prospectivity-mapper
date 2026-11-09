"""Assembly of the feature cube and the design matrix.

All layers are built on the same lattice, so a "sample" is simply the vector of
layer values at one row/column.  Nothing here uses the deposit locations: the
distance-to-deposit raster is intentionally excluded, because it would leak the
label into the features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .geochemistry import geochemistry_layers
from .geology import geology_layers
from .geophysics import geophysics_layers
from .grid import Grid
from .raster import fill_nan_nearest, warp_to_grid
from .sources import (
    dem_mosaic,
    ensure_dem,
    ensure_geochem,
    ensure_geology,
    ensure_gravity,
    ensure_magnetic,
    load_geochem_samples,
)
from .terrain import terrain_layers

# Human-readable groups used by the report and the SHAP plots.
GROUPS: dict[str, tuple[str, ...]] = {
    "terrain": (
        "elevation_m", "slope_deg", "aspect_sin", "aspect_cos",
        "curvature_1e3_per_km", "roughness_m", "tpi_1km_m", "relief_5km_m",
        "roughness_10km_m",
    ),
    "geology": (
        "lith_intrusive_felsic", "lith_intrusive_mafic", "lith_volcanic_felsic",
        "lith_volcanic_mafic", "lith_sedimentary_carbonate",
        "lith_sedimentary_clastic", "lith_sedimentary_chemical", "lith_metamorphic",
        "lith_surficial", "lith_intrusive_prop_3km", "lith_volcanic_prop_3km",
        "lith_carbonate_prop_3km", "lith_clastic_prop_3km", "lith_metamorphic_prop_3km",
        "lith_surficial_prop_3km", "dist_contact_km", "dist_fault_km",
        "fault_density_5km", "dist_intrusive_km", "dist_carbonate_km",
        "dist_volcanic_km",
    ),
    "geophysics": (
        "mag_nt", "mag_hgrad_nt_per_km", "mag_vderiv_nt_per_km",
        "mag_analytic_signal", "mag_residual_nt", "grav_mgal",
        "grav_hgrad_mgal_per_km", "grav_residual_mgal", "structural_complexity",
    ),
    "geochemistry": (
        "geochem_As_log", "geochem_Sb_log", "geochem_Cu_log", "geochem_Pb_log",
        "geochem_Zn_log", "geochem_Ag_log", "geochem_Fe_log", "geochem_Mn_log",
        "geochem_anomaly_score",
    ),
}

# Individual element grids are kept as layers (and shown on maps) but are not
# fed to the classifier: ~60 samples cannot support eight independent surfaces,
# and in spatial hold-out they degraded discrimination.  The composite anomaly
# score carries the same information at the confidence the data supports.
NON_MODEL_LAYERS = {
    n for n in GROUPS["geochemistry"] if n != "geochem_anomaly_score"
}


@dataclass
class FeatureCube:
    """Layer stack + sampling helpers."""

    grid: Grid
    layers: dict[str, np.ndarray]
    names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.names:
            self.names = list(self.layers)
        missing = [n for n in self.names if n not in self.layers]
        if missing:
            raise KeyError(f"missing layers: {missing}")

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    def stack(self) -> np.ndarray:
        """(n_features, height, width) float32 cube with NaN preserved."""
        return np.stack([self.layers[n] for n in self.names]).astype("float32")

    def sample(self, cells: np.ndarray) -> pd.DataFrame:
        """Design matrix at (row, col) cell coordinates."""
        rows, cols = cells[:, 0], cells[:, 1]
        data = {name: self.layers[name][rows, cols] for name in self.names}
        return pd.DataFrame(data, columns=self.names)

    def group_of(self, name: str) -> str:
        for group, members in GROUPS.items():
            if name in members:
                return group
        return "other"


def _fill_layers(layers: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fill residual NaN with the layer median so the model sees a full grid."""
    out: dict[str, np.ndarray] = {}
    for name, arr in layers.items():
        a = np.asarray(arr, dtype="float32")
        if np.isnan(a).any():
            median = float(np.nanmedian(a))
            if not np.isfinite(median):
                median = 0.0
            a = np.where(np.isfinite(a), a, median).astype("float32")
        out[name] = a
    return out


def save_cube(cube: FeatureCube, path) -> None:
    """Cache the aligned layer stack so experiments skip data engineering."""
    arrays = {name: cube.layers[name] for name in cube.layers}
    arrays["__names"] = np.array(cube.names)
    np.savez_compressed(path, **arrays)


def load_cube(path, grid: Grid) -> FeatureCube:
    with np.load(path, allow_pickle=False) as data:
        names = [str(n) for n in data["__names"]]
        layers = {
            key: data[key].astype("float32")
            for key in data.files if key != "__names"
        }
    return FeatureCube(grid=grid, layers=layers, names=names)


def build_feature_cube(cfg: Config, grid: Grid, verbose: bool = True) -> FeatureCube:
    """Download/parse every source and assemble the aligned layer stack."""
    if verbose:
        print("[1/5] Copernicus DEM ...")
    tiles = ensure_dem(cfg)
    mosaic = dem_mosaic(cfg, tiles)
    dem = warp_to_grid(mosaic, grid, method="bilinear")
    dem = fill_nan_nearest(dem)

    if verbose:
        print("[2/5] terrain derivatives ...")
    layers = terrain_layers(dem, grid)

    if verbose:
        print("[3/5] USGS geology + structure ...")
    geo_dir = ensure_geology(cfg)
    layers.update(geology_layers(geo_dir, grid))

    if verbose:
        print("[4/5] USGS magnetics + gravity ...")
    layers.update(geophysics_layers(ensure_magnetic(cfg), ensure_gravity(cfg), grid))

    if verbose:
        print("[5/5] USGS NGDB stream-sediment geochemistry ...")
    ensure_geochem(cfg)
    samples = load_geochem_samples(cfg)
    layers.update(geochemistry_layers(samples, grid, cfg))

    layers = _fill_layers(layers)

    # Drop degenerate layers (constant values carry no information).
    dropped = [n for n, a in layers.items() if float(np.nanstd(a)) == 0.0]
    for name in dropped:
        layers.pop(name)
    if dropped and verbose:
        print(f"  dropped constant layers: {dropped}")

    names = [
        n for group in GROUPS.values() for n in group
        if n in layers and n not in NON_MODEL_LAYERS
    ]
    names += [n for n in layers if n not in names and n not in NON_MODEL_LAYERS]
    return FeatureCube(grid=grid, layers=layers, names=names)
