"""Geology and structure layers from the USGS Nevada state geologic map.

Polygons carry a controlled-vocabulary lithology (``NV_lith.csv``); faults come
from the companion structure layer.  Because a 500 m cell routinely straddles
several map units, we keep both the dominant class and the *proportion* of key
rock types within a 3.5 km neighbourhood -- the latter is what actually
expresses "this area sits in carbonate host rock next to an intrusive".
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from scipy import ndimage

from .grid import Grid
from .sources import LITH_CLASSES, classify_generalize, load_geology_tables

GROUP_MASKS = {
    "intrusive": (LITH_CLASSES["intrusive_felsic"], LITH_CLASSES["intrusive_mafic"]),
    "volcanic": (LITH_CLASSES["volcanic_felsic"], LITH_CLASSES["volcanic_mafic"]),
    "carbonate": (LITH_CLASSES["sedimentary_carbonate"],),
    "clastic": (LITH_CLASSES["sedimentary_clastic"],),
    "metamorphic": (LITH_CLASSES["metamorphic"],),
    "surficial": (LITH_CLASSES["surficial"],),
}


def _bbox_wgs84(grid: Grid) -> tuple[float, float, float, float]:
    tr = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    b = grid.bounds
    lons, lats = tr.transform(
        [b.left, b.right, b.left, b.right], [b.bottom, b.bottom, b.top, b.top]
    )
    return min(lons), min(lats), max(lons), max(lats)


def _crop_and_project(gdf: gpd.GeoDataFrame, grid: Grid) -> gpd.GeoDataFrame:
    west, south, east, north = _bbox_wgs84(grid)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf.cx[west:east, south:north]
    gdf = gdf.to_crs(grid.crs)
    try:
        gdf = gdf.set_geometry(gdf.geometry.make_valid())
    except Exception:  # pragma: no cover - older shapely fallback
        gdf = gdf.set_geometry(gdf.geometry.buffer(0))
    return gdf


def _rasterize(shapes, grid: Grid, dtype: str = "uint8", all_touched: bool = False):
    import rasterio.features

    return rasterio.features.rasterize(
        shapes,
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        dtype=dtype,
        all_touched=all_touched,
    )


def _distance_km(mask: np.ndarray, grid: Grid) -> np.ndarray:
    dist = ndimage.distance_transform_edt(~mask, sampling=(grid.res[1], grid.res[0]))
    return (dist / 1000.0).astype("float32")


def geology_layers(geo_dir: Path, grid: Grid) -> dict[str, np.ndarray]:
    units, _ = load_geology_tables(geo_dir)
    class_of_unit = dict(zip(units["unit_link"].astype(str), units["lith_class"]))

    polys = gpd.read_file(
        geo_dir / "NV_geol_poly.shp", columns=["UNIT_LINK", "GENERALIZE", "geometry"]
    )
    polys = _crop_and_project(polys, grid)
    polys["lith_class"] = (
        polys["UNIT_LINK"].astype(str).map(class_of_unit).fillna(0).astype("int32")
    )
    # Fall back on the map's generalised description for units the unit table
    # does not name.
    fallback = polys["GENERALIZE"].map(classify_generalize).astype("int32")
    polys.loc[polys["lith_class"] == 0, "lith_class"] = fallback[
        polys["lith_class"] == 0
    ]
    shapes = [
        (geom, int(code))
        for geom, code in zip(polys.geometry, polys["lith_class"], strict=True)
        if geom is not None and not geom.is_empty
    ]
    class_map = _rasterize(shapes, grid).astype("int16")

    out: dict[str, np.ndarray] = {}
    for name, code in LITH_CLASSES.items():
        if code == 0:
            continue
        out[f"lith_{name}"] = (class_map == code).astype("float32")

    # Neighbourhood proportions: does this cell sit inside a belt of the rock type?
    for group, codes in GROUP_MASKS.items():
        mask = np.isin(class_map, codes).astype("float32")
        out[f"lith_{group}_prop_3km"] = ndimage.uniform_filter(
            mask, size=7, mode="nearest"
        ).astype("float32")

    # Distance to the nearest exposure of each key rock group (km).  Being
    # 2 km from an intrusive contact means something different from being 2 km
    # from a fault, and the model benefits from seeing both explicitly.
    for group, codes in GROUP_MASKS.items():
        mask = np.isin(class_map, codes)
        if mask.any():
            out[f"dist_{group}_km"] = _distance_km(mask, grid)

    # Distance to the nearest lithological contact.
    edge = np.zeros(grid.shape, dtype=bool)
    edge[:, :-1] |= class_map[:, :-1] != class_map[:, 1:]
    edge[:-1, :] |= class_map[:-1, :] != class_map[1:, :]
    edge[class_map == 0] = False
    out["dist_contact_km"] = _distance_km(edge, grid)

    # Structural layer: faults only (the file also holds folds and lineaments).
    struct = gpd.read_file(geo_dir / "NV_structure.shp", columns=["DESCRIPT", "geometry"])
    faults = struct[struct["DESCRIPT"].fillna("").str.contains("fault", case=False)]
    faults = _crop_and_project(faults, grid)
    fault_mask = _rasterize(
        [(g, 1) for g in faults.geometry if g is not None and not g.is_empty],
        grid,
        all_touched=True,
    ).astype(bool)

    out["dist_fault_km"] = _distance_km(fault_mask, grid)
    # Line density in km of fault per km^2, in a 5.5 km window.
    cell_km = grid.res[0] / 1000.0
    cell_area_km2 = grid.cell_area_km2
    out["fault_density_5km"] = (
        ndimage.uniform_filter(fault_mask.astype("float32"), size=11, mode="nearest")
        * cell_km / cell_area_km2
    ).astype("float32")

    covered = class_map > 0
    print(
        f"  geology: {len(shapes)} polygons, {int(faults.shape[0])} fault features, "
        f"lithology coverage {covered.mean() * 100:.1f}% of grid"
    )
    return out
