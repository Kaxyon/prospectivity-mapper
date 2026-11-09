"""Discovery, download and parsing of the free public datasets.

Every layer comes from a free, no-authentication USGS/Copernicus endpoint:

===============  ==============================================================
layer            source
===============  ==============================================================
DEM              Copernicus DEM GLO-30 on the public AWS Open Data bucket
geology          USGS SGMC-derived Nevada state geologic map (polygons + faults)
geophysics       USGS residual magnetic anomaly grid + complete Bouguer grid
geochemistry     USGS NGDB stream-sediment samples (NURE)
deposits         USGS MRDS mineral deposit records
===============  ==============================================================

Downloaded archives are cached under ``data/raw`` and never re-fetched; parsing
is cached as small CSV files under ``data/processed``.
"""

from __future__ import annotations

import gzip
import io
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

USER_AGENT = "prospectivity-mapper/1.0 (open research project)"
CHUNK = 1 << 20


class DownloadError(RuntimeError):
    """Raised when a required public dataset cannot be obtained."""


# --------------------------------------------------------------------------- io
def download(url: str, dest: Path, min_bytes: int = 1024, retries: int = 3) -> Path:
    """Download *url* to *dest* unless a plausible copy already exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        tmp = dest.with_name(dest.name + ".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as fh:
                shutil.copyfileobj(resp, fh, length=CHUNK)
            size = tmp.stat().st_size
            if size < min_bytes:
                raise DownloadError(f"{url}: got only {size} bytes")
            tmp.replace(dest)
            return dest
        except (urllib.error.URLError, DownloadError, TimeoutError, OSError) as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 * attempt)
    raise DownloadError(f"could not download {url}: {last_error}")


def extract_zip(zip_path: Path, dest_dir: Path, required: str | None = None) -> Path:
    """Extract a zip if needed; return the directory (or the required member)."""
    marker = dest_dir / (required or "")
    if required and marker.exists():
        return marker
    if dest_dir.exists() and any(dest_dir.iterdir()) and required is None:
        return dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    return dest_dir / required if required else dest_dir


# ------------------------------------------------------------------------ DEM
def dem_tile_names(cfg: Config) -> list[str]:
    """Copernicus GLO-30 tile stems covering the AOI."""
    sa = cfg.study_area
    template = cfg.sources.dem.tile_template or "Copernicus_DSM_COG_10_{lat}_{lon}_DEM"
    names = []
    for lat in range(int(np.floor(sa.south)), int(np.ceil(sa.north))):
        for lon in range(int(np.floor(sa.west)), int(np.ceil(sa.east))):
            names.append(
                template.format(lat=f"N{lat:02d}_00", lon=f"W{abs(lon):03d}_00")
            )
    return names


def ensure_dem(cfg: Config) -> list[Path]:
    """Download the Copernicus DEM tiles needed for the AOI."""
    bucket = (cfg.sources.dem.bucket or "").rstrip("/")
    out_dir = cfg.paths.raw / "dem"
    tiles: list[Path] = []
    for name in dem_tile_names(cfg):
        dest = out_dir / f"{name}.tif"
        try:
            download(f"{bucket}/{name}/{name}.tif", dest, min_bytes=1_000_000)
        except DownloadError as exc:  # e.g. a tile that only covers ocean
            print(f"  ! skipping DEM tile {name}: {exc}")
            continue
        tiles.append(dest)
    if not tiles:
        raise DownloadError("no Copernicus DEM tiles could be retrieved")
    return tiles


def dem_mosaic(cfg: Config, tiles: list[Path]) -> Path:
    """Mosaic the tiles and crop them to the AOI; cache the result."""
    import rasterio
    from rasterio.merge import merge

    out = cfg.paths.processed / "dem_mosaic.tif"
    if out.exists():
        return out

    sources = [rasterio.open(t) for t in tiles]
    try:
        mosaic, transform = merge(sources)
        meta = sources[0].meta.copy()
    finally:
        for src in sources:
            src.close()

    meta.update(
        {
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "compress": "deflate",
            "dtype": "float32",
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **meta) as dst:
        dst.write(mosaic.astype("float32"))

    # Crop to the AOI envelope (plus one pixel) to keep the file small.
    from rasterio.windows import from_bounds

    sa = cfg.study_area
    with rasterio.open(out) as src:
        win = from_bounds(
            sa.west - 0.01, sa.south - 0.01, sa.east + 0.01, sa.north + 0.01,
            transform=src.transform,
        )
        data = src.read(1, window=win)
        prof = src.profile.copy()
        prof.update(
            height=data.shape[0], width=data.shape[1],
            transform=src.window_transform(win), dtype="float32",
        )
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(data.astype("float32"), 1)
    return out


# ------------------------------------------------------------------ geology
LITH_CLASSES = {
    "unknown": 0,
    "intrusive_felsic": 1,
    "intrusive_mafic": 2,
    "volcanic_felsic": 3,
    "volcanic_mafic": 4,
    "sedimentary_carbonate": 5,
    "sedimentary_clastic": 6,
    "sedimentary_chemical": 7,
    "metamorphic": 8,
    "surficial": 9,
}

_FELSIC = ("granit", "rhyolit", "tonalit", "granodiorit", "syenit", "felsic",
           "quartz monzonit", "dacit", "aplite", "pegmatit")
_MAFIC = ("gabbro", "basalt", "diorit", "andesit", "mafic", "peridotit",
          "serpentinit", "alkalic", "trachy", "komatiit", "ultramafic")


def _lith_class(row: pd.Series) -> int:
    """Map a controlled-vocabulary lithology description onto a class code."""
    parts = [str(row.get(k, "")) for k in ("lith1", "lith2", "lith3", "rocktype1")]
    text = " ".join(p for p in parts if p and p != "nan").lower()
    if not text.strip():
        return LITH_CLASSES["unknown"]
    if row.get("lith1") == "Unconsolidated" or any(
        k in text for k in ("alluvium", "surficial", "glacial", "eolian", "dune",
                            "sand and gravel", "mass wasting", "playa", "water")
    ):
        return LITH_CLASSES["surficial"]
    if row.get("lith1") == "Metamorphic" or any(
        k in text for k in ("schist", "gneiss", "metamorph", "amphibolit", "marble",
                            "phyllit", "quartzite", "greenstone", "migmatit")
    ):
        return LITH_CLASSES["metamorphic"]
    if "carbonate" in text or "limestone" in text or "dolostone" in text or "dolomite" in text:
        return LITH_CLASSES["sedimentary_carbonate"]
    if "chemical" in text or any(k in text for k in ("evaporite", "gypsum", "halite",
                                                    "barite", "chert")):
        return LITH_CLASSES["sedimentary_chemical"]
    if "plutonic" in text or "hypabyssal" in text or "intrusive" in text:
        return (LITH_CLASSES["intrusive_felsic"]
                if any(k in text for k in _FELSIC)
                else LITH_CLASSES["intrusive_mafic"])
    if "volcanic" in text:
        return (LITH_CLASSES["volcanic_felsic"]
                if any(k in text for k in _FELSIC)
                else LITH_CLASSES["volcanic_mafic"])
    if any(k in text for k in _MAFIC):
        return LITH_CLASSES["volcanic_mafic"]
    if any(k in text for k in _FELSIC):
        return LITH_CLASSES["intrusive_felsic"]
    if row.get("lith1") == "Sedimentary":
        return LITH_CLASSES["sedimentary_clastic"]
    return LITH_CLASSES["unknown"]


def classify_generalize(text: str) -> int:
    """Coarse lithology class from the polygon-level ``GENERALIZE`` field.

    The SGMC unit table only describes the named units; many polygons carry a
    generalised description instead.  Using it recovers most of the map
    coverage without inventing detail.
    """
    t = str(text).strip().lower()
    if not t or t == "nan":
        return LITH_CLASSES["unknown"]
    if "unconsolidated" in t:
        return LITH_CLASSES["surficial"]
    if "carbonate" in t:
        return LITH_CLASSES["sedimentary_carbonate"]
    if "chemical" in t or "evaporite" in t:
        return LITH_CLASSES["sedimentary_chemical"]
    if "clastic" in t:
        return LITH_CLASSES["sedimentary_clastic"]
    if "intrusive" in t:
        return LITH_CLASSES["intrusive_felsic"]
    if "volcanic" in t:
        return LITH_CLASSES["volcanic_felsic"]
    if "metamorphic" in t and "sedimentary" not in t:
        return LITH_CLASSES["metamorphic"]
    if "sedimentary" in t:
        return LITH_CLASSES["sedimentary_clastic"]
    if "metamorphic" in t:
        return LITH_CLASSES["metamorphic"]
    return LITH_CLASSES["unknown"]


def ensure_geology(cfg: Config) -> Path:
    """Geology shapefile directory (polygons + faults + unit tables)."""
    zip_path = download(cfg.sources.geology.url, cfg.paths.raw / "geology" / "NV.zip",
                        min_bytes=1_000_000)
    out_dir = cfg.paths.raw / "geology" / "NV"
    extract_zip(zip_path, out_dir, required="NV_geol_poly.shp")
    return out_dir


def load_geology_tables(geo_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unit table joined to per-unit lithology, with a class code per unit."""
    units = pd.read_csv(geo_dir / "NV_units.csv", dtype=str)
    lith = pd.read_csv(geo_dir / "NV_lith.csv", dtype=str)
    units.columns = [c.strip().lower() for c in units.columns]
    lith.columns = [c.strip().lower() for c in lith.columns]
    if "unit_link" not in lith:
        raise ValueError("NV_lith.csv has no unit_link column")

    # Prefer the dominant ("Major") lithology row for each map unit.
    lith = lith.copy()
    lith["rank_order"] = np.where(
        lith["lith_rank"].str.strip().str.lower() == "major", 0, 1
    )
    lith = lith.sort_values(["unit_link", "rank_order"]).drop_duplicates("unit_link")

    key = units["unit_link"].astype(str) if "unit_link" in units else None
    if key is None:
        raise ValueError("NV_units.csv has no unit_link column")
    units = units.merge(
        lith[["unit_link", "lith1", "lith2", "lith3", "lith_form"]],
        on="unit_link", how="left",
    )
    units["lith_class"] = units.apply(_lith_class, axis=1).astype("int16")
    return units, lith


# ---------------------------------------------------------------- geophysics
def ensure_magnetic(cfg: Config) -> Path:
    """Extract the USGS magnetic ESRI grid; return the grid directory path."""
    zip_path = download(cfg.sources.magnetic.url,
                        cfg.paths.raw / "magnetic" / "USmag_origmrg.zip",
                        min_bytes=1_000_000)
    out_dir = cfg.paths.raw / "magnetic" / "usmag"
    extract_zip(zip_path, out_dir, required="usmag_origmrg")
    return out_dir / "usmag_origmrg"


def ensure_gravity(cfg: Config) -> Path:
    """Download the Bouguer XYZ text grid (gzipped)."""
    return download(cfg.sources.gravity.url,
                    cfg.paths.raw / "gravity" / "bouguer.xyz.gz",
                    min_bytes=100_000)


def load_gravity_xyz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the three-column (lon lat mGal) Bouguer grid."""
    with gzip.open(path, "rt", errors="replace") as fh:
        arr = np.loadtxt(fh, dtype="float64")
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("unexpected Bouguer XYZ layout")
    return arr[:, 0], arr[:, 1], arr[:, 2]


# --------------------------------------------------------------- geochemistry
GEO_UNITS_TO_PPM = {"ppm": 1.0, "ppb": 1e-3, "pct": 1e4, "%": 1e4, "wt%": 1e4}


def ensure_geochem(cfg: Config) -> Path:
    zip_path = download(cfg.sources.geochemistry.url,
                        cfg.paths.raw / "geochem" / "ngdbsed-csv.zip",
                        min_bytes=1_000_000)
    return zip_path


def load_geochem_samples(cfg: Config) -> pd.DataFrame:
    """Stream-sediment samples inside the AOI with the configured elements.

    NGDB is distributed as a long-format table (one row per lab/species/value),
    so we stream ``bestvalue.csv`` and keep only rows belonging to AOI samples.
    A negative value is NGDB's convention for "below the detection limit"; we
    substitute half the reported detection limit.
    """
    cache = cfg.paths.processed / "geochem_samples.csv"
    if cache.exists():
        return pd.read_csv(cache)

    geo_dir = cfg.paths.raw / "geochem"
    zip_path = ensure_geochem(cfg)
    sa = cfg.study_area
    elements = list(cfg.geochemistry.elements)

    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith("main.csv"))
        with zf.open(name) as raw:
            main = pd.read_csv(
                io.TextIOWrapper(raw, encoding="utf-8", errors="replace"),
                usecols=["lab_id", "lat_wgs84", "long_wgs84", "sample_source"],
                low_memory=False,
            )
    main = main.dropna(subset=["lat_wgs84", "long_wgs84"])
    main = main[
        main["long_wgs84"].between(sa.west, sa.east)
        & main["lat_wgs84"].between(sa.south, sa.north)
    ]
    wanted = set(main["lab_id"].astype(str))

    rows: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith("bestvalue.csv"))
        with zf.open(name) as raw:
            reader = pd.read_csv(
                io.TextIOWrapper(raw, encoding="utf-8", errors="replace"),
                usecols=["lab_id", "species", "unit", "qvalue"],
                chunksize=500_000, low_memory=False,
            )
            for chunk in reader:
                chunk = chunk[chunk["species"].isin(elements) & chunk["lab_id"].isin(wanted)]
                if len(chunk):
                    rows.append(chunk)
    if not rows:
        raise ValueError("no NGDB values for the configured elements inside the AOI")
    values = pd.concat(rows, ignore_index=True)

    values["qvalue"] = pd.to_numeric(values["qvalue"], errors="coerce")
    values["factor"] = values["unit"].str.strip().str.lower().map(GEO_UNITS_TO_PPM)
    values = values.dropna(subset=["qvalue", "factor"])
    values["censored"] = values["qvalue"] < 0
    values["value_ppm"] = np.where(
        values["censored"], values["qvalue"].abs() * 0.5, values["qvalue"]
    ) * values["factor"]
    values = values[values["value_ppm"] > 0]

    wide = (values.groupby(["lab_id", "species"])["value_ppm"].median()
            .unstack("species").reset_index())
    out = main.merge(wide, left_on="lab_id", right_on="lab_id", how="inner")
    keep = ["lab_id", "lat_wgs84", "long_wgs84", "sample_source"] + [
        e for e in elements if e in out.columns
    ]
    out = out[keep].drop_duplicates("lab_id")
    out.to_csv(cache, index=False)
    return out


# ------------------------------------------------------------------ deposits
DEPOSIT_COLUMNS = [
    "dep_id", "site_name", "latitude", "longitude", "state", "commod1", "commod2",
    "commod3", "dev_stat", "dep_type", "model", "hrock_type", "arock_type",
    "alteration", "ore_ctrl",
]


def ensure_mrds(cfg: Config) -> Path:
    zip_path = download(cfg.sources.mrds.url, cfg.paths.raw / "mrds" / "mrds-csv.zip",
                        min_bytes=1_000_000)
    out = cfg.paths.raw / "mrds" / "mrds.csv"
    if not out.exists():
        with zipfile.ZipFile(zip_path) as zf:
            member = cfg.sources.mrds.member or "mrds.csv"
            with zf.open(member) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return out


def load_deposits(cfg: Config) -> pd.DataFrame:
    """MRDS records inside the AOI (all commodities, all statuses)."""
    cache = cfg.paths.processed / "deposits_aoi.csv"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)

    csv_path = ensure_mrds(cfg)
    sa = cfg.study_area
    df = pd.read_csv(csv_path, usecols=DEPOSIT_COLUMNS, low_memory=False)
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[
        df["longitude"].between(sa.west, sa.east)
        & df["latitude"].between(sa.south, sa.north)
    ]
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df
