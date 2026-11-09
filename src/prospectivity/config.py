"""Typed configuration loading.

A single YAML file fully describes a run: the study area, the free data
sources, labelling rules, and model hyper-parameters.  Nothing in the
pipeline hard-codes a path or a hyper-parameter, so a different AOI or
commodity is a config edit, not a code edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class StudyAreaConfig:
    name: str
    aoi_wgs84: tuple[float, float, float, float]
    crs: str
    resolution_m: float

    @property
    def west(self) -> float:
        return self.aoi_wgs84[0]

    @property
    def south(self) -> float:
        return self.aoi_wgs84[1]

    @property
    def east(self) -> float:
        return self.aoi_wgs84[2]

    @property
    def north(self) -> float:
        return self.aoi_wgs84[3]


@dataclass(frozen=True)
class SourceConfig:
    url: str | None = None
    member: str | None = None
    bucket: str | None = None
    tile_template: str | None = None


@dataclass(frozen=True)
class SourcesConfig:
    mrds: SourceConfig
    dem: SourceConfig
    geology: SourceConfig
    magnetic: SourceConfig
    gravity: SourceConfig
    geochemistry: SourceConfig


@dataclass(frozen=True)
class LabelsConfig:
    commodities: tuple[str, ...]
    min_confidence: int
    exclusion_buffer_m: float
    negative_ratio: int
    max_positives: int | None
    max_unlabelled: int


@dataclass(frozen=True)
class GeochemistryConfig:
    elements: tuple[str, ...]
    idw_radius_km: float
    idw_power: float


@dataclass(frozen=True)
class ModelConfig:
    algorithm: str
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    max_features: str | float | None
    class_weight: str | None
    pu_learning: bool
    pu_mode: str
    selection: str
    pu_iterations: int
    pu_negative_fraction: float
    spatial_block_km: float
    test_fraction: float
    random_state: int


@dataclass(frozen=True)
class TargetingConfig:
    max_targets: int
    min_separation_km: float
    min_distance_km: float


@dataclass(frozen=True)
class PathsConfig:
    root: Path
    raw: Path
    processed: Path
    output: Path

    @property
    def figures(self) -> Path:
        return self.output / "figures"

    @property
    def models(self) -> Path:
        return self.output / "models"


@dataclass(frozen=True)
class Config:
    project_name: str
    seed: int
    study_area: StudyAreaConfig
    paths: PathsConfig
    sources: SourcesConfig
    labels: LabelsConfig
    geochemistry: GeochemistryConfig
    model: ModelConfig
    targeting: TargetingConfig
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _source(d: dict[str, Any]) -> SourceConfig:
    return SourceConfig(
        url=d.get("url"),
        member=d.get("member"),
        bucket=d.get("bucket"),
        tile_template=d.get("tile_template"),
    )


def load_config(path: str | Path = "configs/study_area.yaml") -> Config:
    """Load and validate a run configuration."""
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    root = cfg_path.parent.parent  # <root>/configs/study_area.yaml
    paths = raw["paths"]
    paths_cfg = PathsConfig(
        root=root,
        raw=(root / paths["raw"]).resolve(),
        processed=(root / paths["processed"]).resolve(),
        output=(root / paths["output"]).resolve(),
    )

    sa = raw["study_area"]
    aoi = tuple(float(v) for v in sa["aoi_wgs84"])
    if len(aoi) != 4 or not (aoi[0] < aoi[2] and aoi[1] < aoi[3]):
        raise ValueError("aoi_wgs84 must be [west, south, east, north] with west<east")

    src = raw["sources"]
    lab = raw["labels"]
    geo = raw["geochemistry"]
    mod = raw["model"]

    return Config(
        project_name=raw.get("project", {}).get("name", "prospectivity"),
        seed=int(raw.get("project", {}).get("seed", 42)),
        study_area=StudyAreaConfig(
            name=sa["name"],
            aoi_wgs84=aoi,  # type: ignore[arg-type]
            crs=sa["crs"],
            resolution_m=float(sa["resolution_m"]),
        ),
        paths=paths_cfg,
        sources=SourcesConfig(
            mrds=_source(src["mrds"]),
            dem=_source(src["dem"]),
            geology=_source(src["geology"]),
            magnetic=_source(src["magnetic"]),
            gravity=_source(src["gravity"]),
            geochemistry=_source(src["geochemistry"]),
        ),
        labels=LabelsConfig(
            commodities=tuple(lab["commodities"]),
            min_confidence=int(lab["min_confidence"]),
            exclusion_buffer_m=float(lab["exclusion_buffer_m"]),
            negative_ratio=int(lab["negative_ratio"]),
            max_positives=lab.get("max_positives"),
            max_unlabelled=int(lab.get("max_unlabelled", 20000)),
        ),
        geochemistry=GeochemistryConfig(
            elements=tuple(geo["elements"]),
            idw_radius_km=float(geo["idw_radius_km"]),
            idw_power=float(geo["idw_power"]),
        ),
        model=ModelConfig(
            algorithm=mod.get("algorithm", "random_forest"),
            n_estimators=int(mod["n_estimators"]),
            max_depth=mod.get("max_depth"),
            min_samples_leaf=int(mod.get("min_samples_leaf", 1)),
            max_features=mod.get("max_features", "sqrt"),
            class_weight=mod.get("class_weight"),
            pu_learning=bool(mod.get("pu_learning", True)),
            pu_mode=str(mod.get("pu_mode", "soft")),
            selection=str(mod.get("selection", "naive")),
            pu_iterations=int(mod.get("pu_iterations", 5)),
            pu_negative_fraction=float(mod.get("pu_negative_fraction", 0.15)),
            spatial_block_km=float(mod.get("spatial_block_km", 8.0)),
            test_fraction=float(mod.get("test_fraction", 0.25)),
            random_state=int(mod.get("random_state", 42)),
        ),
        targeting=TargetingConfig(
            max_targets=int(raw.get("targeting", {}).get("max_targets", 20)),
            min_separation_km=float(
                raw.get("targeting", {}).get("min_separation_km", 3.0)
            ),
            min_distance_km=float(
                raw.get("targeting", {}).get("min_distance_km", 5.0)
            ),
        ),
        raw=raw,
    )
