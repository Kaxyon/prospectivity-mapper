"""Command-line interface.

    python -m prospectivity fetch     # download the free datasets only
    python -m prospectivity labels    # inspect the deposit label set
    python -m prospectivity run       # full pipeline
    python -m prospectivity info      # print the active configuration
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .grid import grid_from_aoi


def _cmd_fetch(cfg) -> int:
    from .sources import (
        ensure_dem, ensure_geochem, ensure_geology, ensure_gravity, ensure_magnetic,
        ensure_mrds,
    )

    tiles = ensure_dem(cfg)
    ensure_geology(cfg)
    ensure_magnetic(cfg)
    ensure_gravity(cfg)
    ensure_geochem(cfg)
    ensure_mrds(cfg)
    print(f"raw data cached in {cfg.paths.raw} ({len(tiles)} DEM tiles)")
    return 0


def _cmd_labels(cfg) -> int:
    from .labels import build_labels
    from .sources import load_deposits

    grid = grid_from_aoi(cfg.study_area.aoi_wgs84, cfg.study_area.crs,
                         cfg.study_area.resolution_m)
    deposits = load_deposits(cfg)
    labels = build_labels(cfg, grid, deposits)
    print(f"positive cells: {labels.n_positives}")
    print(f"unlabelled cells: {len(labels.unlabelled)}")
    print(f"withheld buffer: {labels.exclusion_mask.mean() * 100:.1f}% of grid")
    print(labels.deposits[["site_name", "dev_stat", "confidence"]].head(10).to_string())
    return 0


def _cmd_info(cfg) -> int:
    grid = grid_from_aoi(cfg.study_area.aoi_wgs84, cfg.study_area.crs,
                         cfg.study_area.resolution_m)
    print(f"project      : {cfg.project_name}")
    print(f"study area   : {cfg.study_area.name}")
    print(f"grid         : {grid}")
    print(f"cell area    : {grid.cell_area_km2:.3f} km2")
    print(f"total area   : {grid.width * grid.height * grid.cell_area_km2:.0f} km2")
    print(f"commodities  : {', '.join(cfg.labels.commodities)}")
    print(f"model        : {cfg.model.algorithm}, pu={cfg.model.pu_learning}")
    print(f"raw data dir : {cfg.paths.raw}")
    return 0


def _cmd_run(cfg, quick: bool, reuse_cache: bool) -> int:
    from .pipeline import run

    if quick:
        cfg = replace(
            cfg,
            model=replace(cfg.model, n_estimators=80, pu_iterations=2),
        )
    run(cfg, reuse_cache=reuse_cache)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prospectivity",
        description="ML-based mineral prospectivity mapping with known deposits",
    )
    parser.add_argument("--config", default="configs/study_area.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="download the free datasets")
    sub.add_parser("labels", help="summarise the deposit labels")
    sub.add_parser("info", help="print the active grid configuration")
    run_p = sub.add_parser("run", help="run the full pipeline")
    run_p.add_argument("--quick", action="store_true", help="fewer trees (smoke test)")
    run_p.add_argument("--reuse-cache", action="store_true",
                       help="reuse data/processed/feature_cube.npz")

    args = parser.parse_args(argv)
    config_path = Path(args.config)
    if not config_path.exists():
        alt = Path(__file__).resolve().parents[2] / args.config
        config_path = alt
    cfg = load_config(config_path)

    if args.command == "fetch":
        return _cmd_fetch(cfg)
    if args.command == "labels":
        return _cmd_labels(cfg)
    if args.command == "info":
        return _cmd_info(cfg)
    return _cmd_run(cfg, args.quick, args.reuse_cache)


if __name__ == "__main__":
    sys.exit(main())
