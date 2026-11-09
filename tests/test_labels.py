import numpy as np
import pandas as pd

from prospectivity.grid import grid_from_aoi
from prospectivity.labels import build_labels, distance_raster, select_deposits

AOI = (-117.0, 40.0, -115.5, 41.0)


def _deposits():
    return pd.DataFrame(
        {
            "dep_id": ["a", "b", "c", "d"],
            "site_name": ["Gold A", "Barite B", "Gold Plant", "Silver C"],
            "latitude": [40.5, 40.6, 40.7, 40.8],
            "longitude": [-116.5, -116.4, -116.3, -116.2],
            "state": ["Nevada"] * 4,
            "commod1": ["Gold", "Barium-Barite", "Gold", "Silver"],
            "commod2": [None, None, None, None],
            "commod3": [None, None, None, None],
            "dev_stat": ["Past Producer", "Producer", "Plant", "Occurrence"],
            "dep_type": ["Vein"] * 4,
            "model": [None] * 4,
            "hrock_type": [None] * 4,
            "arock_type": [None] * 4,
            "alteration": [None] * 4,
            "ore_ctrl": [None] * 4,
        }
    )


def test_commodity_and_status_filtering(cfg):
    selected = select_deposits(_deposits(), cfg)
    assert set(selected["dep_id"]) == {"a", "d"}  # barite and plant excluded
    assert (selected["confidence"] > 0).all()


def test_distance_raster_is_zero_at_a_point(cfg):
    g = grid_from_aoi(AOI, cfg.study_area.crs, cfg.study_area.resolution_m)
    x = g.bounds.left + (g.width / 2) * g.res[0]
    y = g.bounds.top - (g.height / 2) * g.res[1]
    dist = distance_raster(np.array([[x, y]]), g)
    assert dist.min() == 0.0
    assert dist.max() > 1.0


def _many_deposits(n: int = 30) -> pd.DataFrame:
    frame = _deposits()
    rows = []
    for i in range(n):
        lon = -116.9 + (i % 6) * 0.2
        lat = 40.15 + (i // 6) * 0.15
        rows.append(
            {
                **frame.iloc[0].to_dict(),
                "dep_id": f"m{i}",
                "site_name": f"Deposit {i}",
                "longitude": lon,
                "latitude": lat,
            }
        )
    return pd.DataFrame(rows)


def test_labels_partition_the_grid(cfg):
    g = grid_from_aoi(AOI, cfg.study_area.crs, cfg.study_area.resolution_m)
    labels = build_labels(cfg, g, _many_deposits())
    assert labels.n_positives == 30
    total = g.width * g.height
    excluded = int(labels.exclusion_mask.sum())
    assert len(labels.unlabelled) == total - excluded
    # The buffer must be much larger than a single cell.
    assert 100 < excluded < total
