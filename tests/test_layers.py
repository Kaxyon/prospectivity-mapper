import numpy as np
import pandas as pd
import pytest

from prospectivity.features import FeatureCube
from prospectivity.geochemistry import idw_interpolate
from prospectivity.grid import grid_from_aoi
from prospectivity.raster import fill_nan_nearest
from prospectivity.sources import _lith_class, classify_generalize


@pytest.fixture
def grid(cfg):
    return grid_from_aoi(cfg.study_area.aoi_wgs84, cfg.study_area.crs, 1000)


def test_idw_returns_sample_value_at_sample(grid):
    x = grid.bounds.left + (grid.width / 2) * grid.res[0]
    y = grid.bounds.top - (grid.height / 2) * grid.res[1]
    pts = np.array([[x, y], [x + 5000, y], [x, y + 5000]])
    vals = np.array([3.0, 1.0, 2.0])
    out = idw_interpolate(vals, pts, grid, k=3)
    row, col = grid.cell_of(x, y)
    assert out[row, col] == pytest.approx(3.0)


def test_idw_handles_missing_values(grid):
    pts = np.array([[grid.bounds.left + 1000, grid.bounds.bottom + 1000]])
    vals = np.array([np.nan])
    out = idw_interpolate(vals, pts, grid)
    assert np.isnan(out).all()


def test_fill_nan_nearest_preserves_finite_cells():
    arr = np.array([[1.0, np.nan, 3.0], [4.0, np.nan, 6.0]], dtype="float32")
    filled = fill_nan_nearest(arr)
    assert np.isfinite(filled).all()
    assert filled[0, 0] == 1.0 and filled[1, 2] == 6.0


def test_lithology_classification():
    assert _lith_class(pd.Series({"lith1": "Unconsolidated"})) == 9
    assert _lith_class(pd.Series({"lith1": "Metamorphic", "lith2": "Schist"})) == 8
    assert _lith_class(
        pd.Series({"lith1": "Sedimentary", "lith2": "Carbonate", "lith3": "Limestone"})
    ) == 5
    assert _lith_class(
        pd.Series({"lith1": "Igneous", "lith2": "Plutonic", "lith3": "Granitic"})
    ) == 1
    assert _lith_class(
        pd.Series({"lith1": "Igneous", "lith2": "Volcanic", "lith3": "Mafic-volcanic"})
    ) == 4


def test_generalised_lithology_fallback():
    assert classify_generalize("Unconsolidated, undifferentiated") == 9
    assert classify_generalize("Sedimentary, carbonate") == 5
    assert classify_generalize("Igneous, intrusive") == 1
    assert classify_generalize("Water") == 0


def test_feature_cube_sampling(grid):
    layers = {
        "elevation_m": np.arange(12, dtype="float32").reshape(3, 4),
        "slope_deg": np.ones((3, 4), dtype="float32"),
    }
    g = type(grid)(
        crs=grid.crs,
        transform=grid.transform,
        width=4,
        height=3,
    )
    cube = FeatureCube(grid=g, layers=layers, names=["elevation_m", "slope_deg"])
    cells = np.array([[0, 0], [2, 3]])
    sample = cube.sample(cells)
    assert list(sample.columns) == cube.names
    assert sample.iloc[0]["elevation_m"] == 0.0
    assert sample.iloc[1]["elevation_m"] == 11.0


def test_feature_cube_rejects_missing_layer(grid):
    with pytest.raises(KeyError):
        FeatureCube(grid=grid, layers={"a": np.zeros(grid.shape)}, names=["b"])
