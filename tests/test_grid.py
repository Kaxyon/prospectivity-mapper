import numpy as np
import pytest

from prospectivity.grid import Grid, grid_from_aoi

AOI = (-117.0, 40.0, -115.5, 41.0)


def test_grid_geometry():
    g = grid_from_aoi(AOI, "EPSG:32611", 500)
    assert g.res == (500.0, 500.0)
    assert g.shape == (g.height, g.width)
    assert 200 < g.width < 300 and 200 < g.height < 260
    assert pytest.approx(g.cell_area_km2, rel=1e-6) == 0.25


def test_cell_of_roundtrip():
    g = grid_from_aoi(AOI, "EPSG:32611", 1000)
    row, col = 7, 11
    x = g.bounds.left + (col + 0.5) * g.res[0]
    y = g.bounds.top - (row + 0.5) * g.res[1]
    assert g.cell_of(x, y) == (row, col)
    assert g.contains(x, y)


def test_points_outside_are_rejected():
    g = grid_from_aoi(AOI, "EPSG:32611", 500)
    assert not g.contains(g.bounds.left - 1000, g.bounds.bottom + 100)
    assert not g.contains(g.bounds.left + 100, g.bounds.top + 1000)


def test_rejects_non_north_up():
    from affine import Affine
    from pyproj import CRS

    with pytest.raises(ValueError):
        Grid(CRS.from_epsg(32611), Affine(500, 10, 0, 0, -500, 0), 10, 10)
