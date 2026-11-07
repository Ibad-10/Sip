import mujoco
import numpy as np

from sip.nav.grid import OccupancyGrid, astar, grid_from_model
from sip.scene.build_house import ROOMS, build_house


def _corridor():
    g = OccupancyGrid(4.0, 2.0, res=0.05)  # x in [-2,2], y in [-1,1]
    for x in np.arange(-2, 2, 0.05):
        g.mark(x, 0.95)
        g.mark(x, -0.95)
    return g


def test_straight_path():
    path = astar(_corridor(), (-1.5, 0.0), (1.5, 0.0))
    assert path is not None
    assert all(abs(y) < 0.1 for _, y in path)


def test_routes_around_box():
    g = _corridor()
    for x in np.arange(-0.2, 0.2, 0.05):
        for y in np.arange(-0.6, 0.3, 0.05):
            g.mark(x, y)
    path = astar(g, (-1.5, 0.0), (1.5, 0.0))
    assert path is not None
    assert max(y for _, y in path) > 0.3


def test_blocked_doorway_returns_none():
    g = _corridor()
    for y in np.arange(-1.0, 1.0, 0.05):
        g.mark(0.0, y)
    assert astar(g, (-1.5, 0.0), (1.5, 0.0)) is None


def test_inflation_keeps_clearance():
    g = _corridor()
    g.mark(0.0, 0.0)
    inflated = g.inflate(0.25)
    path = astar(inflated, (-1.5, 0.0), (1.5, 0.0))
    assert path is not None
    for x, y in path:
        assert np.hypot(x, y) >= 0.25 - 0.05


def test_house_kitchen_to_living_path_exists():
    model = mujoco.MjModel.from_xml_path(build_house("assets/house.xml"))
    grid = grid_from_model(model).inflate(0.25)
    assert not grid.is_free(*ROOMS["kitchen"][:1], 3.0)  # north wall occupied
    path = astar(grid, ROOMS["kitchen"], ROOMS["living"])
    assert path is not None
