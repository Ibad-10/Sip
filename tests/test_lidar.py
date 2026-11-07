import mujoco
import numpy as np
import pytest

from sip.nav.grid import OccupancyGrid
from sip.nav.lidar import scan, update_map
from sip.robot import StretchIO
from sip.scene.build_house import build_house


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(build_house("assets/house.xml"))


def _sim(model, base=(-1.0, 0.0)):
    data = mujoco.MjData(model)
    data.qpos[0:2] = base
    mujoco.mj_forward(model, data)
    return data, StretchIO(model, data)


def _move_chair(model, data, x, y):
    adr = model.joint("chair_hall_free").qposadr[0]
    data.qpos[adr:adr + 3] = [x, y, 0.0]
    mujoco.mj_forward(model, data)


def test_scan_sees_wall(model):
    data, io = _sim(model)
    ranges = scan(model, data, io.base_pose(), n_rays=180)
    # ray 45 of 180 points +90 deg (towards y=+0.8 wall) from base at y=0
    assert abs(ranges[45] - 0.8 + WALL_HALF) < 0.05


WALL_HALF = 0.05


def test_new_chair_marked(model):
    data, io = _sim(model)
    _move_chair(model, data, 0.0, 0.0)
    grid = OccupancyGrid(8.0, 6.0)
    pose = io.base_pose()
    new = update_map(grid, pose, scan(model, data, pose), 4.0)
    assert new
    assert any(abs(x - (-0.25)) < 0.15 and abs(y) < 0.3 for x, y in new)


def test_obstacle_persists_until_seen_gone(model):
    data, io = _sim(model)
    _move_chair(model, data, 0.0, 0.0)
    grid = OccupancyGrid(8.0, 6.0)
    pose = io.base_pose()
    update_map(grid, pose, scan(model, data, pose), 4.0)
    hit = grid.cell(-0.24, 0.0)
    occupied_cells = np.argwhere(grid.occ)
    near = min(occupied_cells, key=lambda c: abs(c[0] - hit[0]) + abs(c[1] - hit[1]))
    _move_chair(model, data, -3.6, -0.5)
    for k in range(3):
        update_map(grid, pose, scan(model, data, pose), 4.0)
        if k < 2:
            assert grid.occ[tuple(near)], f"cleared too early after {k + 1} scans"
    assert not grid.occ[tuple(near)]


def test_scan_ignores_own_gripper(model):
    data, io = _sim(model)
    io.set_arm(lift=0.05)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    ranges = scan(model, data, io.base_pose(), n_rays=180)
    assert ranges[135] > 0.6  # -y: own gripper at ~0.44 must be ignored; wall at ~0.75
