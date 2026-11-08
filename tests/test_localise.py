"""The robot's position comes from wheel encoders + lidar scan matching, not the simulator."""
import mujoco
import numpy as np

from sip.nav.grid import grid_from_model
from sip.nav.lidar import scan
from sip.robot import StretchIO
from sip.scene.build_house import build_house


def _setup():
    m = mujoco.MjModel.from_xml_path(build_house())
    d = mujoco.MjData(m)
    d.qpos[0:2] = [-1.0, 0.0]
    mujoco.mj_forward(m, d)
    io = StretchIO(m, d)
    io.enable_localisation()
    return m, d, io


def _drive(m, d, io, v, w, seconds, grid=None):
    for k in range(int(seconds / 0.05)):
        io.drive(v, w)
        for _ in range(25):
            mujoco.mj_step(m, d)
        io.loc.update()
        if grid is not None and k % 10 == 0:
            io.loc.correct(scan(m, d, io.base_pose()), grid)


def test_odometry_tracks_driving_and_turning():
    m, d, io = _setup()
    _drive(m, d, io, 0.2, 0.0, 4)
    _drive(m, d, io, 0.0, 0.5, 3)
    _drive(m, d, io, 0.15, 0.2, 4)
    t, e = io.true_pose(), io.base_pose()
    assert np.hypot(t.x - e.x, t.y - e.y) < 0.10, (t, e)
    assert abs(np.angle(np.exp(1j * (t.yaw - e.yaw)))) < np.deg2rad(8)   # odometry alone; scan matching fixes the rest


def test_scan_matching_pulls_a_wrong_estimate_back():
    m, d, io = _setup()
    grid = grid_from_model(m)
    io.loc.pose = type(io.loc.pose)(io.loc.pose.x + 0.05, io.loc.pose.y - 0.04, io.loc.pose.yaw + np.deg2rad(2))
    _drive(m, d, io, 0.0, 0.0, 3, grid=grid)
    t, e = io.true_pose(), io.base_pose()
    assert np.hypot(t.x - e.x, t.y - e.y) < 0.03, (t, e)
