import mujoco
import numpy as np

from sip.nav.follow import PathFollower
from sip.nav.grid import astar, grid_from_model
from sip.robot import StretchIO
from sip.scene.build_house import ROOMS, build_house
from sip.types import Pose2D


def test_reaches_goal_in_sim():
    model = mujoco.MjModel.from_xml_path(build_house("assets/house.xml"))
    data = mujoco.MjData(model)
    data.qpos[0:2] = ROOMS["kitchen"]
    mujoco.mj_forward(model, data)
    io = StretchIO(model, data)
    io.stow()
    for _ in range(1500):
        mujoco.mj_step(model, data)
    grid = grid_from_model(model).inflate(0.3)
    path = astar(grid, ROOMS["kitchen"], ROOMS["living"])
    follower = PathFollower(path)
    t = 0.0
    while not follower.done and t < 40.0:
        io.drive(*follower.step(io.base_pose()))
        for _ in range(25):
            mujoco.mj_step(model, data)
        t += 25 * model.opt.timestep
    io.drive(0, 0)
    p = io.base_pose()
    assert follower.done
    assert np.hypot(p.x - ROOMS["living"][0], p.y - ROOMS["living"][1]) < 0.15


def test_slows_at_goal():
    f = PathFollower([(0.0, 0.0), (1.0, 0.0)])
    v_far, _ = f.step(Pose2D(0.0, 0.0, 0.0))
    v_near, _ = f.step(Pose2D(0.95, 0.0, 0.0))
    assert v_near < v_far
