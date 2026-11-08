"""New home: the robot builds its own map from lidar + depth, starting from nothing."""
import mujoco
import numpy as np

from sip.nav.grid import grid_from_model
from sip.robot import StretchIO
from sip.scene.build_house import build_house, seat_person
from sip.tasks.explore import explore


def test_explores_house_from_scratch():
    m = mujoco.MjModel.from_xml_path(build_house())
    d = mujoco.MjData(m)
    d.qpos[0:2] = [-1.0, 0.0]
    seat_person(m, d, "armchair")
    mujoco.mj_forward(m, d)
    io = StretchIO(m, d)
    logs = []
    mb = explore(m, d, io, None, seconds=420, log=logs.append)
    print(logs[:3], logs[-3:], "coverage", mb.coverage())
    truth = grid_from_model(m, 10.0, 8.0)
    # reached every room
    for x, y in ((-2.0, 1.9), (2.0, 1.9), (2.0, -1.9)):
        i, j = mb.grid.cell(x, y)
        assert mb.known[i - 4:i + 5, j - 4:j + 5].any(), (x, y, logs[-3:])
    # what it marked as walls is mostly where walls really are
    found = mb.grid.occ & mb.known
    agree = (found & truth.inflate(0.1).occ).sum() / max(1, found.sum())
    assert agree > 0.7, agree
