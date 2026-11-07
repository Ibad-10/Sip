import mujoco
import numpy as np
import pytest

from sip.config import load_profile
from sip.robot import StretchIO
from sip.scene.build_house import build_house

START = (-1.0, 0.0)  # hall, clear of walls


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(build_house("assets/house.xml"))


@pytest.fixture
def sim(model):
    data = mujoco.MjData(model)
    data.qpos[0:2] = START
    mujoco.mj_forward(model, data)
    return model, data, StretchIO(model, data)


def run(model, data, seconds):
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def test_drive_straight(sim):
    model, data, io = sim
    p0 = io.base_pose()
    io.drive(0.3, 0.0)
    run(model, data, 3.33)
    p1 = io.base_pose()
    assert abs(np.hypot(p1.x - p0.x, p1.y - p0.y) - 1.0) < 0.1


def test_turn_in_place(sim):
    model, data, io = sim
    y0 = io.base_pose().yaw
    io.drive(0.0, 0.5)
    run(model, data, np.pi)
    dyaw = np.angle(np.exp(1j * (io.base_pose().yaw - y0)))
    assert abs(dyaw - np.pi / 2) < 0.15


def test_set_arm_clamps(sim):
    model, data, io = sim
    io.set_arm(lift=2.0, arm=-1.0)
    assert data.ctrl[model.actuator("lift").id] == pytest.approx(1.1)
    assert data.ctrl[model.actuator("arm").id] == pytest.approx(0.0)


def test_attach_release(model):
    data = mujoco.MjData(model)
    io = StretchIO(model, data)
    # park so the retracted gripper sits at the glass
    glass = data.body("glass")
    mujoco.mj_forward(model, data)
    data.qpos[0:3] = [glass.xpos[0], glass.xpos[1] - 0.44 - 0.25, 0]
    data.qpos[3:7] = [0, 0, 0, 1]  # yaw = pi: arm (-y) points to +y world
    io.set_arm(lift=0.86, arm=0.0)
    run(model, data, 2.0)
    io.set_arm(arm=0.25)
    run(model, data, 2.0)
    io.attach("glass")
    z0 = data.body("glass").xpos[2]
    io.set_arm(lift=1.05)
    run(model, data, 2.0)
    assert data.body("glass").xpos[2] > z0 + 0.1
    io.release()
    assert not data.eq_active[model.equality("grasp_weld_glass").id]


def test_freeze_stops(sim):
    model, data, io = sim
    io.drive(0.3, 0.0)
    run(model, data, 1.0)
    io.freeze()
    run(model, data, 0.4)  # wheels stop in <0.1 s; body rocks ~0.2 s after a hard stop
    assert np.hypot(*data.qvel[0:2]) < 0.01


def test_load_profile_default_dev(monkeypatch):
    monkeypatch.delenv("SIP_PROFILE", raising=False)
    p = load_profile()
    assert p.name == "dev" and p.detector == "seg" and p.cloud in (True, False)


def test_stow_tucks_gripper(sim):
    model, data, io = sim
    io.stow()
    run(model, data, 3.0)
    p = io.base_pose()
    g = io.grasp_center()
    assert np.hypot(g[0] - p.x, g[1] - p.y) < 0.25
