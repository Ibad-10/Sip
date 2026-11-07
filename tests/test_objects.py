import mujoco
import numpy as np
import pytest

from sip.perception.objects import SegDetector, aim_head, pixel_to_world, project, render_head
from sip.robot import StretchIO
from sip.scene.build_house import build_house

CAM = "d435i_camera_rgb"


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(build_house("assets/house.xml"))


def _facing_counter(model):
    data = mujoco.MjData(model)
    data.qpos[0:2] = [-2.6, 1.6]
    data.qpos[3:7] = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]  # yaw pi/2, facing +y
    mujoco.mj_forward(model, data)
    io = StretchIO(model, data)
    io.stow()
    renderer = mujoco.Renderer(model, 240, 320)
    return data, io, renderer


def test_glass_world_position_within_5cm(model):
    data, io, r = _facing_counter(model)
    aim_head(model, data, io, data.body("glass").xpos.copy())
    rgb, depth, seg = render_head(model, data, r)
    dets = SegDetector(["glass", "organiser"], noise_m=0.0).detect(model, data, rgb, depth, seg)
    glass = [d for d in dets if d.label == "glass"]
    assert glass and glass[0].conf > 0.5
    assert np.linalg.norm(glass[0].xyz - data.body("glass").xpos) < 0.05


def test_not_visible_returns_empty(model):
    data, io, r = _facing_counter(model)
    aim_head(model, data, io, np.array([-2.6, -3.0, 0.5]))  # look the other way
    rgb, depth, seg = render_head(model, data, r)
    dets = SegDetector(["glass"]).detect(model, data, rgb, depth, seg)
    assert not dets


def test_pixel_to_world_roundtrip(model):
    data, io, r = _facing_counter(model)
    target = data.body("glass").xpos.copy()
    aim_head(model, data, io, target)
    u, v, z = project(model, data, CAM, target, 320, 240)
    back = pixel_to_world(model, data, CAM, u, v, z, 320, 240)
    assert np.linalg.norm(back - target) < 0.01
