import mujoco
import numpy as np
import pytest

from sip.perception.face import FaceTracker
from sip.perception.objects import aim_head, render_head
from sip.robot import StretchIO
from sip.scene.build_house import build_house, park_pose_for, seat_person


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(build_house("assets/house.xml"))


def _near_person(model, look=True):
    data = mujoco.MjData(model)
    facing = seat_person(model, data, "armchair")
    mujoco.mj_forward(model, data)
    mouth = data.site("mouth").xpos.copy()
    x, y, yaw = park_pose_for(mouth, 0.6, facing)
    data.qpos[0:3] = [x, y, 0]
    data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    io = StretchIO(model, data)
    io.stow()
    target = mouth if look else mouth + np.array([0, 0, -3.0])
    aim_head(model, data, io, target)
    r = mujoco.Renderer(model, 240, 320)
    return data, io, r, mouth


def test_sim_mouth_within_2cm(model):
    data, io, r, mouth = _near_person(model)
    rgb, depth, seg = render_head(model, data, r)
    m = FaceTracker("sim").update(rgb, depth, seg, model, data)
    assert m.visible
    assert np.linalg.norm(m.xyz - mouth) < 0.02


def test_head_turned_reports_yaw(model):
    data, io, r, _ = _near_person(model)
    data.qpos[model.joint("head_yaw").qposadr[0]] = 0.7
    mujoco.mj_forward(model, data)
    rgb, depth, seg = render_head(model, data, r)
    m = FaceTracker("sim").update(rgb, depth, seg, model, data)
    assert abs(m.yaw_deg) > 30


def test_face_out_of_view_not_visible(model):
    data, io, r, _ = _near_person(model, look=False)
    rgb, depth, seg = render_head(model, data, r)
    assert not FaceTracker("sim").update(rgb, depth, seg, model, data).visible
