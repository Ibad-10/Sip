"""Real models on rendered camera frames: the robot must find things without simulator labels."""
import mujoco
import numpy as np
import pytest

from sip.perception.objects import aim_head, render_head
from sip.robot import StretchIO
from sip.scene.build_house import build_house, park_pose_for, seat_person

pytest.importorskip("ultralytics")


@pytest.fixture(scope="module")
def models():
    from sip.perception.vision import MediaPipeFace, YoloDetector
    return YoloDetector(asynchronous=False), MediaPipeFace()


def _scene(robot_xy, yaw, look):
    m = mujoco.MjModel.from_xml_path(build_house())
    d = mujoco.MjData(m)
    seat_person(m, d, "armchair")
    d.qpos[0:2] = robot_xy
    d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(m, d)
    io = StretchIO(m, d)
    aim_head(m, d, io, np.asarray(look), snap=True)
    mujoco.mj_forward(m, d)
    return m, d, mujoco.Renderer(m, 480, 640)


def test_person_and_mouth_from_camera_only(models):
    det, face = models
    m0 = mujoco.MjModel.from_xml_path(build_house())
    d0 = mujoco.MjData(m0)
    seat_person(m0, d0, "armchair")
    mujoco.mj_forward(m0, d0)
    mouth = d0.site("mouth").xpos.copy()
    x, y, yaw = park_pose_for(mouth, 0.9, -np.pi / 2, 55.0)
    m, d, r = _scene((x, y), yaw, mouth)
    rgb, depth, _ = render_head(m, d, r, rgb=True)
    dets = det.detect(m, d, rgb, depth)
    assert "person" in det.boxes, [x.label for x in dets]
    ms = face.update(rgb, depth, m, d, "d435i_camera_rgb", person_box=det.boxes["person"])
    assert ms.visible
    assert np.linalg.norm(ms.xyz - mouth) < 0.06, (ms.xyz, mouth)


def test_glass_on_counter_found_by_detector(models):
    det, _ = models
    m, d, r = _scene((-2.0, 1.6), np.pi / 2, (-2.6, 2.6, 0.95))
    rgb, depth, _ = render_head(m, d, r, rgb=True)
    dets = {x.label: x for x in det.detect(m, d, rgb, depth)}
    assert "glass" in dets
    assert np.linalg.norm(dets["glass"].xyz - d.body("glass").xpos) < 0.12
