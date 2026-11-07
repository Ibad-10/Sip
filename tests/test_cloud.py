import time

import mujoco
import numpy as np
import pytest

from sip.cloud.gemini import day_summary, second_opinion
from sip.cloud.verify import recount_log, summary_matches, verify_target
from sip.nav.grid import grid_from_model
from sip.perception.objects import aim_head, project, render_head
from sip.robot import StretchIO
from sip.scene.build_house import build_house
from sip.types import Event

EVENTS = [Event(10, "plan", "drink offered for 3 s"), Event(20, "dose", "taken"),
          Event(30, "plan", "drink offered for 3 s"), Event(40, "dose", "declined"),
          Event(50, "alert", "missed dose: Tue PM")]


def test_recount_log():
    assert recount_log(EVENTS) == {"drinks": 2, "doses_taken": 1, "doses_declined": 1, "alerts": 1}


def test_summary_mismatch_detected():
    counts = recount_log(EVENTS)
    assert summary_matches("Mary had 2 drinks, took 1 dose and declined 1. 1 alert.", counts)
    assert not summary_matches("Mary had 3 drinks today and took 1 dose, declined 1, 1 alert.", counts)


def test_second_opinion_offline_returns_none(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    t0 = time.time()
    assert second_opinion(np.zeros((240, 320, 3), np.uint8), "water", "", ["glass"]) is None
    assert day_summary(EVENTS) is None
    assert time.time() - t0 < 0.5


@pytest.fixture(scope="module")
def scene():
    model = mujoco.MjModel.from_xml_path(build_house("assets/house.xml"))
    data = mujoco.MjData(model)
    data.qpos[0:2] = [-2.6, 1.6]
    data.qpos[3:7] = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
    mujoco.mj_forward(model, data)
    io = StretchIO(model, data)
    aim_head(model, data, io, data.body("glass").xpos.copy())
    r = mujoco.Renderer(model, 240, 320)
    _, depth, _ = render_head(model, data, r, rgb=False)
    return model, data, io, depth, grid_from_model(model)


def test_verify_accepts_glass_pixel(scene):
    model, data, io, depth, grid = scene
    u, v, _ = project(model, data, "d435i_camera_rgb", data.body("glass").xpos)
    ok, checks = verify_target(model, data, int(u), int(v), depth, grid, io, claimed_depth=depth[int(v), int(u)])
    assert ok, checks


def test_verify_rejects_depth_mismatch(scene):
    model, data, io, depth, grid = scene
    u, v, _ = project(model, data, "d435i_camera_rgb", data.body("glass").xpos)
    ok, checks = verify_target(model, data, int(u), int(v), depth, grid, io,
                               claimed_depth=depth[int(v), int(u)] + 0.5)
    assert not ok and "depth" in " ".join(checks)


def test_verify_rejects_unreachable_point(scene):
    model, data, io, depth, grid = scene
    from sip.perception.objects import pixel_to_world
    high = next((u, v) for v in range(0, 240, 4) for u in range(0, 320, 4)
                if pixel_to_world(model, data, "d435i_camera_rgb", u, v, depth[v, u])[2] > 1.3)
    u, v = high  # a point on the wall above the counter, out of the gripper's height range
    ok, checks = verify_target(model, data, u, v, depth, grid, io, claimed_depth=depth[v, u])
    assert not ok and any("height" in c for c in checks)
