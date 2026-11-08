"""Headless integration tests for the water task. Each builds a fresh scene."""
import mujoco
import numpy as np
import pytest

from sip.blackboard import Blackboard
from sip.memory import ObjectMemory
from sip.nav.grid import grid_from_model
from sip.perception.face import FaceTracker
from sip.perception.objects import SegDetector
from sip.robot import StretchIO
from sip.scene.build_house import build_house, seat_person
from sip.tasks.water import TICK_STEPS, WaterTask

MODEL = None


def model():
    global MODEL
    if MODEL is None:
        MODEL = mujoco.MjModel.from_xml_path(build_house("assets/house.xml"))
    return MODEL


class Harness:
    def __init__(self, start=(-1.0, 0.0)):
        self.m = model()
        self.d = mujoco.MjData(self.m)
        self.d.qpos[0:2] = start
        seat_person(self.m, self.d, "armchair")
        mujoco.mj_forward(self.m, self.d)
        self.io = StretchIO(self.m, self.d)
        self.bb = Blackboard()
        self.task = WaterTask(self.io, self.bb, grid_from_model(self.m), ObjectMemory(),
                              SegDetector(["glass"], noise_m=0.005), FaceTracker("sim"))
        self.t = 0.0
        self.hold = 0.0       # longest continuous time straw tip within 2 cm of mouth
        self._run = 0.0
        self.states: list[str] = []

    def move_free(self, joint, xyz):
        adr = self.m.joint(joint).qposadr[0]
        self.d.qpos[adr:adr + 3] = xyz
        self.d.qvel[self.m.joint(joint).dofadr[0]:self.m.joint(joint).dofadr[0] + 6] = 0
        mujoco.mj_forward(self.m, self.d)

    def tick(self):
        state = self.task.step(self.t)
        if not self.states or self.states[-1] != state:
            self.states.append(state)
        for _ in range(TICK_STEPS):
            mujoco.mj_step(self.m, self.d)
        dt = TICK_STEPS * self.m.opt.timestep
        self.t += dt
        tip = self.d.site("straw_tip").xpos
        mouth = self.d.site("mouth").xpos
        self._run = self._run + dt if np.linalg.norm(tip - mouth) <= 0.02 + 1e-3 else 0.0
        self.hold = max(self.hold, self._run)
        return state

    def run(self, seconds, until=None, on=None):
        end = self.t + seconds
        while self.t < end:
            s = self.tick()
            if on:
                on(self, s)
            if until and until(self, s):
                return True
        return False

    def kinds(self, kind):
        return [e.text for e in self.bb.events() if e.kind == kind]


def request_water(h):
    h.bb.set("ready", True)
    h.bb.set("request", "water")


def finished(h, s):
    return "RETURN" in h.states and s == "IDLE"


def test_full_water_run():
    h = Harness()
    request_water(h)
    assert h.run(240, until=finished), h.states
    assert h.hold >= 3.0
    glass = h.d.body("glass").xpos
    assert np.linalg.norm(glass[:2] - np.array([-2.6, 2.6])) < 0.15  # back on the counter
    assert h.io.held is None


def test_glass_moved_triggers_search():
    h = Harness()
    request_water(h)
    moved = {"done": False}

    def on(h, s):
        if s == "NAV_TO_OBJECT" and not moved["done"]:
            h.move_free("glass_free", [3.15, -1.2, 0.57])  # onto the bed, bedroom
            moved["done"] = True

    assert h.run(220, until=finished, on=on), h.states
    assert "SEARCH" in h.states
    assert h.hold >= 3.0


def test_glass_missing_asks():
    h = Harness()
    h.move_free("glass_free", [-2.0, -2.0, 0.06])  # closed store room: unreachable, unseen
    request_water(h)
    assert h.run(200, until=lambda h, s: s == "ASK"), h.states
    assert any("glass" in t for t in h.kinds("say"))


def test_chair_in_hall_reroutes():
    h = Harness()
    h.move_free("chair_hall_free", [-2.0, 0.45, 0.0])  # in front of the kitchen door
    request_water(h)
    assert h.run(220, until=finished), h.states
    assert any("new route" in t for t in h.kinds("plan"))


def test_only_doorway_blocked_asks():
    h = Harness()
    h.move_free("chair_hall_free", [-2.0, 0.45, 0.0])
    for y in np.arange(1.45, 2.6, 0.05):  # also block the kitchen-living door
        h.task.grid.mark(0.0, y)
    request_water(h)
    assert h.run(120, until=lambda h, s: s == "ASK"), h.states
    assert any("way" in t.lower() for t in h.kinds("say"))


def test_stop_freezes_within_one_tick():
    for phase in ("NAV_TO_OBJECT", "APPROACH"):
        h = Harness()
        request_water(h)
        assert h.run(150, until=lambda h, s: s == phase), (phase, h.states)
        h.run(1.0)
        h.bb.set("stop", True)
        assert h.tick() == "HALT"
        ctrl = h.d.ctrl
        assert ctrl[h.io.act["left_wheel_vel"]] == 0 and ctrl[h.io.act["right_wheel_vel"]] == 0
        assert abs(ctrl[h.io.act["arm"]] - h.io.arm_extension()) < 0.02


def test_head_turn_mid_approach_retreats_then_retries():
    h = Harness()
    request_water(h)
    turned = {"t": None}

    def on(h, s):
        if s == "APPROACH" and turned["t"] is None and h.task.dist_to_mouth() < 0.25:
            h.d.ctrl[h.io.act["head_yaw_act"]] = 0.8
            turned["t"] = h.t
        if turned["t"] is not None and h.t - turned["t"] > 3.0:
            h.d.ctrl[h.io.act["head_yaw_act"]] = 0.0

    assert h.run(200, until=finished, on=on), h.states
    assert any("head turned" in t for t in h.kinds("safety"))
    assert h.hold >= 3.0


def test_person_moved_to_bed_retargets():
    h = Harness()
    request_water(h)
    moved = {"done": False}

    def on(h, s):
        if s == "NAV_TO_PERSON" and not moved["done"]:
            seat_person(h.m, h.d, "bed")
            moved["done"] = True

    assert h.run(220, until=finished, on=on), h.states
    assert any("moved" in t for t in h.kinds("plan"))
    assert h.hold >= 3.0


def test_done_during_hold_ends_drink():
    h = Harness()
    request_water(h)
    assert h.run(150, until=lambda h, s: s == "HOLD"), h.states
    h.run(0.5)
    h.bb.set("done", True)
    assert h.run(3, until=lambda h, s: s == "RETREAT"), h.states


def test_no_answer_raises_alert():
    h = Harness()
    h.move_free("glass_free", [-2.0, -2.0, 0.06])
    request_water(h)
    assert h.run(200, until=lambda h, s: s == "ASK")
    h.run(31)
    assert any("no answer" in t for t in h.kinds("alert"))
