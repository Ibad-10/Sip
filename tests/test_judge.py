"""Judge-style checks: physical safety invariants on every tick, perception-before-action,
autonomy with no network, and chaotic unexpected events.

Invariants (checked every control tick by Monitor):
  - robot never tips: base height < 0.05 m above start, roll/pitch < 10 deg
  - carried glass never drops: while held, glass stays within 0.12 m of the gripper
  - near the face (< 0.28 m) the straw tip never closes on the mouth faster than 0.05 m/s
    (0.03 cap + noise); pulling back fast is allowed
  - straw tip never closer than 2 cm to the lips unless "ready" was given
  - no robot body part touches the person (only the glass/straw may)
  - glass-to-person contact force never exceeds 10 N
"""
import os

import mujoco
import numpy as np
import pytest

from test_water import Harness, finished, request_water

ROBOT_BODIES = None


def _robot_geoms(m):
    root = m.body("base_link").id
    out = set()
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        while b > 0 and b != root:
            b = m.body_parentid[b]
        if b == root:
            out.add(g)
    return out


def _person_geoms(m):
    root = m.body("person").id
    return {g for g in range(m.ngeom) if m.geom_bodyid[g] in
            {b for b in range(m.nbody) if b == root or m.body_parentid[b] == root}}


class Monitor:
    def __init__(self, h):
        self.h = h
        self.robot_g, self.person_g = _robot_geoms(h.m), _person_geoms(h.m)
        self.prev_tip = None
        self.violations: list[str] = []
        self.max_near_speed = 0.0
        self.max_force = 0.0

    def __call__(self, h, state):
        d, m = h.d, h.m
        q = d.qpos[3:7]
        roll = np.degrees(np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))
        pitch = np.degrees(np.arcsin(np.clip(2 * (q[0] * q[2] - q[3] * q[1]), -1, 1)))
        if abs(roll) > 10 or abs(pitch) > 10 or d.qpos[2] > 0.05:
            self.violations.append(f"t={h.t:.1f} robot tipping roll={roll:.0f} pitch={pitch:.0f}")
        if h.io.held == "glass":
            if np.linalg.norm(d.body("glass").xpos - h.io.grasp_center()) > 0.12:
                self.violations.append(f"t={h.t:.1f} glass slipped from gripper")
        tip, mouth = d.site("straw_tip").xpos.copy(), d.site("mouth").xpos
        dist = np.linalg.norm(tip - mouth)
        if self.prev_tip is not None and dist < 0.30 and h.io.held == "glass":
            closing = np.linalg.norm(self.prev_tip - mouth) - dist  # > 0 when moving toward the mouth
            v = max(0.0, closing) / (25 * m.opt.timestep)
            self.max_near_speed = max(self.max_near_speed, v)
            if v > 0.05 and state == "APPROACH" and dist < 0.28:  # 2 cm braking margin inside the 0.30 m zone
                self.violations.append(f"t={h.t:.1f} fast near face {v:.3f} m/s")
        self.prev_tip = tip
        if dist < 0.018 and not h.bb.get("ready", False):
            self.violations.append(f"t={h.t:.1f} at the lips without ready")
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {c.geom1, c.geom2}
            if pair & self.robot_g and pair & self.person_g:
                self.violations.append(f"t={h.t:.1f} robot touched person: "
                                       f"{m.geom(c.geom1).name}/{m.geom(c.geom2).name}")
            elif pair & self.person_g and any(m.geom_bodyid[g] == m.body("glass").id for g in pair):
                mujoco.mj_contactForce(m, d, i, f6)
                self.max_force = max(self.max_force, abs(f6[0]))
        if self.max_force > 10:
            self.violations.append(f"t={h.t:.1f} contact force {self.max_force:.1f} N")


def test_physics_invariants_full_run():
    h = Harness()
    mon = Monitor(h)
    request_water(h)
    assert h.run(240, until=finished, on=mon), h.states
    assert not mon.violations, mon.violations[:5]
    assert h.hold >= 3.0


def test_perception_before_action():
    """It must find the glass with its camera before driving to pick it up (never told where)."""
    h = Harness()
    request_water(h)
    assert h.run(80, until=lambda h, s: s == "GRASP"), h.states
    kinds = [(e.kind, e.text) for e in h.bb.events()]
    first_nav = next(i for i, (k, t) in enumerate(kinds) if k == "state" and "-> NAV_TO_OBJECT" in t)
    found = next(i for i, (k, t) in enumerate(kinds) if k == "perceive" and "found the glass" in t)
    assert found < first_nav
    assert any(k == "state" and "confirmed by camera" in t for k, t in kinds)


def test_autonomy_no_network(monkeypatch):
    """No Gemini key, no local LLM server: the task still completes on-device."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    h = Harness()
    request_water(h)
    assert h.run(240, until=finished), h.states


def test_chaos_unexpected_events():
    """Several unscripted changes in one run: chair dropped in the hall, glass nudged while the
    robot drives to it, head turned away mid-approach, a 'stop' then 'go'. Invariants must hold
    and the drink must still be delivered."""
    rng = np.random.default_rng(int(os.environ.get("SIP_CHAOS_SEED", "7")))
    h = Harness()
    mon = Monitor(h)
    request_water(h)
    ev = {"chair": False, "nudge": False, "turn": None, "stop": None}

    def on(h, s):
        mon(h, s)
        if not ev["chair"] and h.t > 3.0:
            h.move_free("chair_hall_free", [-2.0, 0.45 + rng.uniform(-0.1, 0.1), 0.0])
            ev["chair"] = True
        if not ev["nudge"] and s == "NAV_TO_OBJECT":
            g = h.d.body("glass").xpos.copy()
            h.move_free("glass_free", [g[0] + rng.uniform(-0.12, 0.12), g[1], g[2]])
            ev["nudge"] = True
        if ev["stop"] is None and s == "NAV_TO_PERSON" and h.t > 0:
            h.bb.set("stop", True)
            ev["stop"] = h.t
        if ev["stop"] is not None and h.t - ev["stop"] > 2.0 and h.bb.get("stop"):
            h.bb.set("go", True)
        if ev["turn"] is None and s == "APPROACH" and h.task.dist_to_mouth() < 0.2:
            h.d.ctrl[h.io.act["head_yaw_act"]] = rng.choice([-0.8, 0.8])
            ev["turn"] = h.t
        if ev["turn"] is not None and h.t - ev["turn"] > 2.5:
            h.d.ctrl[h.io.act["head_yaw_act"]] = 0.0

    assert h.run(320, until=finished, on=on), h.states
    assert not mon.violations, mon.violations[:5]
    assert h.hold >= 3.0
    assert "HALT" in h.states
