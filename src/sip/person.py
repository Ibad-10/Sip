"""Simulated person: idle gaze (TV, window, around the room), watches the robot when it comes close,
breathes, and is sometimes distracted mid-approach. Unscripted: timing and choices come from a
random generator, so every run differs. The robot never reads this module; it only sees the result
through its cameras."""
import os

import numpy as np

from sip.nav.follow import wrap
from sip.types import Event


def mujoco_has(model, name: str) -> bool:
    try:
        model.geom(name)
        return True
    except KeyError:
        return False

LOOK_AT = {"tv": (1.2, 0.85, 1.0), "window": (4.0, 1.9, 1.4), "door": (2.0, 0.8, 1.4),
           "kitchen": (-0.5, 2.0, 1.2)}
MAX_IDLE_YAW = 0.9      # rad, how far she turns her head when idle
NEAR_ROBOT_M = 1.2


class Person:
    def __init__(self, model, data, bb, seed: int | None = None):
        self.m, self.d, self.bb = model, data, bb
        self.rng = np.random.default_rng(seed)
        self.mid = model.body("person").mocapid[0]
        self.act = model.actuator("head_yaw_act").id
        self.override = False            # /turn from the panel wins
        self.p_distract = float(os.environ.get("SIP_PERSON_DISTRACT", "0.3"))
        self._next, self._yaw, self._base = 0.0, 0.0, None
        self._distract_until, self._approach_seen = -1.0, False
        # taking tablets: Mixamo-derived keyframes shown one at a time
        self.frames = [model.geom(f"grandma_drink{k}").id for k in range(20)
                       if mujoco_has(model, f"grandma_drink{k}")]
        self.rest = model.geom("grandma_body").id if mujoco_has(model, "grandma_body") else None
        self.pills = [model.geom(f"pill{i}").id for i in range(3) if mujoco_has(model, f"pill{i}")]
        self.p_decline = float(os.environ.get("SIP_PERSON_DECLINE", "0"))
        self._take = None    # (t_start, will_take)

    def _facing(self) -> float:
        w, x, y, z = self.d.mocap_quat[self.mid]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def _yaw_to(self, xyz) -> float:
        p = self.d.mocap_pos[self.mid]
        return float(np.clip(wrap(np.arctan2(xyz[1] - p[1], xyz[0] - p[0]) - self._facing()),
                             -MAX_IDLE_YAW, MAX_IDLE_YAW))

    def _show(self, gid) -> None:
        for g in self.frames + ([self.rest] if self.rest is not None else []):
            self.m.geom_group[g] = 1 if g == gid else 5

    def _tablets(self, t: float) -> None:
        """When the robot offers the organiser: reach in, hand to mouth, back; then say so."""
        if not self.bb.get("dose_waiting"):
            if self._take is not None and self.rest is not None:
                self._show(self.rest)
            self._take = None
            return
        if self._take is None:
            self._take = (t + self.rng.uniform(2.0, 4.0), self.rng.random() >= self.p_decline)
            return
        t0, will = self._take
        if t < t0:
            return
        if not will:
            if self.bb.get("dose_reply") is None:
                self.bb.set("dose_reply", "declined")
                self.bb.log(Event(t, "heard", "she says: not now"))
            return
        if not self.frames:
            k = None
        else:
            k = int((t - t0) / 0.45)
        if k is not None and k < len(self.frames):
            self._show(self.frames[k])
            if k >= len(self.frames) // 2:
                for g in self.pills:
                    self.m.geom_rgba[g][3] = 0.0     # tablets are in her mouth now
        elif self.bb.get("dose_reply") is None and t > t0 + 0.45 * len(self.frames) + 1.0:
            if self.rest is not None:
                self._show(self.rest)
            for g in self.pills:
                self.m.geom_rgba[g][3] = 0.0
            self.bb.set("dose_reply", "taken")
            self.bb.log(Event(t, "heard", "she says: I have taken them"))

    def step(self, t: float, robot_state: str = "IDLE") -> None:
        self._tablets(t)
        pos = self.d.mocap_pos[self.mid]
        if self._base is None or np.linalg.norm(pos[:2] - self._base[:2]) > 0.05:
            self._base = pos.copy()       # moved (e.g. to the bed): new resting position
        pos[2] = self._base[2] + 0.004 * np.sin(2 * np.pi * t / 4.5)  # breathing
        if self.override:
            return
        grasp = self.d.body("link_grasp_center").xpos
        near = np.linalg.norm(grasp[:2] - pos[:2]) < NEAR_ROBOT_M
        approaching = robot_state in ("APPROACH", "HOLD")
        if approaching and not self._approach_seen:
            self._approach_seen = True
            if self.rng.random() < self.p_distract:   # e.g. hears something at the door
                start = t + self.rng.uniform(2.0, 6.0)
                self._distract_until = (start, start + self.rng.uniform(2.0, 3.5))
        if not approaching:
            self._approach_seen = False
        if isinstance(self._distract_until, tuple) and self._distract_until[0] <= t < self._distract_until[1]:
            self._yaw = self._yaw_to(LOOK_AT["door"])
        elif near:
            self._yaw = 0.6 * self._yaw_to(grasp)     # watches the robot, mostly facing forward
        elif t >= self._next:
            self._next = t + self.rng.uniform(3.0, 8.0)
            pick = self.rng.choice(list(LOOK_AT) + ["ahead", "ahead"])
            self._yaw = 0.0 if pick == "ahead" else self._yaw_to(LOOK_AT[pick])
        cur = self.d.ctrl[self.act]
        self.d.ctrl[self.act] = cur + float(np.clip(self._yaw - cur, -0.03, 0.03))  # ~1.2 rad/s at 40 Hz
