"""Simulated person: idle gaze (TV, window, around the room), watches the robot when it comes close,
breathes, and is sometimes distracted mid-approach. Unscripted: timing and choices come from a
random generator, so every run differs. The robot never reads this module; it only sees the result
through its cameras."""
import os

import numpy as np

from sip.nav.follow import wrap

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

    def _facing(self) -> float:
        w, x, y, z = self.d.mocap_quat[self.mid]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def _yaw_to(self, xyz) -> float:
        p = self.d.mocap_pos[self.mid]
        return float(np.clip(wrap(np.arctan2(xyz[1] - p[1], xyz[0] - p[0]) - self._facing()),
                             -MAX_IDLE_YAW, MAX_IDLE_YAW))

    def step(self, t: float, robot_state: str = "IDLE") -> None:
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
