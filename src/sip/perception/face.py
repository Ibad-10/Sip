"""Face / mouth tracking -> MouthState.

'sim' mode reads the simulated mouth site with noise and checks it is actually visible in the
head camera (inside the image, not occluded). 'mediapipe' mode is attempted later; it falls back
to 'sim' per frame when no face is found and sets MouthState.source accordingly.
"""
from typing import Literal

import numpy as np

from sip.perception.objects import CAM, _body_geoms, project
from sip.types import MouthState


class FaceTracker:
    def __init__(self, mode: Literal["mediapipe", "sim"] = "sim", noise_m: float = 0.005, seed: int = 0):
        self.mode = mode
        self.noise = noise_m
        self.rng = np.random.default_rng(seed)
        self._person = None

    def update(self, rgb, depth, seg, model, data, hand_raised: bool = False) -> MouthState:
        return self._sim(depth, seg, model, data, hand_raised)

    def _sim(self, depth, seg, model, data, hand_raised: bool) -> MouthState:
        if self._person is None:
            self._person = _body_geoms(model, "person")
        h, w = depth.shape
        mouth = data.site("mouth").xpos.copy()
        yaw_deg = float(np.rad2deg(data.qpos[model.joint("head_yaw").qposadr[0]]))
        u, v, z = project(model, data, CAM, mouth, w, h)
        visible = False
        if z > 0 and 0 <= u < w and 0 <= v < h:
            ui, vi = int(u), int(v)
            visible = bool(np.isin(seg[vi, ui], self._person)) and abs(depth[vi, ui] - z) < 0.08
        xyz = mouth + self.rng.normal(0, self.noise, 3) if visible else None
        return MouthState(visible, xyz, 0.5 if visible else 0.0, yaw_deg, hand_raised, "sim")
