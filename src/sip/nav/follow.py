"""Pure-pursuit path follower for the differential-drive base."""
import numpy as np

from sip.types import Pose2D

GOAL_TOL = 0.1
SLOW_RADIUS = 0.5
TURN_IN_PLACE = np.deg2rad(50)


def wrap(a: float) -> float:
    return float(np.angle(np.exp(1j * a)))


def face(pose: Pose2D, target_xy: tuple[float, float], k: float = 1.5, w_max: float = 0.8) -> float:
    err = wrap(np.arctan2(target_xy[1] - pose.y, target_xy[0] - pose.x) - pose.yaw)
    return float(np.clip(k * err, -w_max, w_max))


class PathFollower:
    def __init__(self, path: list[tuple[float, float]], lookahead: float = 0.3, v_max: float = 0.3):
        self.path = [np.asarray(p, float) for p in path]
        self.lookahead, self.v_max = lookahead, v_max
        self._i = 0
        self.done = len(self.path) == 0

    @property
    def goal(self) -> np.ndarray:
        return self.path[-1]

    def _target(self, p: np.ndarray) -> np.ndarray:
        while self._i < len(self.path) - 1 and np.linalg.norm(self.path[self._i] - p) < self.lookahead:
            self._i += 1
        return self.path[self._i]

    def step(self, pose: Pose2D) -> tuple[float, float]:
        p = np.array([pose.x, pose.y])
        dist_goal = float(np.linalg.norm(self.goal - p))
        if dist_goal < GOAL_TOL:
            self.done = True
            return 0.0, 0.0
        tgt = self._target(p)
        err = wrap(np.arctan2(tgt[1] - p[1], tgt[0] - p[0]) - pose.yaw)
        w = float(np.clip(1.8 * err, -0.8, 0.8))
        if abs(err) > TURN_IN_PLACE:
            return 0.0, w
        v = self.v_max * np.cos(err) * min(1.0, dist_goal / SLOW_RADIUS)
        return float(max(v, 0.05)), w
