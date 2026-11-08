"""Run a trained arm policy on the real robot task, under the safety layer.

The policy was trained with the robot base at the origin facing +x, so the target error is rotated
into the robot's own base frame before it is fed in. The policy only moves lift / arm / wrist yaw;
speed is capped by the caller and the classic controller takes over if the policy stalls.
"""
from pathlib import Path

import numpy as np

from sip.learn.reach_env import MAX_DELTA

POLICY_DIR = Path(__file__).resolve().parents[3] / "assets" / "policy"
NAMES = ("lift", "arm", "wrist_yaw")
# what each learned skill really does (the "pour" skill only learns to put the jug above the glass)
DISPLAY = {"pour": "reach-above-glass", "grasp": "reach-to-grasp", "straw": "straw-to-lips", "offer": "offer-at-chest"}


MIN_SUCCESS = 0.90    # a learned skill only drives the robot if it succeeded this often on held-out tests


def _success(zip_path: Path) -> float:
    """Held-out success rate recorded at training time (per-seed json, or v1's training_results.json)."""
    import json
    js = zip_path.with_suffix(".json")
    if js.exists():
        return float(json.loads(js.read_text())["success_rate"])
    table = zip_path.parent / "training_results.json"
    if table.exists():
        skill, seed = zip_path.stem.rsplit("_seed", 1)
        for row in json.loads(table.read_text()):
            if row["task"] == skill and int(row["seed"]) == int(seed):
                return float(row["success_rate"])
    return 0.0


def best_policy(skill: str) -> Path | None:
    """Best seed across trained versions, only if it passed MIN_SUCCESS on held-out targets."""
    best = (0.0, None)
    for ver in ("v5", "v4", "v3", "v2", "v1"):
        for p in (POLICY_DIR / ver).glob(f"{skill}_seed*.zip"):
            s = _success(p)
            if s > best[0]:
                best = (s, p)
    return best[1] if best[0] >= MIN_SUCCESS else None


class PolicyArm:
    def __init__(self, skill: str, path: Path | None = None):
        from stable_baselines3 import PPO
        self.skill = skill
        self.name = DISPLAY.get(skill, skill)
        self.path = path or best_policy(skill)
        if self.path is None:
            raise FileNotFoundError(f"no trained policy for {skill}")
        self.model = PPO.load(str(self.path), device="cpu")
        self.norm = None
        vn = self.path.with_name(self.path.stem + "_vecnorm.pkl")
        if vn.exists():
            import pickle
            with open(vn, "rb") as f:
                self.norm = pickle.load(f)
        self.prev = np.zeros(3)
        self.speed = 0.0
        self.last_dist = None

    def act(self, io, target: np.ndarray, dt: float, max_speed: float = 0.15) -> float:
        """One control step towards `target` (world xyz). Returns the remaining distance (m)."""
        p = io.base_pose()
        tip = io.grasp_center()
        err_w = np.asarray(target) - tip
        c, s = np.cos(-p.yaw), np.sin(-p.yaw)
        err = np.array([c * err_w[0] - s * err_w[1], s * err_w[0] + c * err_w[1], err_w[2]])   # base frame
        dist = float(np.linalg.norm(err))
        if self.last_dist is not None:
            self.speed = (self.last_dist - dist) / max(dt, 1e-6)
        self.last_dist = dist
        q = [io.arm_targets()[n] for n in NAMES]
        obs = np.concatenate([err, [dist], q, self.prev, [self.speed]]).astype(np.float32)
        if self.norm is not None:
            obs = self.norm.normalize_obs(obs)
        a, _ = self.model.predict(obs, deterministic=True)
        a = np.clip(np.asarray(a, float), -1, 1)
        from sip.learn.reach_env import TASKS
        step = TASKS[self.skill].get("max_step", MAX_DELTA)     # same action limit as in training
        scale = min(1.0, max_speed * dt / step)                 # never faster than the safety layer allows
        cmd = {n: q[i] + a[i] * step * scale for i, n in enumerate(NAMES)}
        io.set_arm(**cmd)
        self.prev = a
        return dist
