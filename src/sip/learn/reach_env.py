"""Learned straw-reach policy: environment for PPO.

The policy moves lift, arm extension and wrist yaw (position-target deltas) so the straw tip (grasp
centre) reaches a mouth point. Reward favours getting close, smooth commands and SLOW closing speed
near the face (the same 0.03 m/s rule the safety gate enforces). The robot base is still; the real
task parks the base first. In the robot, the safety gate still caps speed and can stop the policy.
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parents[3] / "third_party/mujoco_menagerie/hello_robot_stretch_3/scene.xml"
SUBSTEPS = 25            # 0.05 s per policy step at dt 0.002
MAX_DELTA = 0.012        # m (or rad) per step -> 0.24 m/s max command rate
NEAR, SLOW = 0.30, 0.03  # inside 30 cm, close at most 3 cm/s
GOAL = 0.02


# Three skills, same arm, different targets and rules:
#   straw: tip to the lips, slow inside 30 cm, 2 cm goal
#   offer: pill organiser to chest height in front of her, slow inside 30 cm, 4 cm goal
#   pour:  jug spout above the glass, precise 1.5 cm goal, no face rule
TASKS = {
    "straw": {"dy": (0.15, 0.45), "dz": (-0.15, 0.15), "goal": 0.02, "near": 0.30, "slow": 0.03},
    "offer": {"dy": (0.15, 0.45), "dz": (-0.35, -0.05), "goal": 0.04, "near": 0.30, "slow": 0.05},
    "pour":  {"dy": (0.10, 0.40), "dz": (-0.05, 0.25), "goal": 0.015, "near": 0.0, "slow": 1.0},
    # grasp: objects anywhere from a low side table to a high shelf, either side of the arm line
    "grasp": {"dy": (0.05, 0.50), "dz": (-0.45, 0.30), "dx": 0.10, "goal": 0.02, "near": 0.0, "slow": 1.0},
}


class StrawReachEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None, task: str = "straw", noise: float = 0.01, randomise: bool = False):
        self.task, self.cfg, self.noise, self.randomise = task, TASKS[task], noise, randomise
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        self.act = {self.m.actuator(i).name: i for i in range(self.m.nu)}
        self.gc = self.m.body("link_grasp_center").id
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (11,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.names = ("lift", "arm", "wrist_yaw")
        self._nominal = (self.m.actuator_gainprm.copy(), self.m.actuator_biasprm.copy(),
                         self.m.dof_damping.copy(), self.m.body_mass.copy())
        self._pending = np.zeros(3)

    def _tip(self):
        return self.d.xpos[self.gc].copy()

    def _obs(self):
        # perception noise (domain randomisation): the policy sees a jittered target, like a real detector
        err = self.target - self._tip() + self.rng.normal(0, self.noise, 3)
        q = [self.d.ctrl[self.act[n]] for n in self.names]
        return np.concatenate([err, [np.linalg.norm(err)], q, self.prev, [self.speed]]).astype(np.float32)

    def _domain_randomise(self) -> None:
        """Sim-to-real: motor strength +-20 %, joint damping x0.5-2, link masses +-15 %."""
        g, b, dmp, mass = self._nominal
        k = self.rng.uniform(0.8, 1.2, self.m.nu)
        self.m.actuator_gainprm[:] = g
        self.m.actuator_biasprm[:] = b
        self.m.actuator_gainprm[:, 0] *= k
        self.m.actuator_biasprm[:, 1] *= k
        self.m.dof_damping[:] = dmp * self.rng.uniform(0.5, 2.0, self.m.nv)
        self.m.body_mass[:] = mass * self.rng.uniform(0.85, 1.15, self.m.nbody)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.randomise:
            self._domain_randomise()
        mujoco.mj_resetData(self.m, self.d)
        self._pending = np.zeros(3)
        start = {"lift": self.rng.uniform(0.6, 1.0), "arm": self.rng.uniform(0.0, 0.15), "wrist_yaw": 0.0}
        for n, v in start.items():
            self.d.ctrl[self.act[n]] = v
        for _ in range(300):
            mujoco.mj_step(self.m, self.d)
        tip = self._tip()
        # a mouth somewhere ahead of the arm (arm extends along the base's -y)
        c = self.cfg
        dx = c.get("dx", 0.03)
        self.target = tip + np.array([self.rng.uniform(-dx, dx), -self.rng.uniform(*c["dy"]),
                                      self.rng.uniform(*c["dz"])])
        self.prev = np.zeros(3)
        self.speed = 0.0
        self.k = 0
        self.dist = float(np.linalg.norm(self.target - tip))
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1, 1)
        if self.randomise:   # one-step actuation delay, like a real controller loop
            a, self._pending = self._pending, a
        for i, n in enumerate(self.names):
            lo, hi = self.m.actuator_ctrlrange[self.act[n]]
            self.d.ctrl[self.act[n]] = np.clip(self.d.ctrl[self.act[n]] + a[i] * MAX_DELTA, lo, hi)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.m, self.d)
        dist = float(np.linalg.norm(self.target - self._tip()))
        dt = SUBSTEPS * self.m.opt.timestep
        self.speed = (self.dist - dist) / dt          # closing speed
        c = self.cfg
        r = 10.0 * (self.dist - dist) - 0.005 - 0.02 * float(np.sum((a - self.prev) ** 2))
        if dist < c["near"] and self.speed > c["slow"]:
            r -= 20.0 * (self.speed - c["slow"]) * (self.speed - c["slow"]) / c["slow"]   # gentle, grows with overspeed
        done = dist < c["goal"]
        if done:
            r += 5.0
        if dist < 0.005:                              # never poke past the lips
            r -= 2.0
        self.prev, self.dist = a, dist
        self.k += 1
        return self._obs(), float(r), bool(done), self.k >= (700 if c["near"] > 0 else 300), {"dist": dist}
