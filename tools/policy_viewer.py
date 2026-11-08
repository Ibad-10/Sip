"""Watch a trained arm policy drive the Stretch 3 to random targets (red ball).

uv run python tools/policy_viewer.py --skill pour --seed 2
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from sip.learn.reach_env import SUBSTEPS, StrawReachEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", default="pour", choices=["straw", "offer", "pour", "grasp"])
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--path", default=None)
    a = ap.parse_args()
    model = PPO.load(a.path or f"assets/policy/{a.skill}_seed{a.seed}.zip", device="cpu")
    env = StrawReachEnv(seed=int(time.time()), task=a.skill)
    o, _ = env.reset()
    ok = n = 0
    with mujoco.viewer.launch_passive(env.m, env.d) as v:
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 2.2, -20, 210
        while v.is_running():
            a_, _ = model.predict(o, deterministic=True)
            o, r, done, trunc, info = env.step(a_)
            with v.lock():
                v.user_scn.ngeom = 0
                g = v.user_scn.geoms[0]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [0.025, 0, 0], env.target, np.eye(3).flatten(),
                                    np.array([0.1, 0.9, 0.2, 0.8] if done else [0.95, 0.2, 0.2, 0.8], np.float32))
                v.user_scn.ngeom = 1
            v.sync()
            time.sleep(SUBSTEPS * env.m.opt.timestep)
            if done or trunc:
                n += 1
                ok += int(done)
                print(f"{a.skill}: {'reached' if done else 'missed'} (final error {info['dist'] * 100:.1f} cm)  "
                      f"success so far {ok}/{n}", flush=True)
                time.sleep(0.6)
                o, _ = env.reset()


if __name__ == "__main__":
    main()
