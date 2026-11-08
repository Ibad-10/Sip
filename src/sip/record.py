"""Record a headless run to MP4: overview camera following the robot, head camera inset, captions.

uv run python -m sip.record --out runs/water.mp4 --seconds 110 [--chair] [--turn]
"""
import argparse
import os

import cv2
import mujoco
import numpy as np

from sip.app import Sim
from sip.perception.objects import render_head
from sip.tasks.water import TICK_STEPS

W, H, FPS = 960, 540, 10
HD = (1920, 1080, 30)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/water.mp4")
    ap.add_argument("--seconds", type=float, default=110)
    ap.add_argument("--chair", action="store_true", help="drop a chair in the hall at t=6 s")
    ap.add_argument("--turn", action="store_true", help="person turns head away during approach")
    ap.add_argument("--hd", action="store_true", help="1920x1080 at 30 fps (for GPU rendering)")
    a = ap.parse_args(argv)
    global W, H, FPS
    if a.hd:
        W, H, FPS = HD
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sim = Sim()
    sim.bb.set("ready", True)
    sim.bb.set("request", "water")
    r = mujoco.Renderer(sim.m, H, W)
    inset = mujoco.Renderer(sim.m, 180, 240)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 3.2, -35, 135
    opt = mujoco.MjvOption()
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    caption, turned, every = "", None, int(round(1 / (FPS * TICK_STEPS * sim.m.opt.timestep)))
    n = 0
    while sim.t < a.seconds:
        state = sim.task.step(sim.t)
        sim.person.step(sim.t, state)
        if a.chair and 6.0 <= sim.t < 6.06:
            sim.special("/chair", sim.bb)
        if a.turn and state == "APPROACH" and turned is None and sim.task.dist_to_mouth() < 0.25:
            sim.special("/turn", sim.bb)
            turned = sim.t
        if turned is not None and sim.t - turned > 3.0 and sim.person.override:
            sim.special("/turn", sim.bb)
        for _ in range(TICK_STEPS):
            mujoco.mj_step(sim.m, sim.d)
        sim.t += TICK_STEPS * sim.m.opt.timestep
        evs = sim.bb.events()
        if evs:
            caption = f"[{evs[-1].kind}] {evs[-1].text}"
        n += 1
        if n % every:
            continue
        p = sim.io.base_pose()
        cam.lookat[:] = [p.x, p.y, 0.7]
        r.update_scene(sim.d, cam, opt)
        frame = r.render()[:, :, ::-1].copy()
        rgb, _, _ = render_head(sim.m, sim.d, inset, rgb=True)
        frame[10:190, W - 250:W - 10] = rgb[:, :, ::-1]
        cv2.rectangle(frame, (W - 251, 9), (W - 9, 191), (255, 255, 255), 1)
        cv2.putText(frame, "robot head camera", (W - 245, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        pe, _, _ = render_head(sim.m, sim.d, inset, cam="person_eyes", rgb=True)
        frame[220:400, W - 250:W - 10] = pe[:, :, ::-1]
        cv2.rectangle(frame, (W - 251, 219), (W - 9, 401), (255, 255, 255), 1)
        cv2.putText(frame, "person's view", (W - 245, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.rectangle(frame, (0, H - 60), (W, H), (0, 0, 0), -1)
        cv2.putText(frame, f"t={sim.t:5.1f}s  state={state}", (12, H - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 200, 255), 2)
        cv2.putText(frame, caption[:110], (12, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        d = sim.task.dist_to_mouth()
        if d < 0.6:
            cv2.putText(frame, f"straw to mouth: {d * 100:.1f} cm", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if d <= 0.021 else (255, 255, 255), 2)
        vw.write(frame)
        if state == "IDLE" and "RETURN" in "".join(e.text for e in evs[-3:]):
            break
    vw.release()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
