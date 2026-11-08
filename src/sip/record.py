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
    ap.add_argument("--seed", type=int, default=None, help="random home layout")
    ap.add_argument("--task", default="water", choices=["water", "medicine"])
    a = ap.parse_args(argv)
    global W, H, FPS
    if a.hd:
        W, H, FPS = HD
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sim = Sim()
    if hasattr(sim.task.det, "asynchronous"):
        sim.task.det.asynchronous = False      # video: boxes must match each frame exactly
    if a.seed is not None:
        from sip.scene.randomise import randomise_home
        print("layout", randomise_home(sim.m, sim.d, a.seed))
        sim.io.loc = None
        sim.io.enable_localisation()
    sim.bb.set("ready", True)
    sim.bb.set("request", a.task)
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
        if a.task == "medicine" and state == "IDLE" and any("medicine round finished" in e.text for e in sim.bb.events()[-3:]):
            break
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
        why = [e for e in evs[-8:] if e.kind in ("perceive", "plan", "safety", "cloud", "alert", "say", "dose")]
        if why:
            caption = f"[{why[-1].kind}] {why[-1].text}"
        n += 1
        if n % every:
            continue
        # main shot: the fixed room camera of the room the robot is in (tracks the robot; never behind a wall)
        from sip.scene.build_house import room_at
        tp = sim.io.true_pose()
        room = room_at(tp.x, tp.y)
        cname = {"kitchen": "cam_kitchen", "living": "cam_living", "bedroom": "cam_bedroom"}.get(room, "cam_hall")
        r.update_scene(sim.d, camera=cname)
        frame = r.render()[:, :, ::-1].copy()
        # robot head camera: its own latest frame with live detections (label, confidence, ? if unsure)
        tk = sim.task
        head = tk.last_frame if tk.last_frame is not None else render_head(sim.m, sim.d, inset, rgb=True)[0]
        hv = head[:, :, ::-1].copy()
        sy, sx = 1.0, 1.0
        for det in tk.last_dets:
            if det.box:
                u0, v0, u1, v1 = det.box
                col = (40, 200, 60) if det.label == "person" else (30, 150, 255)
                cv2.rectangle(hv, (u0, v0), (u1, v1), col, 2)
        hv = np.ascontiguousarray(np.rot90(hv, 1))                 # D435i is mounted sideways
        for det in tk.last_dets:
            if det.box:
                u0, v0, u1, v1 = det.box
                x, y = v0, hv.shape[0] - 1 - u1
                t = f"{det.label}{'?' if det.unsure else ''} {det.conf:.0%}"
                cv2.putText(hv, t, (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(hv, t, (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        ih = H // 2 - 30
        iw = int(ih * hv.shape[1] / hv.shape[0])
        frame[10:10 + ih, W - iw - 10:W - 10] = cv2.resize(hv, (iw, ih))
        cv2.rectangle(frame, (W - iw - 11, 9), (W - 9, 11 + ih), (255, 255, 255), 1)
        cv2.putText(frame, "robot camera (YOLO-World + CLIP + MediaPipe)", (W - iw - 6, 28 + ih), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        pe, _, _ = render_head(sim.m, sim.d, inset, cam="person_eyes", rgb=True)
        ph, pw = H // 3, int(H // 3 * 4 / 3)
        frame[H - 70 - ph:H - 70, W - pw - 10:W - 10] = cv2.resize(pe[:, :, ::-1], (pw, ph))
        cv2.rectangle(frame, (W - pw - 11, H - 71 - ph), (W - 9, H - 69), (255, 255, 255), 1)
        cv2.putText(frame, "her view", (W - pw - 6, H - 76 - ph), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        dec = tk.decision
        if dec is not None and state in ("APPROACH", "HOLD"):
            col = {"proceed": (60, 200, 60), "pause": (0, 190, 255), "retreat": (40, 40, 230)}.get(dec.action, (200, 200, 200))
            cv2.putText(frame, f"SAFETY: {dec.action.upper()} - {dec.reason}", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.putText(frame, f"arm: {getattr(tk, 'arm_driver', 'classic controller')}", (12, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
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
