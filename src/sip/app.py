"""Launcher: MuJoCo passive viewer + water task + terminal commands + Rerun explain view.

Usage:  uv run python -m sip.app             viewer + Rerun; type commands in this terminal
        uv run python -m sip.app --headless --seconds 30 --no-rerun
Commands: any sentence ("I'm thirsty", "stop", "ready", "go"),
          or /reset /hand /bed /chair /glass /turn  (live changes for the demo)
Toggles:  /lidar (F7)  /path (F8)  /pip (F9: picture-in-picture, robot view + person view)
Cameras:  in the viewer press ] or [ to cycle: person_eyes, d435i_camera_rgb (robot head),
          d405_rgb (wrist), cam_kitchen/living/bedroom/hall (follow robot), cam_overview
"""
import argparse
import os
import sys
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

from sip.blackboard import Blackboard
from sip.types import Event
from sip.config import load_profile
from sip.memory import ObjectMemory
from sip.nav.grid import grid_from_model
from sip.perception.face import FaceTracker
from sip.perception.objects import CAM, SegDetector, project
from sip.robot import StretchIO
from sip.scene.build_house import build_house, seat_person
from sip.tasks.water import TICK_STEPS, WaterTask
from sip.voice.intents import parse_keywords


def handle_command(text: str, bb: Blackboard, llm=None, sim=None) -> None:
    text = text.strip()
    if not text:
        return
    if text.startswith("/"):
        toggles = {"/lidar": "show_lidar", "/path": "show_path", "/pip": "show_pip"}   # view options only
        if text in toggles:
            key = toggles[text]
            bb.set(key, not bb.get(key, True))
        elif sim is not None:
            sim.special(text, bb)
        return
    kw = parse_keywords(text)
    bb.set("heard", text)
    if "stop" in kw:
        bb.set("stop", True)
        return
    if "done" in kw:
        bb.set("done", True)
        return
    if "taken" in kw:
        bb.set("dose_reply", "taken")
        return
    if "no" in kw and bb.get("dose_waiting"):
        bb.set("dose_reply", "declined")
        return
    if "ready" in kw:
        bb.set("ready", True)
    if "go" in kw or "yes" in kw:
        bb.set("go", True)
    task = "water" if "water" in kw else "medicine" if "medicine" in kw else None
    if task is None and llm is not None and not ({"ready", "go", "yes", "no"} & kw):
        out = llm.intent(text)
        task = out["task"] if out["task"] in ("water", "medicine") else None
        bb.set("say", out["reply"])
    elif task is None and not kw:
        # small talk: Gemma 4 replies when online (capped); offline the robot just acknowledges
        def reply():
            from sip.cloud.gemini import chat_reply
            bb.set("say", chat_reply(text) or "I'm here. Would you like a drink or your tablets?")
        threading.Thread(target=reply, daemon=True).start()
    if task:
        bb.set("request", task)


class Sim:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(build_house("assets/house.xml"))
        self.d = mujoco.MjData(self.m)
        self.bb = Blackboard()
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[0:2] = [-1.0, 0.0]
        seat_person(self.m, self.d, "armchair")
        mujoco.mj_forward(self.m, self.d)
        self.io = StretchIO(self.m, self.d)
        if os.environ.get("SIP_DETECTOR", "yolo") == "yolo":   # real models on camera images
            from sip.perception.vision import MediaPipeFace, YoloDetector
            if not hasattr(Sim, "_models"):
                Sim._models = (YoloDetector(), MediaPipeFace())
            det, face = Sim._models
        else:                                                  # simulator oracle (fast tests only)
            det = SegDetector(["glass", "organiser", "jug", "straw2"], noise_m=0.005)
            face = FaceTracker("sim")
        grid, rooms = grid_from_model(self.m), None
        if os.path.exists(os.environ.get("SIP_MAP", "runs/map.npz")):   # map the robot built itself
            from sip.nav.explore import load_map
            grid, rooms = load_map(os.environ.get("SIP_MAP", "runs/map.npz"))
        self.task = WaterTask(self.io, self.bb, grid, ObjectMemory(), det, face,
                              prepare=os.environ.get("SIP_PREPARE", "1") == "1", rooms=rooms)
        from sip.person import Person
        self.person = Person(self.m, self.d, self.bb)
        self.t = 0.0

    def _move_free(self, joint, xyz):
        a = self.m.joint(joint).qposadr[0]
        self.d.qpos[a:a + 3] = xyz
        mujoco.mj_forward(self.m, self.d)

    def special(self, cmd: str, bb: Blackboard):
        if cmd == "/reset":
            for k in ("stop", "go", "request", "ready", "asked_ready"):
                bb.set(k, None)
            self.reset()
        elif cmd == "/bed":
            seat_person(self.m, self.d, "bed")
        elif cmd == "/chair":
            self._move_free("chair_hall_free", [-2.0, 0.45, 0.0])
        elif cmd == "/glass":
            self._move_free("glass_free", [3.15, -1.2, 0.57])
        elif cmd == "/dose":
            bb.set("request", "medicine")
        elif cmd == "/turn":
            a = self.io.act["head_yaw_act"]
            self.person.override = not self.person.override
            self.d.ctrl[a] = 0.8 if self.person.override else 0.0


_PANEL_R = {}


def save_alert_evidence(sim, folder: str = "runs/alerts") -> tuple[str, str]:
    """Snapshot for the carer: robot head camera, person's view and an overview, plus the recent log."""
    import json
    import cv2
    os.makedirs(folder, exist_ok=True)
    r = _PANEL_R.get("ev")
    if r is None:
        r = _PANEL_R["ev"] = mujoco.Renderer(sim.m, 360, 480)
    tiles = []
    for cam in ("d435i_camera_rgb", "person_eyes", "cam_overview"):
        r.update_scene(sim.d, camera=cam)
        tiles.append(r.render()[:, :, ::-1].copy())
    t, text = sim.bb.get("alert") or (sim.t, "alert")
    img = np.hstack(tiles)
    bar = np.zeros((44, img.shape[1], 3), np.uint8)
    bar[:] = (30, 30, 150)
    cv2.putText(bar, f"ALERT t={t:.1f}s: {text}"[:150], (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    img = np.vstack([bar, img])
    stem = os.path.join(folder, f"alert_{int(t * 10):06d}")
    cv2.imwrite(stem + ".jpg", img)
    events = [{"t": e.t, "kind": e.kind, "text": e.text} for e in sim.bb.events()[-30:]]
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump({"alert": text, "t": t, "state": sim.task.state, "events": events}, f, indent=1)
    return stem + ".jpg", stem + ".json"


def publish_panel(panel, sim, st) -> None:
    """Render the camera views (head camera upright, with live detections drawn) and publish the
    robot's beliefs for the web panel. Nothing here is ground truth except the camera images."""
    import cv2
    from sip.cloud import gemini
    from sip.nav.lidar import ray_angles
    from sip.perception.objects import project
    from sip.perception.objects import fast_renderer
    r = _PANEL_R.get("r")
    if r is None:
        r = _PANEL_R["r"] = fast_renderer(sim.m, 240, 320)
    tk = sim.task
    _PANEL_R["n"] = _PANEL_R.get("n", 0) + 1
    views = [("head", "d435i_camera_rgb")]
    if _PANEL_R["n"] % 3 == 1:                       # side views refresh every third update
        views += [("wrist", "d405_rgb"), ("person", "person_eyes")]
    for name, cam in views:
        if name == "head" and tk.last_frame is not None:
            img = tk.last_frame[:, :, ::-1].copy()     # the robot's own latest frame: no extra render
        else:
            r.update_scene(sim.d, camera=cam)
            img = r.render()[:, :, ::-1].copy()
        if name == "head":
            sy, sx = img.shape[0] / tk.renderer.height, img.shape[1] / tk.renderer.width
            for det in tk.last_dets:
                if not det.box:
                    continue
                u0, v0, u1, v1 = det.box
                col = (40, 160, 60) if det.label == "person" else (200, 120, 20)
                cv2.rectangle(img, (int(u0 * sx), int(v0 * sy)), (int(u1 * sx), int(v1 * sy)), col, 2)
            if tk.mouth.visible and tk.mouth.xyz is not None:
                u, v, z = project(sim.m, sim.d, cam, tk.mouth.xyz, img.shape[1], img.shape[0])
                if z > 0:
                    cv2.drawMarker(img, (int(u), int(v)), (40, 40, 230), cv2.MARKER_CROSS, 18, 2)
            img = np.ascontiguousarray(np.rot90(img, 1))   # the D435i is mounted sideways
            for det in tk.last_dets:   # labels after rotating so the text is upright
                if det.box:
                    u0, v0, u1, v1 = det.box
                    x, y = int(v0 * sy), int(img.shape[0] - 1 - u1 * sx)
                    cv2.putText(img, f"{det.label}{'?' if det.unsure else ''} {det.conf:.0%}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (255, 255, 255), 2)
                    cv2.putText(img, f"{det.label}{'?' if det.unsure else ''} {det.conf:.0%}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (30, 30, 30), 1)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            panel.set_frame(name, jpg.tobytes())
    p = sim.io.base_pose()
    lidar = []
    if st.get("ranges") is not None:
        rg, sp = st["ranges"], st.get("scan_pose") or p
        a = ray_angles(len(rg)) + sp.yaw
        lidar = [[round(sp.x + d * np.cos(t), 2), round(sp.y + d * np.sin(t), 2)] for d, t in zip(rg, a) if d < 3.99]
    mem = []
    for label in ("glass", "jug", "organiser"):
        w = tk.memory.where(label)
        if w is not None:
            mem.append({"label": label, "x": float(w[0][0]), "y": float(w[0][1]),
                        "c": float(tk.memory.confidence(label, sim.t))})
    sub = tk.grid.occ[::2, ::2]
    occ = {"nx": int(sub.shape[0]), "ny": int(sub.shape[1]), "d": "".join("1" if v else "0" for v in sub.flatten())}
    alert = sim.bb.get("alert")
    evs = sim.bb.events()
    reason = next((e.text.split(": ", 1)[-1] for e in reversed(evs) if e.kind == "state"), "")
    perc = getattr(tk.det, "needs_rgb", False)
    state = {
        "t": sim.t, "state": tk.state, "reason": reason, "online": bool(os.environ.get("GEMINI_API_KEY")),
        "gemini_calls": gemini._usage["calls"],
        "perception": (f"YOLO-World {tk.det.ms:.0f} ms + MediaPipe" if perc else "sim oracle (test mode)"),
        "decision": {"action": tk.decision.action, "reason": tk.decision.reason,
                     "max_speed": tk.decision.max_speed_mps} if tk.decision else None,
        "sensed": float(tk.sensed_dist()), "water": float(getattr(tk, "water_level", 1.0)),
        "hz": st["ticks"] / max(1e-6, time.time() - st["t0"]),
        "events": [{"t": e.t, "kind": e.kind, "text": e.text} for e in evs[-150:]],
        "alert": alert[1] if alert else None,
        "robot": [p.x, p.y, p.yaw],
        "places": [{"name": k, "x": float(v["xyz"][0]), "y": float(v["xyz"][1])} for k, v in tk.places.items()],
        "path": [list(map(float, q)) for q in tk._nav["follower"].path] if tk._nav else [],
        "lidar": lidar, "memory": mem, "occ": occ,
    }
    panel.set_state(state)


def stdin_thread(sim: Sim, llm):
    for line in sys.stdin:
        handle_command(line, sim.bb, llm, sim)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=1e9)
    ap.add_argument("--no-rerun", action="store_true")
    ap.add_argument("--request", default=None, help="start with a request, e.g. water")
    ap.add_argument("--no-panel", action="store_true", help="disable the web control panel")
    a = ap.parse_args(argv)
    sim = Sim()
    prof = load_profile()
    llm = None
    if os.environ.get("SIP_LLM", "1") == "1":
        from sip.voice.llm import LLM
        llm = LLM(prof.llm_url)
    ex = None
    if not a.no_rerun:
        from sip.explain.rr import Explainer
        os.makedirs("runs", exist_ok=True)
        ex = Explainer(spawn=not a.headless, save_path=f"runs/{time.strftime('%Y%m%d-%H%M%S')}.rrd")
    if a.request:
        sim.bb.set("ready", True)
        sim.bb.set("request", a.request)
    panel = None
    if not a.no_panel:
        from sip.panel import Panel
        from sip.voice import speech

        def on_audio(data: bytes) -> str:
            text = speech.transcribe(data)
            if text:
                sim.bb.log(Event(sim.t, "heard", f"she says: {text}"))
                handle_command(text, sim.bb, llm, sim)
            return text
        panel = Panel(lambda text: handle_command(text, sim.bb, llm, sim), on_audio=on_audio)
        sim.bb.set("show_pip", False)
        print("control panel:", panel.url, flush=True)
        if not a.headless:
            import webbrowser
            webbrowser.open(panel.url)
    if not a.headless:
        threading.Thread(target=stdin_thread, args=(sim, llm), daemon=True).start()
        print(__doc__)

    st = {"n_events": 0, "last_log": -1.0, "ticks": 0, "t0": time.time(), "path_first": None,
          "ranges": None, "last_scan": -1.0, "last_pip": -1.0}
    pip_r = None

    def tick():
        state = sim.task.step(sim.t)
        sim.person.step(sim.t, state)
        for _ in range(TICK_STEPS):
            mujoco.mj_step(sim.m, sim.d)
        sim.t += TICK_STEPS * sim.m.opt.timestep
        st["ticks"] += 1
        say = sim.bb.get("say")
        if say:
            print(f"[robot] {say}", flush=True)
            if not a.headless:
                from sip.voice import speech
                speech.speak(say)
            sim.bb.set("say", None)
        evs = sim.bb.events()
        for e in evs[st["n_events"]:]:
            if a.headless:
                print(f"{e.t:7.1f} {e.kind:8} {e.text}", flush=True)
            if ex:
                ex.log_event(e)
                if e.kind == "state":
                    ex.log_state(e.t, sim.task.state, e.text)
                if e.kind == "plan" and "new route" in e.text:
                    st["path_first"] = st.get("path_cur")
        st["n_events"] = len(evs)
        if sim.t - st["last_scan"] >= 0.5 and (sim.bb.get("show_lidar", True) or ex):
            from sip.nav.lidar import scan
            st["last_scan"] = sim.t
            st["scan_pose"] = sim.io.base_pose()          # draw rays from where they were cast
            st["ranges"] = scan(sim.m, sim.d, st["scan_pose"])
            if ex:
                ex.log_lidar(sim.t, sim.io.base_pose(), st["ranges"])
        alert = sim.bb.get("alert")
        if alert and alert != st.get("alert_saved"):
            st["alert_saved"] = alert
            jpg, _ = save_alert_evidence(sim)
            print(f"[ALERT] {alert[1]} -> evidence {jpg}", flush=True)
            if panel is not None:
                with open(jpg, "rb") as f:
                    panel.set_frame("alert", f.read())
        if panel is not None and time.time() - st.get("last_panel", 0.0) >= 0.4:   # wall clock: no lag build-up
            st["last_panel"] = time.time()
            publish_panel(panel, sim, st)
        if ex and sim.t - st["last_log"] >= 0.2:
            st["last_log"] = sim.t
            tk = sim.task
            mpx = None
            if tk.mouth.visible:
                u, v, z = project(sim.m, sim.d, CAM, sim.d.site("mouth").xpos)
                mpx = (u, v) if z > 0 else None
            ex.log_frame(sim.t, tk.last_frame, tk.last_dets, tk.mouth, mpx, tk.dist_to_mouth())
            nav = tk._nav
            st["path_cur"] = [tuple(p) for p in nav["follower"].path] if nav else None
            ex.log_map(sim.t, tk.grid, sim.io.base_pose(), st["path_first"], st["path_cur"], tk.memory)
            if tk.decision:
                ex.log_safety(sim.t, tk.decision, None, None)
            hz = st["ticks"] / max(1e-6, time.time() - st["t0"])
            ex.log_budget(sim.t, hz, {}, 0.0, online=bool(os.environ.get("GEMINI_API_KEY")))
        return state

    try:
        if a.headless:
            while sim.t < a.seconds:
                tick()
        else:

            import cv2
            from sip import overlay
            keys = {296: "/lidar", 297: "/path", 298: "/pip"}  # F7 F8 F9

            def on_key(code):
                if code in keys:
                    handle_command(keys[code], sim.bb, None, sim)

            with mujoco.viewer.launch_passive(sim.m, sim.d, key_callback=on_key) as v:
                while v.is_running() and sim.t < a.seconds:
                    t0 = time.time()
                    tick()
                    tk = sim.task
                    nav = tk._nav
                    with v.lock():
                        overlay.draw(v.user_scn, st.get("scan_pose") or sim.io.base_pose(), st["ranges"],
                                     [tuple(p) for p in nav["follower"].path] if nav else None,
                                     tk._approach_target() if tk.state in ("APPROACH", "HOLD") else None,
                                     show_lidar=sim.bb.get("show_lidar", True), show_path=sim.bb.get("show_path", True))
                    if sim.bb.get("show_pip", True) and sim.t - st["last_pip"] >= 0.25:
                        st["last_pip"] = sim.t
                        if pip_r is None:
                            pip_r = mujoco.Renderer(sim.m, 270, 360)
                        views = []
                        for cam, label in (("d435i_camera_rgb", "ROBOT SEES (head)"), ("d405_rgb", "ROBOT SEES (wrist)"),
                                           ("person_eyes", "PERSON SEES")):
                            pip_r.update_scene(sim.d, camera=cam)
                            img = pip_r.render()[:, :, ::-1].copy()
                            cv2.rectangle(img, (0, 0), (360, 26), (0, 0, 0), -1)
                            cv2.putText(img, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
                            views.append(img)
                        bar = np.zeros((40, 1080, 3), np.uint8)
                        cv2.putText(bar, f"state: {tk.state}   safety: {tk.decision.action + ' / ' + tk.decision.reason if tk.decision else '-'}",
                                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        cv2.imshow("Sip - what they see", np.vstack([np.hstack(views), bar]))
                        cv2.waitKey(1)
                    elif not sim.bb.get("show_pip", True):
                        cv2.destroyAllWindows()
                    v.sync()
                    dt = TICK_STEPS * sim.m.opt.timestep - (time.time() - t0)
                    if dt > 0:
                        time.sleep(dt)
    finally:
        print(f"loop_hz={st['ticks'] / max(1e-6, time.time() - st['t0']):.1f}", flush=True)
        if ex:
            ex.close()
        if panel is not None:
            panel.close()


if __name__ == "__main__":
    main()
