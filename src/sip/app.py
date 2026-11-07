"""Launcher: MuJoCo passive viewer + water task + terminal commands + Rerun explain view.

Usage:  uv run python -m sip.app             viewer + Rerun; type commands in this terminal
        uv run python -m sip.app --headless --seconds 30 --no-rerun
Commands: any sentence ("I'm thirsty", "stop", "ready", "go"),
          or /reset /hand /bed /chair /glass /turn  (live changes for the demo)
"""
import argparse
import os
import sys
import threading
import time

import mujoco
import mujoco.viewer

from sip.blackboard import Blackboard
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
        if text == "/hand":
            bb.set("hand_raised", not bb.get("hand_raised", False))
        elif sim is not None:
            sim.special(text, bb)
        return
    kw = parse_keywords(text)
    bb.set("heard", text)
    if "stop" in kw:
        bb.set("stop", True)
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
        self.task = WaterTask(self.io, self.bb, grid_from_model(self.m), ObjectMemory(),
                              SegDetector(["glass", "organiser"], noise_m=0.005), FaceTracker("sim"))
        self.t = 0.0

    def _move_free(self, joint, xyz):
        a = self.m.joint(joint).qposadr[0]
        self.d.qpos[a:a + 3] = xyz
        mujoco.mj_forward(self.m, self.d)

    def special(self, cmd: str, bb: Blackboard):
        if cmd == "/reset":
            for k in ("stop", "go", "request", "ready", "hand_raised", "asked_ready"):
                bb.set(k, None)
            self.reset()
        elif cmd == "/bed":
            seat_person(self.m, self.d, "bed")
        elif cmd == "/chair":
            self._move_free("chair_hall_free", [-2.0, 0.45, 0.0])
        elif cmd == "/glass":
            self._move_free("glass_free", [3.15, -1.2, 0.57])
        elif cmd == "/turn":
            a = self.io.act["head_yaw_act"]
            self.d.ctrl[a] = 0.8 if self.d.ctrl[a] == 0 else 0.0


def stdin_thread(sim: Sim, llm):
    for line in sys.stdin:
        handle_command(line, sim.bb, llm, sim)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=1e9)
    ap.add_argument("--no-rerun", action="store_true")
    ap.add_argument("--request", default=None, help="start with a request, e.g. water")
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
    if not a.headless:
        threading.Thread(target=stdin_thread, args=(sim, llm), daemon=True).start()
        print(__doc__)

    st = {"n_events": 0, "last_log": -1.0, "ticks": 0, "t0": time.time(), "path_first": None}

    def tick():
        state = sim.task.step(sim.t)
        for _ in range(TICK_STEPS):
            mujoco.mj_step(sim.m, sim.d)
        sim.t += TICK_STEPS * sim.m.opt.timestep
        st["ticks"] += 1
        say = sim.bb.get("say")
        if say:
            print(f"[robot] {say}", flush=True)
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

            with mujoco.viewer.launch_passive(sim.m, sim.d) as v:
                while v.is_running() and sim.t < a.seconds:
                    t0 = time.time()
                    tick()
                    v.sync()
                    dt = TICK_STEPS * sim.m.opt.timestep - (time.time() - t0)
                    if dt > 0:
                        time.sleep(dt)
    finally:
        print(f"loop_hz={st['ticks'] / max(1e-6, time.time() - st['t0']):.1f}", flush=True)
        if ex:
            ex.close()


if __name__ == "__main__":
    main()
