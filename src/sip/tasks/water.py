"""Water task: deterministic state machine that owns all motion. Every transition is logged
with a plain-English reason so the explain view can show cause and effect."""
import mujoco
import numpy as np

from sip.blackboard import Blackboard
from sip.memory import ObjectMemory
from sip.nav.follow import PathFollower, wrap
from sip.nav.grid import OccupancyGrid, astar
from sip.nav.lidar import scan, update_map
from sip.perception.objects import aim_head, render_head
from sip.robot import StretchIO
from sip.safety import safety_gate
from sip.scene.build_house import ROOMS, SEATS, park_pose_for, room_at
from sip.types import Event, MouthState, SafetyDecision, SafetyInput

TICK_STEPS = 25              # physics steps per control tick (0.05 s)
PERCEIVE_EVERY = 4           # ticks (5 Hz)
LIDAR_EVERY = 10             # ticks (2 Hz)
INFLATE = 0.30
PICK_STANDOFF = 0.69         # base centre to object: grasp offset 0.44 + arm 0.25
LIFT_TO_GRASP_Z = 0.10       # grasp centre z = lift ctrl + 0.10
STRAW_SHORT = 0.015          # straw tip target this far in front of the lips (gate stops at 0.02)
HOLD_S = 3.0
SPIN_W = 0.6
WRIST_CAM = "d405_rgb"
PERSON_STANDOFF = 0.42       # park so the wrist swing ends outside the 0.30 m slow zone
WRIST_SWING = 0.6            # rad/s when swinging the glass out beside the person


class WaterTask:
    def __init__(self, io: StretchIO, bb: Blackboard, grid: OccupancyGrid, memory: ObjectMemory,
                 detector, face, label: str = "glass", prepare: bool = False):
        self.io, self.bb, self.grid, self.memory = io, bb, grid, memory
        self.det, self.face, self.label = detector, face, label
        self.m, self.d = io.m, io.d
        self.renderer = mujoco.Renderer(self.m, 240, 320)
        self.state = "IDLE"
        self.t = 0.0
        self.tick_n = 0
        self.home: np.ndarray | None = None   # where the glass lives (first sighting)
        self.obj_xyz: np.ndarray | None = None
        self.mouth = MouthState(False, None, 0.0, 0.0, False)
        self.decision: SafetyDecision | None = None
        self.reapproach = 0
        self._nav: dict | None = None
        self._sub = 0
        self._timer = 0.0
        self._visited: list[str] = []
        self._saved_state = None
        self._after_ask = None
        self._person_target = None
        self.last_frame = None
        self.last_dets = []
        self.last_mouth: tuple[float, np.ndarray] | None = None
        self.prepare = prepare
        self.seen: dict[str, np.ndarray] = {}
        self._gen = None
        self.water_level = 1.0
        if prepare:
            self._set_water(0.0)
            self._show_straw(False)

    # ---------- helpers ----------
    def log(self, kind: str, text: str) -> None:
        self.bb.log(Event(self.t, kind, text))

    def say(self, text: str) -> None:
        self.bb.set("say", text)
        self.log("say", text)

    def goto(self, state: str, reason: str) -> None:
        if state != self.state:
            self.log("state", f"{self.state} -> {state}: {reason}")
        self.state, self._sub, self._timer = state, 0, self.t

    def pose(self):
        return self.io.base_pose()

    def tip(self) -> np.ndarray:
        return self.d.site("straw_tip").xpos.copy()

    def dist_to_mouth(self) -> float:
        return float(np.linalg.norm(self.tip() - self.d.site("mouth").xpos))

    def _inflated(self) -> OccupancyGrid:
        return self.grid.inflate(INFLATE)

    def _plan(self, goal_xy) -> list | None:
        p = self.pose()
        base = self._inflated()
        for r_start in (0.1, 0.25):  # clear as little of the inflation around us as possible
            g = base.copy()
            for cx, cy, r in ((p.x, p.y, r_start), (goal_xy[0], goal_xy[1], 0.15)):
                for x in np.arange(cx - r, cx + r + 1e-9, g.res):
                    for y in np.arange(cy - r, cy + r + 1e-9, g.res):
                        if np.hypot(x - cx, y - cy) <= r:
                            g.mark(x, y, False)
            path = astar(g, (p.x, p.y), tuple(goal_xy))
            if path is not None:
                return path
        return None

    def _start_nav(self, goal_xy, yaw: float | None, why: str) -> bool:
        path = self._plan(goal_xy)
        if path is None:
            self.log("plan", f"no route to {room_at(*goal_xy)} ({why})")
            return False
        self._nav = {"goal": np.asarray(goal_xy, float), "yaw": yaw,
                     "follower": PathFollower(path, lookahead=0.2), "why": why, "length": _length(path),
                     "progress": (self.t, self.pose()), "backup_until": -1.0}
        self.io.stow()
        self.log("plan", f"route to {room_at(*goal_xy)} for {why}: {_length(path):.1f} m")
        return True

    def _nav_step(self) -> str:
        """Returns 'moving', 'arrived' or 'blocked'."""
        nav = self._nav
        p = self.pose()
        if self.tick_n % LIDAR_EVERY == 0:
            new = update_map(self.grid, p, scan(self.m, self.d, p), 4.0)
            if new and not nav["follower"].done and self._path_blocked(nav["follower"]):
                path = self._plan(nav["goal"])
                if path is None:
                    self.io.drive(0, 0)
                    return "blocked"
                extra = _length(path) - nav["length"]
                nav["follower"], nav["length"] = PathFollower(path, lookahead=0.2), _length(path)
                self.log("plan", f"saw obstacle ahead -> map updated -> new route ({extra:+.1f} m)")
        f = nav["follower"]
        if self.t < nav["backup_until"]:
            self.io.drive(-0.12, 0.0)
            return "moving"
        if not f.done:
            t0, p0 = nav["progress"]
            if np.hypot(p.x - p0.x, p.y - p0.y) > 0.05 or abs(wrap(p.yaw - p0.yaw)) > 0.1:
                nav["progress"] = (self.t, p)
            elif self.t - t0 > 2.0:  # stuck: back off, then re-plan from here
                self.log("plan", "stuck, backing up and re-planning")
                nav["backup_until"] = self.t + 1.0
                nav["progress"] = (self.t + 1.0, p)
                path = self._plan(nav["goal"])
                if path is None:
                    return "blocked"
                nav["follower"] = PathFollower(path, lookahead=0.2)
                return "moving"
            self.io.drive(*f.step(p))
            return "moving"
        if nav["yaw"] is not None:
            err = wrap(nav["yaw"] - p.yaw)
            if abs(err) > 0.03:
                self.io.drive(0.0, float(np.clip(2.0 * err, -0.6, 0.6)))
                return "moving"
        self.io.drive(0, 0)
        return "arrived"

    def _path_blocked(self, f: PathFollower) -> bool:
        g = self._inflated()
        p = self.pose()
        for a, b in zip([np.array([p.x, p.y])] + f.path[f._i:], f.path[f._i:]):
            for s in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / 0.05))):
                q = a + s * (b - a)
                if np.hypot(q[0] - p.x, q[1] - p.y) > 0.35 and not g.is_free(*q):
                    return True
        return False

    def _pick_pose(self, target: np.ndarray) -> tuple[np.ndarray, float] | None:
        g = self._inflated()
        p = self.pose()
        best = None
        for a in np.deg2rad(np.arange(0, 360, 15)):
            b = target[:2] + PICK_STANDOFF * np.array([np.cos(a), np.sin(a)])
            if not g.is_free(*b):
                continue
            u = target[:2] - b
            cost = np.hypot(b[0] - p.x, b[1] - p.y)
            if best is None or cost < best[0]:
                best = (cost, b, float(np.arctan2(u[0], -u[1])))
        return (best[1], best[2]) if best else None

    def _perceive(self) -> None:
        want_rgb = self.tick_n % (4 * PERCEIVE_EVERY) == 0 or self.bb.get("want_frame", False)
        rgb, depth, seg = render_head(self.m, self.d, self.renderer, rgb=want_rgb)
        if rgb is not None:
            self.last_frame = rgb
        self.last_dets = self.det.detect(self.m, self.d, rgb, depth, seg)
        for det in self.last_dets:
            self.seen[det.label] = det.xyz
            if det.label == self.label:
                self.obj_xyz = det.xyz
                self.memory.see(self.label, det.xyz, self.t, room_at(*det.xyz[:2]))
                if self.home is None:
                    self.home = det.xyz.copy()
        hand = bool(self.bb.get("hand_raised", False))
        mopen = self.bb.get("mouth_open", True) is not False
        self.mouth = self.face.update(rgb, depth, seg, self.m, self.d, hand_raised=hand, mouth_open=mopen)
        if not self.mouth.visible and self.state in ("APPROACH", "HOLD"):
            # the arm and glass block the head camera near the face: use the wrist camera
            rgb2, depth2, seg2 = render_head(self.m, self.d, self.renderer, cam=WRIST_CAM, rgb=False)
            self.mouth = self.face.update(rgb2, depth2, seg2, self.m, self.d, hand_raised=hand,
                                          cam=WRIST_CAM, mouth_open=mopen)
        if self.mouth.visible:
            self.last_mouth = (self.t, self.mouth.xyz.copy())

    def _seen(self) -> bool:
        return any(d.label == self.label for d in self.last_dets)

    def _contact_force(self) -> float:
        if not hasattr(self, "_glass_g"):
            from sip.perception.objects import _body_geoms
            self._glass_g = set(_body_geoms(self.m, self.label).tolist())
            self._person_g = set(_body_geoms(self.m, "person").tolist())
        total, f6 = 0.0, np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            pair = {c.geom1, c.geom2}
            if pair & self._glass_g and pair & self._person_g:
                mujoco.mj_contactForce(self.m, self.d, i, f6)
                total += abs(f6[0])
        return total

    def _person_park(self):
        pid = self.m.body("person").id
        mpos = self.d.xpos[pid]
        q = self.d.xquat[pid]
        facing = float(2 * np.arctan2(q[3], q[0]))
        where = min(SEATS, key=lambda k: np.linalg.norm(np.array(SEATS[k][0][:2]) - mpos[:2]))
        mouth = self.mouth.xyz if (self.mouth.visible and self.mouth.xyz is not None) else \
            self.d.site("mouth").xpos.copy()
        front = mouth + 0.05 * np.array([np.cos(facing), np.sin(facing), 0])
        x, y, yaw = park_pose_for(front, PERSON_STANDOFF, facing, SEATS[where][2])
        return np.array([x, y]), yaw, facing, where

    # ---------- main step ----------
    def step(self, t: float) -> str:
        self.t = t
        self.tick_n += 1
        every = PERCEIVE_EVERY if self.state in ("APPROACH", "HOLD", "GRASP", "VERIFY", "SEARCH") else 2 * PERCEIVE_EVERY
        if self.tick_n % every == 0:
            self._perceive()
        if self.bb.get("stop") and self.state != "HALT":
            self.io.freeze()
            self._saved_state = self.state
            self.say("Stopping.")
            self.goto("HALT", "stop heard")
            return self.state
        getattr(self, f"_s_{self.state.lower()}")()
        return self.state

    # ---------- states ----------
    def _s_idle(self):
        if self.bb.get("request") == "water":
            self.bb.set("request", None)
            self._visited = []
            self.reapproach = 0
            self.say("Of course, I'll get you some water.")
            self.goto("LOCATE", "heard: water")

    def _s_locate(self):
        where = self.memory.where(self.label)
        if where is not None and self.memory.confidence(self.label, self.t) >= 0.2:
            self.obj_xyz = where[0]
            self.log("plan", f"remember the {self.label} in the {where[1]} "
                             f"(confidence {self.memory.confidence(self.label, self.t):.2f})")
            return self._go_pick("remembered")
        self.goto("SEARCH", f"no confident memory of the {self.label}")

    def _go_pick(self, why: str):
        pick = self._pick_pose(self.obj_xyz)
        if pick is None or not self._start_nav(pick[0], pick[1], f"picking up the {self.label}"):
            return self._ask(f"I can't find a way to the {self.label}. Could you move something?",
                             resume="LOCATE")
        self.goto("NAV_TO_OBJECT", why)

    def _s_search(self):
        if self._sub == 0:
            rooms = [r for r in self.memory.search_order(self.label, self.t, list(ROOMS))
                     if r not in self._visited]
            if not rooms:
                return self._ask(f"I've looked everywhere but can't find your {self.label}. "
                                 "Where did you leave it?", resume="LOCATE")
            room = rooms[0]
            self._visited.append(room)
            self._room = room
            if not self._start_nav(ROOMS[room], None, f"searching the {room}"):
                return self._ask(f"I can't find a way to the {room}. Could you move something?",
                                 resume="LOCATE")
            self._sub = 1
        elif self._sub == 1:
            r = self._nav_step()
            if r == "blocked":
                return self._ask("Something is blocking the way. Could you move it?", resume="SEARCH")
            if r == "arrived":
                self._sub, self._spin0 = 2, self.pose().yaw
                self._spun = 0.0
        elif self._sub == 2:  # spin in place, looking
            p = self.pose()
            self._spun += abs(wrap(p.yaw - self._spin0))
            self._spin0 = p.yaw
            look = np.array([p.x + 1.5 * np.cos(p.yaw), p.y + 1.5 * np.sin(p.yaw), 0.6])
            aim_head(self.m, self.d, self.io, look, snap=False)
            if self._seen():
                self.io.drive(0, 0)
                self.log("perceive", f"found the {self.label} in the {self._room}")
                return self._go_pick("found while searching")
            if self._spun >= 2 * np.pi:
                self.io.drive(0, 0)
                self.memory.miss(self.label, self._room, self.t)
                self.log("perceive", f"no {self.label} in the {self._room}")
                self._sub = 0
            else:
                self.io.drive(0, SPIN_W)

    def _s_nav_to_object(self):
        if self.obj_xyz is not None:
            aim_head(self.m, self.d, self.io, self.obj_xyz, snap=False)
        r = self._nav_step()
        if r == "blocked":
            return self._ask("Something is blocking the way. Could you move it?", resume="LOCATE")
        if r == "arrived":
            self.goto("VERIFY", "arrived at pick pose")

    def _s_verify(self):
        expected = self.obj_xyz.copy()
        aim_head(self.m, self.d, self.io, expected, snap=False)
        if self.t - self._timer < 1.0:
            return
        if self._seen() and np.linalg.norm(self.obj_xyz - expected) < 0.15:
            if self.prepare and (self.water_level < 0.5 or not self._straw_in):
                self._gen = self._prepare_seq()
                return self.goto("PREPARE", "glass is empty: pour water and add a straw")
            return self.goto("GRASP", f"{self.label} confirmed by camera")
        if self._seen():
            return self._go_pick("it moved a little")
        self.memory.miss(self.label, room_at(*expected[:2]), self.t)
        self.log("perceive", f"the {self.label} is not where I remembered")
        self._visited = []
        self.goto("SEARCH", f"{self.label} missing")

    def _base_bearing_fix(self, target: np.ndarray, gain=2.0, wmax=0.3) -> None:
        p = self.pose()
        b = np.array([p.x, p.y])
        g, mth = self.io.grasp_center()[:2] - b, target[:2] - b
        ang = np.arctan2(g[0] * mth[1] - g[1] * mth[0], g @ mth)
        self.io.drive(0.0, float(np.clip(gain * ang, -wmax, wmax)))

    def _s_grasp(self):
        obj = self.obj_xyz
        dt = TICK_STEPS * self.m.opt.timestep
        if self._sub == 0:
            self.io.set_arm(lift=obj[2] - LIFT_TO_GRASP_Z, arm=0.0, wrist_yaw=0.0, gripper=0.04)
            if abs(self.io.grasp_center()[2] - obj[2]) < 0.03 and self.t - self._timer > 1.0:
                self._sub, self._timer = 1, self.t
        elif self._sub == 1:
            aim_head(self.m, self.d, self.io, obj, snap=False)
            err = obj - self.io.grasp_center()
            arm = self.io.arm_targets()["arm"] + float(np.clip(err @ self.io.arm_dir(), -0.15 * dt, 0.15 * dt))
            self.io.set_arm(arm=arm, lift=self.io.arm_targets()["lift"] + 0.5 * err[2])
            self._base_bearing_fix(obj)
            if np.linalg.norm(err) < 0.03:
                self.io.drive(0, 0)
                self._turn_straw_to_arm()
                self.io.attach(self.label)
                self.log("plan", f"grasped the {self.label}")
                self.io.set_arm(lift=self.io.arm_targets()["lift"] + 0.08)
                self._sub, self._timer = 2, self.t
            elif self.t - self._timer > 12.0:
                self.io.drive(0, 0)
                self._go_pick("grasp timed out, re-parking")
        elif self._sub == 2:
            if self.t - self._timer > 1.0:
                self.io.set_arm(arm=0.0)
            if self.t - self._timer > 2.5 and self.io.arm_extension() < 0.03:
                self.io.stow(lift=max(self.io.arm_targets()["lift"], 0.8))
                self._sub, self._timer = 3, self.t
        elif self._sub == 3 and self.t - self._timer > 2.0:
            self._go_person("bringing the water")

    # ---------- drink preparation (pour from the jug, add a straw) ----------
    _straw_in = True

    def _set_water(self, level: float) -> None:
        """Visual water level in the glass, 0 (empty) .. 1 (full). MuJoCo has no fluids."""
        g = self.m.geom("water").id
        h = max(0.001, 0.04 * level)
        self.m.geom_size[g][1] = h
        self.m.geom_pos[g][2] = -0.055 + h
        self.m.geom_rgba[g][3] = 0.55 if level > 0.02 else 0.0
        self.water_level = level

    def _show_straw(self, on: bool) -> None:
        self.m.geom_rgba[self.m.geom("straw").id][3] = 1.0 if on else 0.0
        self._straw_in = on

    def _s_prepare(self):
        try:
            next(self._gen)
        except StopIteration:
            self.goto("GRASP", "drink ready")

    def _g_wait(self, seconds: float):
        t0 = self.t
        while self.t - t0 < seconds:
            yield

    def _g_nav_to_pick(self, xyz: np.ndarray, why: str):
        pick = self._pick_pose(xyz)
        if pick is None or not self._start_nav(pick[0], pick[1], why):
            self._ask("I can't reach the counter. Could you help?", resume="LOCATE")
            return
        while True:
            aim_head(self.m, self.d, self.io, xyz, snap=False)
            r = self._nav_step()
            if r == "arrived":
                return
            if r == "blocked":
                self._ask("Something is in the way. Could you move it?", resume="LOCATE")
                return
            yield

    def _g_reach(self, xyz: np.ndarray, tol: float = 0.03, timeout: float = 12.0, wrist_yaw: float = 0.0):
        dt = TICK_STEPS * self.m.opt.timestep
        self.io.set_arm(lift=xyz[2] - LIFT_TO_GRASP_Z, arm=0.0, wrist_yaw=wrist_yaw)
        t0 = self.t
        while abs(self.io.grasp_center()[2] - xyz[2]) > 0.03 and self.t - t0 < 3.0:
            yield
        t0 = self.t
        while self.t - t0 < timeout:
            err = xyz - self.io.grasp_center()
            if np.linalg.norm(err) < tol:
                self.io.drive(0, 0)
                return
            arm = self.io.arm_targets()["arm"] + float(np.clip(err @ self.io.arm_dir(), -0.15 * dt, 0.15 * dt))
            self.io.set_arm(arm=arm, lift=self.io.arm_targets()["lift"] + 0.5 * err[2])
            self._base_bearing_fix(xyz)
            yield
        self.io.drive(0, 0)

    def _g_retract(self, lift_up: float = 0.06):
        self.io.set_arm(lift=self.io.arm_targets()["lift"] + lift_up)
        yield from self._g_wait(0.8)
        self.io.set_arm(arm=0.0)
        t0 = self.t
        while self.io.arm_extension() > 0.03 and self.t - t0 < 4.0:
            yield

    def _see(self, label: str, fallback: np.ndarray) -> np.ndarray:
        return np.array(self.seen.get(label, fallback), float).copy()

    def _prepare_seq(self):
        glass = self.obj_xyz.copy()
        jug_home = self._see("jug", self.d.body("jug").xpos)
        # 1. jug
        self.say("Your glass is empty. Let me pour you some water.")
        yield from self._g_nav_to_pick(jug_home, "fetching the jug")
        aim_head(self.m, self.d, self.io, jug_home, snap=False)
        yield from self._g_wait(0.6)
        jug = self._see("jug", jug_home)
        yield from self._g_reach(jug)
        self.io.attach("jug")
        self.log("plan", "picked up the jug")
        yield from self._g_retract(lift_up=0.2)
        # 2. pour: hold the jug just above the glass and tilt it
        yield from self._g_nav_to_pick(glass, "pouring into the glass")
        above = glass + np.array([0, 0, 0.21])
        yield from self._g_reach(above, tol=0.04)
        self.log("plan", "pouring water")
        for k in range(40):
            self.io.d.ctrl[self.io.act["wrist_roll"]] = 0.9 * min(1.0, k / 10)
            if k >= 10:
                self._set_water(min(1.0, (k - 10) / 25))
            yield from self._g_wait(0.1)
        self.io.d.ctrl[self.io.act["wrist_roll"]] = 0.0
        self.log("plan", "poured water into the glass")
        yield from self._g_wait(0.8)
        yield from self._g_retract(lift_up=0.0)
        # 3. put the jug back
        yield from self._g_nav_to_pick(jug_home, "putting the jug back")
        yield from self._g_reach(jug_home + np.array([0, 0, 0.01]))
        self.io.release()
        yield from self._g_retract(lift_up=0.04)
        # 4. straw from the holder into the glass
        straw_home = self._see("straw2", self.d.body("straw2").xpos)
        yield from self._g_nav_to_pick(straw_home, "getting a straw")
        aim_head(self.m, self.d, self.io, straw_home, snap=False)
        yield from self._g_wait(0.6)
        straw = self._see("straw2", straw_home)
        yield from self._g_reach(straw + np.array([0, 0, 0.05]))
        self.io.attach("straw2")
        self.log("plan", "took a straw")
        yield from self._g_retract(lift_up=0.15)
        yield from self._g_nav_to_pick(glass, "putting the straw in")
        yield from self._g_reach(glass + np.array([0, 0, 0.16]), tol=0.04)
        self.io.release()
        a = self.m.joint("straw2_free").qposadr[0]
        self.d.qpos[a:a + 3] = [-3.15, 2.62, -1.0]  # Limits: the straw is handed over to the glass model
        self._show_straw(True)
        self.log("plan", "put a straw in the glass")
        yield from self._g_retract(lift_up=0.0)
        self.say("Your water is ready.")
        # 5. back to a pick pose for the glass, then the normal grasp
        yield from self._g_nav_to_pick(glass, "picking up the glass")

    def _turn_straw_to_arm(self) -> None:
        """Turn the glass about its axis so the straw leans along the arm (Limits: done by
        setting the glass pose, standing in for a regrasp)."""
        ad = self.io.arm_dir()
        yaw = float(np.arctan2(ad[1], ad[0]))
        j = self.m.joint(f"{self.label}_free")
        a = j.qposadr[0]
        self.d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.m, self.d)

    def _go_person(self, why: str):
        goal, yaw, _, where = self._person_park()
        self._person_target = goal
        if not self._start_nav(goal, yaw, why):
            return self._ask("I can't reach you. Could you move something out of the way?",
                             resume="NAV_TO_PERSON")
        self.goto("NAV_TO_PERSON", f"{why} (person at the {where})")

    def _s_nav_to_person(self):
        if self.tick_n % 40 == 0:
            goal, yaw, _, where = self._person_park()
            if np.linalg.norm(goal - self._person_target) > 0.3:
                self.log("plan", f"person moved to the {where} -> new target")
                self._person_target = goal
                if not self._start_nav(goal, yaw, "person moved"):
                    return self._ask("I can't reach you there. Could you help?", resume="NAV_TO_PERSON")
        aim_head(self.m, self.d, self.io, self.d.site("mouth").xpos, snap=False)
        r = self._nav_step()
        if r == "blocked":
            return self._ask("Something is in the way. Could you move it?", resume="NAV_TO_PERSON")
        if r == "arrived":
            self.say("Here's your water. Tell me when you're ready.")
            self.goto("APPROACH", "arrived beside the person")

    def _approach_target(self) -> np.ndarray:
        mouth = self.mouth.xyz if (self.mouth.visible and self.mouth.xyz is not None) else None
        if mouth is None and self.last_mouth is not None and self.dist_to_mouth() > 0.5:
            mouth = self.last_mouth[1]  # far away: last sighting is good enough; gate needs live view < 0.5 m
        if mouth is None:
            return None
        pid = self.m.body("person").id
        q = self.d.xquat[pid]
        yaw = float(2 * np.arctan2(q[3], q[0])) + np.deg2rad(self.mouth.yaw_deg)
        return mouth + STRAW_SHORT * np.array([np.cos(yaw), np.sin(yaw), 0])

    def _gate(self) -> SafetyDecision:
        d = SafetyInput(self.dist_to_mouth(), self.mouth, bool(self.bb.get("ready", False)),
                        bool(self.bb.get("stop", False)), self._contact_force(), self.reapproach,
                        done_heard=bool(self.bb.get("done", False)))
        dec = safety_gate(d)
        if self.decision is None or dec.action != self.decision.action or dec.reason != self.decision.reason:
            self.log("safety", f"{dec.action}: {dec.reason} (d={d.dist_to_mouth_m:.2f} m)")
        self.decision = dec
        return dec

    def _s_approach(self):
        aim_head(self.m, self.d, self.io, self.d.site("mouth").xpos, snap=False)
        dt = TICK_STEPS * self.m.opt.timestep
        if self._sub == 0:  # swing the glass out slowly (away from the face), arm retracted
            near = self.dist_to_mouth() < 0.40
            wy = min(self.io.arm_targets()["wrist_yaw"], float(self.d.joint("joint_wrist_yaw").qpos[0]) + 0.05)
            # limit the straw tip's linear speed, not the joint rate: omega = v_tip / lever
            axis = self.d.body("link_wrist_yaw").xpos
            lever = max(0.05, float(np.hypot(*(self.tip() - axis)[:2])))
            v_tip = 0.025 if near else 0.15
            lag = abs(self.d.joint("joint_wrist_yaw").qpos[0] - wy)
            if lag < 0.15:  # let the joint catch up before commanding further
                self.io.set_arm(wrist_yaw=max(0.0, wy - min(WRIST_SWING, v_tip / lever) * dt))
            tgt = self._approach_target()
            if tgt is not None:
                step = (0.02 if near else 0.15) * dt
                dz = float(np.clip(tgt[2] - self.tip()[2], -step, step))
                self.io.set_arm(lift=self.io.arm_targets()["lift"] + dz)
            if wy <= 1e-3 and abs(self.d.joint("joint_wrist_yaw").qpos[0]) < 0.03 and self.t - self._timer > 2.0:
                self._sub = 1
            return
        dec = self._gate()
        if self._sub == 2:  # backing off after a retreat; wait until clear for 1 s
            if dec.action == "proceed" or (dec.action == "pause" and dec.reason == "waiting for ready"):
                if self.t - self._timer > 1.0:
                    self.log("plan", f"clear again, re-approaching (attempt {self.reapproach + 1})")
                    self._sub = 1
            else:
                self._timer = self.t
            return
        if dec.action == "retreat" and dec.reason == "person finished":
            self.bb.set("done", False)
            self.say("Okay, I'll put it away.")
            return self.goto("RETREAT", "person said they had enough")
        if dec.action == "retreat":
            self.reapproach += 1
            self.io.set_arm(arm=max(0.0, self.io.arm_extension() - 0.10))
            self.io.drive(0, 0)
            self.say("Okay, I'll back off.")
            self._sub, self._timer = 2, self.t
            return
        if dec.action == "ask":
            self.io.set_arm(arm=max(0.0, self.io.arm_extension() - 0.10))
            return self._ask("Shall I try again? Say go when you're ready.", resume="APPROACH")
        if dec.action == "pause":
            self.io.drive(0, 0)
            if dec.reason == "at mouth":
                self.say("Take your time.")
                self.goto("HOLD", "straw at the lips")
            elif dec.reason == "waiting for ready" and not self.bb.get("asked_ready"):
                self.bb.set("asked_ready", True)
                self.say("Ready for a sip? Say ready.")
            return
        tgt = self._approach_target()
        if tgt is None:
            return
        err = tgt - self.tip()
        step = dec.max_speed_mps * dt
        # step from where the joints actually are, so a lagging joint can't overshoot into the slow zone
        ext = self.io.arm_extension()
        cmd = self.io.arm_targets()["arm"] + float(np.clip(err @ self.io.arm_dir(), -step, step))
        lead = 0.02 if dec.max_speed_mps < 0.1 else 0.06  # bound how far the command may run ahead
        self.io.set_arm(arm=float(np.clip(cmd, ext - lead, ext + lead)),
                        lift=self.io.arm_targets()["lift"] + float(np.clip(err[2], -step, step)))
        p = self.pose()
        lever = max(0.3, float(np.hypot(self.tip()[0] - p.x, self.tip()[1] - p.y)))
        self._base_bearing_fix(tgt, gain=1.5, wmax=min(0.15, 0.8 * dec.max_speed_mps / lever))

    def _s_hold(self):
        dec = self._gate()
        if dec.action == "retreat" and dec.reason == "person finished":
            self.bb.set("done", False)
            self.log("plan", f"drink offered for {self.t - self._timer:.0f} s (stopped by person)")
            self.say("Okay, I'll put it away.")
            return self.goto("RETREAT", "person said they had enough")
        if dec.action in ("retreat", "ask"):
            self.goto("APPROACH", dec.reason)
            self._sub = 1
            return
        if self.t - self._timer >= HOLD_S + 0.3:
            self.log("plan", f"drink offered for {HOLD_S:.0f} s")
            self.goto("RETREAT", "drink finished")

    def _s_retreat(self):
        self.io.set_arm(arm=0.0)
        if self.t - self._timer > 2.0 and self.io.arm_extension() < 0.03:
            self.io.stow(lift=max(self.io.arm_targets()["lift"], 0.8))
            self.bb.set("asked_ready", False)
            target = self.home if self.home is not None else self.obj_xyz
            self._place_xyz = target
            pick = self._pick_pose(target)
            if pick is None or not self._start_nav(pick[0], pick[1], "putting the glass back"):
                return self._ask("I can't get back to the counter. Could you help?", resume="RETREAT")
            self.goto("RETURN", "putting the glass back")

    def _s_return(self):
        dt = TICK_STEPS * self.m.opt.timestep
        tgt = self._place_xyz + np.array([0, 0, 0.01])
        if self._sub == 0:
            if self._nav_step() == "arrived":
                self._sub, self._timer = 1, self.t
                self.io.set_arm(lift=tgt[2] - LIFT_TO_GRASP_Z + 0.02, wrist_yaw=0.0)
        elif self._sub == 1 and self.t - self._timer > 2.5:
            err = tgt - self.io.grasp_center()
            self.io.set_arm(arm=self.io.arm_targets()["arm"] + float(np.clip(err @ self.io.arm_dir(), -0.15 * dt, 0.15 * dt)))
            self._base_bearing_fix(tgt)
            if np.linalg.norm(err[:2]) < 0.04 or self.t - self._timer > 12.0:
                self.io.drive(0, 0)
                self.io.release()
                self.log("plan", f"put the {self.label} back")
                self._sub, self._timer = 2, self.t
        elif self._sub == 2:
            self.io.set_arm(arm=0.0)
            if self.t - self._timer > 2.0:
                self.io.stow()
                self.memory.see(self.label, self._place_xyz, self.t, room_at(*self._place_xyz[:2]))
                self.say("All done.")
                self.goto("IDLE", "task complete")

    def _ask(self, question: str, resume: str):
        self.io.drive(0, 0)
        self.say(question)
        self._ask_q = question
        self._alerted = False
        self._after_ask = resume
        self.goto("ASK", question)

    def alert(self, text: str) -> None:
        self.log("alert", text)
        self.bb.set("alert", (self.t, text))

    def _s_ask(self):
        if self.t - self._timer > 30.0 and not getattr(self, "_alerted", False):
            self._alerted = True
            self.alert(f"no answer for 30 s: {self._ask_q} - carer notified")
        if self.bb.get("go"):
            self._alerted = False
            self.bb.set("go", False)
            self.reapproach = 0
            self._visited = []
            resume = self._after_ask or "LOCATE"
            self.goto(resume, "go heard")
            if resume == "APPROACH":
                self._sub = 1

    def _s_halt(self):
        self.io.drive(0, 0)
        if self.bb.get("go"):
            self.bb.set("stop", False)
            self.bb.set("go", False)
            resume = self._saved_state or "IDLE"
            self.say("Carrying on.")
            if resume in ("NAV_TO_OBJECT", "SEARCH", "LOCATE"):
                return self.goto("LOCATE", "go heard, re-planning")
            if resume == "NAV_TO_PERSON":
                return self._go_person("go heard, re-planning")
            self.goto(resume, "go heard")


def _length(path) -> float:
    return float(sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])))
