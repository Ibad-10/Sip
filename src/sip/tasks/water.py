"""Water task: deterministic state machine that owns all motion. Every transition is logged
with a plain-English reason so the explain view can show cause and effect."""
import os
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np

from sip.blackboard import Blackboard
from sip.memory import ObjectMemory
from sip.nav.follow import PathFollower, wrap
from sip.nav.grid import OccupancyGrid, astar
from sip.nav.lidar import scan, update_map
from sip.cloud.verify import verify_target
from sip.perception.objects import CAM, aim_head, pixel_to_world, render_head
from sip.robot import StretchIO
from sip.safety import safety_gate
from sip.scene.build_house import ROOMS, park_pose_for
from sip.types import Event, MouthState, SafetyDecision, SafetyInput

TICK_STEPS = 25              # physics steps per control tick (0.05 s)
PERCEIVE_EVERY = 4           # ticks (5 Hz)
LIDAR_EVERY = 10             # ticks (2 Hz)
INFLATE = 0.30
PICK_STANDOFF = 0.69         # base centre to object: grasp offset 0.44 + arm 0.25
LIFT_TO_GRASP_Z = 0.10       # grasp centre z = lift ctrl + 0.10
STRAW_SHORT = 0.015          # straw tip target this far in front of the lips (gate stops at 0.02)
HOLD_S = 3.0
SPIN_W = 0.4          # slow enough for the camera to sweep near and far at every heading
WRIST_CAM = "d405_rgb"
PERSON_STANDOFF = 0.42       # park so the wrist swing ends outside the 0.30 m slow zone
WRIST_SWING = 0.6            # rad/s when swinging the glass out beside the person


class WaterTask:
    def __init__(self, io: StretchIO, bb: Blackboard, grid: OccupancyGrid, memory: ObjectMemory,
                 detector, face, label: str = "glass", prepare: bool = False, cloud=None, rooms=None):
        self.io, self.bb, self.grid, self.memory = io, bb, grid, memory
        if io.loc is None:
            io.enable_localisation()      # position from wheel odometry + lidar scan matching
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
        self.rooms = dict(rooms or ROOMS)    # named places: learned while exploring, or the setup default
        # where the person is comes from the camera; the map only knows her usual seats
        from sip.perception.objects import SegDetector
        self._person_det = SegDetector(["person"], noise_m=0.02, seed=7)
        self.person_seen: tuple[float, np.ndarray] | None = None
        self.places: dict[str, dict] = {}       # where she has been seen: name -> {xyz, facing}
        self._person_where: str | None = None
        self._seats_tried: list[str] = []
        self.real = bool(getattr(detector, "needs_rgb", False))
        if self.real:
            self.renderer = mujoco.Renderer(self.m, 480, 640)
        # optional cloud second opinion (Gemini Robotics-ER); None = offline, robot still works
        if cloud is None and os.environ.get("GEMINI_API_KEY"):
            from sip.cloud.gemini import second_opinion as cloud
        self.cloud = cloud
        self._cloud_job = None
        self._cloud_found = False
        self.seen: dict[str, np.ndarray] = {}
        self._gen = None
        self.water_level = 1.0
        if prepare:
            self._set_water(0.0)
            self._show_straw(False)

    # ---------- helpers ----------
    def _sweep_point(self, p) -> np.ndarray:
        """Search gaze: alternate near-and-low (side tables, 0.8 m) and far (counters, 1.8 m) every 0.6 s."""
        near = int(self.t / 0.6) % 2 == 0
        dist, z = (0.8, 0.55) if near else (1.8, 0.8)
        return np.array([p.x + dist * np.cos(p.yaw), p.y + dist * np.sin(p.yaw), z])

    def _object_in_hand(self, label: str | None = None) -> bool:
        """Wrist depth camera (D405, looks between the fingers): something solid fills the gap within 30 cm.
        This is how the robot knows a grasp worked; it never reads where the object really is."""
        if not hasattr(self, "_wrist_r"):
            from sip.perception.objects import fast_renderer
            self._wrist_r = fast_renderer(self.m, 120, 160)
        _, dep, _ = render_head(self.m, self.d, self._wrist_r, cam=WRIST_CAM, rgb=False, seg=False)
        gap = dep[50:120, 55:105]
        need = 0.08 if label == "straw2" else 0.35       # a thin straw fills far less of the gap than a glass
        return float((gap < 0.30).mean()) > need

    def _room_of(self, x: float, y: float) -> str:
        """Name of the nearest known room centre (learned while exploring, or set up by the carer)."""
        return min(self.rooms, key=lambda r: np.hypot(self.rooms[r][0] - x, self.rooms[r][1] - y))

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
        """Ground truth, for tests, the judge monitor and the video overlay only."""
        return float(np.linalg.norm(self.tip() - self.d.site("mouth").xpos))

    def sensed_dist(self) -> float:
        """What the robot believes: straw tip to the tracked (or last known) mouth."""
        return float(np.linalg.norm(self.tip() - self._mouth_guess()))

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
            self.log("plan", f"no route to {self._room_of(*goal_xy)} ({why})")
            return False
        self._nav = {"goal": np.asarray(goal_xy, float), "yaw": yaw,
                     "follower": PathFollower(path, lookahead=0.2), "why": why, "length": _length(path),
                     "progress": (self.t, self.pose()), "backup_until": -1.0}
        self.io.stow()
        self.log("plan", f"route to {self._room_of(*goal_xy)} for {why}: {_length(path):.1f} m")
        return True

    def _nav_step(self) -> str:
        """Returns 'moving', 'arrived' or 'blocked'."""
        nav = self._nav
        p = self.pose()
        if self.tick_n % LIDAR_EVERY == 0:
            ranges = scan(self.m, self.d, p)
            if self.io.loc is not None:
                self.io.loc.correct(ranges, self.grid)
                p = self.pose()
            new = update_map(self.grid, p, ranges, 4.0)
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
                nav["stuck"] = nav.get("stuck", 0) + 1
                if nav["stuck"] > 3:
                    self.io.drive(0, 0)
                    self.log("plan", "stuck 3 times on this route -> asking for help")
                    return "blocked"
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
        """Where to park so the arm can reach `target`. The arm reaches 0.44-0.96 m from the base centre, so try
        several distances and directions (corners and walls rule some out); keep the nearest reachable one."""
        g = self._inflated()
        p = self.pose()
        cands = []
        for stand in (PICK_STANDOFF, 0.62, 0.78, 0.56, 0.86):
            for a in np.deg2rad(np.arange(0, 360, 15)):
                b = target[:2] + stand * np.array([np.cos(a), np.sin(a)])
                if not g.is_free(*b):
                    continue
                u = target[:2] - b
                cost = np.hypot(b[0] - p.x, b[1] - p.y) + 2.0 * abs(stand - PICK_STANDOFF)
                cands.append((cost, b, float(np.arctan2(u[0], -u[1]))))
        for cost, b, yaw in sorted(cands, key=lambda c: c[0])[:6]:     # first one the planner can actually reach
            if self._plan(b) is not None:
                return b, yaw
        return None

    def _perceive(self) -> None:
        want_rgb = self.real or self.tick_n % (4 * PERCEIVE_EVERY) == 0 or self.bb.get("want_frame", False)
        rgb, depth, seg = render_head(self.m, self.d, self.renderer, rgb=want_rgb, seg=not self.real)
        if rgb is not None:
            self.last_frame = rgb
        if self.real:
            dets = self.det.detect(self.m, self.d, rgb, depth, seg, t=self.t)
        else:
            dets = self.det.detect(self.m, self.d, rgb, depth, seg)
            dets = dets + self._person_det.detect(self.m, self.d, rgb, depth, seg)
        dets = self._second_opinions(rgb, dets)
        self.last_dets = dets
        for det in dets:
            if det.label == "person":
                self._saw_person(det.xyz)
                continue
            self.seen[det.label] = det.xyz
            if det.label == self.label:
                self.obj_xyz = det.xyz
                self.memory.see(self.label, det.xyz, self.t, self._room_of(*det.xyz[:2]))
                if self.home is None:
                    self.home = det.xyz.copy()
        hand = False
        if self.real:
            box = self.det.boxes.get("person") if self.det.latest_t > self.t - 1.0 else None
            self.mouth = self.face.update(rgb, depth, self.m, self.d, CAM, person_box=box)
            if self.state in ("APPROACH", "HOLD") and box is not None and self.tick_n % 2 == 0:
                if not hasattr(self, "_h2m"):
                    from sip.perception.vision import HandToMouth
                    self._h2m = HandToMouth()
                self.mouth.hand_raised = bool(self._h2m.hand_raised(rgb, box))   # stop gesture, seen by camera
            if not self.mouth.visible and self.state in ("APPROACH", "HOLD"):
                rgb2, depth2, _ = render_head(self.m, self.d, self.renderer, cam=WRIST_CAM, rgb=True)
                self.mouth = self.face.update(rgb2, depth2, self.m, self.d, WRIST_CAM)
        else:
            self.mouth = self.face.update(rgb, depth, seg, self.m, self.d, hand_raised=hand)
            if not self.mouth.visible and self.state in ("APPROACH", "HOLD"):
                # the arm and glass block the head camera near the face: use the wrist camera
                rgb2, depth2, seg2 = render_head(self.m, self.d, self.renderer, cam=WRIST_CAM, rgb=False)
                self.mouth = self.face.update(rgb2, depth2, seg2, self.m, self.d, hand_raised=hand, cam=WRIST_CAM)
        if self.mouth.visible:
            self.last_mouth = (self.t, self.mouth.xyz.copy())
            f = self.face.resting_facing
            if f is not None and self._person_where in self.places:
                self.places[self._person_where]["facing"] = f

    _NAMES = {"glass": "drinking glass", "mug": "mug", "jug": "water jug", "straw2": "cup of straws",
              "organiser": "pill organiser"}

    def _second_opinions(self, rgb, dets):
        """Unsure detections: ask Gemini once per object (by place) what it is and why; remember the answer.
        Offline: keep the detector's best guess, marked unsure, and look again from closer."""
        if not hasattr(self, "_ids"):
            self._ids, self._id_jobs = {}, {}
        out = []
        for det in dets:
            key = None if det.xyz is None else tuple(np.round(det.xyz[:2] / 0.15).astype(int))
            if key in self._ids:
                label, why = self._ids[key]
                if label == "none":
                    continue
                det.why = f"{det.why}; Gemini: {label} ({why})"
                det.label, det.unsure = label, False
            elif det.unsure and det.label in self._NAMES and rgb is not None and self.cloud is not None:
                job = self._id_jobs.get(key)
                if job is None:
                    from sip.cloud.gemini import identify
                    u0, v0, u1, v1 = det.box
                    crop = rgb[max(0, v0 - 10):v1 + 10, max(0, u0 - 10):u1 + 10]
                    self._id_jobs[key] = self._POOL.submit(identify, np.ascontiguousarray(np.rot90(crop, 1)),
                                                           list(self._NAMES.values()))
                    self.log("cloud", f"unsure what this is ({det.why}) -> asking Gemini")
                elif job.done():
                    r = job.result()
                    inv = {v: k for k, v in self._NAMES.items()}
                    if r and r.get("confidence", 0) >= 0.7:       # Gemini overrides only when it is sure
                        lab = inv.get(r["label"], "none")
                        self._ids[key] = (lab, r["why"])
                        self.log("cloud", f"Gemini: it's {'not one of mine' if lab == 'none' else 'a ' + r['label']}"
                                          f" ({r['why']}); detector said {det.label}")
                    else:
                        self._ids[key] = (det.label, "second opinion unsure or offline: kept detector's guess")
            out.append(det)
        return out

    def _fall_check(self, xyz: np.ndarray) -> None:
        """Body centre seen near the floor for 3 s -> possible fall: emergency alert with a photo."""
        low = xyz[2] < 0.20            # body centre at floor level (lying on a bed reads ~0.3 m)
        if not low:
            self._low_since = None
            return
        self._low_since = getattr(self, "_low_since", None) or self.t
        if self.t - self._low_since > 3.0 and not getattr(self, "_fall_alerted", False):
            self._fall_alerted = True
            self.alert(f"POSSIBLE FALL: person seen on the floor in the {self._room_of(*xyz[:2])} for 3 s")
            self.say("Are you all right? I've raised an emergency alert for your carer.")

    def _saw_person(self, xyz: np.ndarray) -> None:
        """Remember where she is; nearby sightings refine the same place."""
        self._fall_check(xyz)
        self.person_seen = (self.t, xyz)
        for name, pl in self.places.items():
            if np.linalg.norm(pl["xyz"][:2] - xyz[:2]) < 1.0:
                pl["xyz"] = 0.8 * pl["xyz"] + 0.2 * xyz
                self._person_where = name
                return
        name = f"{self._room_of(*xyz[:2])} spot {len(self.places) + 1}"
        self.places[name] = {"xyz": xyz.copy(), "facing": None}
        self._person_where = name
        self.log("perceive", f"saw the person in the {self._room_of(*xyz[:2])} -> remembered this place")

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

    def _facing(self) -> float:
        pl = self.places.get(self._person_where) or {}
        if pl.get("facing") is not None:
            return pl["facing"]
        # not seen her face yet: assume she faces the open side of the room (away from the wall)
        xyz = pl.get("xyz", np.zeros(3))
        best, best_free = 0.0, -1
        for a in np.deg2rad(np.arange(0, 360, 30)):
            free = sum(self.grid.is_free(xyz[0] + r * np.cos(a), xyz[1] + r * np.sin(a)) for r in (0.5, 0.8, 1.1))
            if free > best_free:
                best, best_free = a, free
        return float(best)

    def _mouth_guess(self) -> np.ndarray:
        """Best mouth estimate: live face track, else last sighting, else above where she was seen."""
        if self.mouth.visible and self.mouth.xyz is not None:
            return self.mouth.xyz
        pl = self.places.get(self._person_where)
        if self.last_mouth is not None and (pl is None or np.linalg.norm(self.last_mouth[1][:2] - pl["xyz"][:2]) < 0.8):
            return self.last_mouth[1]
        if pl is None:
            p = self.pose()
            return np.array([p.x, p.y, 1.0])
        x, y = pl["xyz"][:2]
        return np.array([x, y, max(pl["xyz"][2] + 0.35, 0.95)])

    def _person_park(self):
        where = self._person_where
        facing = self._facing()
        front = self._mouth_guess() + 0.05 * np.array([np.cos(facing), np.sin(facing), 0])
        for a in (55, -55, 35, -35, 75, -75, 0):   # first side the planner can actually reach
            x, y, yaw = park_pose_for(front, PERSON_STANDOFF, facing, a)
            if self._plan((x, y)) is not None:
                break
        return np.array([x, y]), yaw, facing, where

    def _person_here(self) -> bool:
        if self.person_seen is None or self.t - self.person_seen[0] > 3.0:
            return False
        return np.linalg.norm(self.person_seen[1][:2] - self.places[self._person_where]["xyz"][:2]) < 1.0

    # ---------- main step ----------
    def step(self, t: float) -> str:
        self.t = t
        self.tick_n += 1
        if self.io.loc is not None:          # where am I: wheel odometry (scan matching in _nav_step)
            from sip.perception.objects import set_frame
            self.io.loc.update()
            set_frame(self.io.true_pose(), self.io.loc.pose)
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
        if self.bb.get("request") == "medicine":
            self.bb.set("request", None)
            self._gen = self._medicine_seq()
            return self.goto("MEDICINE", "dose time / asked for tablets")
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
            rooms = [r for r in self.memory.search_order(self.label, self.t, list(self.rooms))
                     if r not in self._visited]
            if not rooms:
                return self._ask(f"I've looked everywhere but can't find your {self.label}. "
                                 "Where did you leave it?", resume="LOCATE")
            room = rooms[0]
            self._visited.append(room)
            self._room = room
            if not self._start_nav(self.rooms[room], None, f"searching the {room}"):
                return self._ask(f"I can't find a way to the {room}. Could you move something?",
                                 resume="LOCATE")
            self._sub = 1
        elif self._sub == 1:
            r = self._nav_step()
            if r == "blocked":
                return self._ask("Something is blocking the way. Could you move it?", resume="SEARCH")
            if r == "arrived":
                self._sub, self._spin0 = 2, self.pose().yaw
                self._spun, self._next_ask = 0.0, 0.0
        elif self._sub == 2:  # spin in place, looking
            p = self.pose()
            self._spun += abs(wrap(p.yaw - self._spin0))
            self._spin0 = p.yaw
            look = self._sweep_point(p)
            aim_head(self.m, self.d, self.io, look, snap=False)
            if self._seen():
                self.io.drive(0, 0)
                why = next((d_.why for d_ in self.last_dets if d_.label == self.label and d_.why), "")
                self.log("perceive", f"found the {self.label} in the {self._room}" + (f" ({why})" if why else ""))
                return self._go_pick("found while searching")
            if self.cloud is not None and self._spun >= self._next_ask and self._spun < 2 * np.pi:
                self._next_ask += np.pi / 2
                return self._ask_cloud()
            if self._spun >= 2 * np.pi:
                self.io.drive(0, 0)
                self.memory.miss(self.label, self._room, self.t)
                self.log("perceive", f"no {self.label} in the {self._room}")
                self._sub = 0
            else:
                self.io.drive(0, SPIN_W)

        elif self._sub == 3:  # waiting for the cloud second opinion (robot stopped, still safe)
            job, rgb, depth, t0 = self._cloud_job
            if not job.done() and self.t - t0 < 10.0:
                return
            r = job.result() if job.done() else None
            self._cloud_job = None
            self._sub = 2
            if not r:
                return
            ok, checks = verify_target(self.m, self.d, r["u"], r["v"], depth, self.grid, self.io)
            if not ok:
                self.log("cloud", f"Gemini pointed at a {r['label']} but local checks failed: {', '.join(checks)}")
                return
            h, w = depth.shape
            self.obj_xyz = pixel_to_world(self.m, self.d, CAM, r["u"], r["v"], float(depth[r["v"], r["u"]]), w, h)
            self.obj_xyz[2] += 0.04  # point lands on the glass surface; aim for its centre
            self.memory.see(self.label, self.obj_xyz, self.t, self._room_of(*self.obj_xyz[:2]))
            if self.home is None:
                self.home = self.obj_xyz.copy()
            self.log("cloud", f"Gemini Robotics-ER pointed at the {self.label}; verified locally ({', '.join(checks)})")
            self._cloud_found = True
            self._go_pick("found with Gemini's help")

    _POOL = ThreadPoolExecutor(max_workers=1)

    def _depth_confirms(self, xyz: np.ndarray, tol: float = 0.08) -> bool:
        """Something solid sits where we expect the object (local check, no cloud)."""
        from sip.perception.objects import project
        _, depth, _ = render_head(self.m, self.d, self.renderer, rgb=False)
        h, w = depth.shape
        u, v, z = project(self.m, self.d, CAM, xyz, w, h)
        if z <= 0 or not (0 <= u < w and 0 <= v < h):
            return False
        return abs(float(depth[int(v), int(u)]) - z) < tol

    def _ask_cloud(self):
        self.io.drive(0, 0)
        rgb, depth, _ = render_head(self.m, self.d, self.renderer, rgb=True)
        transcript = str(self.bb.get("last_utterance", "") or "can I have some water")
        job = self._POOL.submit(self.cloud, rgb, transcript, f"searching the {self._room}", [self.label])
        self._cloud_job = (job, rgb, depth, self.t)
        self.log("cloud", f"local detector unsure: asking Gemini Robotics-ER to look ({self._room})")
        self._sub = 3

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
        if self._cloud_found and self._depth_confirms(expected):
            return self.goto("GRASP", f"depth camera confirms the {self.label} Gemini found")
        self._cloud_found = False
        self.memory.miss(self.label, self._room_of(*expected[:2]), self.t)
        self.log("perceive", f"the {self.label} is not where I remembered")
        self._visited = []
        self.goto("SEARCH", f"{self.label} missing")

    def _base_bearing_fix(self, target: np.ndarray, gain=2.0, wmax=0.3, point=None) -> None:
        """Turn the base so `point` (default: grasp centre) lines up with the target's bearing."""
        p = self.pose()
        b = np.array([p.x, p.y])
        g, mth = (self.io.grasp_center() if point is None else point)[:2] - b, target[:2] - b
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
            if np.linalg.norm(err) < 0.03 or (np.linalg.norm(err) < 0.08 and self._object_in_hand()):
                self.io.drive(0, 0)
                if not (self._object_in_hand() and self.io.attach(self.label)):
                    self.io.drive(0, 0)
                    self.log("perceive", f"wrist camera: nothing between the fingers - the {self.label} is not where I saw it")
                    self.io.set_arm(arm=0.0)
                    return self.goto("VERIFY", "grasp missed, looking again")
                self._turn_straw_to_arm()
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

    def _stream(self, on: bool, k: int = 0) -> None:
        """Water falling from the jug into the glass (visual; MuJoCo has no fluids)."""
        g, mid = self.m.geom("pour_stream_geom").id, self.m.body("pour_stream").mocapid[0]
        if not on:
            self.m.geom_rgba[g][3] = 0.0
            self.d.mocap_pos[mid] = [0, 0, -2]
            return
        top = self.d.body("jug").xpos[2] + 0.02
        surface = self.d.body("glass").xpos[2] - 0.055 + 0.04 * self.water_level
        gx, gy = self.d.body("glass").xpos[:2] + 0.004 * np.sin(k * 1.7)  # slight wobble
        half = max(0.01, (top - surface) / 2)
        self.m.geom_size[g][1] = half
        self.m.geom_rgba[g][3] = 0.55
        self.d.mocap_pos[mid] = [gx, gy, surface + half]

    def _show_straw(self, on: bool) -> None:
        self.m.geom_rgba[self.m.geom("straw").id][3] = 1.0 if on else 0.0
        self._straw_in = on

    def _s_prepare(self):
        try:
            next(self._gen)
        except StopIteration:
            if self.state == "PREPARE":     # the sequence may have asked for help instead
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

    def _policy(self, skill: str):
        """Trained arm policy for a skill, or None (missing, disabled with SIP_POLICY=0)."""
        if os.environ.get("SIP_POLICY", "1") != "1":
            return None
        if not hasattr(self, "_policies"):
            self._policies = {}
        if skill not in self._policies:
            try:
                from sip.learn.runtime import PolicyArm
                self._policies[skill] = PolicyArm(skill)
            except Exception:
                self._policies[skill] = None
        return self._policies[skill]

    def _g_reach(self, xyz: np.ndarray, tol: float = 0.03, timeout: float = 12.0, wrist_yaw: float = 0.0,
                 touch: str | None = None, skill: str | None = None):
        dt = TICK_STEPS * self.m.opt.timestep
        if abs(self.io.arm_targets()["wrist_yaw"] - wrist_yaw) > 0.3:
            # swing the wrist out above everything on the table, then come down to the object
            self.io.set_arm(lift=max(self.io.arm_targets()["lift"], xyz[2] - LIFT_TO_GRASP_Z + 0.18), arm=0.0)
            yield from self._g_wait(1.2)
            self.io.set_arm(wrist_yaw=wrist_yaw)
            yield from self._g_wait(1.2)
        self.io.set_arm(lift=xyz[2] - LIFT_TO_GRASP_Z, arm=0.0, wrist_yaw=wrist_yaw,
                        gripper=None if self.io.held else 0.04)
        t0 = self.t
        while abs(self.io.grasp_center()[2] - xyz[2]) > 0.03 and self.t - t0 < 3.0:
            yield
        t0 = self.t
        pol = self._policy(skill or ("grasp" if touch else "pour"))
        best, best_t = 9.0, self.t
        self.arm_driver = f"learned policy ({pol.name}, {pol.path.parent.name or 'v1'})" if pol else "classic controller"
        if pol:
            self.log("plan", f"arm driven by the trained {pol.name} policy ({pol.path.parent.name}/{pol.path.name}); safety layer caps speed")
        while self.t - t0 < timeout:
            err = xyz - self.io.grasp_center()
            d = float(np.linalg.norm(err))
            if d < tol or (touch and d < 0.08 and self._object_in_hand(touch)):
                self.io.drive(0, 0)
                return
            if d < best - 0.01:
                best, best_t = d, self.t
            if pol and self.t - best_t > 3.0:            # policy stalled: hand over to the classic controller
                self.log("plan", f"{pol.name} policy stalled at {d * 100:.0f} cm -> classic controller takes over")
                pol, self.arm_driver = None, "classic controller (policy stalled)"
            p = self.pose()
            ad = self.io.arm_dir()
            lateral = float(abs(err[0] * ad[1] - err[1] * ad[0]))       # sideways error, out of the policy's range
            if pol and lateral > 0.03:
                self._base_bearing_fix(xyz)                              # line the base up first
                best_t = self.t
                yield
                continue
            if pol:
                pol.act(self.io, xyz, dt, max_speed=0.15)
            else:
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

    def _g_spin_look(self, label: str):
        spun, y0 = 0.0, self.pose().yaw
        while label not in self.seen and spun < 2 * np.pi:
            p = self.pose()
            spun += abs(wrap(p.yaw - y0))
            y0 = p.yaw
            aim_head(self.m, self.d, self.io, self._sweep_point(p), snap=False)
            self.io.drive(0, SPIN_W)
            yield
        self.io.drive(0, 0)

    def _g_find(self, label: str, what: str):
        """Look around here; if not seen, visit rooms (likeliest first, from memory) and look again."""
        if label in self.seen:
            return
        self.log("perceive", f"looking for the {what}")
        yield from self._g_spin_look(label)
        for room in self.memory.search_order(label, self.t, list(self.rooms)):
            if label in self.seen:
                return
            if not self._start_nav(self.rooms[room], None, f"looking for the {what} in the {room}"):
                continue
            while label not in self.seen:
                p = self.pose()
                aim_head(self.m, self.d, self.io, np.array([p.x + 1.2 * np.cos(p.yaw), p.y + 1.2 * np.sin(p.yaw), 0.9]), snap=False)
                if self._nav_step() != "moving":
                    break
                yield
            yield from self._g_spin_look(label)
        if label in self.seen:
            self.log("perceive", f"found the {what}")

    def _g_grab(self, label: str, what: str, offset=(0, 0, 0)):
        """Reach where the camera saw it and close the fingers; on a miss, look again (twice)."""
        for attempt in range(3):
            target = self.seen[label] + np.array(offset)
            yield from self._g_reach(target, touch=label)
            if self._object_in_hand(label) and self.io.attach(label):
                self.log("plan", f"picked up the {what}")
                return
            self.log("perceive", f"missed the {what}; looking again")
            yield from self._g_retract(lift_up=0.0)
            self.seen.pop(label, None)
            aim_head(self.m, self.d, self.io, target, snap=False)
            yield from self._g_wait(0.8)
            if label not in self.seen:
                yield from self._g_find(label, what)
            if label not in self.seen:
                break
        self._grab_failed = what

    def _prepare_seq(self):
        glass = self.obj_xyz.copy()
        self._grab_failed = None
        self.say("Your glass is empty. Let me pour you some water.")
        yield from self._g_find("jug", "jug of water")
        if "jug" not in self.seen:
            self._ask("I can't see the water jug. Where is it?", resume="LOCATE")
            return
        jug_home = self.seen["jug"].copy()
        # 1. jug
        yield from self._g_nav_to_pick(jug_home, "fetching the jug")
        aim_head(self.m, self.d, self.io, jug_home, snap=False)
        yield from self._g_wait(0.6)
        yield from self._g_grab("jug", "jug")
        if self._grab_failed:
            self._ask("I can't pick up the jug. Could you help me?", resume="LOCATE")
            return
        yield from self._g_retract(lift_up=0.2)
        # 2. pour: hold the jug just above the glass and tilt it
        yield from self._g_nav_to_pick(glass, "pouring into the glass")
        above = glass + np.array([0, 0, 0.21])
        yield from self._g_reach(above, tol=0.04)
        self.log("plan", "pouring water")
        for k in range(40):
            self.io.d.ctrl[self.io.act["wrist_roll"]] = 0.9 * min(1.0, k / 10)
            if 10 <= k < 36:
                self._set_water(min(1.0, (k - 10) / 25))
                self._stream(True, k)
            else:
                self._stream(False)
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
        yield from self._g_find("straw2", "straws")
        if "straw2" not in self.seen:
            self._ask("I can't see the straws. Where are they?", resume="LOCATE")
            return
        straw_home = self.seen["straw2"].copy()
        yield from self._g_nav_to_pick(straw_home, "getting a straw")
        aim_head(self.m, self.d, self.io, straw_home, snap=False)
        yield from self._g_wait(0.6)
        yield from self._g_grab("straw2", "straw")
        if self._grab_failed:
            self._ask("I can't pick up a straw. Could you help me?", resume="LOCATE")
            return
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

    # ---------- medicine: remind, fetch the organiser, check the day, log, alert ----------
    def _s_medicine(self):
        try:
            next(self._gen)
        except StopIteration:
            if self.state == "MEDICINE":
                self.goto("IDLE", "medicine round finished")

    def _today_compartment(self) -> tuple[str, str]:
        """Which compartment to use: the clock gives the day; Gemini (if online) reads the organiser
        on the wrist camera to confirm. Returns (day, how it was checked)."""
        import datetime
        day = datetime.date.today().strftime("%A")
        if self.cloud is None:
            return day, "clock (offline)"
        from sip.cloud.gemini import read_compartment
        rgb, _, _ = render_head(self.m, self.d, self.renderer, cam=WRIST_CAM, rgb=True)
        seen = read_compartment(rgb)
        if seen is None:
            return day, "clock (Gemini unavailable)"
        if seen.lower()[:3] != day.lower()[:3]:
            return day, f"MISMATCH: camera shows {seen}"
        return day, f"clock + Gemini read '{seen}' on the organiser"

    def _pills_seen(self) -> int:
        """Count tablet pixels in the head camera around the organiser it is holding (it knows where
        its own gripper is). Pure camera measurement."""
        from sip.perception.objects import project
        from sip.perception.vision import pill_pixels
        aim = self.io.grasp_center()
        rgb, _, _ = render_head(self.m, self.d, self.renderer, rgb=True)
        h, w = rgb.shape[:2]
        u, v, z = project(self.m, self.d, CAM, aim, w, h)
        if z <= 0:
            return 0
        r = int(0.18 * (h / 2) / np.tan(np.deg2rad(self.m.cam_fovy[self.m.camera(CAM).id]) / 2) / z)
        return pill_pixels(rgb, (u - r, v - r, u + r, v + r))

    def _hand_at_mouth(self) -> bool:
        if not self.real:
            return False
        if not hasattr(self, "_h2m"):
            from sip.perception.vision import HandToMouth
            self._h2m = HandToMouth()
        rgb, _, _ = render_head(self.m, self.d, self.renderer, rgb=True)
        return bool(self._h2m.check(rgb, self.det.boxes.get("person")))

    def _streak_up(self) -> int:
        """Days in a row with the dose taken (kept on the robot, runs/streak.json)."""
        import datetime
        import json
        path = os.path.join("runs", "streak.json")
        today = datetime.date.today()
        try:
            data = json.load(open(path))
        except (OSError, ValueError):
            data = {"last": None, "count": 0}
        last = datetime.date.fromisoformat(data["last"]) if data.get("last") else None
        if last == today:
            return data["count"]
        data["count"] = data["count"] + 1 if last == today - datetime.timedelta(days=1) else max(1, data.get("count", 0) and 1)
        data["last"] = today.isoformat()
        os.makedirs("runs", exist_ok=True)
        json.dump(data, open(path, "w"))
        return data["count"]

    def _medicine_seq(self):
        self.say("It's time for your tablets. I'll bring your pill organiser.")
        self._grab_failed = None
        yield from self._g_find("organiser", "pill organiser")
        if "organiser" not in self.seen:
            self.alert("pill organiser not found - dose may be missed")
            self._ask("I can't find your pill organiser. Where is it?", resume="IDLE")
            return
        home = self.seen["organiser"].copy()
        yield from self._g_nav_to_pick(home, "fetching the pill organiser")
        aim_head(self.m, self.d, self.io, home, snap=False)
        yield from self._g_wait(0.6)
        yield from self._g_grab("organiser", "pill organiser")
        if self._grab_failed:
            self.alert("could not pick up the pill organiser - dose may be missed")
            self._ask("I can't pick up the organiser. Could you help?", resume="IDLE")
            return
        day, how = self._today_compartment()
        self.log("perceive", f"today's compartment: {day} (checked by {how})")
        if how.startswith("MISMATCH"):
            # never offer tablets that may be the wrong day's: put the organiser back and alert
            self.alert(f"organiser compartment does not match today ({day}): {how} - not offered")
            self.say("This doesn't look like today's tablets, so I won't give them. I've raised an alert for your carer.")
            yield from self._g_retract(lift_up=0.0)
            yield from self._g_reach(home + np.array([0, 0, 0.01]))
            self.io.release()
            yield from self._g_retract(lift_up=0.04)
            return
        yield from self._g_retract(lift_up=0.15)
        self.io.stow(lift=max(self.io.arm_targets()["lift"], 0.8))
        yield from self._g_wait(1.5)
        # bring it to her: same person search and parking as for drinks, offered in front of her
        if self._person_where is None:
            yield from self._find_person_seq()
        goal, yaw, facing, where = self._person_park()
        if not self._start_nav(goal, yaw, "bringing the pill organiser"):
            self._ask("I can't reach you. Could you move something?", resume="IDLE")
            return
        while True:
            aim_head(self.m, self.d, self.io, self._mouth_guess(), snap=False)
            r = self._nav_step()
            if r == "arrived":
                break
            if r == "blocked":
                self._ask("Something is in the way. Could you move it?", resume="IDLE")
                return
            yield
        facing = self._facing()
        mouth = self._mouth_guess()
        offer = mouth + 0.30 * np.array([np.cos(facing), np.sin(facing), 0]) + np.array([0, 0, -0.30])
        yield from self._g_reach(offer, tol=0.05)
        for _ in range(30):   # look at the organiser in its hand (and her, just above it)
            aim_head(self.m, self.d, self.io, self.io.grasp_center(), snap=False)
            yield
        self.say(f"Here are your {day} tablets. Say 'taken' when you've had them, or 'not now'.")
        self.bb.set("dose_reply", None)
        self.bb.set("dose_waiting", True)
        before = self._pills_seen()
        self.log("perceive", f"camera: {before} tablet pixels in today's compartment")
        asked, declines, hand_seen = self.t, 0, False
        while True:
            aim_head(self.m, self.d, self.io, self.io.grasp_center() + np.array([0, 0, 0.12]), snap=False)
            if not hand_seen and self.tick_n % 6 == 0 and self._hand_at_mouth():
                hand_seen = True
                self.log("perceive", "camera: her hand is at her mouth (taking the tablets)")
            reply = self.bb.get("dose_reply")
            if reply == "taken":
                after = self._pills_seen()
                empty = before > 0 and after < 0.3 * before
                self.log("perceive", f"camera: {after} tablet pixels left ({'empty' if empty else 'tablets still there'})")
                if before > 0 and not empty:
                    self.log("dose", f"unconfirmed: she said taken but tablets are still in the {day} compartment")
                    self.alert(f"{day} tablets still in the organiser after 'taken' - please check")
                    self.say("It looks like the tablets are still in the box. I've raised an alert for your carer.")
                    break
                how = ("camera (compartment empty)" if before > 0 else "her word (camera could not see the tablets)")                     + (" + hand seen at mouth" if hand_seen else "")
                streak = self._streak_up()
                self.log("dose", f"taken: {day} dose at t={self.t:.0f}s (confirmed by {how})")
                self.say(f"Well done! That's {streak} day{'s' if streak != 1 else ''} in a row. Keep the streak going!")
                break
            if reply == "declined":
                declines += 1
                self.bb.set("dose_reply", None)
                self.log("dose", f"declined: {day} dose ({declines})")
                if declines >= 2:
                    self.alert(f"{day} dose declined twice")
                    self.say("That's okay. I've raised an alert for your carer.")
                    break
                self.say("Okay. I'll ask again in a little while.")
                asked = self.t
            if self.t - asked > 30.0:
                self.log("dose", f"missed: no answer for 30 s ({day} dose)")
                self.alert(f"no answer at dose time ({day})")
                self.say("I've raised an alert for your carer.")
                break
            yield
        self.bb.set("dose_waiting", False)
        yield from self._g_retract(lift_up=0.0)
        self.io.stow(lift=max(self.io.arm_targets()["lift"], 0.8))
        yield from self._g_wait(1.0)
        yield from self._g_nav_to_pick(home, "putting the organiser back")
        yield from self._g_reach(home + np.array([0, 0, 0.01]))
        self.io.release()
        yield from self._g_retract(lift_up=0.04)
        self.io.stow(lift=max(self.io.arm_targets()["lift"], 0.5))
        self.say("All done.")

    def _turn_straw_to_arm(self) -> None:
        """Turn the glass about its axis so the straw leans along the arm (Limits: done by
        setting the glass pose, standing in for a regrasp)."""
        ad = self.io.arm_dir()
        yaw = float(np.arctan2(ad[1], ad[0]))
        j = self.m.joint(f"{self.label}_free")
        a = j.qposadr[0]
        self.d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.m, self.d)

    def _start_find_person(self, why: str):
        self._find_resume = why
        self._gen = self._find_person_seq()
        self.goto("FIND_PERSON", why)

    def _s_find_person(self):
        try:
            next(self._gen)
        except StopIteration:
            if self._person_where is None:
                self.alert("can't find the person anywhere in the house")
                return self._ask("Where are you? I have your water.", resume="NAV_TO_PERSON")
            self._go_person(self._find_resume)

    def _find_person_seq(self):
        """Visit rooms (living areas first) and look around until the camera sees a person."""
        self.say("Where are you? I'm coming to find you.")
        order = sorted(self.rooms, key=lambda r: {"living": 0, "bedroom": 1, "kitchen": 2, "hall": 3}.get(r, 9))
        for room in order:
            if not self._start_nav(self.rooms[room], None, f"looking for the person in the {room}"):
                continue
            while True:
                if self._person_where is not None:
                    self.io.drive(0, 0)
                    return
                p = self.pose()
                aim_head(self.m, self.d, self.io, np.array([p.x + 1.5 * np.cos(p.yaw), p.y + 1.5 * np.sin(p.yaw), 0.9]), snap=False)
                if self._nav_step() != "moving":
                    break
                yield
            spun, y0 = 0.0, self.pose().yaw
            while spun < 2 * np.pi:
                if self._person_where is not None:
                    self.io.drive(0, 0)
                    self.log("perceive", f"found the person in the {room}")
                    return
                p = self.pose()
                spun += abs(wrap(p.yaw - y0))
                y0 = p.yaw
                aim_head(self.m, self.d, self.io, np.array([p.x + 1.5 * np.cos(p.yaw), p.y + 1.5 * np.sin(p.yaw), 0.9]), snap=False)
                self.io.drive(0, SPIN_W)
                yield
            self.io.drive(0, 0)

    def _go_person(self, why: str):
        if self._person_where is None:
            return self._start_find_person(why)
        goal, yaw, _, where = self._person_park()
        self._person_target = goal
        if not self._start_nav(goal, yaw, why):
            return self._ask("I can't reach you. Could you move something out of the way?",
                             resume="NAV_TO_PERSON")
        self.goto("NAV_TO_PERSON", f"{why} (person at the {where})")

    def _s_nav_to_person(self):
        if self.tick_n % 40 == 0:
            goal, yaw, _, where = self._person_park()
            if np.linalg.norm(goal - self._person_target) > (0.1 if self.mouth.visible else 0.3):
                self.log("plan", f"person moved / seen better at the {where} -> new target")
                self._person_target = goal
                if not self._start_nav(goal, yaw, "person moved"):
                    return self._ask("I can't reach you there. Could you help?", resume="NAV_TO_PERSON")
        aim_head(self.m, self.d, self.io, self._mouth_guess(), snap=False)
        r = self._nav_step()
        if r == "blocked":
            return self._ask("Something is in the way. Could you move it?", resume="NAV_TO_PERSON")
        if r == "arrived":
            if not self._person_here():
                if self._sub == 0:
                    self._sub, self._timer = 1, self.t   # look for a moment before deciding
                if self.t - self._timer < 2.0:
                    self.io.drive(0, 0)
                    return
                self._seats_tried.append(self._person_where)
                left = [k for k in self.places if k not in self._seats_tried]
                if not left:
                    self._seats_tried = []
                    self._person_where = None
                    return self._start_find_person("not at any place I've seen her")
                self.log("plan", f"person moved: not at the {self._person_where}, checking the {left[0]}")
                self._person_where = left[0]
                self._sub = 0
                return self._go_person("looking for the person")
            self._seats_tried = []
            if hasattr(self.face, "set_reference"):
                self.face.set_reference()
            self.say("Here's your water. Tell me when you're ready.")
            self.goto("APPROACH", "arrived beside the person")

    def _approach_target(self) -> np.ndarray:
        mouth = self.mouth.xyz if (self.mouth.visible and self.mouth.xyz is not None) else None
        if mouth is None and self.last_mouth is not None and self.sensed_dist() > 0.5:
            mouth = self.last_mouth[1]  # far away: last sighting is good enough; gate needs live view < 0.5 m
        if mouth is None:
            return None
        yaw = self._facing() + np.deg2rad(self.mouth.yaw_deg)
        return mouth + STRAW_SHORT * np.array([np.cos(yaw), np.sin(yaw), 0])

    def _perception_age(self) -> float:
        """Seconds since the camera models last produced a result (real models run in a background thread)."""
        if not self.real:
            return 0.0
        return self.t - getattr(self.det, "latest_t", self.t)

    def _gate(self) -> SafetyDecision:
        mouth = self.mouth
        if self._perception_age() > 1.0:          # watchdog: never move near the face on old data
            mouth = MouthState(False, None, 0.0, 0.0, False, "stale")
            if not getattr(self, "_stale_logged", False):
                self.log("safety", f"perception is {self._perception_age():.1f} s old -> treating face as unseen")
                self._stale_logged = True
        else:
            self._stale_logged = False
        d = SafetyInput(self.sensed_dist(), mouth, bool(self.bb.get("ready", False)),
                        bool(self.bb.get("stop", False)), self._contact_force(), self.reapproach,
                        done_heard=bool(self.bb.get("done", False)))
        dec = safety_gate(d)
        if self.decision is None or dec.action != self.decision.action or dec.reason != self.decision.reason:
            self.log("safety", f"{dec.action}: {dec.reason} (d={d.dist_to_mouth_m:.2f} m)")
        self.decision = dec
        return dec

    def _s_approach(self):
        aim_head(self.m, self.d, self.io, self._mouth_guess(), snap=False)
        dt = TICK_STEPS * self.m.opt.timestep
        if self._sub == 0:  # swing the glass out slowly (away from the face), arm retracted
            near = self.sensed_dist() < 0.40
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
        lateral = float(np.linalg.norm((err - (err @ self.io.arm_dir()) * self.io.arm_dir())[:2]))
        if self.io.arm_extension() > 0.48 and err @ self.io.arm_dir() > 0.01:
            if getattr(self, "_reparks", 0) < 2:
                self._reparks = getattr(self, "_reparks", 0) + 1
                self.io.set_arm(arm=0.0)
                self.log("plan", "mouth is out of the arm's reach from here -> moving the base closer")
                return self._go_person("re-parking closer")
        step = dec.max_speed_mps * dt
        # step from where the joints actually are, so a lagging joint can't overshoot into the slow zone
        ext = self.io.arm_extension()
        cmd = self.io.arm_targets()["arm"] + float(np.clip(err @ self.io.arm_dir(), -step, step))
        lead = 0.02 if dec.max_speed_mps < 0.1 else 0.06  # bound how far the command may run ahead
        self.io.set_arm(arm=float(np.clip(cmd, ext - lead, ext + lead)),
                        lift=self.io.arm_targets()["lift"] + float(np.clip(err[2], -step, step)))
        p = self.pose()
        lever = max(0.3, float(np.hypot(self.tip()[0] - p.x, self.tip()[1] - p.y)))
        self._base_bearing_fix(tgt, gain=1.5, wmax=min(0.15, 0.8 * dec.max_speed_mps / lever), point=self.tip())

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
                self.memory.see(self.label, self._place_xyz, self.t, self._room_of(*self._place_xyz[:2]))
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
        """Raise a carer alert: always on the carer dashboard (with photo evidence saved by the app);
        also sent to the carer's webhook (e.g. a phone notification service) if CARER_WEBHOOK_URL is set."""
        from sip.notify import send_carer
        sent = send_carer(text)
        self.log("alert", f"{text} [{'sent to carer' if sent else 'carer dashboard only - no carer channel configured'}]")
        self.bb.set("alert", (self.t, text))

    def _s_ask(self):
        if self.t - self._timer > 30.0 and not getattr(self, "_alerted", False):
            self._alerted = True
            self.alert(f"no answer for 30 s: {self._ask_q}")
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
