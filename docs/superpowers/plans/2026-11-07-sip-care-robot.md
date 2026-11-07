# Sip Care Robot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A simulated Stretch 3 in a four-area MuJoCo house that hears "can I have some water?", finds the glass by camera, avoids live obstacles, and presents the straw safely at a seated person's mouth; runs a medicine round; talks offline; asks Gemini for a verified second opinion when online; and shows its reasoning live in Rerun.

**Architecture:** One Python package `sip`. A passive MuJoCo viewer loop owns physics; perception, voice and cloud run in worker threads and post results to a shared `Blackboard`. A deterministic state machine owns all motion, behind a pure `SafetyGate` that can veto any command. Models are swapped by a `dev`/`demo` profile.

**Tech Stack:** Python 3.12.10, uv 0.13.1, mujoco 4.0.0, numpy, rerun-sdk 0.42.0, mediapipe 1.2.2, opencv-python, pytest; later: faster-whisper, piper-tts, llama.cpp `llama-server` (Gemma 4), google-genai >= 2.3.0, transformers (OWLv2, demo profile only).

**Spec:** `DESIGN.md` (sections 10b–10g override earlier sections). Research: `RESEARCH-CARE-ROBOT.md`. Brief: `BRIEF.md`.

## Global Constraints
- Deadline: Sunday 8 Nov 2026, 13:00. Feature freeze 11:00. Must-haves before nice-to-haves.
- Python 3.12 via `uv run`; tests with `uv run pytest -q`.
- Robot model: `third_party/mujoco_menagerie/hello_robot_stretch_3/stretch.xml` (actuators: `left_wheel_vel`, `right_wheel_vel` [-6, 6]; `lift` [0, 1.1]; `arm` [0, 0.52]; `wrist_yaw` [-1.39, 4.42]; `wrist_pitch` [-1.57, 0.56]; `wrist_roll` [-3.14, 3.14]; `gripper` [-0.02, 0.04]; `head_pan` [-4.04, 1.73]; `head_tilt` [-1.53, 0.79]; cameras `d435i_camera_rgb`, `d435i_camera_depth`, `d405_rgb`, `nav_camera_rgb`; body `link_grasp_center`).
- Core robot behaviour works with no network. Cloud calls always have a timeout and a local fallback.
- Safety numbers (from DESIGN.md): within 0.30 m of the mouth, speed ≤ 0.03 m/s; stop 0.02 m short of the lips; hold 3 s; at most 2 re-approaches, then ask; never approach without "ready"; on "stop", a raised hand or head yaw > 30°, stop and back off 0.10 m.
- Keyword intents `stop`, `ready`, `go` never wait on an LLM.
- No API keys in the repo: `GEMINI_API_KEY` from the environment only.
- Grasp = weld equality constraint activated on contact; stated under Limits.
- Profiles: `SIP_PROFILE=dev` (CPU, small models, segmentation detector) or `demo` (A2000: Gemma 4 E4B, OWLv2).
- Commits authored as the user's local git config; no Claude attribution lines.

## Review Focus
1. Face lost mid-approach (person turns or is occluded) → the arm stops and holds, never continues to a stale mouth point. Test in Task 7 and Task 10.
2. Glass neither where memory says nor visible anywhere → search the rooms in confidence order once, then ask the person; never loop forever. Test in Task 10.
3. "stop" during any phase (driving, grasping, approaching) → all motion targets frozen within one control tick. Test in Task 7 and Task 10.
4. The only doorway blocked by a dragged chair → no route, so the robot asks for help and does not oscillate. Test in Task 3 and Task 10.
5. LLM or Gemini timeout, error or malformed output → local fallback; the control loop never blocks. Test in Task 12 and Task 13.

---

## File Structure
```
sip/
  __init__.py
  config.py            # Profile dataclass, load_profile()
  types.py             # Pose2D, Detection, MouthState, SafetyInput, SafetyDecision, Event
  blackboard.py        # thread-safe latest-value store + event log
  scene/build_house.py # writes assets/house.xml (house, furniture, glass, organiser, person, Stretch include)
  robot.py             # StretchIO: read state, set targets, base drive, grasp weld
  nav/grid.py          # OccupancyGrid, astar()
  nav/lidar.py         # scan() with mj_ray, mark_obstacles()
  nav/follow.py        # PathFollower (diff-drive pure pursuit)
  memory.py            # ObjectMemory (fading confidence), search order
  safety.py            # safety_gate(SafetyInput) -> SafetyDecision
  perception/objects.py # SegDetector (dev), OwlDetector (demo), pixel_to_world()
  perception/face.py    # FaceTracker: MediaPipe or sim fallback -> MouthState
  tasks/water.py       # WaterTask state machine
  tasks/medicine.py    # MedicineTask state machine + organiser check
  voice/intents.py     # parse_keywords(text)
  voice/stt.py voice/tts.py voice/llm.py
  cloud/gemini.py      # second_opinion(), day_summary()
  cloud/verify.py      # verify_target(), recount_log()
  explain/rr.py        # Rerun logging of all panels
  app.py               # launcher: viewer loop, threads, reset key
tests/ (one test file per module, named test_<module>.py)
assets/ (generated house.xml)
```

---

### Task 1: House scene and reach spike

**Files:** Create `sip/__init__.py`, `sip/scene/build_house.py`, `assets/house.xml` (generated), `tests/test_scene.py`

**Interfaces:**
- Produces: `build_house(path: str = "assets/house.xml") -> str` writing an MJCF that `<include>`s the Menagerie Stretch; named bodies `glass` (free joint), `organiser` (free joint), `chair_hall` (free joint), `armchair`, `bed`, `kitchen_counter`, `side_table`, `person` (mocap body) with child `person_head` (hinge `head_yaw` driven by position actuator `head_yaw_act`) and site `mouth`; rooms `living`, `kitchen`, `bedroom`, `hall` as named sites at room centres; walls with 1.0 m open doorways; floor 8 m × 6 m.
- Produces: constant `ROOMS: dict[str, tuple[float, float]]` in `sip/scene/build_house.py`.

- [ ] **Step 1: Write failing tests** `test_house_loads` (model compiles, all named bodies above exist), `test_seated_mouth_reachable` (with `person` in the armchair, set lift/arm to reach the `mouth` site: solve lift = mouth_z − grasp-centre offset, arm = horizontal distance from the mast; assert both within actuator ranges and `link_grasp_center` ends within 0.03 m of the mouth after 3 s of simulation with the base parked 0.45 m beside the armchair), `test_bed_mouth_reachable` (same, person sitting up in bed).
- [ ] **Step 2: Run** `uv run pytest tests/test_scene.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement `build_house`.** Primitives only (boxes, cylinders, capsules) with plain materials. Person = mocap torso plus capsule limbs plus a head sphere with a face texture plane; seat height 0.45 m. Glass = cylinder r 0.035 m, h 0.12 m, mass 0.25 kg, with a thin straw capsule. If a reach test fails, adjust chair height or base parking offset and record the values in the file header.
- [ ] **Step 4: Run tests.** Expected: PASS. Open `uv run python -m mujoco.viewer --mjcf=assets/house.xml`; check visually that the glass and chair can be dragged with the mouse.
- [ ] **Step 5: Commit** `feat: house scene and reach spike`.

### Task 2: Robot interface and base driving

**Files:** Create `sip/types.py`, `sip/config.py` (the `Profile` and `load_profile()` defined in Task 15's Interfaces, created here so Tasks 8–13 can use it), `sip/robot.py`, `tests/test_robot.py`

**Interfaces:**
- Consumes: `assets/house.xml`.
- Produces: `Pose2D(x: float, y: float, yaw: float)` dataclass; class `StretchIO(model, data)` with `base_pose() -> Pose2D`, `drive(v: float, w: float) -> None` (sets wheel velocities from linear and angular speed), `set_arm(lift: float | None = None, arm: float | None = None, wrist_yaw: float | None = None, gripper: float | None = None) -> None` (clamped to ranges), `grasp_center() -> np.ndarray`, `attach(body: str) -> None` / `release() -> None` (toggle a pre-declared weld `grasp_weld_<body>`), `freeze() -> None` (zero base velocity, hold current joint targets).

- [ ] **Step 1: Failing tests** `test_drive_straight` (drive v = 0.3 for 3.33 s → base moved 1.0 ± 0.1 m), `test_turn_in_place` (w = 0.5 for π s → yaw changes π/2 ± 0.15), `test_set_arm_clamps` (lift 2.0 → clamped to 1.1), `test_attach_release` (attach glass, lift 0.2 m → glass z rises; release → glass falls), `test_freeze_stops` (after freeze, base speed < 0.01 m/s within 0.2 s).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Wheel radius and separation read from the model geometry; weld equalities declared in `build_house` (inactive at start) and toggled through `data.eq_active`.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat: Stretch IO and base driving`.

### Task 3: Occupancy grid and A*

**Files:** Create `sip/nav/__init__.py`, `sip/nav/grid.py`, `tests/test_grid.py`

**Interfaces:**
- Produces: class `OccupancyGrid(width_m: float, height_m: float, res: float = 0.05)` with `mark(x: float, y: float, occupied: bool = True)`, `inflate(radius_m: float) -> "OccupancyGrid"`, `is_free(x, y) -> bool`; `astar(grid, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]] | None` (None when unreachable); `grid_from_model(model) -> OccupancyGrid` (static walls and furniture from geoms below 1.5 m height).

- [ ] **Step 1: Failing tests** `test_straight_path`, `test_routes_around_box`, `test_blocked_doorway_returns_none` (only doorway marked occupied → None), `test_inflation_keeps_clearance` (path cells ≥ 0.25 m from obstacles), `test_house_kitchen_to_living_path_exists`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** 8-connected A* on the inflated grid with a Euclidean heuristic; simplify the path by dropping collinear points.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat: grid and A*`.

### Task 4: Lidar scan and map memory

**Files:** Create `sip/nav/lidar.py`, `tests/test_lidar.py`

**Interfaces:**
- Consumes: `OccupancyGrid`, `StretchIO.base_pose()`.
- Produces: `scan(model, data, pose: Pose2D, n_rays: int = 180, max_range: float = 4.0, height: float = 0.2) -> np.ndarray` (ranges, using `mujoco.mj_ray`, excluding robot geoms); `update_map(grid, pose, ranges, max_range) -> list[tuple[float, float]]` returns newly occupied cells; cleared cells are freed only after 3 consecutive free observations (persistence = learning on the fly).

- [ ] **Step 1: Failing tests** `test_scan_sees_wall` (range to a known wall within 0.05 m), `test_new_chair_marked` (move `chair_hall` into the hallway, scan, cell becomes occupied), `test_obstacle_persists_until_seen_gone` (chair removed: still occupied after 2 scans, free after 3).
- [ ] **Step 2–4: Run FAIL, implement, run PASS.**
- [ ] **Step 5: Commit** `feat: lidar and persistent obstacles`.

### Task 5: Path follower

**Files:** Create `sip/nav/follow.py`, `tests/test_follow.py`

**Interfaces:**
- Consumes: `StretchIO`, `astar` path.
- Produces: class `PathFollower(path: list[tuple[float, float]], lookahead: float = 0.3, v_max: float = 0.3)` with `step(pose: Pose2D) -> tuple[float, float]` (v, w) and property `done: bool` (within 0.1 m of the goal); `face(pose, target_xy) -> float` (angular command to face a point).

- [ ] **Step 1: Failing tests** `test_reaches_goal_in_sim` (kitchen to living room via A*, within 0.15 m in < 40 s sim time), `test_slows_at_goal`.
- [ ] **Step 2–4: FAIL, implement pure pursuit, PASS.**
- [ ] **Step 5: Commit** `feat: path follower`.

### Task 6: Object memory

**Files:** Create `sip/memory.py`, `tests/test_memory.py`

**Interfaces:**
- Produces: class `ObjectMemory(half_life_s: float = 600.0)` with `see(label: str, xyz: np.ndarray, t: float, room: str)`, `miss(label: str, room: str, t: float)` (looked there, not found: confidence for that room ×0.2), `confidence(label, t) -> float`, `search_order(label: str, t: float, rooms: list[str]) -> list[str]` (best remembered room first, then the rest by prior: kitchen, living, bedroom, hall).

- [ ] **Step 1: Failing tests** `test_confidence_halves_at_half_life`, `test_miss_demotes_room`, `test_search_order_prefers_last_seen`, `test_unknown_object_uses_prior`.
- [ ] **Step 2–4: FAIL, implement, PASS.**
- [ ] **Step 5: Commit** `feat: object memory`.

### Task 7: Safety gate

**Files:** Create `sip/safety.py`, `tests/test_safety.py`

**Interfaces:**
- Produces: `MouthState(visible: bool, xyz: np.ndarray | None, open_ratio: float, yaw_deg: float, hand_raised: bool)`, `SafetyInput(dist_to_mouth_m: float, mouth: MouthState, ready_heard: bool, stop_heard: bool, contact_force_n: float, reapproach_count: int)` and `SafetyDecision(action: Literal["proceed", "pause", "retreat", "ask", "halt"], max_speed_mps: float, reason: str)` in `sip/types.py`; `safety_gate(inp: SafetyInput) -> SafetyDecision` (pure).

- [ ] **Step 1: Failing tests** with the exact values: `test_stop_halts` (stop_heard → halt, speed 0), `test_no_ready_pauses_near_face` (0.25 m, not ready → pause), `test_speed_capped_within_30cm` (0.29 m, ready → proceed, max 0.03), `test_far_full_speed` (0.6 m → max 0.3), `test_face_lost_pauses` (visible False within 0.5 m → pause, reason "face lost"), `test_head_turned_retreats` (yaw 35° → retreat), `test_hand_raised_retreats`, `test_contact_force_retreats` (force > 10 N → retreat), `test_third_reapproach_asks` (reapproach_count 2 and retreat condition → ask), `test_stop_short_at_2cm` (dist 0.02 → pause, reason "at mouth").
- [ ] **Step 2–4: FAIL, implement as an ordered rule list (halt rules first), PASS.**
- [ ] **Step 5: Commit** `feat: safety gate`.

### Task 8: Object perception

**Files:** Create `sip/perception/__init__.py`, `sip/perception/objects.py`, `tests/test_objects.py`

**Interfaces:**
- Consumes: model cameras `d435i_camera_rgb` and `d435i_camera_depth`.
- Produces: `Detection(label: str, conf: float, box: tuple[int, int, int, int], xyz: np.ndarray | None)` in `sip/types.py`; `render_head(model, data, renderer, w: int = 320, h: int = 240) -> tuple[np.ndarray, np.ndarray, np.ndarray]` (rgb, depth, segmentation); `pixel_to_world(model, data, cam: str, u: int, v: int, depth_m: float) -> np.ndarray`; class `SegDetector(labels: list[str], noise_m: float = 0.01)` and class `OwlDetector(labels)` (demo profile; lazy import of transformers), both with `detect(rgb, depth, seg) -> list[Detection]`.

- [ ] **Step 1: Failing tests** `test_glass_world_position_within_3cm` (SegDetector, robot facing the counter), `test_not_visible_returns_empty` (head turned away), `test_pixel_to_world_roundtrip`.
- [ ] **Step 2–4: FAIL, implement, PASS.** Time-box OwlDetector to 1 hour (DESIGN Q14); if it fails on renders, the demo profile uses SegDetector and the pitch says so.
- [ ] **Step 5: Commit** `feat: object perception`.

### Task 9: Face and mouth tracking

**Files:** Create `sip/perception/face.py`, `tests/test_face.py`

**Interfaces:**
- Consumes: `render_head`, the `mouth` site, `head_yaw` joint.
- Produces: class `FaceTracker(mode: Literal["mediapipe", "sim"])` with `update(rgb, depth, model, data) -> MouthState`. The "sim" mode uses the mouth site plus 0.01 m noise, yaw from the joint, visible only if the mouth projects inside the image and is unoccluded (check the segmentation id). "mediapipe" uses Face Landmarker (mouth centre from inner-lip landmarks, open ratio, yaw), with depth for 3D; it falls back to "sim" per frame when no face is found, and sets a `source` field for the overlay.

- [ ] **Step 1: Failing tests** `test_sim_mouth_within_2cm`, `test_head_turned_reports_yaw` (head_yaw 0.7 rad → yaw_deg > 30), `test_face_out_of_view_not_visible`.
- [ ] **Step 2–4: FAIL, implement, PASS.** Time-box MediaPipe on the rendered face to 1 hour.
- [ ] **Step 5: Commit** `feat: face tracker`.

### Task 10: Water task state machine

**Files:** Create `sip/blackboard.py`, `sip/tasks/__init__.py`, `sip/tasks/water.py`, `tests/test_water.py`

**Interfaces:**
- Consumes: everything in Tasks 2–9.
- Produces: class `Blackboard` (thread-safe `set(key, value)`, `get(key, default)`, `log(event: Event)`, `events() -> list[Event]`; `Event(t: float, kind: str, text: str)` in `sip/types.py`); class `WaterTask(io: StretchIO, bb: Blackboard, grid, memory, detector, face)` with `step(t: float) -> str` returning the state name. States: `IDLE → LOCATE → NAV_TO_OBJECT → GRASP → NAV_TO_PERSON → APPROACH → HOLD → RETREAT → RETURN → IDLE`, plus `SEARCH`, `ASK`, `HALT`. Every transition logs an `Event` with a plain-English reason.

- [ ] **Step 1: Failing headless integration tests** (each uses a fresh scene and a simulated "ready" on the blackboard):
  - `test_full_water_run` (glass on counter → straw within 0.02 m of the mouth for ≥ 3 s, then returned to the counter; < 120 s sim time)
  - `test_glass_moved_triggers_search` (move the glass to the bedroom before grasp → `SEARCH` → found → completes)
  - `test_glass_missing_asks` (glass deleted → after one pass of all rooms → `ASK`)
  - `test_chair_in_hall_reroutes` (event "new route" logged; completes)
  - `test_only_doorway_blocked_asks`
  - `test_stop_freezes_within_one_tick` (stop at NAV and at APPROACH)
  - `test_head_turn_mid_approach_retreats_then_retries`
  - `test_person_moved_to_bed_retargets`
- [ ] **Step 2–4: FAIL, implement, PASS.** Speeds: base 0.3 m/s; arm moves fast until 0.30 m from the mouth, then the SafetyGate cap applies.
- [ ] **Step 5: Commit** `feat: water task`.

### Task 11: Medicine task

**Files:** Create `sip/tasks/medicine.py`, `tests/test_medicine.py`; modify `sip/scene/build_house.py` (organiser = 7×2 compartments, coloured by day, lid boxes with label text textures; today's PM compartment full).

**Interfaces:**
- Produces: class `MedicineTask(io, bb, grid, memory, detector, schedule: list[tuple[str, str]])` (day, time) with `step(t: float, now: datetime) -> str`; `check_compartment(seg_or_rgb, expected: str) -> tuple[bool, str]` (colour check plus label; reason text); `log_dose(bb, status: Literal["taken", "declined", "missed"])`; a missed dose 30 min after reminder → Event kind "alert".

- [ ] **Step 1: Failing tests** `test_reminder_at_time`, `test_wrong_compartment_refuses` (yesterday's compartment open → refuse, alert "wrong compartment"), `test_declined_logged`, `test_missed_alerts_after_30_min`.
- [ ] **Step 2–4: FAIL, implement, PASS.**
- [ ] **Step 5: Commit** `feat: medicine task`.

### Task 12: Voice

**Files:** Create `sip/voice/__init__.py`, `sip/voice/intents.py`, `sip/voice/stt.py`, `sip/voice/tts.py`, `sip/voice/llm.py`, `tests/test_voice.py`

**Interfaces:**
- Produces: `parse_keywords(text: str) -> set[Literal["stop", "ready", "go", "water", "medicine", "yes", "no"]]` (case-insensitive, includes "wait", "no no", "hold on" → stop); class `STT(profile)` → `listen_once(audio: np.ndarray) -> str` (faster-whisper; demo can use Gemma audio); class `TTS(profile)` → `say(text: str) -> None` non-blocking (Piper); class `LLM(base_url: str, timeout_s: float = 4.0)` → `intent(text: str) -> dict` returning `{"task": "water" | "medicine" | "chat" | "none", "reply": str}`, falling back to `parse_keywords` on timeout, error or invalid JSON.

- [ ] **Step 1: Failing tests** `test_stop_variants`, `test_ready_go`, `test_water_request_phrasings` ("could I get a drink", "I'm thirsty"), `test_llm_timeout_falls_back` (unreachable URL → returns keyword result within 4.5 s), `test_llm_garbage_falls_back`.
- [ ] **Step 2–4: FAIL, implement, PASS.** `uv add faster-whisper piper-tts`; llama.cpp prebuilt binary documented in the README (dev: Gemma 4 E2B q4 on CPU; demo: E4B on CUDA).
- [ ] **Step 5: Commit** `feat: voice`.

### Task 13: Gemini second opinion, verifier and carer summary

**Files:** Create `sip/cloud/__init__.py`, `sip/cloud/gemini.py`, `sip/cloud/verify.py`, `tests/test_cloud.py`

**Interfaces:**
- Produces: `second_opinion(rgb: np.ndarray, transcript: str, map_summary: str, labels: list[str], timeout_s: float = 8.0) -> dict | None` (`{"label", "u", "v", "plan"}` or None when offline, timed out or malformed); `verify_target(model, data, u, v, depth, grid, io) -> tuple[bool, list[str]]` (checks: a real surface at the pixel within 0.05 m of the depth reading, reachable by lift and arm ranges from a free base pose, an A* path exists, safety gate not halted; returns which checks passed); `day_summary(events: list[Event]) -> str | None`; `recount_log(events) -> dict` and `summary_matches(summary: str, counts: dict) -> bool`.
- Model ids from the builder guide (`gemini-4-argon`; Robotics-ER per AI Studio); verify in AI Studio before use; key from `GEMINI_API_KEY`.

- [ ] **Step 1: Failing tests** `test_verify_rejects_unreachable_point`, `test_verify_rejects_depth_mismatch`, `test_verify_accepts_glass_pixel`, `test_second_opinion_offline_returns_none` (no key → None fast), `test_summary_mismatch_detected` ("3 drinks" vs counted 2 → False).
- [ ] **Step 2–4: FAIL, implement, PASS.** Wire into WaterTask: in `SEARCH`/`ASK`, if online and local confidence < 0.5, call `second_opinion`; act only if `verify_target` passes; log both on the blackboard.
- [ ] **Step 5: Commit** `feat: Gemini second opinion with verifier`.

### Task 14: Explain view (Rerun)

**Files:** Create `sip/explain/__init__.py`, `sip/explain/rr.py`, `tests/test_explain.py`

**Interfaces:**
- Produces: class `Explainer(app_id: str = "sip", spawn: bool = True)` with `log_frame(t, rgb, detections, mouth)`, `log_map(grid, pose, path_old, path_new, memory)`, `log_state(state: str, reason: str)` (state graph as Rerun `GraphNodes`/`GraphEdges` with the active node highlighted), `log_safety(decision, gemini: dict | None, verified: tuple[bool, list[str]] | None)`, `log_budget(loop_hz, latencies: dict, mem_mb)`, `log_event(Event)`; saves `runs/<timestamp>.rrd` for replay.

- [ ] **Step 1: Failing test** `test_explainer_writes_rrd` (spawn=False, log one of each, file exists and is > 0 bytes).
- [ ] **Step 2–4: FAIL, implement, PASS.** Check the Rerun 0.42 API for graph and time-series archetypes before writing.
- [ ] **Step 5: Commit** `feat: explain view`.

### Task 15: App launcher, live demo mode and profiles

**Files:** Create `sip/config.py`, `sip/app.py`, `README.md`, `tests/test_app.py`

**Interfaces:**
- Produces: `Profile(name, detector: str, face_mode: str, stt_model: str, llm_url: str, cloud: bool)` and `load_profile() -> Profile` (env `SIP_PROFILE`, default "dev"); `main()` runs `mujoco.viewer.launch_passive` with key callbacks: `R` reset to the start state, `W` simulated "water" request, `M` medicine time, `S` stop, `Y` ready, `T` typed-input prompt in the terminal; worker threads for perception (5 Hz), voice and cloud; a `--headless --seconds N` flag for tests and recording.

- [ ] **Step 1: Failing test** `test_headless_runs_10s` (`uv run python -m sip.app --headless --seconds 10` exits 0, log shows loop rate > 50 Hz in dev).
- [ ] **Step 2–4: FAIL, implement, PASS.** README: clone, `uv sync`, `SIP_PROFILE=demo`, llama-server command, Gemini key, keys, troubleshooting.
- [ ] **Step 5: Run the full suite** `uv run pytest -q`, all pass. **Commit** `feat: app launcher and demo mode`. Create the GitHub repo (public) and push only with the user's go-ahead.

### Task 16: Nice-to-haves, in order, each its own commit with tests
1. Feeding: `sip/tasks/feeding.py` (spoon with rigid food pieces, same approach, hold and safety path as water; test `test_spoon_stops_2cm_short`).
2. Preference learning: `ObjectMemory` gets `prefer(person, item)`; test `test_offers_preferred_cup`.
3. Fall check: person lying on the floor (mocap pose) → alert; test `test_fall_alert`.
4. Closed door: a door geom closed → no route → ask "please open the door"; test.
5. Hydration count: offer water after 2 h without a drink; test.
6. Gemini Live family check-in: manual demo only, needs the person's "yes" before any motion.

### Task 17: Deliverables (no tests)
- [ ] 60 s video: record the viewer plus Rerun side by side (OBS or Windows Game Bar) on the demo laptop; one take; driving at 2× labelled on screen; live chair drag plus head turn; pre-recorded voice clip played into STT.
- [ ] One-page PDF: user and problem, features, architecture diagram, where each model runs with measured latency and memory, limits (simulated sensors, weld grasp, sim person), critical failure mode and safety rules, evidence figures from RESEARCH-CARE-ROBOT.md.
- [ ] 5-minute pitch plus 2-minute live demo script; rehearse twice; backup recording on the laptop.
- [ ] Credits file for any CC-BY assets.
