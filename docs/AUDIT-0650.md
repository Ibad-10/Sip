# Audit, 06:50 Sun 8 Nov (no code changed)

Rule being checked: everything the robot knows must come from its sensors, the AI models reading them, or what the person or carer says. Nothing fixed, nothing read from the simulator.

## 1. Robot reads simulator truth (cheats)

| Where | What it reads | Should come from |
|---|---|---|
| `robot.py:20` `base_pose()` | The robot's exact position (qpos) | Wheel-encoder odometry, corrected by lidar against the map (localisation) |
| `robot.py:60` `can_grasp()` | The true object position, to decide whether the grip worked | Finger contact / force sensors, or the gripper opening after closing |
| `water.py:774` | Teleports the spare straw away and switches the glass's straw on | Real placement: release the straw into the glass and let it stay there by physics |
| `water.py:962` `_turn_straw_to_arm` | Rotates the glass by setting its pose | Physical regrasp, or approach from the straw side (detect the straw direction) |
| `water.py:592-598` | Jug/glass truth, but only to draw the water stream | OK: display only, not used for decisions |
| `face.py:36` `FaceTracker("sim")` | Mouth site and head joint | Oracle used only in fast tests; the app uses MediaPipe. Keep, labelled as a test oracle |
| `app.py:98` `SegDetector` | Simulator object labels | Same: test oracle only (`SIP_DETECTOR=seg`) |
| `water.py:110` `dist_to_mouth()` | True mouth position | Used only by tests, the judge monitor and the video overlay. OK, but must never drive decisions (verify) |

## 2. Fixed knowledge written into the code or scene

- Room centres `ROOMS` (`build_house.py`) are the default when no map has been explored. Exploration exists (`sip.tasks.explore`) but is not part of the app flow; `runs/map.npz` is used only if present.
- The occupancy grid at start-up comes from `grid_from_model` (scene geometry) unless the explored map is present.
- The detector vocabulary is a fixed prompt list (`vision.DEFAULT_LABELS`). That is acceptable as "what this robot looks for", but there is no bootstrap from the scene (e.g. Gemini listing what it sees in a new home).
- The glass ("water" means glass), jug, straw holder and organiser are fixed task targets. That is acceptable as the task definition.
- Robot start pose `[-1, 0]` and the live-change positions (chair, glass, bed) are simulation set-up. Acceptable, but the tests always use the same layout: **no randomised scenes**. So nothing proves detection works when things are elsewhere.
- Pill compartment day: the clock, plus Gemini when online. There is no local label reading (OCR).

## 3. Integrations missing or half-done

- **Policies (RL):** nothing at runtime loads a policy. v1: reach-above-glass 97–100% (misnamed "pour"), grasp ~60%, straw 0%, offer 0%. v2 and v3 crashed (unstable physics with randomisation, broken pipe with 32 workers). No viewer of policies inside the real task.
- **Medicine:**
  - the panel has **no "dose time / tablets" button** (only typing "my tablets", or `/dose`, which is not on the panel)
  - no dose schedule (time-based reminder)
  - the streak file is shared across runs
  - the hand-to-mouth pose check works only with real models
- **ElevenLabs:** speaks only in viewer (non-headless) mode. `record.py` (the video) has no voice track. Gemma small-talk replies take about 10 s and have no "one moment" filler.
- **Exploration:** not wired into the app or panel (no "Explore this home" button). Rooms are named only by the detector's furniture labels, so when it sees no stove there is no kitchen.
- **Fake sensor toggles still in the code:** `/mouth` and `/hand` (removed from the panel buttons, but still in `app.py` and the task inputs). `hand_raised` should come from MediaPipe Pose (wrist above shoulder).
- **Video recorder** (`record.py`): no detection boxes, not using the new person behaviour features, no captions for "why", no voice.
- **Rerun explain view:** the spawn hang was never fixed (always `--no-rerun`).

## 4. Correctness and quality bugs found

- **Dashboard blank:** JS syntax error (fixed at 06:45, test added).
- **Real-model runs mostly time out:** 4/18 scenarios passed on Modal (older size-rule detector). Glass/mug/straw confusion makes the robot search forever.
- CLIP on small crops is unreliable (it called the straw cup a "jug" at 0.97).
- Gemini identify can relabel a correct detection, and has no confidence threshold before overriding.
- Speed: loop about 7 Hz with the viewer (good enough), but the sim is slower than real time and a drink takes 2.5–4 min of sim time. No single figure for the time each task takes with real models.
- Fresh-clone headless test fails (probably the YOLO weights download inside the test timeout).
- The `pour` policy env never tests pouring (only reaching a point above the glass).
- Training environment physics goes unstable with domain randomisation (damping x0.5 plus masses) and gives NaN.

## 5. Deliverables (required, not started)

- 1-minute single-take video with perception on screen and one unscripted change
- 1-page PDF
- 5-minute pitch plus live demo script
