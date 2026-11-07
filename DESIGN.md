# Design: Home Care Robot (working name "Sip")

Status: grilled and agreed in chat, 7 November 2026, about 19:45 on Saturday; waiting for the user's final approval of this document. Items still marked **(proposed)** were accepted in grill round 3 unless sections 10b–10d say otherwise; those sections take priority. Research is in RESEARCH-CARE-ROBOT.md; the track brief is in BRIEF.md.

## 1. Intent
An older or disabled person who cannot easily lift a glass, reach their medicines or feed themselves asks out loud for help. A mobile robot with an arm finds the item, brings it, and presents it safely at the mouth. It reminds them about medicines and fetches the pharmacist-filled organiser. It talks like a patient carer, runs entirely on the device, and treats the person's face as the most dangerous place it ever goes.

Why it can win: an everyday task anyone understands in five seconds; surprises happen naturally (things get moved, people turn their heads, cough or raise a hand); "Assistive perception" is the first area listed in the brief; a gap exists (companions cannot carry, arms cannot move around the home, research robots are not products).

## 2. Users
- **Primary (proposed):** an older person living at home or in a care home, seated, who can speak but has weak or shaky arms. Evidence: Age UK, 2 million people aged 65+ in England with unmet care needs, 1.1 million who struggle to dress.
- **Secondary:** people with high spinal cord injury or motor neurone disease (MND); later work, since their interaction needs (sip-and-puff, very limited head movement) differ.
- **Carer or family:** receives alerts and a daily log. Never in the loop for routine actions, always in the loop for anything unusual.

## 3. Goals
1. **Water (core):** "Can I have some water?" → robot finds the glass with its camera, grasps it, drives over, tracks the face, brings the straw to the lips, holds, waits, retreats, puts the glass back.
2. **Medicine round (core):** at the set time the robot reminds the person, fetches the organiser, checks with its camera that today's compartment is the right one and still full, presents it, logs "taken" or "declined", and alerts the carer after a missed dose.
3. **Conversation (core glue):** local speech in and out; short, calm, simple phrases.
4. **Feeding (stretch, user requested):** spoon with rigid food pieces brought to the mouth, using the same face-tracking and safety layer as water. Built only after 1–3 work end to end.

## 4. Non-goals and anti-goals
- Non-goals: medical advice, dosing decisions, dispensing loose pills, lifting or moving the person, cloth or dressing, real hardware.
- **Anti-goals (failure even if it "works"):** the robot moves toward a face that is not ready; it hands over the wrong medicine compartment; it calls itself a nurse; it keeps trying after the person says stop.

## 5. How it meets the judging criteria
| Criterion | Shown by |
|---|---|
| Perception-driven action (25) | Every move comes from a detection shown on screen: glass box and depth point, face mesh and mouth point, compartment label. If the glass is not where it was, it searches |
| Adaptation (25) | Unscripted change during the take: a teammate moves the glass, or the person turns their head away mid-approach. The robot re-plans or pauses on screen |
| Autonomy (15) | Everything runs on one laptop (RTX A2000 8 GB, measured latency on screen). Unsure, so it asks or waits. It keeps working with Wi-Fi off |
| Relevance (20) | Age UK unmet-need figures; medication harm (about 1 in 3 older adults after discharge, about £400m a year to the NHS). Critical failure (approaching a face that is not ready) shown being handled |
| Ambition (15) | A single autonomous system combining mobile manipulation, mouth-safe hand-over, medicine checks and local voice |

## 6. System (proposed)
### Perception
- Objects: open-vocabulary detector (OWLv2 or YOLO-World) on Stretch's head camera, plus depth, gives a 3D point. Fallback: MuJoCo segmentation with noise.
- Face and mouth: MediaPipe Face Landmarker gives the mouth centre, an "open" ratio and head yaw. Fallback: simulated head pose with noise.
- Organiser compartment: Gemma 4 vision reads the day label ("Tue PM").
- Hands: detect a raised hand near the face, which stops the arm (MediaPipe hands or a pose check).

### Decision
- A deterministic state machine (find, grasp, navigate, approach, hold, retreat, return) owns all motion.
- Gemma 4 (E4B or E2B, local through llama.cpp) handles speech understanding, choosing which task, and wording; it can never command motion directly.
- **Safety layer, always on, overrides everything:**
  - Near the face (under 30 cm), speed is capped at about 3 cm/s (to tune).
  - Approach only when the face is toward the robot, the mouth is visible and the person has said or signalled "ready".
  - Stop and back off 10 cm when the head turns away, a hand rises, a cough is heard, or contact force passes a limit (ISO/TS 15066 transient-contact guidance; feeding research uses a 65 N breakaway).
  - Maximum two re-approaches, then ask; after a "stop", wait for an explicit "go".
  - Never hand over a compartment that fails the label check.

### Action
- Robot: Hello Robot Stretch 3 through `stretch_mujoco` (mobile base, vertical lift, telescoping arm, wrist yaw, gripper; head colour and depth camera; 2D lidar).
- Navigation: occupancy grid from lidar plus A* in Python.
- Grasp: scripted approach, close and lift from the perceived pose. Fallback: weld on contact, stated under Limits.

### Voice
- In: Gemma 4 E4B audio; fallback faster-whisper.
- Out: Piper TTS.

### Cloud (optional, not core)
- Gemini, for a daily carer summary built from the local log, sent only when online. This is for the Best Use of Gemini prize and is never needed for the robot to act.

## 7. Scene (proposed)
- One room: bedroom or living area with a seated person in an armchair or patient chair, a side table, and a kitchen counter or cabinet three or four metres away holding the glass and the organiser.
- Kitchen pieces from RoboCasa (native MuJoCo). Care props from SimReady (patient chair, hospital bed, drug cabinet, 8 oz glass), converted from USD: extract meshes with `pxr`, simplify them, and use primitive collision shapes. Credits list kept for CC-BY.
- Person: MuJoCo humanoid, seated, with a textured face; the head is driven by a small script (looks around, turns away, opens mouth) and can be dragged in the viewer for unscripted changes.

## 8. Video (60 s, one take; draft, depends on scope)
0–10 s: the person asks for water; the speech transcript appears on screen. 10–25 s: the robot drives to the counter with the detection overlay and grasps the glass. 25–35 s: unscripted change, a teammate drags the glass or the person turns their head; the robot re-plans. 35–50 s: face mesh and mouth point on screen, slow approach, the person raises a hand, the robot stops and backs off, then on "ready" presents the straw. 50–60 s: retreat; on-screen log shows latency and "all on-device".

## 9. Hardware and budget
- Brother's laptop: Dell Precision 5570, RTX A2000 8 GB. Runs MuJoCo, the detector, MediaPipe and Gemma. Budget to state: VRAM in use, latency per loop and per Gemma call, all measured.
- User's laptop (Iris Xe): docs, pitch, PDF, possibly MuJoCo without Gemma.

## 10. Deliverables
- One-page technical summary PDF (user and problem, features, architecture, where each model runs with hardware and latency, limits, critical failure mode, safety).
- 60 s video.
- 5-minute pitch.
- Public GitHub repo (no keys).

## 10b. Calls made after the builder check (Claude's calls, for the user to check)
- **Own scene, not the stretch_mujoco app:** load the Stretch 3 MJCF (Menagerie or stretch_mujoco's model file) into our own scene and run our own `mujoco.viewer.launch_passive` loop, so we control the humanoid, props, overlays and live mouse drags. Simple joint and base commands; no inverse kinematics needed.
- **Primitives first:** cylinder glass, box tables, counter and chair, with plain textures. At most one SimReady prop as visual only, if time allows. RoboCasa is dropped (5 GB plus robosuite).
- **Grasp:** weld the glass to the gripper on contact from the start (stated under Limits).
- **Navigation:** known map grid plus A*; lidar used only to stop or re-route when something is closer than a set distance.
- **Compute:** MediaPipe on the CPU; detector at 2–5 Hz in a separate thread; head camera rendered at low resolution.
- **Voice safety:** "stop", "ready" and "go" are keyword intents that never wait on the LLM. faster-whisper is the speech-in fallback.
- **Cough detection dropped;** triggers are "stop", a raised hand and head turned away.
- **Organiser labels:** colour-coded compartments with a printed label; the label is shown on screen. Gemma reading text is optional.
- **Reach test in hour one** with a capsule person; use a high-backed chair with no arms if the base cannot get close.

## 10c. User decisions, grill round 1 (7 Nov 2026, 19:24)
- **Q1 agreed:** primary user is an older person, seated, who can speak but has weak or shaky arms.
- **No cutting:** plan the full product in tiers (must-have, nice-to-have, future), not a reduced scope.
- **Whole house, not one room:** several rooms, so that things change and get in the way, the robot avoids them, and it learns (where things are, new obstacles).
- **Deadline: Sunday 13:00.**
- **Team: the user alone for now;** the brother is busy. Everything must run on the user's laptop: Intel Core i7-1255U (10 cores, 12 threads, 15 W), 16 GB RAM, Intel Iris Xe graphics, no NVIDIA GPU, Python 3.12.10.
- Time left at 19:24 Saturday: about 17.5 hours including sleep.

## 10c-2. User decisions, grill round 2 (7 Nov 2026)
- **Two model profiles:** "dev" uses small models on the user's laptop (CPU) for building and testing; "demo" uses the best models that fit on the brother's RTX A2000 8 GB laptop, which clones the repo from GitHub. One config switch. Video and budget numbers come from the demo profile.
- **Live changes:** (a) glass moved, (b) chair dragged into the hallway, (c) head turned away, and **(d) the person moves from the armchair to the bed**: all must-have.
- **Video:** one take of the water task through the house (agreed).
- **Grok (xAI) for video:** the user wants to use Grok to create demo video. Constraint: the submitted 60 s simulation video must be a real, continuous screen capture of the running simulation (the brief requires it to be captured from the stream, with perception on screen and an unscripted change). Grok-generated footage may only be used for pitch extras such as a title card or a "what it could look like in a real home" clip, clearly labelled as AI-generated concept footage.
- **All recommendations accepted:** Q3 own scene and viewer; Q5 success = straw within 2 cm of the lips for 3 s after "ready"; Q6 fully offline, cloud optional; Q10 medicines in the live demo and pitch; Q11 real mouse drags; Q12 driving sped up 2× and labelled; Q13 stop 2 cm short, no lip contact; Q14 one hour on real detection, then simulator positions plus noise, said openly; Q15 **reversed by the user: go for the Best Use of Gemini prize** (see 10e); Q16 pre-recorded voice clip in the video; Q17 four areas (living room with armchair, kitchen, bedroom with bed, hallway), doorways open; Q19 learning = object memory with fading confidence plus persistent new obstacles.

## 10e. Gemini prize design (proposed)
Rule kept: the robot never needs the cloud to act safely. Offline-first, cloud-enhanced, as in the builder guide's "offline-first Gemma edge planner... verifying... against Gemini 4 Argon... whenever connectivity returns" blueprint.

Prize criteria and how we meet them:
1. **Two or more modalities in one connected loop:** camera image + the person's speech + the robot's map, in one "second opinion" request.
2. **Gemini output checked by a real verifier:** nothing Gemini says is acted on until deterministic checks pass.
3. **Beyond a text box, people in control:** the person and the carer talk to it; Gemini never moves the arm directly.

Parts:
- **Second opinion when unsure (must-have):** when local confidence is low (an object not found, two candidate glasses, an unexpected scene) and Wi-Fi is up, the robot sends the camera frame, the transcript and its map summary to Gemini (Gemini Robotics-ER for points and boxes, or `gemini-4-argon` for reasoning). Gemini returns a target point and a short plan. **Verifier:** the point must agree with the depth camera (a real surface at that pixel), be reachable by Stretch's joints in MuJoCo, give a collision-free path on the map, and pass every safety rule. Pass: act and show "verified by depth, reach and path check". Fail or offline: ask the person instead. The overlay shows Gemini's suggestion and the verifier result side by side.
- **Carer daily summary (must-have):** `gemini-4-argon` writes a plain-English day summary from the local log (drinks, doses taken or declined, alerts, short clips). **Verifier:** a Python check recounts the doses and drinks from the raw log; the summary is shown only if every number matches, otherwise it is regenerated or the numbers are corrected.
- **Family voice check-in (nice-to-have):** a relative talks to the robot through `gemini-3.8-live` and sees its camera view. Any request that moves the robot needs the person's spoken "yes" first.
- **Build tool (counts toward the prize, costs nothing at runtime):** use `gemini-4-argon` to help generate the house MJCF and critique multi-angle renders of it.
- **Setup:** claim onsite Gemini API credits at the Google DeepMind booth. The key lives in an environment variable, never in the repo. Model IDs from the builder guide; check them against AI Studio before use.

## 10d. Feature tiers
### Must-have (by Sunday 13:00)
1. House scene: living room with armchair, kitchen, bedroom and hallway, with open doorways; built from primitives.
2. Stretch 3 navigates between rooms on a known map with A*, and stops or re-routes when lidar sees something new.
3. Object memory: remembers where it last saw each item (glass, organiser) with a confidence that fades over time; searches likely places when an item is not where it expected.
4. Map memory: a new obstacle seen once stays in the map until seen gone (learning on the fly).
5. Water task end to end, with face tracking and the mouth-safe approach.
6. Safety layer (speed near the face, stop on "stop", a raised hand or head turned away, two re-approaches maximum, no approach unless "ready").
7. Medicine task: timed reminder, fetch the organiser, colour and label check, present, log taken or declined, alert after a missed dose (alert shown on screen).
8. Voice: keyword intents for safety words plus a local model for requests and replies; offline speech in and out.
9. Overlay: detections, mouth point, plan, safety state, latency and "on-device" shown next to the viewer.
10. Live changes: a glass moved, a chair dragged into the hallway, a person who turned their head, all by mouse in the viewer.
11. Deliverables: 60 s video, one-page PDF, 5-minute pitch, public repo.

### Nice-to-have (if must-haves are done)
- Feeding: spoon with rigid food pieces, same face tracking and safety layer.
- Preference learning: remembers "cold water in the blue cup" and offers it next time.
- Fall check: notices the person on the floor in any room and calls the carer.
- Closed door: asks the person to open it, or routes another way.
- Daily carer summary through cloud Gemini when online (Best Use of Gemini prize).
- Hydration tracking: counts drinks and offers water if the person has not drunk for a while.

### Future (pitch slide)
- Real Stretch hardware trial in a care home.
- Alternative inputs for people with spinal injury or MND (sip-and-puff, eye gaze, chin joystick).
- Dressing and bathing help.
- Several languages (Gemma is multilingual).
- Pharmacy and GP integration, with consent, for refill alerts.
- A fleet across care-home rooms, with a nurse dashboard.

## 10f. Live demo (user decision)
- Live run during the 5-minute pitch, on the brother's laptop (demo profile). The judge or a teammate makes the unscripted change by mouse.
- Suggested flow: about 2 minutes live. A spoken request, then the water task with one live change, then the medicine check if time allows. Gemini second opinion shown only if venue Wi-Fi works.
- Backups:
  - a local screen recording of a full run (the builder guide's Wi-Fi advice)
  - typed input if the microphone fails in a noisy room
  - keyword fallback if the local model is slow
  - an offline mode that still works with Wi-Fi off
- A demo-mode launcher: one command opens the scene, the overlay and the voice, with a reset key to restart from a known state.
- Rehearse the live run at least twice before 13:00.

## 10g. Explain view for judges (user request)
A second window, next to the MuJoCo viewer, that shows what the robot is thinking, live and in replay.
- **Decision flow graph:** the state machine (find, grasp, navigate, approach, hold, retreat, plus "ask", "search" and "safety stop") drawn as boxes and arrows, with the current state lit up and the last transition labelled with its reason ("glass not at counter").
- **Cause and effect timeline:** every event in plain words with a timestamp, for example "19:52:31 saw: chair in hallway → map updated → new route via living room (+1.8 m)".
- **Map panel:** top-down house map with the old path dashed and the new path solid, obstacles learned on the fly, and remembered object positions with fading confidence.
- **Camera panel:** head camera with detection boxes, the face mesh and mouth point, and the distance to the mouth.
- **Safety and trust panel:** current safety rule state (speed cap on or off, "ready" heard, hand raised), detector confidence, and Gemini's suggestion next to the verifier's pass or fail.
- **Budget strip:** loop rate, model latency per call, memory use, and "offline" or "online".
- **Replay:** scrub back through a run after it ends, to walk judges through each decision.
- **Proposed tool:** Rerun (open-source robotics visualiser, `pip install rerun-sdk`), which already has image, 2D/3D, text-log and time-series views plus a timeline scrubber. It saves writing a dashboard by hand. Check the current version before use.
- **Static diagram** of the architecture for the pitch and the PDF, drawn once.

## 11. Left to the builder
Exact thresholds, which detector, state-machine library, file layout, prompt wording, colours of the overlay.

## 12. Open questions (for the grill)
See the grill rounds in chat. Known unknowns: submission deadline time; whether Stretch reaches a seated mouth; whether MediaPipe works on a rendered face; Gemma audio quality; VRAM headroom for all models together; how to let a teammate make a live change inside the stretch_mujoco viewer.
