# Sip — where we are and what's next (Sun 8 Nov, 03:10; deadline 13:00)

Rule from Ibad: **no hardcoding.** Everything the robot knows comes from its sensors (cameras,
depth, lidar, wheel encoders, microphone) or from AI models reading those sensors, or from what
the person or carer tells it.

## 1. Done and tested

- House in MuJoCo with real furniture (Poly Haven, SimReady), painted walls and pictures, and the real grandma (Sketchfab, re-posed seated, head turns).
- Stretch 3 robot: drives, plans routes (A*), replans when lidar sees something new, and backs out when stuck.
- Water task, end to end:
  - finds the glass
  - fetches the jug and pours (visible stream, level rises)
  - takes a straw from the holder and puts it in
  - brings the drink and holds the straw at her lips, with the safety gate controlling the approach
  - puts the glass back afterwards
- Safety gate:
  - stops on "stop"; pauses on "wait"
  - "that's enough" ends the drink
  - head turned or face lost: backs off or waits
  - slows down near the face; contact force limit
- Alerts with evidence: photos plus log saved, shown in the panel.
- Gemini: Argon checks the summary; Gemini points at the glass when the local detector is unsure, and the robot checks the point with its own depth camera before acting.
- Web control panel: live camera views, map, buttons for live changes.
- Modal: 16 slow scenarios plus 77 fast tests run in parallel. Last run: 14/16 (2 broke after the person change; being fixed).
- Real-model perception started:
  - YOLO-World finds the person (0.88) and the glass (as "cup"/"mug" at 0.78)
  - MediaPipe finds her mouth from the camera to within 6 cm

## 2. Issues you raised, and status

| # | Issue | Status / fix |
|---|---|---|
| 1 | Hardcoding: objects come from sim labels | **In progress:** YOLO-World on the camera image. The glass is found as "cup"; the mug is a second "cup", so the robot must tell them apart (memory, second look, Gemini) |
| 2 | Hardcoding: mouth position from the sim | **In progress:** MediaPipe face landmarks plus depth (tested, 6 cm) |
| 3 | Hardcoding: where she sits (seat list) | Next: find "person" with the detector, remember places she was seen |
| 4 | Hardcoding: map from sim geometry | Next: map built by lidar in a setup walk-through, saved and loaded |
| 5 | Hardcoding: robot position from the sim | Next: wheel-encoder odometry |
| 6 | Room names | Kept as setup input ("this is the kitchen"), like naming rooms on a Roomba. Say so on the Limits slide |
| 7 | "Raise hand" and "close mouth" buttons fake a sensor | Remove. Stop by voice instead; mouth open/closed read from the face landmarks |
| 8 | Head camera image looks sideways | The real D435i is mounted sideways. Show it upright in the panel and video |
| 9 | Sim and panel laggy | Fewer and smaller panel renders, detector in a background thread, run on the uni PC for the demo |
| 10 | Water and gripper realism | Fingers close; water is visual. Optional: physics droplets |
| 11 | Person too still | Done: looks around, watches the robot, breathes, sometimes distracted |
| 12 | 2 tests broke after the person change | Fix while switching to real perception |

## 3. Must-haves still to build

1. Real perception wired into the robot (items 1–5 above), all scenarios passing again.
2. Medicine task with the pill box: reminder, fetch, check the label by camera, log taken or declined, alert on a miss.
3. Speech in and out: microphone to faster-whisper, Gemma 4 for intent, Piper voice out. Typed text works today.
4. Deliverables:
   - 1-minute video (one continuous take, perception on screen, one unscripted change), rendered in HD on Modal
   - 1-page PDF
   - 5-minute pitch plus live demo script
   - README and credits

## 4. UI improvements (panel and viewer)

- **Help button (?)** on the panel: a short "what you can do" card:
  - say things to her (water, ready, that's enough, stop)
  - drag objects in the viewer (double-click to select, Ctrl+right-drag to move)
  - what each live-change button tests
  - what the colours on the map mean
- Upright head camera, larger main view, a toggle for which camera is big.
- "What it's thinking" turned into a clear timeline:
  - icons per kind (sees, plans, safety, alert)
  - current goal at the top
  - perception shown live: the detector's boxes and the mouth point drawn on the camera feed, with confidence
- Safety status as a big traffic light (go / slow / wait / back off / stop) with the reason.
- Remove fake-sensor buttons (raise hand, close mouth). Add "Gemini: on/off" and "person behaviour: calm / distracted".
- Viewer: on-screen key help (F1), labels over objects the robot currently sees, route and lidar toggles (F7–F9 exist).

## 5. Gemini and Modal: what we can add

| Idea | Value | Cost |
|---|---|---|
| Gemini Argon names the glass vs the mug when YOLO sees two "cups" (photo plus question) | Shows "sensible when unsure"; Best Use of Gemini | Low |
| Gemini Argon carer summary at the end of the day, checked against the log | Already built; show it in the panel | Low |
| Gemma 4 (API now, local on the A2000 later) for voice intent | Local model = autonomy points | Medium |
| Gemini reads the pill box label / day compartment from the camera | Medicine task check | Medium |
| Modal GPU: bigger detector (OWLv2 / Grounding DINO) for the HD video | Better boxes in the video | Medium |
| Modal: run all tests with real models in parallel | Proves it works, fast | Low |
| Modal: HD 1080p video render | The submission video | Low |
| VLA model | **No:** not trained on the Stretch, black box, and network latency breaks autonomy | — |

## 6. Nice to have (after must-haves)

- Physics water droplets.
- Feeding with the spoon.
- Preference learning (her usual drink time, how close she likes the straw).
- Fall check (person on the floor seen by the camera, alert).
- Closed door: ask for help.
- Hydration tracking across the day.

## 7. Order of work (next steps)

1. 03:15–06:00: real perception wired in (detector, mouth, person places), panel fixes (upright camera, boxes, help button, remove fake buttons), tests green on Modal.
2. 06:00–07:30: lidar-built map and wheel odometry.
3. 07:30–09:30: medicine task (plus Gemini reading the pill box).
4. 09:30–10:30: speech in and out.
5. 10:30–12:30: video on Modal, PDF, pitch, rehearsal.
6. 12:30: submit.

Dropped first if time runs out (in this order): physics droplets, feeding, speech out, odometry (stated as a limit).
