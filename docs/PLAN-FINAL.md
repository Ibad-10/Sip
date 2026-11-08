# Final plan: build until 11:00, then deliverables

Standard: if this robot went into a real home tomorrow, it would know nothing in advance. Everything comes from its sensors, the models reading them, or what the person or carer tells it. Every failure needs a safe fallback and a carer alert. Sources: docs/AUDIT-0650.md plus a second pass (dashboard, real-world readiness).

## Second-pass findings (added to the audit)

**Dashboard**
- The map assumes the house spans x −4..4, y −3..3 (`W()`/`H()` in panel.html). Size it from the map's own extent instead.
- "online" means a key exists in `.env`, not that the network works. Measure it from the last cloud call (ok / failed / none).
- Black cameras with no message when the sim isn't running. Show "robot not connected" when /state stops updating.
- There's no dose / tablets button, no "Explore this home" button, and no voice-engine status (ElevenLabs or offline).
- There's no home settings view (who lives here, dose times, carer channel).

**Real-world readiness**
- **Stale perception watchdog:** if the detector or face tracker stops updating during the straw approach, the gate must pause. Today only the face's visibility is checked, not the age of the data.
- **Crash safety:** an exception in the loop kills the app with the arm wherever it is. Wrap the tick: on error, freeze motors, log, alert, keep the dashboard alive.
- **Dose schedule** isn't configurable: there's no home profile (names, dose times, carer channel). Add `home.json`, editable from the dashboard, with reminders from the real clock.
- **No person-fall detection:** the pose model is available, so on-floor or lying in a wrong place should trigger an emergency alert.
- **Privacy:** images go to Gemini without a switch. Add a `CLOUD_IMAGES=off` option in the home profile and show it on the dashboard.
- **Escalation:** an alert with no acknowledgement should repeat or escalate (dashboard "acknowledge" button; repeat every N minutes).

## Priorities

### P0: correctness and honesty (07:00–08:15)
1. Grasp success from the gripper itself (finger opening after closing, or contact force), not the object's true position. *Test: the grasp misses when the object is moved away just before closing.*
2. Straw: release it into the glass physically, with no teleport. Approach the glass from the side where the camera sees the straw instead of rotating the glass. *Fallback if physics is too fiddly:* keep the snap but **say it** in the limits and the video.
3. Robot position: wheel-encoder odometry plus lidar scan-match correction against the map. *Test: drift stays under 10 cm after the full water run.*
4. Remove the `/mouth` and `/hand` fake toggles. `hand_raised` comes from MediaPipe Pose (wrist above shoulder).
5. Stale-data watchdog in the safety gate plus crash-safe loop (freeze, alert).

### P1: real models proven on unseen layouts (08:15–09:30)
6. **Randomised homes:** each run places the glass, jug, straw cup, organiser and person at random valid spots (counters, tables, seats), with random robot start. No position is known to the robot.
7. Detection: YOLO-World scores → CLIP second opinion on large crops → Gemini only when CLIP is unsure, with a confidence threshold before Gemini can override. Show the confidence and "why" everywhere.
8. Measure: N randomised real-model runs in parallel on Modal (success rate, time per task, failure reasons). That's the evidence table for the PDF.

### P2: policies done properly and actually used (in parallel on Modal, 07:15 onwards)
9. Training fixes:
   - clip randomisation ranges so the physics stays stable, and auto-reset on NaN
   - observation and reward normalisation (VecNormalize)
   - curriculum (targets start near, then widen)
   - 8–16 workers (stable)
   - 3 seeds
   - learning curves
   - evaluate on held-out targets
10. Rename "pour" honestly to "reach above glass". Add a real pour skill only if time allows.
11. **Use them in the robot:** a policy proposes the arm motion, the safety gate caps speed and can veto, and the classic controller takes over if the policy stalls or hasn't converged. The dashboard shows which one is driving.

### P3: integration and product (09:30–11:00)
12. Home profile `home.json`: residents, dose times, carer webhook, cloud-images on/off. Edited from the dashboard. Dose reminders from the clock.
13. Dashboard:
    - map auto-sized
    - real online status
    - "robot not connected" state
    - buttons: tablets now, explore this home
    - voice engine status
    - alert acknowledge plus escalation
14. Exploration in the flow: "Explore this home" builds the map and names rooms. The app uses it instead of scene geometry.
15. Person-fall check → emergency alert with photo.
16. Voice: "one moment" filler before slow Gemma replies; voice track in the recorded video.
17. Fresh-clone headless test fix (weights download).

### 11:00–12:45 deliverables
- Video on Modal: one continuous take, perception boxes, her view, captions, one unscripted change, voice.
- 1-page PDF: user and problem, features, architecture, where each model runs plus latency, limits, critical failure mode and how it's handled, safety.
- 5-minute pitch plus live-demo script and backup recording.
- README and limits updated with measured numbers.

## Contingency
- Real models fail on randomised scenes: show the success rate honestly, demo on the best-performing layout, and keep the sim-label mode as a labelled backup for the live demo only.
- Policies still fail for straw/offer: they drive reach-above-glass and grasp only, with the classic controller elsewhere, and the table goes in the PDF.
- Network down at the venue: everything core runs locally; ElevenLabs falls back to the offline voice; Gemini calls are skipped. Demonstrate it on purpose by cutting the wifi mid-run.
- Laptop too slow live: run on the uni PC, or play the recorded video and show the dashboard replay.
