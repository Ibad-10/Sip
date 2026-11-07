# Agentverse 2026 Robotics Track Brief (UCL, 7-8 Nov 2026)

Sponsor: NVIDIA. Tagline: "If it feels like science fiction, you're heading the right way."

## Challenge: give an agent a body
Build an embodied agent that perceives a simulated world through its sensors, reasons about what it finds, and acts on it. It makes its own decisions, with nothing to phone home to.
- Perceive: input from at least one simulated sensor.
- Decide: choose actions from what it senses, not from a pre-scripted path.
- Act: move, navigate, manipulate objects or communicate.

Suggested directions: assistive perception, learning on the fly, adaptive autonomy, coordinated fleets, or your own idea.

## Stack
- Simulator: NVIDIA Isaac Sim is recommended, but it needs an RTX GPU with NVENC. MuJoCo, PyBullet, Gazebo and Webots run without one.
- Sensors: RGB or depth camera, LiDAR, microphone, IMU, contact sensors.
- Streaming: run Isaac Sim headless with the WebRTC streaming client.
- Do this first: get the stream or viewer running in the first hour, before writing agent code.
- Keep it on the device: core decisions that depend on a remote service score lowest on autonomy. Run models locally.

## Deliverables
1. Technical summary, one-page PDF: user and problem, key features, system architecture, where each model runs (hardware and latency), limits, critical failure mode, and safety.
2. Simulation video, up to 1 minute, one continuous take: perception shown on screen next to the actions, plus one unscripted change.
3. Pitch, 5 minutes live: what problem the robot solves, how it works, what it can and cannot do, and why it matters to its user.

## Judging
| Weight | Criterion | What a top score looks like |
|---|---|---|
| 25% | Perception-driven action | Every decision traces to what it perceived, and it corrects mid-run. |
| 25% | Adaptation to change | The scene changes mid-sequence and the agent visibly re-plans. |
| 15% | Operational autonomy | Runs within a stated edge-device budget and behaves sensibly when unsure. |
| 20% | Relevance and impact | A well-evidenced need, with the most critical failure mode handled on screen. |
| 15% | Ambition and technical quality | An original approach to a demanding problem, done to a high standard. |

## Organiser follow-up (UCL AI Society)
- Option A, recommended: MuJoCo, running locally (`pip install mujoco`, then `python -m mujoco.viewer`). Use Modal only for heavy GPU work such as training.
  - Robots: https://github.com/google-deepmind/mujoco_menagerie
  - MuJoCo + Gemini walkthrough: https://dev.to/googleai/building-a-gemini-powered-robotics-simulator-in-the-browser-with-mujoco-wasm-hjj
  - Fast policy training: https://playground.mujoco.org/
  - Gemini Robotics-ER (public in AI Studio): https://github.com/google-gemini/robotics-samples/blob/main/Getting%20Started/gemini_robotics_er.ipynb
- Option B, stretch: Isaac Sim on a Modal L40S GPU (https://github.com/TheFloatingString/isaac-lab-on-modal). The first launch takes 10+ minutes. There is no live GUI over Modal: you save frames and turn them into a video.
- Modal skills for Antigravity: run `modal skills install`. Docs: https://modal.com/docs

### What this means for us
- Use MuJoCo locally. With Isaac Sim on Modal there is no live view, which makes a live unscripted change in one continuous take hard to film, and that change is worth 25%.
- Gemini Robotics-ER is a cloud API, and the brief says core decisions that depend on a remote service score lowest on autonomy. Plan: the core loop runs on local Gemma 4. Gemini Robotics-ER is an optional extra (for example the end-of-mission report to crews), and the robot carries on when the network drops. Demo idea: cut the wifi mid-run and show the robot keep working.

## Builder guide ("Build with Gemini 4 Argon", 12-page PDF)
The user says this is a guide to building with Gemini 4 Argon, not the hackathon's rules. The official requirements are the robotics track slides above. The guide's advice is still useful:
- It suggests freezing features 2 hours before the deadline and keeping a local screen recording as a Wi-Fi backup.
- Keep API keys out of any public repo.
- It suggests keeping people in control of actions that affect physical hardware, which helps the safety story.
- Best Use of Gemini favours a real Gemini call, not mocked output.

### Models (IDs as given in the guide)
- `gemini-4-argon`: lead reasoning, coding and verification. It can generate MuJoCo scenes and critique multi-angle renders.
- `gemini-3.8-live`: real-time voice and camera.
- Gemma 4 E2B and E4B: run locally on laptops and robots, with native audio input. These are our on-robot brain.
- `gemini-3.5-flash-lite`: fastest routing and checks.
- Install: `pip install -U "google-genai>=2.3.0"`, plus `npx skills add google-gemini/gemini-skills --skill gemini-api-dev`.
- Free onsite Gemini API credits from the Google DeepMind booth.

### Robotics blueprint that matches our plan
"An offline-first Gemma 4 edge planner for a mobile robot... processes local audio and obstacle cues on a laptop, while verifying multi-step recovery plans against Gemini 4 Argon safety constraints whenever connectivity returns." That is Smoke Search's architecture, so other teams may build something similar. Our edge is the fire domain: smoke-degraded sensing, heat-aware routing and the hearing feature.

### Best Use of Gemini prize (separate from the track prize)
1. Two or more modalities in one connected loop (for us: thermal and camera vision plus audio).
2. Gemini outputs checked by a real verifier (for us: an A* check on the heat cost map proves the crew route is safe; multi-angle render critique of the scene).
3. An interactive experience beyond a text box, with people kept in control (for us: the incident commander talks to the robot and approves risky moves).

## Constraints on this laptop
- GPU: Intel Iris Xe, no NVIDIA card, so Isaac Sim will not run locally. Use MuJoCo or PyBullet, or ask organisers whether cloud RTX machines are available.
- RAM: 16 GB.
