# Sip: a home care robot for drinks and medicines

Sip is a simulated Hello Robot Stretch 3 that helps an older person who can speak but has weak or shaky arms. It brings water and holds the straw at her lips, and it brings the right day's tablets. Built for AgentVerse 2026 (UCL), robotics track.

Everything the robot knows comes from its own sensors: cameras, depth, lidar, wheel encoders and the microphone. AI models read those sensors. Nothing about where things are is written into the code.

## What it does

- **Water.** Finds the glass by camera, pours from the jug, adds a straw from the holder, finds her, then brings the straw to her lips. A safety gate decides every step of the final approach. Afterwards it puts the glass back.
- **Medicines.** Reminds her, fetches the pill organiser and checks it is today's compartment. It watches her take the tablets: her hand at her mouth (pose model), then the compartment empty (camera). It waits for her to say she has taken them and keeps a streak. If anything is wrong it raises a carer alert with photo evidence.
- **Safety.** It stops on "stop" or "don't go" (negation-aware). "That's enough" ends the drink. If she turns her head or the face is lost, it backs off or waits. It slows near the face and stops on contact force. It never offers tablets that may be the wrong day's.
- **Adapting.** If the glass is moved, it searches. If a chair falls in the hall, lidar sees it and it re-routes. If she moves to the bed, it notices and looks for her. If a grasp misses, it looks again. If it is stuck, it asks for help.
- **New home.** It explores from an empty map using lidar and the depth camera, and names rooms from what it sees (stove means kitchen, bed means bedroom).

## Models and where they run

| Model | Job | Runs on |
|---|---|---|
| YOLO-World v2 (open vocabulary) | glass, jug, straws, pill box, person, furniture | robot computer (CPU, ~200 ms, background thread) |
| MediaPipe Face Landmarker | mouth point, lips open, head turn | robot computer |
| MediaPipe Pose | hand at mouth (taking tablets) | robot computer |
| faster-whisper base.en | speech to text | robot computer |
| Gemini 4 Argon | second opinion when the detector is unsure, pill-box day check, carer summary | cloud, optional, capped at 30 calls |
| Gemma 4 | small talk | cloud API now; can run on a local GPU |
| ElevenLabs (falls back to the offline Windows voice) | robot voice | cloud, optional |
| PPO policies (Stable-Baselines3) | arm skills: reach-above-glass, grasp; straw and offer retraining | trained on Modal, run on the robot |

The robot works with no network. Cloud models only add second opinions and a nicer voice.

## Set up

You need `git` and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Ibad-10/Sip.git && cd Sip
bash scripts/setup.sh          # installs deps, fetches the pinned Stretch 3 model, checks the scene
```

The setup script creates `.env` from `.env.example`. Keys are optional: `GEMINI_API_KEY`, `ELEVENLABS_API_KEY` and `CARER_WEBHOOK_URL`. Never commit `.env`.

## Run

```bash
uv run python -m sip.app --request water        # 3D viewer + control panel at http://127.0.0.1:8765
uv run python -m sip.app --request medicine
uv run python -m sip.tasks.explore              # explore a new home, save runs/map.npz
uv run python -m sip.record --out runs/demo.mp4 # record a video
```

In the panel you can:

- talk with **Hold to talk**, or type
- use the buttons for live changes
- press **?** for help

In the viewer, double-click an object, then press Ctrl and right-drag to move it.

## Tests

```bash
uv run pytest -q --ignore=tests/test_water.py --ignore=tests/test_judge.py   # fast checks (~1 min)
uv run pytest -q tests/test_water.py tests/test_judge.py                     # full scenarios (slow)
SIP_REAL=1 uv run pytest -q -s tests/test_water.py                           # scenarios on the real AI models
uv run modal run modal_app.py::tests --real                                  # all scenarios in parallel on Modal
```

`SIP_REAL=1` runs the robot on YOLO-World and MediaPipe. Without it, scenario tests use the simulator's own object labels. That is quicker, and it checks the control logic only.

## Limits (honest)

- The grip is a physics weld once the fingers touch, and the water is visual: MuJoCo has no fluids.
- Her body is animated keyframes (Mixamo motion retargeted), not physics. Her head turns on a real joint.
- The robot's position comes from the simulator, standing in for wheel odometry. A real robot would use SLAM localisation.
- Two of four learned arm skills work well (reach-above-glass 97–100%, grasp ~60%). The straw and offer skills are being retrained, so the classic controller does that motion today.
- A carer alert is sent only if `CARER_WEBHOOK_URL` is set. Otherwise it appears on the dashboard only.

## Credits

See [CREDITS.md](CREDITS.md). Grandma model: wolfgar74 (CC-BY-4.0). Pill box: MidiPoly (CC-BY-4.0). Furniture: Poly Haven (CC0). Kitchen items: NVIDIA SimReady (CC-BY-4.0). Robot: MuJoCo Menagerie (Apache-2.0). Motion: Mixamo.
