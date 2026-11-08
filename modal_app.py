"""Offload heavy, non-interactive work to Modal. Nothing here is on the robot's decision path.

  uv run modal run modal_app.py::tests              # every slow scenario in parallel, one container each
  uv run modal run modal_app.py::video --scenario chaos   # 1080p demo render on a GPU, saved to runs/

The robot's own decisions still run locally (autonomy); Modal only runs tests and renders.
"""
from pathlib import Path

import modal

ROOT = Path(__file__).parent
IGNORE = ["**/.venv/**", "**/raw/**", "**/__pycache__/**", "**/runs/**", "**/.git/**", "**/.superpowers/**",
          "**/*.rrd", "**/*.mp4"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libegl1", "libgl1", "libglib2.0-0", "libosmesa6", "ffmpeg")
    .pip_install("mujoco==4.0.0", "numpy", "opencv-python-headless", "pytest", "rerun-sdk==0.42.0",
                 "mediapipe", "trimesh", "pillow")
    .env({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "PYTHONPATH": "/root/sip/src", "SIP_LLM": "0"})
    .add_local_dir(ROOT / "src", "/root/sip/src", ignore=IGNORE)
    .add_local_dir(ROOT / "tests", "/root/sip/tests", ignore=IGNORE)
    .add_local_dir(ROOT / "assets", "/root/sip/assets", ignore=IGNORE)
    .add_local_dir(ROOT / "third_party", "/root/sip/third_party", ignore=IGNORE)
)
app = modal.App("sip-care-robot", image=image)

SLOW = [
    "tests/test_water.py::test_full_water_run",
    "tests/test_water.py::test_glass_moved_triggers_search",
    "tests/test_water.py::test_glass_missing_asks",
    "tests/test_water.py::test_chair_in_hall_reroutes",
    "tests/test_water.py::test_only_doorway_blocked_asks",
    "tests/test_water.py::test_stop_freezes_within_one_tick",
    "tests/test_water.py::test_head_turn_mid_approach_retreats_then_retries",
    "tests/test_water.py::test_person_moved_to_bed_retargets",
    "tests/test_water.py::test_done_during_hold_ends_drink",
    "tests/test_water.py::test_no_answer_raises_alert",
    "tests/test_water.py::test_prepare_drink_pours_and_adds_straw",
    "tests/test_water.py::test_gemini_points_when_local_detector_misses",
    "tests/test_judge.py::test_physics_invariants_full_run",
    "tests/test_judge.py::test_perception_before_action",
    "tests/test_judge.py::test_autonomy_no_network",
    "tests/test_judge.py::test_chaos_unexpected_events",
]


@app.function(timeout=900, cpu=4, memory=4096)
def run_test(nodeid: str) -> tuple[str, bool, str]:
    import subprocess
    r = subprocess.run(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", nodeid],
                       cwd="/root/sip", capture_output=True, text=True)
    return nodeid, r.returncode == 0, (r.stdout + r.stderr)[-1500:]


@app.function(timeout=600, cpu=8)
def run_fast() -> tuple[bool, str]:
    import subprocess
    r = subprocess.run(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "--ignore=tests/test_water.py", "--ignore=tests/test_judge.py"],
                       cwd="/root/sip", capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr)[-1500:]


@app.local_entrypoint()
def tests():
    fast = run_fast.spawn()
    results = list(run_test.map(SLOW))
    ok_fast, out_fast = fast.get()
    print(("PASS" if ok_fast else "FAIL"), "fast suite\n", out_fast.splitlines()[-1] if out_fast else "")
    for nodeid, ok, out in results:
        print(("PASS " if ok else "FAIL ") + nodeid + ("" if ok else "\n" + out))
    print(f"{sum(ok for _, ok, _ in results)}/{len(results)} slow scenarios passed")


@app.function(timeout=1800, cpu=8, memory=8192)
def render(scenario: str, seconds: float) -> bytes:
    import subprocess
    args = ["python", "-m", "sip.record", "--out", "/tmp/out.mp4", "--seconds", str(seconds), "--hd"]
    if scenario in ("chaos", "chair"):
        args.append("--chair")
    if scenario in ("chaos", "turn"):
        args.append("--turn")
    subprocess.run(args, cwd="/root/sip", check=True)
    subprocess.run(["ffmpeg", "-y", "-i", "/tmp/out.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "/tmp/out_h264.mp4"], check=True, capture_output=True)
    return Path("/tmp/out_h264.mp4").read_bytes()


@app.local_entrypoint()
def video(scenario: str = "chaos", seconds: float = 240):
    data = render.remote(scenario, seconds)
    out = ROOT / "runs" / f"demo_{scenario}.mp4"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(data)
    print("saved", out, len(data) // 1_000_000, "MB")


# ---------- learned straw-reach policy (PPO, many parallel simulations) ----------
train_image = image.pip_install("gymnasium", "stable-baselines3")


@app.function(image=train_image, cpu=32, memory=16384, timeout=3600)
def train_policy(steps: int = 3_000_000) -> tuple[bytes, dict]:
    import sys
    sys.path.insert(0, "/root/sip/src")
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from sip.learn.reach_env import NEAR, StrawReachEnv

    envs = SubprocVecEnv([lambda i=i: StrawReachEnv(seed=i) for i in range(32)])
    model = PPO("MlpPolicy", envs, n_steps=256, batch_size=4096, learning_rate=3e-4, gamma=0.98,
                ent_coef=0.002, policy_kwargs={"net_arch": [128, 128]}, verbose=1, seed=0)
    model.learn(total_timesteps=steps)
    # evaluate on fresh targets: success rate and closing speed near the face
    env, ok, fast, n = StrawReachEnv(seed=999), 0, [], 200
    for _ in range(n):
        o, _ = env.reset()
        for _ in range(300):
            a, _ = model.predict(o, deterministic=True)
            o, r, done, trunc, info = env.step(a)
            if info["dist"] < NEAR:
                fast.append(env.speed)
            if done or trunc:
                break
        ok += int(done)
    model.save("/tmp/straw_reach")
    stats = {"success_rate": ok / n, "max_close_speed_near_face": float(np.max(fast)) if fast else 0.0,
             "p95_close_speed_near_face": float(np.percentile(fast, 95)) if fast else 0.0, "steps": steps}
    return Path("/tmp/straw_reach.zip").read_bytes(), stats


@app.local_entrypoint()
def train(steps: int = 3_000_000):
    import json
    data, stats = train_policy.remote(steps)
    out = ROOT / "assets" / "policy"
    out.mkdir(parents=True, exist_ok=True)
    (out / "straw_reach.zip").write_bytes(data)
    (out / "straw_reach_stats.json").write_text(json.dumps(stats, indent=1))
    print("saved", out / "straw_reach.zip", stats)
