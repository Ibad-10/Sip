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
    .apt_install("libegl1", "libgl1", "libglib2.0-0", "libosmesa6", "ffmpeg", "git", "libgles2", "libegl-mesa0")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")   # CPU build: the CUDA one segfaults here
    .pip_install("mujoco==4.0.0", "numpy", "opencv-python-headless", "pytest", "rerun-sdk==0.42.0",
                 "mediapipe", "trimesh", "pillow", "gymnasium", "stable-baselines3", "ultralytics",
                 "git+https://github.com/ultralytics/CLIP.git")
    .run_commands("python -c \"import clip; clip.load('ViT-B/32', device='cpu')\"")   # cache text encoder
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
    "tests/test_water.py::test_medicine_taken_is_logged_and_organiser_returned",
    "tests/test_water.py::test_medicine_no_answer_alerts_carer",
    "tests/test_judge.py::test_physics_invariants_full_run",
    "tests/test_judge.py::test_perception_before_action",
    "tests/test_judge.py::test_autonomy_no_network",
    "tests/test_judge.py::test_chaos_unexpected_events",
]


@app.function(timeout=1800, cpu=4, memory=6144)
def run_test(job) -> tuple[str, bool, str]:
    import os
    import subprocess
    nodeid, real = job if isinstance(job, tuple) else (job, False)
    env = {**os.environ, **({"SIP_REAL": "1"} if real else {})}
    r = subprocess.run(["python", "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider", nodeid],
                       cwd="/root/sip", capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    timing = " | ".join(l for l in out.splitlines() if l.startswith("TIMING"))
    return nodeid, r.returncode == 0, timing + ("" if r.returncode == 0 else chr(10) + out[-1500:])


@app.function(timeout=600, cpu=8)
def run_fast() -> tuple[bool, str]:
    import subprocess
    r = subprocess.run(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "--ignore=tests/test_water.py", "--ignore=tests/test_judge.py"],
                       cwd="/root/sip", capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr)[-1500:]


@app.local_entrypoint()
def tests(real: bool = False):
    fast = run_fast.spawn()
    results = list(run_test.map([(n, real) for n in SLOW]))
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


# ---------- learned arm skills (PPO), trained in parallel: 3 skills x 3 seeds ----------
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("mujoco==4.0.0", "numpy", "gymnasium", "stable-baselines3", "torch",
                 extra_index_url="https://download.pytorch.org/whl/cpu")
    .env({"PYTHONPATH": "/root/sip/src"})
    .add_local_dir(ROOT / "src", "/root/sip/src", ignore=IGNORE)
    .add_local_dir(ROOT / "third_party", "/root/sip/third_party", ignore=IGNORE)
)


@app.function(image=train_image, cpu=8, memory=8192, timeout=3600, retries=0)
def train_policy(job: tuple) -> tuple[str, int, bytes, dict]:
    task, seed, steps = job
    from functools import partial
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from sip.learn.reach_env import StrawReachEnv

    envs = SubprocVecEnv([partial(StrawReachEnv, seed=seed * 100 + i, task=task) for i in range(8)],
                         start_method="spawn")
    model = PPO("MlpPolicy", envs, n_steps=512, batch_size=2048, learning_rate=3e-4, gamma=0.98,
                ent_coef=0.002, policy_kwargs={"net_arch": [128, 128]}, verbose=0, seed=seed)
    model.learn(total_timesteps=steps)
    env, ok, fast, errs, n = StrawReachEnv(seed=10_000 + seed, task=task), 0, [], [], 200
    for _ in range(n):
        o, _ = env.reset()
        for _ in range(300):
            a, _ = model.predict(o, deterministic=True)
            o, r, done, trunc, info = env.step(a)
            if info["dist"] < env.cfg["near"]:
                fast.append(env.speed)
            if done or trunc:
                break
        ok += int(done)
        errs.append(info["dist"])
    model.save("/tmp/p")
    stats = {"task": task, "seed": seed, "steps": steps, "success_rate": ok / n,
             "median_final_error_cm": float(np.median(errs) * 100),
             "p95_close_speed_near_face_mps": float(np.percentile(fast, 95)) if fast else None}
    return task, seed, Path("/tmp/p.zip").read_bytes(), stats


@app.local_entrypoint()
def train(steps: int = 2_000_000):
    import json
    out = ROOT / "assets" / "policy"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(t, s, steps) for t in ("straw", "offer", "pour", "grasp") for s in (0, 1, 2)]
    table = []
    for task, seed, data, stats in train_policy.map(jobs, return_exceptions=False):
        (out / f"{task}_seed{seed}.zip").write_bytes(data)
        table.append(stats)
        print(stats, flush=True)
    (out / "training_results.json").write_text(json.dumps(table, indent=1))


@app.local_entrypoint()
def one(nodeid: str, real: bool = True):
    name, ok, out = run_test.remote((nodeid, real))
    print(("PASS " if ok else "FAIL ") + name)
    print(out)
