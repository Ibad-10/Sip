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
    .apt_install("libegl1", "libgl1", "libglib2.0-0", "libosmesa6", "ffmpeg", "git", "libgles2", "libegl-mesa0",
                 "espeak-ng")
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
results = modal.Volume.from_name("sip-results", create_if_missing=True)   # survives detached runs

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
    "tests/test_water.py::test_medicine_mismatch_is_not_offered",
    "tests/test_water.py::test_wrist_camera_sees_object_in_hand_and_empty_gap",
    "tests/test_explore.py::test_explores_house_from_scratch",
    "tests/test_policy_runtime.py::test_reach_policy_drives_the_arm_in_the_house",
    "tests/test_judge.py::test_physics_invariants_full_run",
    "tests/test_judge.py::test_perception_before_action",
    "tests/test_judge.py::test_autonomy_no_network",
    "tests/test_judge.py::test_chaos_unexpected_events",
]


@app.function(timeout=1800, cpu=4, memory=6144, volumes={"/results": results})
def run_test(job) -> tuple[str, bool, str]:
    import os
    import subprocess
    nodeid, real = job if isinstance(job, tuple) else (job, False)
    env = {**os.environ, **({"SIP_REAL": "1"} if real else {})}
    r = subprocess.run(["python", "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider", nodeid],
                       cwd="/root/sip", capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    timing = " | ".join(l for l in out.splitlines() if l.startswith("TIMING"))
    import pathlib as _p
    d = _p.Path("/results/tests") / ("real" if real else "oracle")
    d.mkdir(parents=True, exist_ok=True)
    (d / (nodeid.split("::")[-1] + ".txt")).write_text(("PASS" if r.returncode == 0 else "FAIL") + chr(10) + timing + chr(10) + out[-6000:])
    results.commit()
    return nodeid, r.returncode == 0, timing + ("" if r.returncode == 0 else chr(10) + out[-1500:])


@app.function(timeout=1500, cpu=8)
def run_fast() -> tuple[bool, str]:
    import subprocess
    r = subprocess.run(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "--ignore=tests/test_water.py", "--ignore=tests/test_judge.py",
                        "--ignore=tests/test_benchmark.py", "--ignore=tests/test_explore.py",
                        "--ignore=tests/test_policy_runtime.py"],
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


@app.function(image=train_image, cpu=16, memory=16384, timeout=5400, retries=0, volumes={"/results": results})
def train_policy(job: tuple) -> tuple[str, int, bytes, dict]:
    """PPO with observation/reward normalisation, curriculum, stable domain randomisation, held-out eval."""
    task, seed, steps, nenv, dr, ver = job
    import json
    import os
    from functools import partial
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from sip.learn.reach_env import StrawReachEnv

    envs = VecNormalize(SubprocVecEnv([partial(StrawReachEnv, seed=seed * 100 + i, task=task, randomise=dr)
                                       for i in range(nenv)], start_method="spawn"), norm_obs=True, norm_reward=True,
                        clip_obs=10.0)
    curve = []

    class Trainer(BaseCallback):
        def _on_rollout_end(self):
            frac = self.num_timesteps / steps
            self.training_env.env_method("set_curriculum", 0.2 + frac / 0.5)   # full range by half-way
            ep = [e["r"] for e in self.model.ep_info_buffer]
            if ep:
                curve.append([int(self.num_timesteps), float(np.mean(ep))])

        def _on_step(self):
            return True

    model = PPO("MlpPolicy", envs, n_steps=512, batch_size=2048, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                ent_coef=0.003, clip_range=0.2, policy_kwargs={"net_arch": [256, 256]}, verbose=0, seed=seed)
    model.learn(total_timesteps=steps, callback=Trainer())
    envs.training = False
    out = f"/results/policy/{ver}"
    os.makedirs(out, exist_ok=True)
    model.save(f"{out}/{task}_seed{seed}")
    envs.save(f"{out}/{task}_seed{seed}_vecnorm.pkl")
    # held-out evaluation on fresh targets (full range), with the same normaliser
    env, ok, fast, errs, n = StrawReachEnv(seed=10_000 + seed, task=task, randomise=dr), 0, [], [], 200
    for _ in range(n):
        o, _ = env.reset()
        for _ in range(1500):
            a, _ = model.predict(envs.normalize_obs(o), deterministic=True)
            o, r, done, trunc, info = env.step(a)
            if info["dist"] < env.cfg["near"]:
                fast.append(env.speed)
            if done or trunc:
                break
        ok += int(done)
        errs.append(info["dist"])
    stats = {"task": task, "seed": seed, "steps": steps, "success_rate": ok / n,
             "median_final_error_cm": float(np.median(errs) * 100),
             "p95_close_speed_near_face_mps": float(np.percentile(fast, 95)) if fast else None,
             "parallel_worlds": nenv, "domain_randomisation": dr, "learning_curve": curve}
    json.dump(stats, open(f"{out}/{task}_seed{seed}.json", "w"))
    results.commit()
    return task, seed, open(f"{out}/{task}_seed{seed}.zip", "rb").read(), stats


@app.local_entrypoint()
def train(steps: int = 2_000_000, version: str = "v1", worlds: int = 8, dr: bool = False, skills: str = "straw,offer,pour,grasp"):
    import json
    out = ROOT / "assets" / "policy"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(t, s, steps, worlds, dr, version) for t in skills.split(",") for s in (0, 1, 2)]
    print("launching", len(jobs), "training jobs", flush=True)
    out = out / version
    out.mkdir(parents=True, exist_ok=True)
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


@app.local_entrypoint()
def train_spawn(steps: int = 3_000_000, version: str = "v4", worlds: int = 16, dr: bool = True,
                skills: str = "straw,offer,pour,grasp"):
    """Fire-and-forget: each policy trains in its own call and saves to the sip-results volume.
    Use with `modal run --detach`; closing this terminal does not cancel the jobs."""
    for t in skills.split(","):
        for s in (0, 1, 2):
            call = train_policy.spawn((t, s, steps, worlds, dr, version))
            print("spawned", t, s, call.object_id, flush=True)


@app.function(timeout=2400, cpu=4, memory=6144, volumes={"/results": results})
def bench_one(job) -> dict:
    import json
    import os
    import subprocess
    seed, real = job
    env = {**os.environ, "SIP_SEEDS": str(seed), **({"SIP_REAL": "1"} if real else {})}
    r = subprocess.run(["python", "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider", "tests/test_benchmark.py"],
                       cwd="/root/sip", capture_output=True, text=True, env=env)
    line = next((l for l in r.stdout.splitlines() if l.startswith("BENCH ")), None)
    res = json.loads(line[6:]) if line else {"seed": seed, "success": False, "error": (r.stdout + r.stderr)[-800:]}
    res["real_models"] = real
    os.makedirs("/results/bench", exist_ok=True)
    json.dump(res, open(f"/results/bench/{'real' if real else 'oracle'}_seed{seed}.json", "w"))
    results.commit()
    return res


@app.local_entrypoint()
def bench(seeds: int = 12, real: bool = True):
    """Randomised homes in parallel; prints a success table."""
    import json
    rows = list(bench_one.map([(s, real) for s in range(seeds)], return_exceptions=True))
    ok = [r for r in rows if isinstance(r, dict) and r.get("success")]
    for r in rows:
        print(json.dumps(r)[:400] if isinstance(r, dict) else f"error: {r}")
    print(f"{len(ok)}/{len(rows)} random homes succeeded (real models={real})")


# ---------- live demo in the cloud: the whole robot runs here, you open the dashboard link ----------
live_image = image.apt_install("espeak-ng") if False else image


@app.function(gpu="T4", cpu=8, memory=16384, timeout=3 * 3600)
def live(minutes: float, keys: dict) -> None:
    import os
    import subprocess
    import time
    env = {**os.environ, **{k: v for k, v in keys.items() if v}, "SIP_PANEL_HOST": "0.0.0.0", "SIP_VOICE_BROWSER": "1",
           "SIP_LLM": "0", "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
    with modal.forward(8765) as tunnel:
        print("DASHBOARD:", tunnel.url, flush=True)
        p = subprocess.Popen(["python", "-m", "sip.app", "--headless", "--realtime", "--no-rerun",
                              "--seconds", str(minutes * 60)], cwd="/root/sip", env=env)
        t0 = time.time()
        while p.poll() is None and time.time() - t0 < minutes * 60 + 120:
            time.sleep(5)
        p.kill()


@app.local_entrypoint()
def demo(minutes: float = 30):
    """uv run modal run modal_app.py::demo --minutes 30   (prints the dashboard link; stops itself)"""
    from pathlib import Path as _P
    keys = {}
    envf = _P(__file__).with_name(".env")
    if envf.exists():
        for line in envf.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() in ("GEMINI_API_KEY", "ELEVENLABS_API_KEY") and v.strip():
                keys[k.strip()] = v.strip()
    live.remote(minutes, keys)
