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
