import subprocess
import sys

from sip.app import handle_command
from sip.blackboard import Blackboard


def test_handle_command_flags():
    bb = Blackboard()
    handle_command("I'm thirsty", bb, llm=None)
    assert bb.get("request") == "water"
    handle_command("stop", bb, llm=None)
    assert bb.get("stop") is True
    handle_command("go", bb, llm=None)
    assert bb.get("go") is True
    handle_command("ready", bb, llm=None)
    assert bb.get("ready") is True
    handle_command("/hand", bb, llm=None)
    assert bb.get("hand_raised") is True


def test_headless_runs_10s():
    r = subprocess.run([sys.executable, "-m", "sip.app", "--headless", "--seconds", "10", "--no-rerun"],
                       capture_output=True, text=True, timeout=300, env={**__import__("os").environ, "SIP_LLM": "0"})
    assert r.returncode == 0, r.stderr[-2000:]
    rate = [line for line in r.stdout.splitlines() if line.startswith("loop_hz=")]
    assert rate and float(rate[-1].split("=")[1]) > 5.0
