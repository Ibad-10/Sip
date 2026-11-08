"""Randomised homes: the robot is not told where anything is. SIP_REAL=1 uses the real AI models.

  SIP_REAL=1 uv run pytest -q -s tests/test_benchmark.py -k "seed3"
  uv run modal run modal_app.py::bench --seeds 20           # many homes in parallel on Modal
"""
import json
import os

import mujoco
import numpy as np
import pytest

from sip.scene.randomise import randomise_home
from test_water import Harness, perception, request_water
from sip.tasks.water import WaterTask

SEEDS = [int(s) for s in os.environ.get("SIP_SEEDS", "0,1,2").split(",")]


@pytest.mark.parametrize("seed", SEEDS, ids=[f"seed{s}" for s in SEEDS])
def test_drink_in_a_random_home(seed):
    h = Harness()
    layout = randomise_home(h.m, h.d, seed)
    h.io.loc = None
    h.task = WaterTask(h.io, h.bb, h.task.grid, h.task.memory, *perception(["glass", "jug", "straw2", "organiser"]),
                       prepare=True)
    request_water(h)
    ok = h.run(420, until=lambda h, s: "RETURN" in h.states and s == "IDLE")
    tp, ep = h.io.true_pose(), h.io.base_pose()
    result = {"seed": seed, "layout": layout, "success": bool(ok and h.hold >= 2.0), "sim_seconds": round(h.t, 1),
              "held_at_lips_s": round(h.hold, 1), "loc_error_cm": round(float(np.hypot(tp.x - ep.x, tp.y - ep.y)) * 100, 1),
              "states": h.states[-6:], "alerts": h.kinds("alert")[-2:]}
    print("BENCH " + json.dumps(result))
    assert result["success"], result
