"""Trained policies actually drive the arm in the robot task (with fallback)."""
import numpy as np
import pytest

from sip.learn.runtime import best_policy


@pytest.mark.skipif(best_policy("pour") is None, reason="no trained policy")
def test_reach_policy_drives_the_arm_in_the_house():
    import sys
    sys.path.insert(0, "tests")
    from test_water import Harness, perception, request_water
    from sip.tasks.water import WaterTask
    h = Harness()
    h.task = WaterTask(h.io, h.bb, h.task.grid, h.task.memory, *perception(["glass", "jug", "straw2"]), prepare=True)
    request_water(h)
    h.run(120, until=lambda h, s: any("poured water" in t for t in h.kinds("plan")))
    plans = h.kinds("plan")
    assert any("arm driven by the trained" in t for t in plans), plans[-10:]
    assert any("poured water" in t for t in plans)
