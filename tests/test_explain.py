import numpy as np

from sip.explain.rr import STATES, Explainer
from sip.memory import ObjectMemory
from sip.nav.grid import OccupancyGrid
from sip.types import Detection, Event, MouthState, Pose2D, SafetyDecision


def test_explainer_writes_rrd(tmp_path):
    out = tmp_path / "run.rrd"
    ex = Explainer(spawn=False, save_path=str(out))
    mem = ObjectMemory()
    mem.see("glass", np.array([-2.6, 2.6, 0.96]), 0.0, "kitchen")
    ex.log_frame(1.0, np.zeros((240, 320, 3), np.uint8),
                 [Detection("glass", 0.9, (10, 10, 40, 60), np.zeros(3))],
                 MouthState(True, np.zeros(3), 0.5, 0.0, False), mouth_px=(160, 120), dist_to_mouth=0.3)
    ex.log_map(1.0, OccupancyGrid(8, 6), Pose2D(0, 0, 0), [(0, 0), (1, 1)], [(0, 0), (1, -1)], mem)
    ex.log_state(1.0, "APPROACH", "arrived beside the person")
    ex.log_safety(1.0, SafetyDecision("proceed", 0.03, "near face, slow"), {"label": "glass", "plan": "x"},
                  (True, ["depth ok", "reach ok", "path ok"]))
    ex.log_budget(1.0, 20.0, {"render": 12.0}, 500.0, online=False)
    ex.log_event(Event(1.0, "plan", "new route"))
    ex.close()
    assert out.exists() and out.stat().st_size > 0
    assert "APPROACH" in STATES
