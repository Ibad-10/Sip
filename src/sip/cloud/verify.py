"""Deterministic checks that gate anything Gemini suggests, plus log recounts for summaries."""
import re

import numpy as np

from sip.nav.grid import astar
from sip.perception.objects import CAM, pixel_to_world

GRASP_Z = (0.15, 1.25)       # grasp-centre heights Stretch can reach
PICK_STANDOFF = 0.69
DEPTH_TOL = 0.05


def verify_target(model, data, u: int, v: int, depth: np.ndarray, grid, io,
                  claimed_depth: float | None = None, inflate: float = 0.30) -> tuple[bool, list[str]]:
    checks: list[str] = []
    h, w = depth.shape
    if not (0 <= u < w and 0 <= v < h):
        return False, ["pixel outside image"]
    z = float(depth[v, u])
    surface = 0.1 < z < 5.0
    if claimed_depth is not None and abs(claimed_depth - z) > DEPTH_TOL:
        surface = False
    checks.append("depth ok" if surface else "depth mismatch: no surface where Gemini pointed")
    xyz = pixel_to_world(model, data, CAM, u, v, z, w, h)
    reach_z = GRASP_Z[0] <= xyz[2] <= GRASP_Z[1]
    g = grid.inflate(inflate)
    p = io.base_pose()
    path_ok = False
    if reach_z:
        for a in np.deg2rad(np.arange(0, 360, 15)):
            b = xyz[:2] + PICK_STANDOFF * np.array([np.cos(a), np.sin(a)])
            if g.is_free(*b) and astar(g, (p.x, p.y), tuple(b)) is not None:
                path_ok = True
                break
    checks.append("reach ok" if reach_z else f"unreachable height {xyz[2]:.2f} m")
    checks.append("path ok" if path_ok else "no collision-free route to a pick pose")
    ok = surface and reach_z and path_ok
    return ok, checks


def recount_log(events) -> dict:
    return {
        "drinks": sum(1 for e in events if e.kind == "plan" and e.text.startswith("drink offered")),
        "doses_taken": sum(1 for e in events if e.kind == "dose" and e.text.startswith("taken")),
        "doses_declined": sum(1 for e in events if e.kind == "dose" and e.text.startswith("declined")),
        "alerts": sum(1 for e in events if e.kind == "alert"),
    }


_PATTERNS = {
    "drinks": r"(\d+)\s+(?:drinks?|sips?)",
    "doses_taken": r"(?:took|taken)\s+(\d+)",
    "doses_declined": r"declined\s+(\d+)|(\d+)\s+declined",
    "alerts": r"(\d+)\s+alerts?",
}


def summary_matches(summary: str, counts: dict) -> bool:
    """Every number the summary states for a counted item must equal the recount."""
    s = summary.lower()
    for key, pat in _PATTERNS.items():
        for m in re.finditer(pat, s):
            n = int(next(g for g in m.groups() if g))
            if n != counts[key]:
                return False
    return True
