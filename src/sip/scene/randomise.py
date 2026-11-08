"""World-side set-up: a different home layout every run (the robot is never told any of this).

Places the glass, jug, cup of straws and pill organiser on random free spots of the real surfaces,
seats the person in a random seat, and starts the robot somewhere free. Used to prove that the
robot finds things with its cameras instead of relying on fixed positions.
"""
import mujoco
import numpy as np

from sip.scene.build_house import seat_person

# (x range, y range, top height) of surfaces objects can stand on
SURFACES = {
    "kitchen counter": ((-3.5, -1.6), (2.5, 2.8), 0.90),
    "side table": ((2.22, 2.48), (2.32, 2.58), 0.60),
    "nightstand": ((2.47, 2.83), (-2.88, -2.52), 0.62),
}
HALF_HEIGHT = {"glass": 0.06, "jug": 0.10, "organiser": 0.023, "straw_cup": 0.05}
WHERE = {   # which surfaces each thing may be on (supplies stay in the kitchen, like a real home)
    "glass": ["kitchen counter", "side table", "nightstand"],
    "jug": ["kitchen counter"],
    "straw_cup": ["kitchen counter"],
    "organiser": ["kitchen counter", "side table", "nightstand"],
}
# decor already standing on those surfaces (from scene/visuals.py): never place objects inside them
DECOR = [(2.3, 2.5, 0.10), (2.42, 2.38, 0.12), (2.65, -2.65, 0.08), (-3.4, 2.62, 0.08), (-1.1, 2.72, 0.35)]
ROBOT_STARTS = [(-1.0, 0.0), (0.5, 0.0), (1.5, 1.6), (-2.0, 1.4), (1.0, -1.5)]


def _set_free(m, d, joint: str, xyz, yaw: float = 0.0) -> None:
    a = m.joint(joint).qposadr[0]
    d.qpos[a:a + 3] = xyz
    d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    v = m.joint(joint).dofadr[0]
    d.qvel[v:v + 6] = 0


def randomise_home(m, d, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    placed: list[tuple[str, np.ndarray]] = []
    layout = {}
    for item in ("jug", "straw_cup", "glass", "organiser"):
        for _ in range(200):
            surf = WHERE[item][rng.integers(len(WHERE[item]))]
            (x0, x1), (y0, y1), top = SURFACES[surf]
            p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1), top + HALF_HEIGHT[item]])
            clear = all(np.linalg.norm(p[:2] - q[:2]) > 0.2 for _, q in placed) and \
                all(np.hypot(p[0] - dx, p[1] - dy) > r + 0.06 for dx, dy, r in DECOR)   # not inside decor/stove
            if clear:
                break
        placed.append((item, p))
        layout[item] = (surf, p.round(2).tolist())
        if item == "straw_cup":
            m.body_pos[m.body("straw_cup").id] = p
            _set_free(m, d, "straw2_free", p + np.array([0, 0, 0.07]))
        else:
            _set_free(m, d, f"{item}_free", p, yaw=rng.uniform(-np.pi, np.pi) if item == "organiser" else 0.0)
    seat = ["armchair", "bed"][rng.integers(2)]
    seat_person(m, d, seat)
    layout["person"] = seat
    sx, sy = ROBOT_STARTS[rng.integers(len(ROBOT_STARTS))]
    yaw = rng.uniform(-np.pi, np.pi)
    d.qpos[0:2] = [sx, sy]
    d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    layout["robot_start"] = [sx, sy, round(float(yaw), 2)]
    mujoco.mj_forward(m, d)
    return layout
