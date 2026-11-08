"""Simulated 2D lidar (mj_ray) and persistent obstacle map updates."""
import mujoco
import numpy as np

from sip.nav.grid import OccupancyGrid
from sip.types import Pose2D

RAY_START = 0.25   # start rays outside the robot footprint
CLEAR_AFTER = 3    # consecutive free observations before an occupied cell is cleared


def ray_angles(n_rays: int) -> np.ndarray:
    return np.linspace(0, 2 * np.pi, n_rays, endpoint=False)


_ROBOT_GEOMS: dict[int, np.ndarray] = {}


def _robot_geom_mask(model) -> np.ndarray:
    key = id(model)
    if key not in _ROBOT_GEOMS:
        root = model.body("base_link").id
        bodies = set()
        for b in range(model.nbody):
            c = b
            while c > 0 and c != root:
                c = model.body_parentid[c]
            if c == root:
                bodies.add(b)
        _ROBOT_GEOMS[key] = np.isin(model.geom_bodyid, list(bodies))
    return _ROBOT_GEOMS[key]


def scan(model, data, pose: Pose2D, n_rays: int = 180, max_range: float = 4.0,
         height: float = 0.2) -> np.ndarray:
    """Planar lidar using one batched mj_multiRay call; the few rays that hit the robot's own
    geoms are re-cast past them."""
    robot = _robot_geom_mask(model)
    ang = pose.yaw + ray_angles(n_rays)
    dirs = np.c_[np.cos(ang), np.sin(ang), np.zeros(n_rays)]
    origin = np.array([pose.x, pose.y, height])
    geomid = np.zeros(n_rays, dtype=np.int32)
    dist = np.zeros(n_rays)
    mujoco.mj_multiRay(model, data, origin, dirs.flatten(), None, 1, -1, geomid, dist, None, n_rays, max_range)
    ranges = np.full(n_rays, max_range)
    one = np.zeros(1, dtype=np.int32)
    for k in range(n_rays):
        d, g = dist[k], geomid[k]
        start = 0.0
        while g >= 0 and robot[g] and start < max_range:
            start += d + 0.01
            d = mujoco.mj_ray(model, data, origin + start * dirs[k], dirs[k], None, 1, -1, one)
            g = one[0] if d >= 0 else -1
        if g >= 0 and d >= 0 and start + d < max_range and start + d >= RAY_START:
            ranges[k] = start + d
    return ranges


def _free_counts(grid: OccupancyGrid) -> np.ndarray:
    if not hasattr(grid, "free_count"):
        grid.free_count = np.zeros_like(grid.occ, dtype=np.int16)
    return grid.free_count


def update_map(grid: OccupancyGrid, pose: Pose2D, ranges: np.ndarray,
               max_range: float) -> list[tuple[float, float]]:
    """Mark lidar hits occupied; free a cell only after CLEAR_AFTER consecutive misses.
    Returns world centres of newly occupied cells."""
    counts = _free_counts(grid)
    new: list[tuple[float, float]] = []
    hits: set[tuple[int, int]] = set()
    free: set[tuple[int, int]] = set()
    for a, r in zip(ray_angles(len(ranges)), ranges):
        c, s = np.cos(pose.yaw + a), np.sin(pose.yaw + a)
        for t in np.arange(RAY_START, min(r, max_range) - grid.res, grid.res):
            free.add(grid.cell(pose.x + t * c, pose.y + t * s))
        if r < max_range:
            hits.add(grid.cell(pose.x + r * c, pose.y + r * s))
    for cell in hits:
        if grid.inside(*cell):
            if not grid.occ[cell]:
                new.append(grid.center(*cell))
            grid.occ[cell] = True
            counts[cell] = 0
    for cell in free - hits:
        if grid.inside(*cell) and grid.occ[cell]:
            counts[cell] += 1
            if counts[cell] >= CLEAR_AFTER:
                grid.occ[cell] = False
                counts[cell] = 0
    return new
