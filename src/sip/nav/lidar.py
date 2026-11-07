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
    """Planar lidar; rays skip the robot's own geoms."""
    robot = _robot_geom_mask(model)
    ranges = np.full(n_rays, max_range)
    geomid = np.zeros(1, dtype=np.int32)
    for k, a in enumerate(ray_angles(n_rays)):
        d = np.array([np.cos(pose.yaw + a), np.sin(pose.yaw + a), 0.0])
        start = RAY_START
        while start < max_range:
            p = np.array([pose.x, pose.y, height]) + start * d
            dist = mujoco.mj_ray(model, data, p, d, None, 1, -1, geomid)
            if dist < 0:
                break
            if robot[geomid[0]]:
                start += dist + 0.01
                continue
            if start + dist < max_range:
                ranges[k] = start + dist
            break
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
