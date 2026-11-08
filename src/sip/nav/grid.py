"""2D occupancy grid and A* planner. World frame, origin at grid centre."""
import heapq

import mujoco
import numpy as np


class OccupancyGrid:
    def __init__(self, width_m: float, height_m: float, res: float = 0.05):
        self.res = res
        self.w, self.h = width_m, height_m
        self.nx, self.ny = int(round(width_m / res)), int(round(height_m / res))
        self.occ = np.zeros((self.nx, self.ny), dtype=bool)

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int((x + self.w / 2) / self.res), int((y + self.h / 2) / self.res)

    def center(self, i: int, j: int) -> tuple[float, float]:
        return (i + 0.5) * self.res - self.w / 2, (j + 0.5) * self.res - self.h / 2

    def inside(self, i: int, j: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny

    def mark(self, x: float, y: float, occupied: bool = True) -> None:
        i, j = self.cell(x, y)
        if self.inside(i, j):
            self.occ[i, j] = occupied

    def is_free(self, x: float, y: float) -> bool:
        i, j = self.cell(x, y)
        return self.inside(i, j) and not self.occ[i, j]

    def inflate(self, radius_m: float) -> "OccupancyGrid":
        out = OccupancyGrid(self.w, self.h, self.res)
        r = int(np.ceil(radius_m / self.res))
        offs = [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)
                if (di * di + dj * dj) * self.res ** 2 <= radius_m ** 2]
        src = self.occ
        for di, dj in offs:
            shifted = np.zeros_like(src)
            xs = slice(max(di, 0), self.nx + min(di, 0))
            xd = slice(max(-di, 0), self.nx + min(-di, 0))
            ys = slice(max(dj, 0), self.ny + min(dj, 0))
            yd = slice(max(-dj, 0), self.ny + min(-dj, 0))
            shifted[xs, ys] = src[xd, yd]
            out.occ |= shifted
        return out

    def copy(self) -> "OccupancyGrid":
        out = OccupancyGrid(self.w, self.h, self.res)
        out.occ = self.occ.copy()
        return out


def astar(grid: OccupancyGrid, start: tuple[float, float],
          goal: tuple[float, float]) -> list[tuple[float, float]] | None:
    s, g = grid.cell(*start), grid.cell(*goal)
    if not (grid.inside(*s) and grid.inside(*g)) or grid.occ[g]:
        return None
    occ = grid.occ.copy()
    occ[s] = False  # the robot's own cell may be inside inflation
    steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    openq = [(0.0, s)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {s: 0.0}
    while openq:
        _, cur = heapq.heappop(openq)
        if cur == g:
            break
        for di, dj, c in steps:
            nb = (cur[0] + di, cur[1] + dj)
            if not grid.inside(*nb) or occ[nb]:
                continue
            nc = cost[cur] + c
            if nc < cost.get(nb, 1e18):
                cost[nb], came[nb] = nc, cur
                heapq.heappush(openq, (nc + np.hypot(nb[0] - g[0], nb[1] - g[1]), nb))
    if g not in cost:
        return None
    cells = [g]
    while cells[-1] != s:
        cells.append(came[cells[-1]])
    cells.reverse()
    # drop collinear points
    keep = [cells[0]]
    for a, b, c in zip(cells, cells[1:], cells[2:]):
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            keep.append(b)
    keep.append(cells[-1])
    return [grid.center(*c) for c in keep]


STATIC_SKIP = ("person", "glass", "organiser", "chair_hall")


def _robot_bodies(model) -> set[int]:
    root = model.body("base_link").id
    return {b for b in range(model.nbody) if _is_descendant(model, b, root)}


def _is_descendant(model, b: int, root: int) -> bool:
    while b > 0:
        if b == root:
            return True
        b = model.body_parentid[b]
    return False


def grid_from_model(model, width_m: float = 8.0, height_m: float = 6.0,
                    res: float = 0.05) -> OccupancyGrid:
    """Static walls and furniture (box geoms below 1.5 m) rasterised into a grid."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    grid = OccupancyGrid(width_m, height_m, res)
    skip = _robot_bodies(model)
    skip |= {model.body(n).id for n in STATIC_SKIP}
    skip |= {b for b in range(model.nbody) for s in STATIC_SKIP if _is_descendant(model, b, model.body(s).id)}
    xs = np.arange(-width_m / 2 + res / 2, width_m / 2, res)
    ys = np.arange(-height_m / 2 + res / 2, height_m / 2, res)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    for gi in range(model.ngeom):
        if model.geom_bodyid[gi] in skip or model.geom_type[gi] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        if model.geom_contype[gi] == 0 and model.geom_conaffinity[gi] == 0:
            continue  # visual-only (decals, worktop trims)
        pos, rot, half = data.geom_xpos[gi], data.geom_xmat[gi].reshape(3, 3), model.geom_size[gi]
        if pos[2] - abs(rot[2] @ half) > 1.5:
            continue
        # point-in-oriented-box test in the plane
        dx, dy = gx - pos[0], gy - pos[1]
        lx = dx * rot[0, 0] + dy * rot[1, 0]
        ly = dx * rot[0, 1] + dy * rot[1, 1]
        grid.occ |= (np.abs(lx) <= half[0]) & (np.abs(ly) <= half[1])
    return grid
