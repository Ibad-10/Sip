"""First visit to a new home: build the map from sensors only.

The robot starts knowing nothing. Lidar rays mark free space up to each hit and the hit as occupied;
the head depth camera adds obstacles above lidar height (table tops, shelves). It then drives to the
nearest frontier (known free cell next to unknown space) until no reachable frontier is left.
Rooms are named from what the object detector sees in them (stove -> kitchen, bed -> bedroom...).
"""
import numpy as np

from sip.nav.grid import OccupancyGrid, astar
from sip.nav.lidar import ray_angles
from sip.types import Pose2D

ROOM_HINTS = {"stove": "kitchen", "sink": "kitchen", "bed": "bedroom", "sofa": "living", "tv": "living"}


class MapBuilder:
    def __init__(self, width_m: float = 10.0, height_m: float = 8.0, res: float = 0.05):
        self.grid = OccupancyGrid(width_m, height_m, res)       # occupied cells
        self.known = np.zeros_like(self.grid.occ)               # seen (free or occupied)
        self.hits = np.zeros(self.grid.occ.shape, np.int16)     # evidence counts (noise-robust)
        self.rooms: dict[str, tuple[float, float]] = {}

    def add_scan(self, pose: Pose2D, ranges: np.ndarray, max_range: float) -> None:
        g = self.grid
        for a, r in zip(pose.yaw + ray_angles(len(ranges)), ranges):
            hit = r < max_range - 1e-3
            n = int(min(r, max_range) / g.res)
            xs = pose.x + np.arange(n) * g.res * np.cos(a)
            ys = pose.y + np.arange(n) * g.res * np.sin(a)
            for x, y in zip(xs, ys):
                i, j = g.cell(x, y)
                if g.inside(i, j):
                    self.known[i, j] = True
                    if self.hits[i, j] > 0:
                        self.hits[i, j] -= 1          # seen through: probably free after all
                        g.occ[i, j] = self.hits[i, j] > 1
            if hit:
                i, j = g.cell(pose.x + r * np.cos(a), pose.y + r * np.sin(a))
                if g.inside(i, j):
                    self.known[i, j] = True
                    self.hits[i, j] = min(self.hits[i, j] + 2, 6)
                    g.occ[i, j] = self.hits[i, j] > 1

    def add_points(self, pts: np.ndarray, zmin: float = 0.25, zmax: float = 1.6) -> None:
        """Depth-camera points (world xyz): anything at body height is an obstacle."""
        for x, y, z in pts:
            if zmin < z < zmax:
                i, j = self.grid.cell(x, y)
                if self.grid.inside(i, j):
                    self.known[i, j] = True
                    self.hits[i, j] = min(self.hits[i, j] + 2, 6)
                    self.grid.occ[i, j] = self.hits[i, j] > 1

    def see_object(self, label: str, xyz) -> None:
        room = ROOM_HINTS.get(label)
        if room and room not in self.rooms:
            self.rooms[room] = (float(xyz[0]), float(xyz[1]))

    def frontiers(self) -> list[tuple[float, float]]:
        free = self.known & ~self.grid.occ
        unk = ~self.known
        f = np.zeros_like(free)
        f[1:-1, 1:-1] = free[1:-1, 1:-1] & (unk[2:, 1:-1] | unk[:-2, 1:-1] | unk[1:-1, 2:] | unk[1:-1, :-2])
        ij = np.argwhere(f)
        return [self.grid.center(i, j) for i, j in ij]

    def next_goal(self, pose: Pose2D, inflate: float = 0.3, tried=()) -> tuple[float, float] | None:
        """Nearest reachable frontier cluster (unknown space counts as blocked for the path)."""
        pts = self.frontiers()
        if not pts:
            return None
        g = self.grid.inflate(inflate)
        g.occ |= ~self.known
        pts = sorted(pts, key=lambda p: np.hypot(p[0] - pose.x, p[1] - pose.y))
        for x, y in pts[::5][:40]:
            if np.hypot(x - pose.x, y - pose.y) < 0.6 or any(np.hypot(x - a, y - b) < 0.5 for a, b in tried):
                continue
            # step back from the frontier into known free space so the goal itself is reachable
            for back in (0.0, 0.25, 0.5):
                d = np.array([x - pose.x, y - pose.y])
                gx, gy = np.array([x, y]) - back * d / (np.linalg.norm(d) + 1e-9)
                c = g.cell(gx, gy)
                if g.inside(*c) and not g.occ[c] and astar(g, (pose.x, pose.y), (gx, gy)) is not None:
                    return float(gx), float(gy)
        return None

    def coverage(self) -> float:
        return float(self.known.mean())

    def save(self, path: str) -> None:
        np.savez_compressed(path, occ=self.grid.occ, known=self.known, w=self.grid.w, h=self.grid.h,
                            res=self.grid.res, rooms=np.array([[k, v[0], v[1]] for k, v in self.rooms.items()], dtype=object))


def load_map(path: str) -> tuple[OccupancyGrid, dict]:
    z = np.load(path, allow_pickle=True)
    g = OccupancyGrid(float(z["w"]), float(z["h"]), float(z["res"]))
    g.occ = z["occ"] & z["known"]
    rooms = {str(r[0]): (float(r[1]), float(r[2])) for r in z["rooms"]}
    return g, rooms
