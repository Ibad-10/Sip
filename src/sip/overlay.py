"""Viewer overlays (toggleable): lidar rays and hits, planned route, mouth target."""
import mujoco
import numpy as np

from sip.nav.lidar import RAY_START, ray_angles

LINE = mujoco.mjtGeom.mjGEOM_LINE
SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE


def _add_line(scn, a, b, rgba, width=2.0):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, LINE, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, LINE, width, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1


def _add_dot(scn, p, r, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], SPHERE, np.array([r, 0, 0]), np.asarray(p, float),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def draw(scn, pose, ranges, path, mouth_target, show_lidar=True, show_path=True, height=0.2, max_range=4.0):
    scn.ngeom = 0
    if show_lidar and ranges is not None:
        for a, r in zip(ray_angles(len(ranges)), ranges):
            d = np.array([np.cos(pose.yaw + a), np.sin(pose.yaw + a), 0.0])
            o = np.array([pose.x, pose.y, height])
            hit = r < max_range - 1e-6
            _add_line(scn, o + RAY_START * d, o + r * d, (1.0, 0.25, 0.2, 0.35) if hit else (0.3, 0.8, 1.0, 0.15), 1.0)
            if hit:
                _add_dot(scn, o + r * d, 0.015, (1.0, 0.2, 0.1, 0.9))
    if show_path and path:
        pts = [np.array([x, y, 0.03]) for x, y in path]
        for a, b in zip(pts, pts[1:]):
            _add_line(scn, a, b, (0.1, 0.9, 0.3, 0.9), 5.0)
        _add_dot(scn, pts[-1], 0.06, (0.1, 0.9, 0.3, 0.8))
    if mouth_target is not None:
        _add_dot(scn, mouth_target, 0.012, (1.0, 0.9, 0.1, 0.9))
