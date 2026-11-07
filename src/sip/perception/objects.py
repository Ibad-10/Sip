"""Head-camera rendering, pixel<->world geometry and object detectors."""
import mujoco
import numpy as np

from sip.types import Detection

CAM = "d435i_camera_rgb"


def render_head(model, data, renderer: mujoco.Renderer, cam: str = CAM):
    """Returns rgb (h,w,3 uint8), depth (h,w metres), seg (h,w geom id, -1 = none)."""
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    s = renderer.render()
    renderer.disable_segmentation_rendering()
    seg = np.where(s[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM), s[:, :, 0], -1)
    return rgb, depth, seg


def _focal(model, cam: str, h: int) -> float:
    fovy = np.deg2rad(model.cam_fovy[model.camera(cam).id])
    return (h / 2) / np.tan(fovy / 2)


def pixel_to_world(model, data, cam: str, u: float, v: float, depth_m: float,
                   w: int = 320, h: int = 240) -> np.ndarray:
    f = _focal(model, cam, h)
    c = model.camera(cam).id
    local = np.array([(u - w / 2) / f * depth_m, -(v - h / 2) / f * depth_m, -depth_m])
    return data.cam_xpos[c] + data.cam_xmat[c].reshape(3, 3) @ local


def project(model, data, cam: str, xyz: np.ndarray, w: int = 320, h: int = 240):
    """World point -> (u, v, depth). depth <= 0 means behind the camera."""
    f = _focal(model, cam, h)
    c = model.camera(cam).id
    local = data.cam_xmat[c].reshape(3, 3).T @ (np.asarray(xyz) - data.cam_xpos[c])
    z = -local[2]
    if z <= 1e-6:
        return -1.0, -1.0, z
    return w / 2 + f * local[0] / z, h / 2 - f * local[1] / z, z


def _view_error(model, data, target, cam=CAM):
    c = model.camera(cam).id
    fwd = -data.cam_xmat[c].reshape(3, 3)[:, 2]
    d = np.asarray(target) - data.cam_xpos[c]
    yaw_err = np.angle(np.exp(1j * (np.arctan2(d[1], d[0]) - np.arctan2(fwd[1], fwd[0]))))
    pitch_err = np.arctan2(d[2], np.hypot(d[0], d[1])) - np.arctan2(fwd[2], np.hypot(fwd[0], fwd[1]))
    return np.array([yaw_err, pitch_err])


def aim_head(model, data, io, target: np.ndarray, snap: bool = True) -> tuple[float, float]:
    """Point the head camera at a world point. Solves pan/tilt kinematically on a scratch copy
    (numeric Jacobian), then commands the head actuators (and snaps the joints if snap)."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    jp, jt = model.joint("joint_head_pan").qposadr[0], model.joint("joint_head_tilt").qposadr[0]
    lo_p, hi_p = model.jnt_range[model.joint("joint_head_pan").id]
    lo_t, hi_t = model.jnt_range[model.joint("joint_head_tilt").id]
    q = np.array([scratch.qpos[jp], scratch.qpos[jt]])
    for _ in range(25):
        scratch.qpos[jp], scratch.qpos[jt] = q
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_camlight(model, scratch)
        e = _view_error(model, scratch, target)
        if np.abs(e).max() < 1e-3:
            break
        J = np.zeros((2, 2))
        for k in range(2):
            dq = q.copy()
            dq[k] += 1e-3
            scratch.qpos[jp], scratch.qpos[jt] = dq
            mujoco.mj_kinematics(model, scratch)
            mujoco.mj_camlight(model, scratch)
            J[:, k] = (e - _view_error(model, scratch, target)) / 1e-3
        q = q + np.linalg.lstsq(J, e, rcond=None)[0]
        q = np.array([np.clip(q[0], lo_p, hi_p), np.clip(q[1], lo_t, hi_t)])
    io.d.ctrl[io.act["head_pan"]], io.d.ctrl[io.act["head_tilt"]] = q
    if snap:
        data.qpos[jp], data.qpos[jt] = q
        mujoco.mj_forward(model, data)
    return float(q[0]), float(q[1])


def _body_geoms(model, body: str) -> np.ndarray:
    root = model.body(body).id
    ids = []
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        while b > 0 and b != root:
            b = model.body_parentid[b]
        if b == root:
            ids.append(g)
    return np.array(ids)


class SegDetector:
    """Dev-profile detector: segmentation mask from the renderer plus depth, with optional noise.
    Stands in for an open-vocabulary detector; stated under Limits."""

    MIN_PIXELS = 30

    def __init__(self, labels: list[str], noise_m: float = 0.01, cam: str = CAM, seed: int = 0):
        self.labels, self.noise, self.cam = labels, noise_m, cam
        self.rng = np.random.default_rng(seed)
        self._geoms: dict[str, np.ndarray] = {}

    def detect(self, model, data, rgb, depth, seg) -> list[Detection]:
        h, w = depth.shape
        out = []
        for label in self.labels:
            if label not in self._geoms:
                self._geoms[label] = _body_geoms(model, label)
            mask = np.isin(seg, self._geoms[label])
            n = int(mask.sum())
            if n < self.MIN_PIXELS:
                continue
            vs, us = np.nonzero(mask)
            pts = np.array([pixel_to_world(model, data, self.cam, u, v, depth[v, u], w, h)
                            for u, v in zip(us[::4], vs[::4])])
            xyz = np.median(pts, axis=0) + self.rng.normal(0, self.noise, 3)
            out.append(Detection(label, min(1.0, n / 200), (int(us.min()), int(vs.min()),
                                                            int(us.max()), int(vs.max())), xyz))
        return out
