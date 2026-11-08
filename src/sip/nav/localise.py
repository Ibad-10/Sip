"""Where am I? Wheel odometry + lidar scan matching against the map (no simulator position).

Odometry integrates the wheel encoder angles (radius x angle -> distance, difference -> turn).
Every second a correlative scan match nudges the estimate so lidar hits land on mapped walls.
The map frame is the one the robot built (its dock / start pose is the origin it was given at setup).
"""
import numpy as np

from sip.nav.grid import OccupancyGrid
from sip.nav.lidar import ray_angles
from sip.types import Pose2D

WHEEL_R = 0.0487        # effective wheel radius (m), calibrated: drive 1 m straight, compare
WHEEL_SEP = 0.3095      # effective wheel spacing (m), calibrated: spin on the spot, compare


class Localiser:
    def __init__(self, model, data, start: Pose2D):
        self.m, self.d = model, data
        self.jl = model.joint("joint_left_wheel").qposadr[0]
        self.jr = model.joint("joint_right_wheel").qposadr[0]
        self.last = (float(data.qpos[self.jl]), float(data.qpos[self.jr]))
        self.pose = Pose2D(start.x, start.y, start.yaw)
        self.sign = 1.0          # wheel joint direction, calibrated on the first motion
        self._calib = None
        self.corrections = []    # (dx, dy, dyaw) applied by scan matching, for the dashboard
        try:                     # IMU gyro: heading from integrated yaw rate (wheels skid when turning)
            sid = model.sensor("base_gyro").id
            self.gyro = model.sensor_adr[sid]
            # IMU mounting: rotation of the IMU site relative to the robot base (fixed, from the robot model)
            import mujoco
            data_ = mujoco.MjData(model)
            mujoco.mj_forward(model, data_)
            site = model.sensor_objid[sid]
            base = model.body("base_link").id
            self.mount = data_.xmat[base].reshape(3, 3).T @ data_.site_xmat[site].reshape(3, 3)
        except KeyError:
            self.gyro = None
        self.t_last = float(data.time)

    def update(self) -> Pose2D:
        ql, qr = float(self.d.qpos[self.jl]), float(self.d.qpos[self.jr])
        dl, dr = (ql - self.last[0]) * WHEEL_R * self.sign, (qr - self.last[1]) * WHEEL_R * self.sign
        self.last = (ql, qr)
        ds, dth = (dl + dr) / 2, (dr - dl) / WHEEL_SEP
        dt = float(self.d.time) - self.t_last
        self.t_last = float(self.d.time)
        if self.gyro is not None and dt > 0:
            w_base = self.mount @ self.d.sensordata[self.gyro:self.gyro + 3]   # gyro in the base frame
            dth = float(w_base[2]) * dt                                       # yaw rate x elapsed time
        p = self.pose
        mid = p.yaw + dth / 2
        self.pose = Pose2D(p.x + ds * np.cos(mid), p.y + ds * np.sin(mid), float(np.angle(np.exp(1j * (p.yaw + dth)))))
        return self.pose

    def correct(self, ranges: np.ndarray, grid: OccupancyGrid, max_range: float = 4.0) -> None:
        """Correlative scan match: small search around the estimate, keep the pose whose lidar hits
        fall on the most occupied map cells."""
        hit = ranges < max_range - 1e-3
        if hit.sum() < 20:
            return
        ang = ray_angles(len(ranges))[hit]
        r = ranges[hit]
        if getattr(self, "_field_src", None) is not grid or getattr(self, "_field_n", -1) != int(grid.occ.sum()):
            import cv2
            free = (~grid.occ).astype(np.uint8)
            self._field = cv2.distanceTransform(free, cv2.DIST_L2, 5) * grid.res   # metres to nearest wall
            self._field_src, self._field_n = grid, int(grid.occ.sum())
        field = self._field
        p = self.pose
        sigma, prior_xy, prior_a = 0.05, 0.05, np.deg2rad(3)

        def score(dx, dy, da):
            x, y, yaw = p.x + dx, p.y + dy, p.yaw + da
            px, py = x + r * np.cos(yaw + ang), y + r * np.sin(yaw + ang)
            i = np.clip(((px + grid.w / 2) / grid.res).astype(int), 0, grid.nx - 1)
            j = np.clip(((py + grid.h / 2) / grid.res).astype(int), 0, grid.ny - 1)
            fit = float(np.exp(-0.5 * (field[i, j] / sigma) ** 2).mean())
            # trust odometry for small moves: penalise corrections it cannot explain (corridor aliasing)
            return fit - 0.01 * ((dx * dx + dy * dy) / prior_xy ** 2 + (da / prior_a) ** 2)

        best = (score(0, 0, 0), 0.0, 0.0, 0.0)
        for step_xy, span_xy, step_a, span_a in ((0.02, 0.08, 1.0, 4.0), (0.005, 0.015, 0.25, 0.75)):
            cx, cy, ca = best[1], best[2], best[3]
            for da in np.deg2rad(np.arange(-span_a, span_a + 1e-6, step_a)) + ca:
                for dx in np.arange(-span_xy, span_xy + 1e-6, step_xy) + cx:
                    for dy in np.arange(-span_xy, span_xy + 1e-6, step_xy) + cy:
                        sc = score(dx, dy, da)
                        if sc > best[0]:
                            best = (sc, dx, dy, da)
        _, dx, dy, da = best
        if dx or dy or da:
            self.pose = Pose2D(p.x + dx, p.y + dy, p.yaw + da)
            self.corrections.append((dx, dy, da))

    def frame_correction(self, true: Pose2D) -> tuple[float, float, float]:
        """Rigid 2D transform taking positions computed from the robot's kinematic chain (sensors relative
        to its base) into the map frame of the estimate: est = R(dyaw) * (p - true) + est."""
        return self.pose.x, self.pose.y, self.pose.yaw - true.yaw
