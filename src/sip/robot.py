"""Thin control interface over the Stretch 3 model."""
import mujoco
import numpy as np

from sip.types import Pose2D

WHEEL_R = 0.05
WHEEL_SEP = 0.34
WHEEL_SIGN = 1.0  # set from the model: +1 if positive wheel velocity drives forward


class StretchIO:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.m, self.d = model, data
        self.act = {model.actuator(i).name: i for i in range(model.nu)}
        self.base = model.body("base_link").id
        self.gc = model.body("link_grasp_center").id
        self._held: str | None = None

    def base_pose(self) -> Pose2D:
        q = self.d.qpos
        w, z = q[3], q[6]
        return Pose2D(float(q[0]), float(q[1]), float(2 * np.arctan2(z, w)))

    def drive(self, v: float, w: float) -> None:
        left = (v - w * WHEEL_SEP / 2) / WHEEL_R
        right = (v + w * WHEEL_SEP / 2) / WHEEL_R
        for name, omega in (("left_wheel_vel", left), ("right_wheel_vel", right)):
            a = self.act[name]
            lo, hi = self.m.actuator_ctrlrange[a]
            self.d.ctrl[a] = np.clip(WHEEL_SIGN * omega * self.m.actuator_gear[a, 0], lo, hi)

    def _set(self, name: str, value: float | None) -> None:
        if value is None:
            return
        lo, hi = self.m.actuator_ctrlrange[self.act[name]]
        self.d.ctrl[self.act[name]] = float(np.clip(value, lo, hi))

    def set_arm(self, lift=None, arm=None, wrist_yaw=None, gripper=None) -> None:
        self._set("lift", lift)
        self._set("arm", arm)
        self._set("wrist_yaw", wrist_yaw)
        self._set("gripper", gripper)

    def stow(self, lift: float | None = None) -> None:
        """Tuck the gripper over the base (wrist yaw pi) for driving through doorways."""
        cur = self.d.ctrl[self.act["lift"]]
        self.set_arm(lift=max(0.3, cur if lift is None else lift), arm=0.0, wrist_yaw=np.pi)

    def arm_targets(self) -> dict[str, float]:
        return {k: float(self.d.ctrl[self.act[k]]) for k in ("lift", "arm", "wrist_yaw", "gripper")}

    def grasp_center(self) -> np.ndarray:
        return self.d.xpos[self.gc].copy()

    def arm_dir(self) -> np.ndarray:
        """World direction the arm extends (base -y)."""
        return -self.d.xmat[self.base].reshape(3, 3)[:, 1].copy()

    def can_grasp(self, body: str, tol: float = 0.06) -> bool:
        """Physical check: the fingers must actually reach the object's surface (like finger contact).
        Distance is to the object's centre, so its radius is allowed for."""
        b = self.m.body(body).id
        radius = max((self.m.geom_size[g][0] for g in range(self.m.ngeom)
                      if self.m.geom_bodyid[g] == b and self.m.geom_contype[g]), default=0.0)
        return float(np.linalg.norm(self.d.body(body).xpos - self.grasp_center())) < tol + radius

    def attach(self, body: str) -> bool:
        if not self.can_grasp(body):
            return False
        eq = self.m.equality(f"grasp_weld_{body}").id
        # weld at the current relative pose: relpose = object (body2) pose in the grasp (body1)
        # frame, anchor at the body origin (the spec default anchor is not zero).
        bpos, bquat = self.d.xpos[self.gc], self.d.xquat[self.gc]
        opos, oquat = self.d.body(body).xpos, self.d.body(body).xquat
        inv = np.zeros(4)
        mujoco.mju_negQuat(inv, bquat)
        rel_pos = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_pos, opos - bpos, inv)
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, inv, oquat)
        # eq_data layout for weld: anchor(3), relpos(3), relquat(4), torquescale(1)
        self.d.ctrl[self.act["gripper"]] = self.GRIP.get(body, 0.0)
        self.m.eq_data[eq, 0:3] = 0.0
        self.m.eq_data[eq, 3:6] = rel_pos
        self.m.eq_data[eq, 6:10] = rel_quat
        self.d.eq_active[eq] = 1
        self._held = body
        # carried object must not collide with the robot (it rides over the base when stowed)
        self._saved_contype = {}
        root = self.m.body(body).id
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] == root:
                self._saved_contype[g] = (self.m.geom_contype[g], self.m.geom_conaffinity[g])
                self.m.geom_contype[g] = self.m.geom_conaffinity[g] = 0
        return True

    # gripper command that visibly closes the fingers on each object (the weld does the holding)
    GRIP = {"glass": 0.01, "jug": 0.03, "straw2": -0.015, "organiser": 0.0}

    def release(self) -> None:
        if self._held:
            self.d.ctrl[self.act["gripper"]] = 0.04
            self.d.eq_active[self.m.equality(f"grasp_weld_{self._held}").id] = 0
            for g, (ct, ca) in getattr(self, "_saved_contype", {}).items():
                self.m.geom_contype[g], self.m.geom_conaffinity[g] = ct, ca
        self._held = None

    @property
    def held(self) -> str | None:
        return self._held

    def freeze(self) -> None:
        self.drive(0.0, 0.0)
        for name in ("lift", "arm", "wrist_yaw"):
            j = {"lift": "joint_lift", "wrist_yaw": "joint_wrist_yaw"}.get(name)
            if j:
                self._set(name, self.d.joint(j).qpos[0])
        self._set("arm", self.arm_extension())

    def arm_extension(self) -> float:
        return float(sum(self.d.joint(f"joint_arm_l{i}").qpos[0] for i in range(4)))
