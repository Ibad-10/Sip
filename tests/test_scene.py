import mujoco
import numpy as np
import pytest

from sip.scene.build_house import ROOMS, build_house, seat_person

NAMED_BODIES = [
    "glass", "organiser", "chair_hall", "armchair", "bed", "kitchen_counter",
    "side_table", "person", "person_head",
]


@pytest.fixture(scope="module")
def model():
    path = build_house("assets/house.xml")
    return mujoco.MjModel.from_xml_path(path)


def test_house_loads(model):
    for name in NAMED_BODIES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0, name
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mouth") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_yaw_act") >= 0
    for room in ("living", "kitchen", "bedroom", "hall"):
        assert room in ROOMS
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, room) >= 0


def _act(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def _park_and_reach(model, where):
    """Park the base beside the person, then servo lift/arm onto the mouth."""
    data = mujoco.MjData(model)
    facing = seat_person(model, data, where)
    mujoco.mj_forward(model, data)
    # target: 5 cm in front of the lips (the gripper cannot occupy the face)
    mouth = data.site("mouth").xpos.copy() + 0.05 * np.array([np.cos(facing), np.sin(facing), 0.0])
    gc = model.body("link_grasp_center").id
    lift, arm = _act(model, "lift"), _act(model, "arm")
    lo, hi = model.actuator_ctrlrange[lift], model.actuator_ctrlrange[arm]

    # Arm extends along the base's -y axis. Place the base so that axis points at the mouth.
    from sip.scene.build_house import park_pose_for
    from sip.scene.build_house import SEATS
    x, y, yaw = park_pose_for(mouth, standoff=0.30, facing=facing, approach_deg=SEATS[where][2])
    data.qpos[0:3] = [x, y, 0.0]
    data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    data.ctrl[lift], data.ctrl[arm] = np.clip(mouth[2] - 0.1, *lo), 0.0  # lift first, arm retracted
    for _ in range(10):
        for _ in range(500):
            mujoco.mj_step(model, data)
        err = mouth - data.xpos[gc]
        data.ctrl[lift] = np.clip(data.ctrl[lift] + err[2], *lo)
        arm_dir = -data.xmat[model.body("base_link").id].reshape(3, 3)[:, 1]
        data.ctrl[arm] = np.clip(data.ctrl[arm] + np.dot(err, arm_dir), *hi)
        # rotate the base in place so the grasp centre swings onto the mouth bearing
        b = data.qpos[0:2]
        g, mth = data.xpos[gc][:2] - b, mouth[:2] - b
        yaw += np.arctan2(g[0] * mth[1] - g[1] * mth[0], g @ mth)
        data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        data.qvel[:6] = 0
    for _ in range(12):  # 3 s hold, closed loop on lift/arm only (as the real approach does)
        for _ in range(125):
            mujoco.mj_step(model, data)
        err = mouth - data.xpos[gc]
        arm_dir = -data.xmat[model.body("base_link").id].reshape(3, 3)[:, 1]
        data.ctrl[lift] = np.clip(data.ctrl[lift] + 0.5 * err[2], *lo)
        data.ctrl[arm] = np.clip(data.ctrl[arm] + 0.5 * np.dot(err, arm_dir), *hi)
    return np.linalg.norm(mouth - data.xpos[gc]), data, lift, arm, lo, hi


def test_seated_mouth_reachable(model):
    dist, data, lift, arm, lo, hi = _park_and_reach(model, "armchair")
    assert lo[0] <= data.ctrl[lift] <= lo[1]
    assert hi[0] <= data.ctrl[arm] <= hi[1]
    assert dist < 0.03


def test_bed_mouth_reachable(model):
    dist, *_ = _park_and_reach(model, "bed")
    assert dist < 0.03
