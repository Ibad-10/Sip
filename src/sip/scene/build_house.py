"""Build the four-area house MJCF around the Menagerie Stretch 3.

Layout (metres, origin at house centre, 8 x 6 m):
  kitchen  x[-4,0]  y[0.8,3]    doors: hall (x=-2), living (x=0, y=2)
  living   x[0,4]   y[0.8,3]    door: hall (x=2)
  bedroom  x[0,4]   y[-3,-0.8]  door: hall (x=2)
  hall     x[-4,4]  y[-0.8,0.8] (plus a closed store room x[-4,0] y[-3,-0.8])
Reach test values: park standoff 0.45 m (arm extension budget); seat pelvis z 0.50, bed pelvis z 0.55.
"""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STRETCH = ROOT / "third_party/mujoco_menagerie/hello_robot_stretch_3/stretch.xml"

ROOMS: dict[str, tuple[float, float]] = {
    "kitchen": (-2.0, 1.9),
    "living": (2.0, 1.9),
    "bedroom": (2.0, -1.9),
    "hall": (-1.0, 0.0),
}

# where -> (pelvis xyz, facing yaw, approach angle from facing in deg). Person faces their +x axis.
SEATS = {
    "armchair": ((3.0, 2.35, 0.50), -np.pi / 2, 55.0),
    "bed": ((3.15, -2.0, 0.55), np.pi, -40.0),
}

WALL_H, WALL_T = 2.4, 0.1
GRASP_OFFSET = 0.44  # base centre to grasp centre along -y with arm retracted

BOX, CYL, CAP, SPH, PLANE = (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER,
                             mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE,
                             mujoco.mjtGeom.mjGEOM_PLANE)


def _box(body, name, center, half, rgba=(0.8, 0.8, 0.8, 1), **kw):
    return body.add_geom(name=name, type=BOX, pos=list(center), size=list(half), rgba=list(rgba), **kw)


def _wall(world, name, x0, y0, x1, y1):
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hx = max(abs(x1 - x0) / 2, WALL_T / 2)
    hy = max(abs(y1 - y0) / 2, WALL_T / 2)
    _box(world, name, (cx, cy, WALL_H / 2), (hx, hy, WALL_H / 2), (0.93, 0.91, 0.86, 1))


def _walls(world):
    # outer
    _wall(world, "wall_n", -4, 3, 4, 3)
    _wall(world, "wall_s", -4, -3, 4, -3)
    _wall(world, "wall_w", -4, -3, -4, 3)
    _wall(world, "wall_e", 4, -3, 4, 3)
    # y = 0.8 line: doors 1.0 m wide at x=-2 (kitchen) and x=2 (living)
    _wall(world, "wall_k1", -4, 0.8, -2.5, 0.8)
    _wall(world, "wall_k2", -1.5, 0.8, 1.5, 0.8)
    _wall(world, "wall_k3", 2.5, 0.8, 4, 0.8)
    # y = -0.8 line: door at x=2 (bedroom); store room closed
    _wall(world, "wall_b1", -4, -0.8, 1.5, -0.8)
    _wall(world, "wall_b2", 2.5, -0.8, 4, -0.8)
    # kitchen | living at x=0, door at y in [1.5, 2.5]
    _wall(world, "wall_kl1", 0, 0.8, 0, 1.5)
    _wall(world, "wall_kl2", 0, 2.5, 0, 3)
    # bedroom | store at x=0
    _wall(world, "wall_bs", 0, -3, 0, -0.8)


def _furniture(world):
    wood, fabric, white = (0.55, 0.38, 0.24, 1), (0.35, 0.45, 0.6, 1), (0.95, 0.95, 0.95, 1)
    b = world.add_body(name="kitchen_counter", pos=[-2.5, 2.7, 0])
    _box(b, "counter_top", (0, 0, 0.45), (1.1, 0.3, 0.45), white)
    b = world.add_body(name="side_table", pos=[2.35, 2.45, 0])
    _box(b, "side_table_top", (0, 0, 0.3), (0.2, 0.2, 0.3), wood)
    b = world.add_body(name="armchair", pos=[3.0, 2.45, 0])
    _box(b, "armchair_seat", (0, 0, 0.225), (0.38, 0.35, 0.225), fabric)
    _box(b, "armchair_back", (0, 0.3, 0.6), (0.38, 0.08, 0.4), fabric)
    b = world.add_body(name="bed", pos=[3.45, -1.95, 0])
    _box(b, "bed_base", (0, 0, 0.25), (0.5, 0.95, 0.25), (0.9, 0.9, 1.0, 1))
    b = world.add_body(name="chair_hall", pos=[-3.6, -0.5, 0])
    b.add_freejoint(name="chair_hall_free")
    _box(b, "chair_seat", (0, 0, 0.45), (0.22, 0.22, 0.03), wood, mass=3.0)
    _box(b, "chair_back", (0, 0.2, 0.7), (0.22, 0.02, 0.25), wood, mass=1.0)
    for i, (dx, dy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        _box(b, f"chair_leg{i}", (0.19 * dx, 0.19 * dy, 0.21), (0.02, 0.02, 0.21), wood, mass=0.3)


def _items(world):
    b = world.add_body(name="glass", pos=[-2.6, 2.6, 0.96])
    b.add_freejoint(name="glass_free")
    b.add_geom(name="glass_geom", type=CYL, size=[0.035, 0.06, 0], rgba=[0.6, 0.85, 1.0, 0.7],
               mass=0.25, friction=[1.5, 0.05, 0.01])
    # straw leans out along the glass's local +x; the robot turns the glass so +x faces the arm
    b.add_geom(name="straw", type=CAP, fromto=[0, 0, 0.0, 0.07, 0, 0.14], size=[0.004, 0, 0],
               rgba=[1, 0.3, 0.3, 1], mass=0.001)       # real collider: straw-face contact is measured
    b.add_site(name="straw_tip", pos=[0.07, 0, 0.14], size=[0.006, 0, 0], group=4)
    # jug of water (collider cylinder; visual added in visuals.dress) and a straw holder with a straw
    b = world.add_body(name="jug", pos=[-1.72, 2.62, 1.0])
    b.add_freejoint(name="jug_free")
    b.add_geom(name="jug_geom", type=CYL, size=[0.065, 0.1, 0], rgba=[0.7, 0.72, 0.75, 1], mass=0.8,
               friction=[1.5, 0.05, 0.01])
    b = world.add_body(name="pour_stream", pos=[0, 0, -2], mocap=True)  # visual water stream while pouring
    b.add_geom(name="pour_stream_geom", type=CYL, size=[0.006, 0.05, 0], rgba=[0.6, 0.8, 1.0, 0.0],
               contype=0, conaffinity=0, group=1)
    b = world.add_body(name="straw_cup", pos=[-3.15, 2.62, 0.95])
    b.add_geom(name="straw_cup_geom", type=CYL, size=[0.03, 0.05, 0], rgba=[0.95, 0.95, 0.93, 1])
    b = world.add_body(name="straw2", pos=[-3.15, 2.62, 1.02])
    b.add_freejoint(name="straw2_free")
    b.add_geom(name="straw2_geom", type=CAP, size=[0.004, 0.1, 0], rgba=[1, 0.3, 0.3, 1], mass=0.005,
               contype=0, conaffinity=0)
    b.add_geom(name="straw2_grip", type=BOX, size=[0.012, 0.012, 0.03], pos=[0, 0, 0.05], rgba=[1, 0.3, 0.3, 0],
               mass=0.005, group=3)
    b = world.add_body(name="organiser", pos=[-2.2, 2.66, 0.93])   # weekly pill organiser (pharmacist-filled)
    b.add_freejoint(name="organiser_free")
    _box(b, "organiser_geom", (0, 0, 0), (0.1, 0.058, 0.023), (0.1, 0.55, 0.35, 1), mass=0.3)
    for i, (dx, dy) in enumerate(((-0.012, -0.008), (0.006, 0.01), (0.014, -0.012))):   # today's tablets
        b.add_geom(name=f"pill{i}", type=CAP, pos=[0.07 + dx, dy, 0.03], size=[0.006, 0.006, 0],
                   euler=[0, 1.5708, 0.6 * i], rgba=[0.98, 0.45, 0.1, 1], contype=0, conaffinity=0, mass=0)


def _person(spec, world):
    skin, cloth = (0.87, 0.72, 0.6, 1), (0.3, 0.5, 0.35, 1)
    p = world.add_body(name="person", pos=[3.0, 2.35, 0.5], mocap=True)
    p.add_geom(name="torso", type=CAP, fromto=[0, 0, 0.05, 0, 0, 0.45], size=[0.12, 0, 0], rgba=list(cloth),
               contype=1, conaffinity=1)
    for s in (-1, 1):
        p.add_geom(name=f"thigh{s}", type=CAP, fromto=[0, 0.1 * s, 0.02, 0.42, 0.1 * s, 0.02],
                   size=[0.07, 0, 0], rgba=list(cloth))
        p.add_geom(name=f"shin{s}", type=CAP, fromto=[0.42, 0.1 * s, 0.0, 0.45, 0.1 * s, -0.42],
                   size=[0.05, 0, 0], rgba=list(skin))
        p.add_geom(name=f"uarm{s}", type=CAP, fromto=[0, 0.2 * s, 0.42, 0.05, 0.22 * s, 0.15],
                   size=[0.045, 0, 0], rgba=list(cloth))
    h = p.add_body(name="person_head", pos=[0.06, 0, 0.6])  # head slightly forward of the chest
    h.add_joint(name="head_yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], range=[-1.2, 1.2],
                damping=2.0)
    h.add_geom(name="head", type=SPH, size=[0.1, 0, 0], rgba=list(skin), mass=4.0)
    h.add_geom(name="face_plate", type=BOX, pos=[0.085, 0, 0], size=[0.012, 0.07, 0.08],
               rgba=[0.95, 0.8, 0.7, 1], contype=0, conaffinity=0, mass=0.01)
    h.add_camera(name="person_eyes", pos=[0.09, 0, 0.03], xyaxes=[0, -1, 0, 0, 0, 1], fovy=75)
    h.add_site(name="mouth", pos=[0.1, 0, -0.045], size=[0.01, 0, 0], rgba=[0.8, 0.2, 0.2, 1], group=4)
    a = spec.add_actuator(name="head_yaw_act", target="head_yaw", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    a.set_to_position(kp=20.0)
    a.ctrlrange = [-1.2, 1.2]


def build_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(STRETCH))
    spec.modelname = "sip_house"
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
    # Menagerie wheel damping (30) with a 100 N.m force limit caps the base near 0.1 m/s;
    # real Stretch drives ~0.3 m/s, so soften it.
    for j in spec.joints:
        if j.name in ("joint_left_wheel", "joint_right_wheel"):
            j.damping = [2.0, 0.0, 0.0]
    for a in spec.actuators:
        if a.name in ("left_wheel_vel", "right_wheel_vel"):
            a.ctrlrange = [-20.0, 20.0]  # gear 3: 20 -> 6.7 rad/s -> 0.33 m/s
            a.gainprm[0], a.biasprm[2] = 30.0, -30.0  # kv 100 rings and skids in turns
    world = spec.worldbody
    world.add_geom(name="floor", type=PLANE, size=[5, 4, 0.1], rgba=[0.75, 0.7, 0.62, 1])
    world.add_light(name="sun", pos=[0, 0, 5], dir=[0, 0, -1], diffuse=[0.35, 0.35, 0.35], castshadow=False)
    for name, (x, y) in ROOMS.items():
        world.add_site(name=name, pos=[x, y, 0.01], size=[0.05, 0, 0], rgba=[0, 0, 0, 0], group=4)
    _walls(world)
    _furniture(world)
    _items(world)
    _person(spec, world)
    from sip.scene.visuals import dress
    dress(spec, world)
    # graspable objects never collide with the robot's own links (they ride in/over the gripper), but
    # they DO collide with the person and the furniture, so contact-force safety checks are real
    tmp = spec.compile()
    root = tmp.body("base_link").id
    robot_bodies = [tmp.body(i).name for i in range(tmp.nbody)
                    if _descends(tmp, i, root) and tmp.body(i).name]
    for item in ("glass", "organiser", "jug", "straw2"):
        for rb in robot_bodies:
            spec.add_exclude(bodyname1=item, bodyname2=rb)
    for item in ("glass", "organiser", "jug", "straw2"):
        spec.add_equality(name=f"grasp_weld_{item}", type=mujoco.mjtEq.mjEQ_WELD,
                          objtype=mujoco.mjtObj.mjOBJ_BODY, name1="link_grasp_center", name2=item,
                          active=False)
    return spec


def build_house(path: str = "assets/house.xml") -> str:
    out = (ROOT / path) if not Path(path).is_absolute() else Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    spec = build_spec()
    spec.compile()
    xml = spec.to_xml()
    asset_dir = Path("..") / STRETCH.parent.relative_to(ROOT) / "assets"
    rel = asset_dir.as_posix() + "/"
    xml = xml.replace('meshdir="assets"', f'meshdir="{rel}"').replace('texturedir="assets"', f'texturedir="{rel}"')
    out.write_text(xml, encoding="utf-8")
    return str(out)


def room_at(x: float, y: float) -> str:
    if y > 0.8:
        return "kitchen" if x < 0 else "living"
    if y < -0.8:
        return "bedroom" if x > 0 else "store"
    return "hall"


def _descends(model, i: int, root: int) -> bool:
    while i > 0:
        if i == root:
            return True
        i = model.body_parentid[i]
    return False


def seat_person(model, data, where: str) -> float:
    """Place the person mocap body at a seat ('armchair' or 'bed'); returns facing yaw."""
    (x, y, z), yaw, _ = SEATS[where]
    mid = model.body("person").mocapid[0]
    data.mocap_pos[mid] = [x, y, z]
    data.mocap_quat[mid] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    return yaw


def park_pose_for(mouth: np.ndarray, standoff: float, facing: float,
                  approach_deg: float = 55.0) -> tuple[float, float, float]:
    """Base pose whose -y axis points at the mouth, approaching approach_deg off the person's
    facing direction, mouth at GRASP_OFFSET + standoff horizontally."""
    ang = facing + np.deg2rad(approach_deg)
    dist = GRASP_OFFSET + standoff
    bx, by = mouth[0] + dist * np.cos(ang), mouth[1] + dist * np.sin(ang)
    ux, uy = mouth[0] - bx, mouth[1] - by
    yaw = float(np.arctan2(ux, -uy))
    return float(bx), float(by), yaw
