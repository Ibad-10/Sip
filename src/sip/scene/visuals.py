"""Realistic look for the house: textured floors/walls, Poly Haven CC0 furniture meshes
(visual only: contype 0, the physics keeps the simple colliders), lights with shadows and
cameras (room views that track the robot, the person's eyes).

Collider boxes that the meshes replace are moved to geom group 3 (not drawn by default)."""
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PROPS = ROOT / "assets" / "props"
TEX = ROOT / "assets" / "textures"
MESH = mujoco.mjtGeom.mjGEOM_MESH
BOX = mujoco.mjtGeom.mjGEOM_BOX
RGB = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
HIDDEN = 3

_done_tex: set[str] = set()


def _material(spec, name: str, png: Path | None, rgba=(1, 1, 1, 1), repeat=(1, 1), uniform=False,
              rough=0.6, spec_=0.2):
    m = spec.add_material(name=name)
    m.rgba = list(rgba)
    m.specular, m.shininess = spec_, 1 - rough
    if png is not None and png.exists():
        tname = f"tex_{name}"
        spec.add_texture(name=tname, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(png))
        tex = list(m.textures)
        tex[RGB] = tname
        m.textures = tex
        m.texrepeat = list(repeat)
        m.texuniform = uniform
    return name


def add_prop(spec, parent, asset: str, pos, yaw: float = 0.0, scale=(1, 1, 1), name: str | None = None,
             fit: tuple[float, float, float] | None = None):
    """Place a Poly Haven prop with its base centred at pos. fit=(x,y,z) scales it to that size."""
    man_path = PROPS / asset / "manifest.json"
    if not man_path.exists():
        return None
    man = json.loads(man_path.read_text())
    lo, hi = np.array(man["min"]), np.array(man["max"])
    size = hi - lo
    s = np.array(scale, float)
    if fit is not None:
        s = np.array([f / sz if f else 1.0 for f, sz in zip(fit, size)])
        if fit[2] is None or fit[2] == 0:
            s[2] = s[0]
    center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    name = name or asset
    b = parent.add_body(name=f"vis_{name}", pos=list(pos), quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    for k, part in enumerate(man["parts"]):
        mesh = f"m_{name}_{k}"
        spec.add_mesh(name=mesh, file=str(PROPS / asset / part["obj"]), scale=list(s),
                      refpos=list(center), inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        mat = _material(spec, f"mat_{name}_{k}", PROPS / asset / part["png"] if part["png"] else None,
                        rgba=tuple(part.get("rgba", (1, 1, 1, 1))),
                        rough=0.25 if "metal" in part.get("material", "") else 0.6,
                        spec_=0.6 if "metal" in part.get("material", "") else 0.2)
        b.add_geom(name=f"g_{name}_{k}", type=MESH, meshname=mesh, material=mat, contype=0,
                   conaffinity=0, group=1, mass=0)
    return b


def _hide(spec, names):
    for g in spec.geoms:
        if g.name in names:
            g.group = HIDDEN


def dress(spec, world) -> None:
    v = spec.visual
    v.quality.shadowsize, v.quality.offsamples = 4096, 4
    v.headlight.ambient, v.headlight.diffuse, v.headlight.specular = [0.32] * 3, [0.35] * 3, [0.05] * 3
    v.map.znear = 0.01
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.95, 0.94, 0.91], rgb2=[0.86, 0.84, 0.8],
                     width=256, height=256)

    floor = _material(spec, "floor_laminate", TEX / "laminate_floor_02.png", repeat=(1.2, 1.2), uniform=True, rough=0.5)
    tiles = _material(spec, "floor_tiles", TEX / "floor_tiles_06.png", repeat=(1.5, 1.5), uniform=True, rough=0.3)
    wall = _material(spec, "wall_plaster", TEX / "beige_wall_001.png", repeat=(0.5, 0.5), uniform=True, rough=0.9)
    worktop = _material(spec, "worktop", TEX / "kitchen_wood.png", repeat=(2, 1), rough=0.4)
    carpet = _material(spec, "carpet", TEX / "dirty_carpet.png", repeat=(1, 1), uniform=True, rough=1.0,
                       rgba=(0.75, 0.75, 0.85, 1))
    for g in spec.geoms:
        if g.name == "floor":
            g.material, g.rgba = floor, [1, 1, 1, 1]
        elif g.name.startswith("wall_"):
            g.material, g.rgba = wall, [1, 1, 1, 1]
    # kitchen tiles and bedroom rug as thin visual decals
    world.add_geom(name="kitchen_tiles", type=BOX, pos=[-2.0, 1.9, 0.001], size=[1.95, 1.05, 0.001],
                   material=tiles, contype=0, conaffinity=0, group=1)
    world.add_geom(name="bedroom_rug", type=BOX, pos=[1.6, -1.9, 0.002], size=[0.9, 0.6, 0.002],
                   material=carpet, contype=0, conaffinity=0, group=1)

    # kitchen: base cabinets under a wooden worktop (collider: counter_top, now hidden)
    _hide(spec, {"counter_top", "armchair_seat", "armchair_back", "side_table_top"})
    counter = spec.body("kitchen_counter")
    add_prop(spec, counter, "facilitySystem_kitchenette_module_a01", (0, 0.0, 0), yaw=-np.pi / 2,
             fit=(0.6, 2.2, None), name="kitchenette")
    # glass: SimReady 8 oz glass as the visual, with water inside; the cylinder collider is hidden
    _hide(spec, {"glass_geom"})
    gl = spec.body("glass")
    add_prop(spec, gl, "dishware_glass_8oz_a01", (0, 0, -0.06), fit=(0.07, 0.07, 0.12), name="glass_vis")
    gl.add_geom(name="water", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, -0.015], size=[0.03, 0.04, 0],
                rgba=[0.55, 0.75, 0.95, 0.55], contype=0, conaffinity=0, group=1, mass=0)
    add_prop(spec, world, "dishware_mug_a01", (-3.0, 2.62, 0.9), name="mug")
    add_prop(spec, world, "electric_stove", (-1.1, 2.72, 0), yaw=np.pi, name="stove")
    world.add_geom(name="stove_col", type=BOX, pos=[-1.1, 2.72, 0.43], size=[0.26, 0.26, 0.43],
                   group=HIDDEN)
    add_prop(spec, world, "tea_set_01", (-3.3, 2.7, 0.9), yaw=np.pi, fit=(0.55, None, None), name="tea_set")
    add_prop(spec, world, "wooden_bowl_01", (-2.05, 2.75, 0.9), name="bowl")
    add_prop(spec, world, "food_apple_01", (-2.05, 2.75, 0.93), name="apple")
    add_prop(spec, world, "potted_plant_04", (-3.75, 1.0, 0), fit=(0.35, None, None), name="kitchen_plant")

    # living room
    add_prop(spec, spec.body("armchair"), "ArmChair_01", (0, 0.03, 0), yaw=0.0, fit=(0.8, 0.78, None))
    add_prop(spec, spec.body("side_table"), "side_table_01", (0, 0, 0), fit=(0.42, 0.42, 0.6))
    add_prop(spec, world, "standing_picture_frame_01", (2.3, 2.5, 0.6), yaw=-0.3, name="photo")
    add_prop(spec, world, "potted_plant_04", (2.42, 2.38, 0.6), name="table_plant")
    add_prop(spec, world, "Sofa_01", (1.15, 2.62, 0), yaw=0.0, name="sofa")
    world.add_geom(name="sofa_col", type=BOX, pos=[1.15, 2.62, 0.4], size=[0.8, 0.34, 0.4], group=HIDDEN)
    add_prop(spec, world, "throw_pillows_01", (1.15, 2.7, 0.42), name="pillows")
    add_prop(spec, world, "wall_clock", (2.0, 2.94, 1.7), name="clock")
    add_prop(spec, world, "desk_lamp_arm_01", (3.75, 2.75, 0), name="reading_lamp")

    # bedroom
    add_prop(spec, spec.body("bed"), "old_bed_frame", (0, 0, 0), fit=(1.0, 1.9, None))
    add_prop(spec, world, "painted_wooden_nightstand", (2.65, -2.7, 0), yaw=np.pi / 2, name="nightstand")
    world.add_geom(name="nightstand_col", type=BOX, pos=[2.65, -2.7, 0.31], size=[0.25, 0.25, 0.31], group=HIDDEN)
    add_prop(spec, world, "round_spectacles", (2.65, -2.65, 0.63), name="spectacles")
    add_prop(spec, world, "drawer_cabinet", (0.65, -2.7, 0), yaw=0.0, fit=(1.0, 0.45, 1.6), name="wardrobe")
    world.add_geom(name="wardrobe_col", type=BOX, pos=[0.65, -2.7, 0.8], size=[0.5, 0.23, 0.8], group=HIDDEN)

    # hall
    add_prop(spec, world, "wooden_bookshelf_worn", (0.27, -1.75, 0), yaw=-np.pi / 2, fit=(1.0, 0.4, 1.6),
             name="bookshelf")
    world.add_geom(name="bookshelf_col", type=BOX, pos=[0.27, -1.75, 0.8], size=[0.2, 0.5, 0.8], group=HIDDEN)

    # lights (one per room, shadows on)
    for name, pos in {"light_kitchen": (-2.0, 1.9, 2.3), "light_living": (2.0, 1.9, 2.3),
                      "light_bedroom": (2.0, -1.9, 2.3), "light_hall": (0.0, 0.0, 2.3)}.items():
        world.add_light(name=name, pos=list(pos), dir=[0, 0, -1], diffuse=[0.45, 0.43, 0.4],
                        specular=[0.1] * 3, castshadow=True, cutoff=80, exponent=1)

    # cameras: room views that track the robot, plus an overview
    track = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    for name, pos in {"cam_kitchen": (-3.7, 0.95, 2.1), "cam_living": (3.8, 0.95, 2.1),
                      "cam_bedroom": (3.8, -0.95, 2.1), "cam_hall": (-3.8, 0.0, 1.9)}.items():
        world.add_camera(name=name, pos=list(pos), mode=track, targetbody="base_link", fovy=70)
    world.add_camera(name="cam_overview", pos=[0, -6.0, 7.5], xyaxes=[1, 0, 0, 0, 0.78, 0.62], fovy=50)
