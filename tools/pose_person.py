"""Pose the rigged Sketchfab grandma (CC-BY-4.0, wolfgar74) into a seated pose with linear blend
skinning, split off the head (so the sim can turn it) and write OBJ + texture for MuJoCo.

  uv run python tools/pose_person.py
Output: assets/props/grandma/{body.obj, head.obj, base.png, manifest.json}
Frame: metres, z up, x forward (where she looks), origin = mouth (the sim's mouth site).
"""
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets/sketchfab/grandma"
OUT = ROOT / "assets/props/grandma"
MOUTH_HEIGHT = 1.05  # sim: seat 0.50 + head 0.60 - mouth 0.045 above the floor
CT = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def reader(g, binary):
    def acc(i):
        a = g["accessors"][i]
        bv = g["bufferViews"][a["bufferView"]]
        dt, n = np.dtype(CT[a["componentType"]]), NC[a["type"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride", dt.itemsize * n)
        raw = np.frombuffer(binary, np.uint8, count=stride * (a["count"] - 1) + dt.itemsize * n, offset=off)
        rows = np.lib.stride_tricks.as_strided(raw, (a["count"], dt.itemsize * n), (stride, 1))
        out = rows.copy().view(dt).reshape(a["count"], n)
        if a.get("normalized"):
            out = out.astype(np.float64) / np.iinfo(dt).max
        return out
    return acc


def rot_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def main(arm_dirs=None, tag: str = ""):
    """arm_dirs: optional {'R_Upperarm': (f,u,l), 'R_Forearm': (f,u,l)} from a retargeted animation frame."""
    g = json.loads((SRC / "scene.gltf").read_text())
    acc = reader(g, (SRC / g["buffers"][0]["uri"]).read_bytes())
    skin = g["skins"][0]
    joints = skin["joints"]
    names = [g["nodes"][j].get("name", "") for j in joints]
    ibm = acc(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)  # column-major
    parent = {}
    for i, n in enumerate(g["nodes"]):
        for c in n.get("children", []):
            parent[c] = i
    jidx = {j: k for k, j in enumerate(joints)}
    # world-space bind pose from the scene graph; skin matrix S_j = G_j @ IBM_j
    Gw = {}

    def world(n):
        if n not in Gw:
            pn = parent.get(n)
            Gw[n] = (world(pn) if pn is not None else np.eye(4)) @ _local(g["nodes"][n])
        return Gw[n]
    G = {k: world(j) for k, j in enumerate(joints)}
    S = np.stack([G[k] @ ibm[k] for k in range(len(joints))])
    piv = np.stack([G[k][:3, 3] for k in range(len(joints))])

    def J(name):
        return next(k for k, n in enumerate(names) if n.startswith("CC_Base_" + name + "_"))

    up = piv[J("Head")] - piv[J("Pelvis")]
    up /= np.linalg.norm(up)
    left = piv[J("L_Thigh")] - piv[J("R_Thigh")]
    left -= (left @ up) * up
    left /= np.linalg.norm(left)
    fwd = np.cross(left, up)
    eyes = (piv[J("L_Eye")] + piv[J("R_Eye")]) / 2 - piv[J("Head")]
    if eyes @ fwd < 0:
        fwd = -fwd

    def d(f=0.0, u=0.0, l=0.0):
        return f * fwd + u * up + l * left

    # target bone directions for a relaxed seated pose (bone = joint -> its child joint)
    pose = {
        "L_Thigh": ("L_Calf", d(1, -0.12, 0.08)), "R_Thigh": ("R_Calf", d(1, -0.12, -0.08)),
        "L_Calf": ("L_Foot", d(0.2, -1)), "R_Calf": ("R_Foot", d(0.2, -1)),
        "L_Upperarm": ("L_Forearm", d(0.25, -1, 0.12)), "R_Upperarm": ("R_Forearm", d(0.25, -1, -0.12)),
        "L_Forearm": ("L_Hand", d(1, -0.25, -0.35)), "R_Forearm": ("R_Forearm", None),
    }
    pose["R_Forearm"] = ("R_Hand", d(1, -0.25, 0.35))
    for bone, (f_, u_, l_) in (arm_dirs or {}).items():
        pose[bone] = (pose[bone][0], d(f_, u_, l_))
    D = {}
    order = sorted(range(len(joints)), key=lambda k: len(_chain(joints[k], parent)))
    for k in order:
        p = parent.get(joints[k])
        Dp = D.get(jidx.get(p), np.eye(4)) if p in jidx else np.eye(4)
        Dk = Dp.copy()
        key = names[k][len("CC_Base_"):].rsplit("_", 1)[0] if names[k].startswith("CC_Base_") else None
        if key in pose:
            child, target = pose[key]
            pj = Dp[:3, :3] @ piv[k] + Dp[:3, 3]
            pc = Dp[:3, :3] @ piv[J(child)] + Dp[:3, 3]
            R = rot_between(pc - pj, target)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, pj - R @ pj
            Dk = T @ Dp
        D[k] = Dk
    Dm = np.stack([D[k] for k in range(len(joints))])
    Sk = np.einsum('kij,kjl->kil', Dm, S)

    head_set = {k for k in range(len(joints)) if J("Head") in _chain_idx(k, joints, parent, jidx)}
    parts = {"body": ([], [], []), "head": ([], [], [])}
    for node in g["nodes"]:
        if "mesh" not in node or g["meshes"][node["mesh"]].get("name", "").startswith("Cube_002"):
            continue  # Cube_002 is the bench
        for prim in g["meshes"][node["mesh"]]["primitives"]:
            a = prim["attributes"]
            V = acc(a["POSITION"]).astype(np.float64)
            UV = acc(a["TEXCOORD_0"]).astype(np.float64)
            Jn = acc(a["JOINTS_0"]).astype(int)
            W = acc(a["WEIGHTS_0"]).astype(np.float64)
            W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
            M = np.einsum("vk,vkij->vij", W, Sk[Jn])
            P = np.einsum("vij,vj->vi", M[:, :3, :3], V) + M[:, :3, 3]
            F = acc(prim["indices"]).reshape(-1, 3).astype(int)
            dom = Jn[np.arange(len(V)), W.argmax(1)]
            is_head = np.isin(dom, list(head_set))
            fh = is_head[F].sum(1) >= 2
            for name, mask in (("head", fh), ("body", ~fh)):
                vs, uvs, fs = parts[name]
                base = sum(len(x) for x in vs)
                vs.append(P)
                uvs.append(UV)
                fs.append(F[mask] + base)

    # to the sim frame: metres, x fwd, y left, z up; origin at the mouth
    B = np.stack([fwd, left, up])
    head_piv = Dm[J("Head")][:3, :3] @ piv[J("Head")] + Dm[J("Head")][:3, 3]
    allv = np.vstack([np.vstack(parts[p][0]) for p in parts])
    height_raw = (allv @ up).max() - (allv @ up).min()
    s = 1.25 / height_raw  # provisional; rescaled below so the feet touch the floor
    mouth_raw = head_piv + _mouth(piv, J, fwd, up, Dm)
    feet = ((np.vstack(parts["body"][0]) - mouth_raw) @ up).min() * s
    s *= MOUTH_HEIGHT / -feet
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    for name, (vs, uvs, fs) in parts.items():
        if tag and name != "body":
            continue
        V = (np.vstack(vs) - mouth_raw) @ B.T * s
        UV = np.vstack(uvs)
        F = np.vstack(fs)
        used = np.unique(F)
        remap = -np.ones(len(V), int)
        remap[used] = np.arange(len(used))
        V, UV, F = V[used], UV[used], remap[F]
        with open(OUT / f"{name}{tag}.obj", "w") as f:
            f.write("".join(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in V))
            f.write("".join(f"vt {u:.5f} {1 - v:.5f}\n" for u, v in UV))
            f.write("".join(f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}\n" for a, b, c in F))
        meta[name] = {"faces": int(len(F)), "min": V.min(0).round(3).tolist(), "max": V.max(0).round(3).tolist()}
    if tag:
        return
    hp = (head_piv - mouth_raw) @ B.T * s
    meta["head_pivot"] = hp.round(4).tolist()
    tex = g["images"][g["textures"][g["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"]]["source"]]["uri"]
    shutil.copy(SRC / tex, OUT / "base.png")
    (OUT / "manifest.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def _mouth(piv, J, fwd, up, Dm):
    """Mouth point: between the upper and lower jaw joints, pushed forward to the lips."""
    k = J("UpperJaw")
    p = Dm[k][:3, :3] @ piv[k] + Dm[k][:3, 3]
    h = Dm[J("Head")][:3, :3] @ piv[J("Head")] + Dm[J("Head")][:3, 3]
    return p - h


def _local(n):
    if "matrix" in n:
        return np.array(n["matrix"], float).reshape(4, 4).T
    x, y, z, w = n.get("rotation", [0, 0, 0, 1])
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3] = R * np.array(n.get("scale", [1, 1, 1]))
    T[:3, 3] = n.get("translation", [0, 0, 0])
    return T


def _chain(node, parent):
    out = []
    while node in parent:
        node = parent[node]
        out.append(node)
    return out


def _chain_idx(k, joints, parent, jidx):
    return {k} | {jidx[n] for n in _chain(joints[k], parent) if n in jidx}


def retarget_frames(frames=(181, 205, 217, 229, 241, 289, 337, 349, 361, 385)):
    """Mixamo 'Sitting Drinking' right-arm motion -> our grandma's rig, one OBJ per keyframe."""
    J = json.loads((ROOT / "assets/mixamo/sitting_drinking_bones.json").read_text())
    by_f = {fr["f"]: {k: np.array(v) for k, v in fr["pos"].items()} for fr in J["frames"]}
    for k, f in enumerate(frames):
        p = by_f[min(by_f, key=lambda x: abs(x - f))]
        up = p["Head"] - p["Hips"]
        up /= np.linalg.norm(up)
        left = p["LeftArm"] - p["RightArm"]
        left -= (left @ up) * up
        left /= np.linalg.norm(left)
        fwd = np.cross(left, up)
        if (p["LeftUpLeg"] + p["RightUpLeg"] - 2 * p["Hips"]) @ fwd < 0 and (p["RightHand"] - p["Hips"]) @ fwd < 0:
            fwd = -fwd

        def comp(v):
            v = v / np.linalg.norm(v)
            return float(v @ fwd), float(v @ up), float(v @ left)
        # Mixamo holds a cup out; for tablets the hand must reach the lips, so blend the forearm
        # toward the mouth in proportion to how close Mixamo's hand comes to the head.
        mouth = p["Head"] + 16 * fwd + 4 * up
        w = float(np.clip((60 - np.linalg.norm(p["RightHand"] - p["Head"])) / 28, 0, 1))
        fa = p["RightHand"] - p["RightForeArm"]
        to_mouth = mouth - p["RightForeArm"]
        fa = (1 - w) * fa / np.linalg.norm(fa) + w * to_mouth / np.linalg.norm(to_mouth)
        ua = p["RightForeArm"] - p["RightArm"]
        ua = ua / np.linalg.norm(ua) + w * 0.6 * fwd     # elbow a little more forward
        main({"R_Upperarm": comp(ua), "R_Forearm": comp(fa)}, tag=f"_drink{k}")
    (OUT / "drink_frames.json").write_text(json.dumps({"count": len(frames), "mixamo_frames": list(frames)}))
    print("wrote", len(frames), "drinking keyframes")


if __name__ == "__main__":
    import sys
    if "--drink" in sys.argv:
        retarget_frames()
    else:
        main()
