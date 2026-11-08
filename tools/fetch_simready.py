"""Download an NVIDIA SimReady asset's USD files only (no multi-GB textures), flatten its meshes
(world transforms, instance proxies), merge them per material, and write MuJoCo-ready OBJ parts
with a flat colour per material taken from the material (or its name). CC-BY-4.0: credit NVIDIA.

uv run python tools/fetch_simready.py facilitySystem_kitchenette_module_a01 ...
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "props"
API = "https://huggingface.co/api/datasets/nvidia/simready-assets/tree/main/"
RAW = "https://huggingface.co/datasets/nvidia/simready-assets/resolve/main/"
H = {"User-Agent": "sip-hackathon-asset-fetch"}
MAX_FACES = 60000
SKIP = ("instances_newton", "instances_physx", "/Newton/", "/PhysX/", "source_assets")
COLORS = [("stainless", (0.78, 0.79, 0.8)), ("steel", (0.72, 0.73, 0.75)), ("chrome", (0.85, 0.86, 0.88)),
          ("aluminum", (0.8, 0.8, 0.82)), ("bronze", (0.62, 0.46, 0.27)), ("gold", (0.8, 0.65, 0.3)),
          ("black", (0.07, 0.07, 0.08)), ("white", (0.93, 0.93, 0.92)), ("gray", (0.5, 0.5, 0.52)),
          ("grey", (0.5, 0.5, 0.52)), ("rubber", (0.1, 0.1, 0.1)), ("wood", (0.55, 0.38, 0.24)),
          ("oak", (0.62, 0.47, 0.3)), ("walnut", (0.38, 0.25, 0.16)), ("fabric", (0.42, 0.47, 0.56)),
          ("leather", (0.32, 0.2, 0.13)), ("glass", (0.8, 0.9, 1.0)), ("blue", (0.25, 0.4, 0.7)),
          ("red", (0.7, 0.15, 0.12)), ("green", (0.25, 0.5, 0.3)), ("beige", (0.85, 0.78, 0.65)),
          ("plastic", (0.6, 0.6, 0.62)), ("metal", (0.65, 0.66, 0.68)), ("ceramic", (0.92, 0.92, 0.9))]


def _list(asset: str) -> list[dict]:
    return json.loads(urllib.request.urlopen(urllib.request.Request(API + asset + "?recursive=true", headers=H),
                                             timeout=60).read())


def _get(path: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(RAW + path, headers=H), timeout=600) as r:
        dest.write_bytes(r.read())


def _color(mat_prim) -> tuple[float, float, float, float]:
    from pxr import UsdShade
    if mat_prim is not None and mat_prim.IsValid():
        for p in mat_prim.GetAllDescendants() if hasattr(mat_prim, "GetAllDescendants") else []:
            sh = UsdShade.Shader(p)
            for name in ("diffuseColor", "diffuse_color_constant", "base_color", "albedo_add"):
                inp = sh.GetInput(name) if sh else None
                if inp and inp.Get() is not None:
                    c = inp.Get()
                    try:
                        return (float(c[0]), float(c[1]), float(c[2]), 1.0)
                    except Exception:
                        pass
        key = str(mat_prim.GetPath()).lower()
        for p in mat_prim.GetAllDescendants() if hasattr(mat_prim, "GetAllDescendants") else []:
            for a in p.GetAttributes():
                v = a.Get()
                if isinstance(v, str) or hasattr(v, "path"):
                    key += " " + str(v).lower()
        for word, c in COLORS:
            if word in key:
                return (*c, 0.35 if word == "glass" else 1.0)
    return (0.7, 0.7, 0.7, 1.0)


def fetch(asset: str) -> dict:
    from pxr import Usd, UsdGeom, UsdShade
    files = [f for f in _list(asset) if f["type"] == "file" and f["path"].endswith((".usd", ".usda", ".usdc"))
             and not any(s in f["path"] for s in SKIP)]
    raw = OUT / asset / "raw"
    for f in files:
        _get(f["path"], raw / f["path"])
    roots = [f["path"] for f in files if f["path"].count("/") == 2 and "/simready_usd/" in f["path"]]
    stage = Usd.Stage.Open(str(raw / roots[0]))
    up = UsdGeom.GetStageUpAxis(stage)
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    cache = UsdGeom.XformCache()
    groups: dict[str, dict] = {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
            continue
        purpose = UsdGeom.Imageable(prim).ComputePurpose()
        if purpose in ("guide", "proxy"):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts:
            continue
        M = np.array(cache.GetLocalToWorldTransform(prim)).T  # USD row-vector -> column
        P = np.c_[np.array(pts, float), np.ones(len(pts))] @ M.T
        P = P[:, :3] * mpu
        if up == "Y":
            P = P[:, [0, 2, 1]] * np.array([1, -1, 1])
        tris, k = [], 0
        idx = np.array(idx)
        for c in counts:
            for j in range(1, c - 1):
                tris.append((idx[k], idx[k + j], idx[k + j + 1]))
            k += c
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        key = str(mat.GetPath()) if mat else "none"
        g = groups.setdefault(key, {"verts": [], "faces": [], "rgba": _color(mat.GetPrim() if mat else None)})
        off = sum(len(v) for v in g["verts"])
        g["verts"].append(P)
        g["faces"].append(np.array(tris) + off)
    parts, allv = [], []
    total = sum(sum(len(f) for f in g["faces"]) for g in groups.values())
    keep = min(1.0, MAX_FACES / max(total, 1))
    for k, (key, g) in enumerate(groups.items()):
        V, F = np.vstack(g["verts"]), np.vstack(g["faces"])
        if keep < 1.0 and len(F) > 200:
            import fast_simplification
            V, F = fast_simplification.simplify(V.astype(np.float32), F.astype(np.int32), target_reduction=1 - keep)
        allv.append(V)
        obj = OUT / asset / f"part{k}.obj"
        with open(obj, "w") as f:
            f.writelines(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in V)
            f.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in F)
        parts.append({"obj": obj.name, "png": None, "rgba": list(g["rgba"]), "faces": int(len(F)),
                      "material": key.split("/")[-1]})
    allv = np.vstack(allv)
    man = {"asset": asset, "parts": parts, "min": allv.min(0).tolist(), "max": allv.max(0).tolist(),
           "license": "CC-BY-4.0 (NVIDIA SimReady)"}
    (OUT / asset / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


if __name__ == "__main__":
    for a in sys.argv[1:]:
        try:
            m = fetch(a)
            size = np.array(m["max"]) - np.array(m["min"])
            print(f"{a}: {len(m['parts'])} parts, {sum(p['faces'] for p in m['parts'])} faces, size {size.round(2)}",
                  [p["material"] + ":" + str([round(c, 2) for c in p["rgba"][:3]]) for p in m["parts"]][:8])
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{a}: FAILED {e!r}")
