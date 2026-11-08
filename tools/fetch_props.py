"""Download Poly Haven (CC0) glTF models + textures and convert them into MuJoCo-ready
OBJ parts (one per material, Z-up, metres) with their diffuse texture as PNG.

uv run python tools/fetch_props.py ArmChair_01 Sofa_01 ...
Writes assets/props/<id>/part<k>.obj + part<k>.png and assets/props/<id>/manifest.json
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "props"
API = "https://api.polyhaven.com/files/"


def _get(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "sip-hackathon-asset-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r:
        dest.write_bytes(r.read())


def fetch(asset: str, res: str = "1k") -> dict:
    files = json.loads(urllib.request.urlopen(urllib.request.Request(
        API + asset, headers={"User-Agent": "sip-hackathon-asset-fetch"}), timeout=60).read())
    g = files["gltf"].get(res) or files["gltf"]["1k"]
    g = g["gltf"]
    raw = OUT / asset / "raw"
    _get(g["url"], raw / Path(g["url"]).name)
    for rel, info in g.get("include", {}).items():
        if rel.endswith(".bin") or "diff" in rel.lower() or "albedo" in rel.lower() or "color" in rel.lower():
            _get(info["url"], raw / rel)
    scene = trimesh.load(raw / Path(g["url"]).name, force="scene")
    parts = []
    yup_to_zup = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    for k, (name, geom) in enumerate(scene.dump(concatenate=False) and
                                     [(n, scene.geometry[scene.graph[n][1]].copy()) for n in scene.graph.nodes_geometry]):
        T = scene.graph[name][0]
        geom.apply_transform(yup_to_zup @ T)
        part = OUT / asset / f"part{k}.obj"
        tex = None
        vis = geom.visual
        img = None
        if isinstance(vis, trimesh.visual.texture.TextureVisuals) and vis.material is not None:
            mat = vis.material
            img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            if img is None and getattr(mat, "baseColorFactor", None) is not None:
                c = np.array(mat.baseColorFactor[:3])
                img = Image.new("RGB", (4, 4), tuple(int(x) for x in (c if c.max() > 1 else c * 255)))
        if img is not None:
            tex = OUT / asset / f"part{k}.png"
            img.convert("RGB").resize((min(1024, img.size[0]), min(1024, img.size[1]))).save(tex)
        # OBJ with UVs only (no mtl): MuJoCo reads v/vt/f
        uv = vis.uv if isinstance(vis, trimesh.visual.texture.TextureVisuals) and vis.uv is not None else None
        with open(part, "w") as f:
            for v in geom.vertices:
                f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
            if uv is not None:
                for t in uv:
                    f.write(f"vt {t[0]:.5f} {t[1]:.5f}\n")
            for a, b, c in geom.faces + 1:
                f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n" if uv is not None else f"f {a} {b} {c}\n")
        parts.append({"obj": part.name, "png": tex.name if tex else None, "faces": int(len(geom.faces))})
    allv = np.vstack([trimesh.load(OUT / asset / p["obj"], force="mesh").vertices for p in parts])
    man = {"asset": asset, "parts": parts, "min": allv.min(0).tolist(), "max": allv.max(0).tolist(),
           "license": "CC0 (Poly Haven)"}
    (OUT / asset / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


if __name__ == "__main__":
    for a in sys.argv[1:]:
        try:
            m = fetch(a)
            size = np.array(m["max"]) - np.array(m["min"])
            print(f"{a}: {len(m['parts'])} parts, {sum(p['faces'] for p in m['parts'])} faces, size {size.round(2)}")
        except Exception as e:
            print(f"{a}: FAILED {e!r}")
