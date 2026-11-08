"""Few-shot object classifier: a linear probe on CLIP image features, trained on detector crops.

Training data: robot-camera views in randomised homes; YOLO-World proposes boxes and the simulator
provides the TRUE label of each box (training only). At runtime the robot only sees the crop.
Held-out homes measure accuracy.

  uv run python tools/train_probe.py            # writes assets/models/clip_probe.pt + prints accuracy
"""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from sip.perception.objects import CAM, aim_head, project, render_head
from sip.perception.vision import ClipCheck, YoloDetector, unrotate_box
from sip.robot import StretchIO
from sip.scene.build_house import build_house
from sip.scene.randomise import randomise_home

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["glass", "jug", "straw2", "organiser", "mug", "other"]
BODY = {"glass": "glass", "jug": "jug", "straw2": "straw_cup", "organiser": "organiser"}
MUG_XY = np.array([-3.4, 2.62])


def crops_for_home(m, d, io, det, clip, r, seed):
    layout = randomise_home(m, d, seed)
    feats, labels = [], []
    targets = {k: d.body(b).xpos.copy() for k, b in BODY.items()}
    for name, xyz in targets.items():
        for dist, ang in ((0.8, -0.5), (1.0, 0.0), (1.3, 0.5), (0.9, 2.8)):
            base = xyz[:2] + dist * np.array([np.cos(ang + np.pi), np.sin(ang + np.pi)]) * np.sign(1.5 - xyz[1] + 0.01)
            d.qpos[0:2] = base
            yaw = np.arctan2(xyz[1] - base[1], xyz[0] - base[0])
            d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            mujoco.mj_forward(m, d)
            aim_head(m, d, io, xyz, snap=True)
            mujoco.mj_forward(m, d)
            rgb, depth, _ = render_head(m, d, r, rgb=True, seg=False)
            h, w = depth.shape
            up = np.ascontiguousarray(np.rot90(rgb, 1)[:, :, ::-1])
            res = det.model.predict(up, conf=0.05, verbose=False, imgsz=640)[0]
            proj = {k: project(m, d, CAM, p, w, h) for k, p in targets.items()}
            mp = project(m, d, CAM, np.array([*MUG_XY, 0.95]), w, h)
            for box in res.boxes.xyxy.cpu().numpy():
                b = unrotate_box(box, w)
                cu, cv = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                lab = "other"
                best = 30.0
                for k, (u, v, z) in list(proj.items()) + [("mug", mp)]:
                    if z > 0 and np.hypot(u - cu, v - cv) < best:
                        best, lab = np.hypot(u - cu, v - cv), k
                u0, v0, u1, v1 = b
                pad = max(12, int(0.8 * max(u1 - u0, v1 - v0)))
                crop = np.ascontiguousarray(np.rot90(rgb[max(0, v0 - pad):v1 + pad, max(0, u0 - pad):u1 + pad], 1))
                if crop.size == 0:
                    continue
                feats.append(clip.embed(crop))
                labels.append(CLASSES.index(lab))
    return feats, labels


def main():
    m = mujoco.MjModel.from_xml_path(build_house())
    d = mujoco.MjData(m)
    io = StretchIO(m, d)
    det = YoloDetector(asynchronous=False)
    clip = ClipCheck()
    r = mujoco.Renderer(m, 480, 640)
    data = {}
    for seed in range(14):
        f, lab = crops_for_home(m, d, io, det, clip, r, seed)
        data[seed] = (f, lab)
        print("home", seed, "crops", len(f), "labels", np.bincount(lab, minlength=len(CLASSES)).tolist(), flush=True)
    train = [s for s in data if s < 10]
    test = [s for s in data if s >= 10]
    X = torch.tensor(np.concatenate([np.array(data[s][0]) for s in train]), dtype=torch.float32)
    y = torch.tensor(np.concatenate([data[s][1] for s in train]), dtype=torch.long)
    Xt = torch.tensor(np.concatenate([np.array(data[s][0]) for s in test]), dtype=torch.float32)
    yt = torch.tensor(np.concatenate([data[s][1] for s in test]), dtype=torch.long)
    probe = torch.nn.Linear(X.shape[1], len(CLASSES))
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-3)
    counts = torch.bincount(y, minlength=len(CLASSES)).float().clamp(min=1)
    loss_fn = torch.nn.CrossEntropyLoss(weight=counts.sum() / counts)
    for _ in range(400):
        opt.zero_grad()
        loss = loss_fn(probe(X), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc_tr = (probe(X).argmax(1) == y).float().mean().item()
        pred = probe(Xt).argmax(1)
        acc_te = (pred == yt).float().mean().item()
        per = {c: float(((pred == i) & (yt == i)).sum() / max(1, int((yt == i).sum()))) for i, c in enumerate(CLASSES)}
    out = ROOT / "assets/models/clip_probe.pt"
    torch.save({"state": probe.state_dict(), "classes": CLASSES}, out)
    stats = {"train_homes": len(train), "test_homes": len(test), "train_crops": int(len(y)), "test_crops": int(len(yt)),
             "train_accuracy": round(acc_tr, 3), "held_out_accuracy": round(acc_te, 3), "per_class_recall": per}
    (ROOT / "assets/models/clip_probe_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
