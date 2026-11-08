"""Real-model perception on the robot's camera images (no simulator ground truth).

- Objects and people: YOLO-World open-vocabulary detector on the head camera RGB, 3D from the
  aligned depth image. Runs in a worker thread; results carry the camera pose at capture time.
- Mouth: MediaPipe Face Landmarker on an upscaled crop of the detected person's head (or the whole
  wrist-camera image up close). Mouth pixel + depth -> 3D; lip gap -> open ratio; face pose ->
  head turn relative to her resting direction, which is learned from observation.

The Stretch head camera (D435i) is mounted sideways, so frames are rotated upright first.
"""
import threading
from pathlib import Path

import numpy as np

from sip.types import Detection, MouthState

MODELS = Path(__file__).resolve().parents[3] / "assets" / "models"
# label -> (text prompt for the open-vocabulary model, minimum confidence)
DEFAULT_LABELS = {
    "glass": ("drinking glass", 0.12),
    "jug": ("metal jug", 0.10),
    "straw2": ("drinking straw", 0.20),
    "person": ("person", 0.40),
    "organiser": ("pill box", 0.10),
}


def _focal(fovy_deg: float, h: int) -> float:
    return (h / 2) / np.tan(np.deg2rad(fovy_deg) / 2)


def to_world(u, v, z, cam_pos, cam_mat, fovy_deg, w, h) -> np.ndarray:
    f = _focal(fovy_deg, h)
    u, v, z = np.asarray(u, float), np.asarray(v, float), np.asarray(z, float)
    local = np.stack([(u - w / 2) / f * z, -(v - h / 2) / f * z, -z], axis=-1)
    return cam_pos + local @ cam_mat.T


def box_depth(depth: np.ndarray, box) -> float:
    """Robust depth of an object: nearest-quarter median of the central part of its box."""
    u0, v0, u1, v1 = box
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    hu, hv = max(1, (u1 - u0) * 0.25), max(1, (v1 - v0) * 0.25)
    patch = depth[int(cv - hv):int(cv + hv) + 1, int(cu - hu):int(cu + hu) + 1]
    vals = patch[(patch > 0.05) & (patch < 8.0)]
    if vals.size == 0:
        return 0.0
    near = np.sort(vals)[: max(1, vals.size // 4)]
    return float(np.median(near))


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def unrotate_box(box, w: int) -> tuple[int, int, int, int]:
    """Box in the upright (np.rot90 k=1) image -> box in the raw camera image of width w."""
    x1, y1, x2, y2 = box
    return int(w - 1 - y2), int(x1), int(w - 1 - y1), int(x2)


class YoloDetector:
    """Same interface as SegDetector.detect(model, data, rgb, depth, seg) but uses only rgb + depth.
    Asynchronous: each call submits the newest frame if the worker is free and returns the latest
    finished detections (already in world coordinates)."""

    needs_rgb = True

    def __init__(self, labels: dict | None = None, cam: str = "d435i_camera_rgb", asynchronous: bool = True,
                 weights: str = "yolov8s-worldv2.pt"):
        from ultralytics import YOLOWorld
        self.labels = dict(labels or DEFAULT_LABELS)
        self.cam = cam
        self.model = YOLOWorld(str(MODELS / weights))
        self.prompts = [p for p, _ in self.labels.values()]
        self.model.set_classes(self.prompts)
        self.by_prompt = {p: (k, c) for k, (p, c) in self.labels.items()}
        self.asynchronous = asynchronous
        self._lock = threading.Lock()
        self._busy = False
        self.latest: list[Detection] = []
        self.latest_t = -1.0
        self.boxes: dict[str, tuple] = {}   # label -> best raw-image box, for the face tracker
        self.ms = 0.0

    def detect(self, model, data, rgb, depth, seg=None, t: float = 0.0) -> list[Detection]:
        if rgb is None:
            return list(self.latest)
        c = model.camera(self.cam).id
        job = (rgb.copy(), depth.copy(), data.cam_xpos[c].copy(), data.cam_xmat[c].reshape(3, 3).copy(),
               float(model.cam_fovy[c]), t)
        if not self.asynchronous:
            self._run(job)
            return list(self.latest)
        with self._lock:
            if not self._busy:
                self._busy = True
                threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return list(self.latest)

    def _run(self, job) -> None:
        import time
        rgb, depth, pos, mat, fovy, t = job
        h, w = depth.shape
        t0 = time.time()
        try:
            up = np.ascontiguousarray(np.rot90(rgb, 1)[:, :, ::-1])
            res = self.model.predict(up, conf=0.05, verbose=False, imgsz=640)[0]
            best: dict[str, tuple] = {}
            cand = []
            for box, cls, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(),
                                      res.boxes.conf.cpu().numpy()):
                label, thr = self.by_prompt[self.prompts[int(cls)]]
                if conf >= thr:
                    cand.append((float(conf / thr), box, label, float(conf)))
            kept = []   # one object gets one label: the most confident (relative to its threshold)
            for score, box, label, conf in sorted(cand, key=lambda c: -c[0]):
                if all(_iou(box, k[1]) < 0.5 for k in kept):
                    kept.append((score, box, label, conf))
            for _, box, label, conf in kept:
                if label not in best or conf > best[label][1]:
                    best[label] = (unrotate_box(box, w), float(conf))
            dets, boxes = [], {}
            for label, (b, conf) in best.items():
                z = box_depth(depth, b)
                if z <= 0:
                    continue
                xyz = to_world((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, z, pos, mat, fovy, w, h)
                if label != "person":
                    xyz = xyz + np.array([0, 0, -0.02])  # centre of a small object, not its front skin
                dets.append(Detection(label, conf, tuple(int(x) for x in b), xyz, source="yolo-world"))
                boxes[label] = b
            self.latest, self.boxes, self.latest_t = dets, boxes, t
        finally:
            self.ms = (time.time() - t0) * 1000
            with self._lock:
                self._busy = False


class MediaPipeFace:
    """Mouth tracking from camera images only."""

    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")),
            output_facial_transformation_matrixes=True, num_faces=1)
        self.fl = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
        self._facing: list[float] = []      # observed head directions (world yaw)
        self._ref: list[float] | None = None  # head direction when she said she was ready
        self.rot = {}                         # camera -> rotation that worked last

    def _landmarks(self, img: np.ndarray, cam: str):
        order = [self.rot.get(cam, 1)] + [k for k in (1, 0, -1) if k != self.rot.get(cam, 1)]
        for k in order:
            up = np.ascontiguousarray(np.rot90(img, k))
            res = self.fl.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=up))
            if res.face_landmarks:
                self.rot[cam] = k
                return k, up.shape, res
        return None

    def update(self, rgb, depth, model, data, cam: str, person_box=None, hand_raised=False) -> MouthState:
        import cv2
        none = MouthState(False, None, 0.0, 0.0, hand_raised, f"mediapipe:{cam}")
        if rgb is None:
            return none
        h, w = depth.shape
        # crop the head: top part of the person box, or the whole image (wrist camera up close)
        if person_box is not None:
            u0, v0, u1, v1 = person_box
            # camera is sideways: "top" of the person is the high-u side in the raw image
            size = int(max(u1 - u0, v1 - v0) * 0.45)
            cu, cv_ = u1 - size // 2, (v0 + v1) // 2
            x0, y0 = max(0, cu - size // 2 - size // 4), max(0, cv_ - size // 2 - size // 4)
            x1, y1 = min(w, x0 + int(size * 1.5)), min(h, y0 + int(size * 1.5))
        else:
            x0, y0, x1, y1 = 0, 0, w, h
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            return none
        s = max(1.0, 320 / max(crop.shape[:2]))
        big = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        found = self._landmarks(big, cam)
        if found is None:
            return none
        k, (H, W, _), res = found
        lm = res.face_landmarks[0]

        def raw(i):  # landmark in the upright upscaled crop -> raw image pixel
            x, y = lm[i].x * W, lm[i].y * H
            # undo np.rot90(big, k): (x, y) in rotated -> (cx, cy) in big
            bh, bw = big.shape[:2]
            if k == 1:
                cx, cy = bw - 1 - y, x
            elif k == -1:
                cx, cy = y, bh - 1 - x
            else:
                cx, cy = x, y
            return x0 + cx / s, y0 + cy / s

        mu, mv = np.mean([raw(13), raw(14)], axis=0)
        if not (0 <= mu < w and 0 <= mv < h):
            return none
        patch = depth[max(0, int(mv) - 2):int(mv) + 3, max(0, int(mu) - 2):int(mu) + 3]
        vals = patch[(patch > 0.05) & (patch < 5)]
        if vals.size == 0:
            return none
        c = model.camera(cam).id
        xyz = to_world(mu, mv, float(np.median(vals)), data.cam_xpos[c], data.cam_xmat[c].reshape(3, 3),
                       float(model.cam_fovy[c]), w, h)
        gap = np.hypot(lm[13].x - lm[14].x, lm[13].y - lm[14].y)
        face_h = np.hypot(lm[10].x - lm[152].x, lm[10].y - lm[152].y) + 1e-6
        # head direction in the world: face normal from the transform, through the camera pose
        M = np.array(res.facial_transformation_matrixes[0])[:3, :3]
        n_up = M @ np.array([0, 0, 1.0])            # face normal in the upright image frame (x right, y up, z out)
        rot = {1: np.array([[0, -1], [1, 0]]), -1: np.array([[0, 1], [-1, 0]]), 0: np.eye(2)}[k]
        nx, ny = rot.T @ n_up[:2]                   # back to the raw camera frame
        n_cam = np.array([nx, ny, n_up[2]])
        n_world = data.cam_xmat[c].reshape(3, 3) @ n_cam
        facing = float(np.arctan2(n_world[1], n_world[0]))
        self._facing.append(facing)
        self._facing = self._facing[-200:]
        if self._ref is not None and len(self._ref) < 5:
            self._ref.append(facing)
        rest = float(np.median(self._ref)) if self._ref else float(np.median(self._facing))
        yaw = float(np.rad2deg(np.angle(np.exp(1j * (facing - rest)))))
        return MouthState(True, xyz, float(min(1.0, gap / face_h * 4)), yaw, hand_raised, f"mediapipe:{cam}")

    def set_reference(self) -> None:
        """Head turns are measured from where she looks when the approach starts."""
        self._ref = []

    @property
    def resting_facing(self) -> float | None:
        if not self._facing:
            return None
        a = np.array(self._facing)
        return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))
