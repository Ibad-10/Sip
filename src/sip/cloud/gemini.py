"""Optional cloud calls. Never needed for the robot to act; every call has a timeout and returns
None when offline, unconfigured or malformed. Key from GEMINI_API_KEY only."""
import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

import numpy as np

MODEL = os.environ.get("GEMINI_MODEL", "gemini-4-argon")  # builder-guide id; check in AI Studio
# Gemini Robotics-ER: embodied-reasoning model that points at objects ([y, x] normalised to 0-1000)
ER_MODEL = os.environ.get("GEMINI_ER_MODEL", "gemini-4-argon")
_POOL = ThreadPoolExecutor(max_workers=2)


def _client():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=key)
    except Exception:
        return None


def _call(fn, timeout_s: float):
    fut = _POOL.submit(fn)
    try:
        return fut.result(timeout=timeout_s)
    except (FutTimeout, Exception):
        return None


def parse_points(raw: str, labels: list[str], w: int, h: int) -> dict | None:
    """First pointed object whose label is wanted, converted to pixel (u, v)."""
    try:
        pts = json.loads(raw[raw.find("["):raw.rfind("]") + 1])
    except ValueError:
        return None
    for p in pts if isinstance(pts, list) else []:
        try:
            y, x = (float(c) for c in p["point"])
        except (KeyError, TypeError, ValueError):
            continue
        if p.get("label") in labels and 0 <= x <= 1000 and 0 <= y <= 1000:
            return {"label": p["label"], "u": min(w - 1, int(x / 1000 * w)), "v": min(h - 1, int(y / 1000 * h))}
    return None


def second_opinion(rgb: np.ndarray, transcript: str, map_summary: str, labels: list[str],
                   timeout_s: float = 8.0) -> dict | None:
    """Ask Gemini Robotics-ER to point at the object. Result must still pass sip.cloud.verify."""
    client = _client()
    if client is None:
        return None
    import cv2
    from google.genai import types
    ok, jpg = cv2.imencode(".jpg", rgb[:, :, ::-1])
    if not ok:
        return None
    h, w = rgb.shape[:2]
    prompt = (f"A home care robot is looking for: {', '.join(labels)}. The person said: '{transcript}'. "
              f"Map: {map_summary}. Point to the item if visible. Answer as JSON "
              '[{"point": [y, x], "label": <label>}] with points normalised to 0-1000, or [] if not visible.')

    def run():
        r = client.models.generate_content(model=ER_MODEL, contents=[
            types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg"), prompt],
            config=types.GenerateContentConfig(temperature=0.5,
                                               thinking_config=types.ThinkingConfig(thinking_budget=0)))
        return parse_points(r.text or "", labels, w, h)

    return _call(run, timeout_s)


def day_summary(events, timeout_s: float = 15.0) -> str | None:
    client = _client()
    if client is None:
        return None
    lines = "\n".join(f"{e.t:.0f}s {e.kind}: {e.text}" for e in events if e.kind in ("plan", "dose", "alert", "say"))
    prompt = ("Write a short, warm summary for a family carer of this home care robot's day log. "
              "State counts as digits (e.g. '2 drinks', 'took 1', 'declined 1', '1 alert'). "
              "No medical advice.\n" + lines)
    return _call(lambda: client.models.generate_content(model=MODEL, contents=prompt).text, timeout_s)
