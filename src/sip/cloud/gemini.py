"""Optional cloud calls. Never needed for the robot to act; every call has a timeout and returns
None when offline, unconfigured or malformed. Key from GEMINI_API_KEY only."""
import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

import numpy as np

MODEL = os.environ.get("GEMINI_MODEL", "gemini-4-argon")  # builder-guide id; check in AI Studio
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


def second_opinion(rgb: np.ndarray, transcript: str, map_summary: str, labels: list[str],
                   timeout_s: float = 8.0) -> dict | None:
    client = _client()
    if client is None:
        return None
    import cv2
    from google.genai import types
    ok, jpg = cv2.imencode(".jpg", rgb[:, :, ::-1])
    if not ok:
        return None
    h, w = rgb.shape[:2]
    prompt = (f"You help a home care robot. The person said: '{transcript}'. Map: {map_summary}. "
              f"Find one of {labels} in this {w}x{h} image. Reply ONLY JSON "
              '{"label": str, "u": int, "v": int, "plan": "<one sentence>"} with pixel coordinates, '
              'or {"label": null} if none is visible.')

    def run():
        r = client.models.generate_content(model=MODEL, contents=[
            types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg"), prompt])
        raw = r.text
        out = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        if out.get("label") in labels and 0 <= int(out["u"]) < w and 0 <= int(out["v"]) < h:
            return {"label": out["label"], "u": int(out["u"]), "v": int(out["v"]),
                    "plan": str(out.get("plan", ""))}
        return None

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
