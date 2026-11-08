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


def _client(images: bool = False):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    if images:
        from sip.home import load
        if not load().get("cloud_images_allowed", True):   # privacy: the carer can keep camera images local
            return None
    try:
        from google import genai
        return genai.Client(api_key=key)
    except Exception:
        return None


# Fair-use guard (organisers: no abuse): hard cap per session and minimum spacing between calls.
MAX_CALLS = int(os.environ.get("GEMINI_MAX_CALLS", "30"))
MIN_GAP_S = 5.0
_usage = {"calls": 0, "last": 0.0, "ok": None}   # ok: did the last cloud call succeed (None = never tried)


def _call(fn, timeout_s: float):
    import time
    now = time.time()
    if _usage["calls"] >= MAX_CALLS or now - _usage["last"] < MIN_GAP_S:
        return None   # robot carries on with local models
    _usage["calls"] += 1
    _usage["last"] = now
    fut = _POOL.submit(fn)
    try:
        out = fut.result(timeout=timeout_s)
        _usage["ok"] = True
        return out
    except (FutTimeout, Exception):
        _usage["ok"] = False
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
    client = _client(images=True)
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


def identify(crop_rgb: np.ndarray, options: list[str], timeout_s: float = 10.0) -> dict | None:
    """Second opinion on an object the local detector is unsure about: which of `options` is it, and why."""
    client = _client(images=True)
    if client is None or crop_rgb is None or crop_rgb.size == 0:
        return None
    import cv2
    from google.genai import types
    ok, jpg = cv2.imencode(".jpg", crop_rgb[:, :, ::-1])
    if not ok:
        return None
    prompt = (f"A home robot's camera crop. Which one is it: {', '.join(options)}, or none? Reply ONLY JSON "
              '{"label": <one option or "none">, "confidence": <0-1>, "why": "<visual reason, max 12 words>"}')

    def run():
        r = client.models.generate_content(model=MODEL, contents=[
            types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg"), prompt])
        raw = r.text or ""
        out = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        if out.get("label") in options + ["none"]:
            return {"label": out["label"], "confidence": float(out.get("confidence", 0.5)), "why": str(out.get("why", ""))[:120]}
        return None

    return _call(run, timeout_s)


def name_objects(rgb: np.ndarray, timeout_s: float = 10.0) -> list[str] | None:
    """New home: which household objects are in view (for the robot's detector vocabulary)."""
    client = _client(images=True)
    if client is None:
        return None
    import cv2
    from google.genai import types
    ok, jpg = cv2.imencode(".jpg", rgb[:, :, ::-1])
    if not ok:
        return None
    prompt = ("A home-care robot is exploring a new home. List up to 8 distinct household objects visible here that "
              "matter for helping an older person (furniture, cups, containers, medicine, obstacles). "
              'Reply ONLY JSON {"objects": ["short noun phrase", ...]}')

    def run():
        r = client.models.generate_content(model=MODEL, contents=[
            types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg"), prompt])
        raw = r.text or ""
        out = json.loads(raw[raw.find("{"):raw.rfind("}") + 1]).get("objects", [])
        return [str(o).strip().lower()[:30] for o in out if str(o).strip()][:8]

    return _call(run, timeout_s)


def read_compartment(rgb: np.ndarray, timeout_s: float = 8.0) -> str | None:
    """Read which day's compartment of a weekly pill organiser is in view. One word or None."""
    client = _client(images=True)
    if client is None:
        return None
    import cv2
    from google.genai import types
    ok, jpg = cv2.imencode(".jpg", rgb[:, :, ::-1])
    if not ok:
        return None
    prompt = ("This photo from a care robot's wrist camera shows a weekly pill organiser. Which day's compartment "
              "is nearest the camera or open? Reply with only the day name, or 'unknown'.")

    def run():
        r = client.models.generate_content(model=MODEL, contents=[
            types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg"), prompt])
        word = (r.text or "").strip().split()[0].strip(".,").capitalize() if r.text else ""
        return None if not word or word.lower() == "unknown" else word

    return _call(run, timeout_s)


CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemma-4-26b-a4b-it")   # open Gemma 4 model (also runs locally)


def chat_reply(text: str, timeout_s: float = 15.0) -> str | None:
    """One short, kind spoken reply when what she said is not a command (small talk, worries)."""
    client = _client()
    if client is None:
        return None
    prompt = ("You are Sip, a gentle home-care robot talking to an older woman. Reply in one or two short spoken "
              "sentences, warm and plain. Never give medical advice; suggest asking her carer or GP instead. "
              f"She said: \"{text}\"")

    def run():
        r = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
        return (r.text or "").strip().replace("*", "")[:240] or None

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
