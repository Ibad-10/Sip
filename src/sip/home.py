"""The home profile: who lives here and what the carer set up (home.json). Defaults if missing."""
import datetime
import json
import os
from pathlib import Path

PATH = Path(os.environ.get("SIP_HOME", Path(__file__).resolve().parents[2] / "home.json"))
DEFAULT = {"resident": "", "dose_times": [], "carer_webhook_url": "", "cloud_images_allowed": True}


def load() -> dict:
    try:
        return {**DEFAULT, **json.loads(PATH.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return dict(DEFAULT)


def due_dose(profile: dict, done_today: set, now: datetime.datetime | None = None) -> str | None:
    """The dose time that is due now (within 30 min after it) and not yet handled today."""
    now = now or datetime.datetime.now()
    for hhmm in profile.get("dose_times", []):
        h, m = (int(x) for x in hhmm.split(":"))
        t = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if t <= now <= t + datetime.timedelta(minutes=30) and hhmm not in done_today:
            return hhmm
    return None


def next_dose(profile: dict, now: datetime.datetime | None = None) -> str | None:
    now = now or datetime.datetime.now()
    later = [t for t in profile.get("dose_times", []) if t > now.strftime("%H:%M")]
    return later[0] if later else (profile.get("dose_times") or [None])[0]
