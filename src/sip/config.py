"""Model profiles: 'dev' (CPU laptop) and 'demo' (RTX A2000 laptop)."""
import os
from dataclasses import dataclass
from pathlib import Path


def load_env(path: Path = Path(__file__).resolve().parents[2] / ".env") -> None:
    """Read KEY=value lines from the repo's .env (git-ignored). Real environment wins."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.strip().partition("=")
        if sep and k and not k.startswith("#"):
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()


@dataclass(frozen=True)
class Profile:
    name: str
    detector: str      # "seg" (sim segmentation + noise) or "owl" (OWLv2)
    face_mode: str     # "sim" or "mediapipe"
    stt_model: str     # faster-whisper model size, or "gemma" for Gemma audio
    llm_url: str       # llama.cpp llama-server OpenAI-compatible endpoint
    cloud: bool        # allow Gemini second opinion / summary when online


PROFILES = {
    "dev": Profile("dev", "seg", "mediapipe", "base.en", "http://127.0.0.1:8080", True),
    "demo": Profile("demo", "owl", "mediapipe", "gemma", "http://127.0.0.1:8080", True),
}


def load_profile() -> Profile:
    return PROFILES[os.environ.get("SIP_PROFILE", "dev")]
