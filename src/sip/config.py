"""Model profiles: 'dev' (CPU laptop) and 'demo' (RTX A2000 laptop)."""
import os
from dataclasses import dataclass


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
