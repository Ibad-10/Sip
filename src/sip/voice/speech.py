"""Speech in and out.

out: ElevenLabs (cloud, natural voice) -> falls back to the Windows built-in voice (offline).
in:  audio recorded by the panel's push-to-talk -> faster-whisper on this computer (offline).
Speaking never blocks the robot: a background thread plays queued sentences.
"""
import json
import os
import queue
import subprocess
import threading
import urllib.request

ELEVEN_VOICE = os.environ.get("ELEVENLABS_VOICE", "EXAVITQu4vr4xnSDxMaL")   # "Sarah - mature, reassuring"
ELEVEN_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_flash_v2_5")
_q: "queue.Queue[str]" = queue.Queue()
_started = False
last_engine = {"name": None}


def _eleven(text: str) -> bool:
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        return False
    try:
        import numpy as np
        import sounddevice as sd
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}?output_format=pcm_16000",
            data=json.dumps({"text": text, "model_id": ELEVEN_MODEL}).encode(),
            headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/pcm"})
        pcm = urllib.request.urlopen(req, timeout=8).read()
        sd.play(np.frombuffer(pcm, dtype=np.int16), 16000, blocking=True)
        last_engine["name"] = "elevenlabs"
        return True
    except Exception:
        return False


def _offline(text: str) -> bool:
    """Windows SAPI voice, no network."""
    if os.name != "nt":
        return False
    safe = text.replace("'", "''")
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                        f"$s.Rate = -1; $s.Speak('{safe}')"], timeout=30, capture_output=True)
        last_engine["name"] = "offline"
        return True
    except Exception:
        return False


def _worker() -> None:
    while True:
        text = _q.get()
        if not _eleven(text):
            _offline(text)


def speak(text: str) -> None:
    global _started
    if os.environ.get("SIP_VOICE", "1") != "1" or not text:
        return
    if not _started:
        threading.Thread(target=_worker, daemon=True).start()
        _started = True
    _q.put(text)


_whisper = {}


def transcribe(audio_bytes: bytes, suffix: str = ".webm") -> str:
    """Local speech-to-text (faster-whisper 'base.en', CPU int8)."""
    import tempfile
    from faster_whisper import WhisperModel
    if "m" not in _whisper:
        _whisper["m"] = WhisperModel(os.environ.get("SIP_WHISPER", "base.en"), device="cpu", compute_type="int8")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        segs, _ = _whisper["m"].transcribe(path, language="en", vad_filter=True)
        return " ".join(s.text.strip() for s in segs).strip()
    finally:
        os.unlink(path)
