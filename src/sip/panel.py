"""Local control panel (stdlib only): live camera feeds, map, state, safety, events, and buttons
for live edits during the demo. Served on http://127.0.0.1:8765 by sip.app --panel."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = (__import__("pathlib").Path(__file__).with_name("panel.html")).read_text(encoding="utf-8")


class Panel:
    def __init__(self, on_command, port: int = 8765, on_audio=None):
        self.on_command = on_command
        self.on_audio = on_audio
        self.frames: dict[str, bytes] = {}
        self.state: dict = {}
        self.lock = threading.Lock()
        panel = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body: bytes, ctype: str):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/" or self.path.startswith("/?"):
                    return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                if self.path.startswith("/frame/"):
                    name = self.path.split("/")[2].split(".")[0]
                    with panel.lock:
                        img = panel.frames.get(name)
                    return self._send(200, img, "image/jpeg") if img else self._send(204, b"", "text/plain")
                if self.path.startswith("/speech.wav"):
                    with panel.lock:
                        snd = panel.frames.get("speech_wav")
                    return self._send(200, snd, "audio/wav") if snd else self._send(204, b"", "text/plain")
                if self.path.startswith("/state"):
                    with panel.lock:
                        body = json.dumps(panel.state).encode()
                    return self._send(200, body, "application/json")
                self._send(404, b"not found", "text/plain")

            def do_POST(self):
                if self.path == "/audio" and panel.on_audio is not None:
                    n = int(self.headers.get("Content-Length", "0") or 0)
                    if n > 5_000_000:
                        return self._send(413, b"too large", "text/plain")
                    text = panel.on_audio(self.rfile.read(n))
                    return self._send(200, json.dumps({"text": text}).encode(), "application/json")
                if self.path == "/cmd":
                    n = int(self.headers.get("Content-Length", "0") or 0)
                    text = self.rfile.read(min(n, 2000)).decode("utf-8", "replace")
                    panel.on_command(text)
                    return self._send(204, b"", "text/plain")
                self._send(404, b"not found", "text/plain")

        import os
        self.server = ThreadingHTTPServer((os.environ.get("SIP_PANEL_HOST", "127.0.0.1"), port), H)
        self.url = f"http://127.0.0.1:{port}/"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def set_frame(self, name: str, jpg: bytes) -> None:
        with self.lock:
            self.frames[name] = jpg

    def set_state(self, state: dict) -> None:
        with self.lock:
            self.state = state

    def close(self) -> None:
        self.server.shutdown()
