"""Local language model (llama.cpp llama-server, OpenAI-compatible) for intent and replies.
Never on the safety path: any timeout, error or bad JSON falls back to keyword intents."""
import json

import urllib.request

from sip.voice.intents import parse_keywords

SYSTEM = ("You are Sip, a calm home care robot for an older person. Reply ONLY with JSON: "
          '{"task": "water" | "medicine" | "chat" | "none", "reply": "<one short, kind sentence>"}. '
          "Never give medical advice or dosing; suggest asking the carer or GP instead.")


class LLM:
    def __init__(self, base_url: str, timeout_s: float = 4.0, model: str = "gemma"):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.timeout, self.model = timeout_s, model

    def _chat(self, text: str) -> str:
        body = json.dumps({"model": self.model, "temperature": 0.2, "max_tokens": 80,
                           "messages": [{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": text}]}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    def intent(self, text: str) -> dict:
        try:
            raw = self._chat(text)
            start, end = raw.find("{"), raw.rfind("}")
            out = json.loads(raw[start:end + 1])
            if out.get("task") in ("water", "medicine", "chat", "none") and isinstance(out.get("reply"), str):
                return {"task": out["task"], "reply": out["reply"], "source": "llm"}
        except Exception:
            pass
        kw = parse_keywords(text)
        task = "water" if "water" in kw else "medicine" if "medicine" in kw else "none"
        reply = {"water": "Of course, I'll get you some water.",
                 "medicine": "Let me check your medicines."}.get(task, "Sorry, could you say that again?")
        return {"task": task, "reply": reply, "source": "keywords"}
