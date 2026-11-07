"""Keyword intents. Safety words never wait on a language model."""
import re

_STOP = r"\b(stop|wait|hold on|no no|enough|careful)\b"
_READY = r"\b(ready|i'?m ready)\b"
_GO = r"\b(go|go ahead|carry on|continue|try again)\b"
_WATER = r"\b(water|drink|thirsty|sip|glass)\b"
_MEDICINE = r"\b(medicine|medicines|pill|pills|tablet|tablets|medication)\b"
_YES = r"\b(yes|yeah|yep|please do)\b"
_NO = r"\b(no|nope|don'?t)\b"


def parse_keywords(text: str) -> set[str]:
    t = text.lower()
    out: set[str] = set()
    if re.search(_STOP, t):
        out.add("stop")
    for name, pat in (("ready", _READY), ("go", _GO), ("water", _WATER), ("medicine", _MEDICINE),
                      ("yes", _YES), ("no", _NO)):
        if re.search(pat, t):
            out.add(name)
    if "stop" in out:  # stop overrides anything that would start motion
        out -= {"ready", "go", "yes"}
    if re.search(r"\bnot ready\b", t):
        out.discard("ready")
    return out
