"""Keyword intents. Safety words never wait on a language model.

Negation-aware: "I haven't taken them" is not 'taken', "don't go" is a stop, "not ready" is not ready.
"""
import re

_STOP = r"\b(stop|wait|hold on|no no|careful)\b"
_DONE = r"(that'?s enough|i'?ve had enough|i'?m done|\bdone\b|no more|finished)"
_READY = r"\b(ready|i'?m ready)\b"
_GO = r"\b(go|go ahead|carry on|continue|try again)\b"
_WATER = r"\b(water|drink|thirsty|sip|glass)\b"
_MEDICINE = r"\b(medicine|medicines|pill|pills|tablet|tablets|medication)\b"
_YES = r"\b(yes|yeah|yep|please do)\b"
_NO = r"\b(no|nope|don'?t|not now|later)\b"
_TAKEN = r"\b(taken|took them|took it|had them|had it|swallowed)\b"
_TAKE_VERB = r"\b(take|taken|took|had|have|swallowed)\b"

# a negation within the three words before a keyword flips it
_NEG = r"\b(not|never|no|dont|havent|didnt|cant|wont)\b|n't\b"
_NEGATABLE = ("go", "ready", "yes", "taken", "done")


def _negated(t: str, pat: str) -> bool:
    for m in re.finditer(pat, t):
        before = " ".join(re.findall(r"[\w']+", t[:m.start()])[-3:])
        if re.search(_NEG, before):
            return True
    return False


def parse_keywords(text: str) -> set[str]:
    t = text.lower().replace("’", "'")
    out: set[str] = set()
    if re.search(_STOP, t):
        out.add("stop")
    for name, pat in (("done", _DONE), ("ready", _READY), ("go", _GO), ("water", _WATER), ("medicine", _MEDICINE),
                      ("yes", _YES), ("no", _NO), ("taken", _TAKEN)):
        if not re.search(pat, t):
            continue
        if name in _NEGATABLE and _negated(t, pat if name != "taken" else _TAKE_VERB):
            if name == "go":
                out.add("stop")        # "don't go" means stop: never move on a negated command
            continue
        out.add(name)
    if "stop" in out:  # stop overrides anything that would start motion
        out -= {"ready", "go", "yes"}
    return out
