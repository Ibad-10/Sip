import time

import pytest

from sip.voice.intents import parse_keywords
from sip.voice.llm import LLM


@pytest.mark.parametrize("text", ["Stop!", "wait", "no no no", "hold on a second", "STOP please"])
def test_stop_variants(text):
    assert "stop" in parse_keywords(text)


def test_ready_go():
    assert parse_keywords("I'm ready") == {"ready"}
    assert "go" in parse_keywords("okay go ahead")


@pytest.mark.parametrize("text", ["could I get a drink", "I'm thirsty", "Can I have some water?"])
def test_water_request_phrasings(text):
    assert "water" in parse_keywords(text)


def test_medicine_phrasing():
    assert "medicine" in parse_keywords("have I taken my pills?")


def test_stop_wins_over_ready():
    kw = parse_keywords("no wait, not ready")
    assert "stop" in kw and "ready" not in kw


def test_llm_timeout_falls_back():
    llm = LLM("http://127.0.0.1:9", timeout_s=4.0)  # nothing listens on port 9
    t0 = time.time()
    out = llm.intent("I'm thirsty")
    assert time.time() - t0 < 4.5
    assert out["task"] == "water" and out["source"] == "keywords"


def test_llm_garbage_falls_back(monkeypatch):
    llm = LLM("http://127.0.0.1:9")
    monkeypatch.setattr(llm, "_chat", lambda text: "sure! {not json")
    out = llm.intent("could I get a drink")
    assert out["task"] == "water" and out["source"] == "keywords"


def test_llm_valid_json_used(monkeypatch):
    llm = LLM("http://127.0.0.1:9")
    monkeypatch.setattr(llm, "_chat", lambda text: '{"task": "chat", "reply": "Hello Mary."}')
    out = llm.intent("hello")
    assert out == {"task": "chat", "reply": "Hello Mary.", "source": "llm"}


@pytest.mark.parametrize("text", ["that's enough", "I'm done thanks", "no more please", "finished"])
def test_done_is_not_emergency_stop(text):
    kw = parse_keywords(text)
    assert "done" in kw and "stop" not in kw
