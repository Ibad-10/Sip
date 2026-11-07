import numpy as np
import pytest

from sip.memory import ObjectMemory

ROOMS = ["kitchen", "living", "bedroom", "hall"]
XYZ = np.array([1.0, 2.0, 0.9])


def test_confidence_halves_at_half_life():
    m = ObjectMemory(half_life_s=600.0)
    m.see("glass", XYZ, t=0.0, room="kitchen")
    assert m.confidence("glass", 0.0) == pytest.approx(1.0)
    assert m.confidence("glass", 600.0) == pytest.approx(0.5, abs=1e-6)


def test_miss_demotes_room():
    m = ObjectMemory()
    m.see("glass", XYZ, t=0.0, room="kitchen")
    m.miss("glass", "kitchen", t=1.0)
    assert m.confidence("glass", 1.0) < 0.25
    assert m.search_order("glass", 1.0, ROOMS)[0] != "kitchen"


def test_search_order_prefers_last_seen():
    m = ObjectMemory()
    m.see("glass", XYZ, t=0.0, room="bedroom")
    assert m.search_order("glass", 10.0, ROOMS)[0] == "bedroom"
    assert m.where("glass") is not None


def test_unknown_object_uses_prior():
    m = ObjectMemory()
    assert m.search_order("glass", 0.0, ROOMS) == ["kitchen", "living", "bedroom", "hall"]
    assert m.confidence("glass", 0.0) == 0.0
