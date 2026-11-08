import numpy as np
import pytest

from sip.safety import safety_gate
from sip.types import MouthState, SafetyInput


def face(**kw):
    base = dict(visible=True, xyz=np.zeros(3), open_ratio=0.5, yaw_deg=0.0, hand_raised=False)
    base.update(kw)
    return MouthState(**base)


def inp(dist, mouth=None, **kw):
    base = dict(dist_to_mouth_m=dist, mouth=mouth or face(), ready_heard=True, stop_heard=False,
                contact_force_n=0.0, reapproach_count=0)
    base.update(kw)
    return SafetyInput(**base)


def test_stop_halts():
    d = safety_gate(inp(0.6, stop_heard=True))
    assert d.action == "halt" and d.max_speed_mps == 0.0


def test_no_ready_pauses_near_face():
    assert safety_gate(inp(0.25, ready_heard=False)).action == "pause"


def test_speed_capped_within_30cm():
    d = safety_gate(inp(0.29))
    assert d.action == "proceed" and d.max_speed_mps == pytest.approx(0.03)


def test_far_full_speed():
    d = safety_gate(inp(0.6))
    assert d.action == "proceed" and d.max_speed_mps == pytest.approx(0.3)


def test_face_lost_pauses():
    d = safety_gate(inp(0.4, mouth=face(visible=False, xyz=None)))
    assert d.action == "pause" and d.reason == "face lost"


def test_head_turned_retreats():
    assert safety_gate(inp(0.2, mouth=face(yaw_deg=35.0))).action == "retreat"


def test_hand_raised_retreats():
    assert safety_gate(inp(0.2, mouth=face(hand_raised=True))).action == "retreat"


def test_contact_force_retreats():
    assert safety_gate(inp(0.05, contact_force_n=11.0)).action == "retreat"


def test_third_reapproach_asks():
    assert safety_gate(inp(0.2, mouth=face(yaw_deg=35.0), reapproach_count=2)).action == "ask"


def test_stop_short_at_2cm():
    d = safety_gate(inp(0.02))
    assert d.action == "pause" and d.reason == "at mouth"


def test_stop_beats_everything():
    d = safety_gate(inp(0.02, stop_heard=True, mouth=face(hand_raised=True)))
    assert d.action == "halt"


def test_mouth_closed_pauses_near_face():
    d = safety_gate(inp(0.1, mouth=face(open_ratio=0.0)))
    assert d.action == "pause" and d.reason == "mouth closed"


def test_done_retreats():
    d = safety_gate(inp(0.05, done_heard=True))
    assert d.action == "retreat" and d.reason == "person finished"


def test_slowdown_band_before_near_zone():
    d = safety_gate(inp(0.4))
    assert d.action == "proceed" and d.max_speed_mps == pytest.approx(0.06)
