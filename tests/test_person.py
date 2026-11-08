import mujoco
import numpy as np

from sip.blackboard import Blackboard
from sip.person import Person
from sip.scene.build_house import build_house, seat_person


def _sim():
    m = mujoco.MjModel.from_xml_path(build_house())
    d = mujoco.MjData(m)
    seat_person(m, d, "armchair")
    mujoco.mj_forward(m, d)
    return m, d


def test_idle_person_looks_around_and_breathes():
    m, d = _sim()
    p = Person(m, d, Blackboard(), seed=1)
    yaws, zs = [], []
    for k in range(800):  # 20 s at 40 Hz
        p.step(k * 0.025)
        yaws.append(d.ctrl[m.actuator("head_yaw_act").id])
        zs.append(d.mocap_pos[m.body("person").mocapid[0]][2])
    assert np.ptp(yaws) > 0.2 and np.ptp(zs) > 0.005


def test_distracted_during_approach_turns_away():
    m, d = _sim()
    p = Person(m, d, Blackboard(), seed=2)
    p.p_distract = 1.0
    yaws = []
    for k in range(480):
        p.step(k * 0.025, "APPROACH")
        yaws.append(abs(d.ctrl[m.actuator("head_yaw_act").id]))
    assert max(yaws) > np.deg2rad(30)  # beyond the safety gate's yaw limit


def test_override_freezes_gaze():
    m, d = _sim()
    p = Person(m, d, Blackboard(), seed=3)
    p.override = True
    d.ctrl[m.actuator("head_yaw_act").id] = 0.8
    for k in range(200):
        p.step(k * 0.025)
    assert d.ctrl[m.actuator("head_yaw_act").id] == 0.8
