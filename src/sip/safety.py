"""Safety gate: pure function that can veto any arm motion near the face. Halt rules first."""
from sip.types import SafetyDecision, SafetyInput

NEAR_FACE_M = 0.30
WATCH_M = 0.50        # face checks apply inside this distance
STOP_SHORT_M = 0.02
SLOW_MPS = 0.03
FAST_MPS = 0.30
MAX_YAW_DEG = 30.0
MAX_FORCE_N = 10.0
MAX_REAPPROACH = 2


def _retreat(inp: SafetyInput, reason: str) -> SafetyDecision:
    if inp.reapproach_count >= MAX_REAPPROACH:
        return SafetyDecision("ask", 0.0, f"{reason}; asking before trying again")
    return SafetyDecision("retreat", SLOW_MPS, reason)


def safety_gate(inp: SafetyInput) -> SafetyDecision:
    if inp.stop_heard:
        return SafetyDecision("halt", 0.0, "stop heard")
    if inp.contact_force_n > MAX_FORCE_N:
        return _retreat(inp, f"contact {inp.contact_force_n:.0f} N")
    d, m = inp.dist_to_mouth_m, inp.mouth
    if d < WATCH_M:
        if m.hand_raised:
            return _retreat(inp, "hand raised")
        if not m.visible:
            return SafetyDecision("pause", 0.0, "face lost")
        if abs(m.yaw_deg) > MAX_YAW_DEG:
            return _retreat(inp, "head turned away")
    if d <= STOP_SHORT_M:
        return SafetyDecision("pause", 0.0, "at mouth")
    if d < NEAR_FACE_M:
        if not inp.ready_heard:
            return SafetyDecision("pause", 0.0, "waiting for ready")
        return SafetyDecision("proceed", SLOW_MPS, "near face, slow")
    return SafetyDecision("proceed", FAST_MPS, "clear")
