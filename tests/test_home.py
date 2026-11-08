import datetime

from sip.home import due_dose, next_dose


def test_dose_due_only_in_its_window_and_once():
    p = {"dose_times": ["09:00", "18:00"]}
    at = datetime.datetime(2026, 11, 8, 9, 10)
    assert due_dose(p, set(), at) == "09:00"
    assert due_dose(p, {"09:00"}, at) is None
    assert due_dose(p, set(), datetime.datetime(2026, 11, 8, 10, 0)) is None
    assert next_dose(p, at) == "18:00"
