"""Where things were last seen, with confidence that fades over time."""
from dataclasses import dataclass, field

import numpy as np

PRIOR = ["kitchen", "living", "bedroom", "hall"]
MISS_FACTOR = 0.2


@dataclass
class _Sighting:
    xyz: np.ndarray
    t: float
    room: str
    scale: float = 1.0
    misses: dict[str, float] = field(default_factory=dict)


class ObjectMemory:
    def __init__(self, half_life_s: float = 600.0):
        self.half_life = half_life_s
        self._seen: dict[str, _Sighting] = {}

    def see(self, label: str, xyz: np.ndarray, t: float, room: str) -> None:
        self._seen[label] = _Sighting(np.asarray(xyz, float).copy(), t, room)

    def miss(self, label: str, room: str, t: float) -> None:
        s = self._seen.get(label)
        if s is None:
            self._seen[label] = s = _Sighting(np.zeros(3), t, "", scale=0.0)
        s.misses[room] = t
        if s.room == room:
            s.scale *= MISS_FACTOR

    def confidence(self, label: str, t: float) -> float:
        s = self._seen.get(label)
        if s is None or not s.room:
            return 0.0
        return float(s.scale * 0.5 ** ((t - s.t) / self.half_life))

    def where(self, label: str) -> tuple[np.ndarray, str] | None:
        s = self._seen.get(label)
        return (s.xyz.copy(), s.room) if s and s.room else None

    def search_order(self, label: str, t: float, rooms: list[str]) -> list[str]:
        s = self._seen.get(label)
        prior = [r for r in PRIOR if r in rooms] + [r for r in rooms if r not in PRIOR]
        if s is None:
            return prior
        rest = sorted((r for r in prior if r != s.room), key=lambda r: r in s.misses)
        if s.room and s.room in rooms and self.confidence(label, t) >= MISS_FACTOR:
            return [s.room] + rest
        return rest + ([s.room] if s.room in rooms else [])
