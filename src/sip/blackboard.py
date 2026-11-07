"""Thread-safe latest-value store plus an append-only event log."""
import threading
from typing import Any

from sip.types import Event


class Blackboard:
    def __init__(self):
        self._lock = threading.Lock()
        self._vals: dict[str, Any] = {}
        self._events: list[Event] = []

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._vals[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._vals.get(key, default)

    def log(self, event: Event) -> None:
        with self._lock:
            self._events.append(event)

    def events(self) -> list[Event]:
        with self._lock:
            return list(self._events)
