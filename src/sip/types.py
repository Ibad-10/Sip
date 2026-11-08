"""Shared plain data types."""
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float



@dataclass
class MouthState:
    visible: bool
    xyz: np.ndarray | None
    open_ratio: float
    yaw_deg: float
    hand_raised: bool
    source: str = "sim"


@dataclass
class SafetyInput:
    dist_to_mouth_m: float
    mouth: MouthState
    ready_heard: bool
    stop_heard: bool
    contact_force_n: float
    reapproach_count: int
    done_heard: bool = False


@dataclass
class SafetyDecision:
    action: Literal["proceed", "pause", "retreat", "ask", "halt"]
    max_speed_mps: float
    reason: str


@dataclass
class Detection:
    label: str
    conf: float
    box: tuple[int, int, int, int]  # u0, v0, u1, v1
    xyz: np.ndarray | None
    source: str = "seg"
    why: str = ""                 # how it was decided (scores, runner-up, second opinion)
    unsure: bool = False          # top labels too close or confidence low: ask for a second opinion


@dataclass
class Event:
    t: float
    kind: str   # state | plan | say | safety | perceive | alert | dose
    text: str
