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


@dataclass
class SafetyDecision:
    action: Literal["proceed", "pause", "retreat", "ask", "halt"]
    max_speed_mps: float
    reason: str
