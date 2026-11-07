"""Shared plain data types."""
from dataclasses import dataclass


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float
