"""Arenas: the shape the bots live in, and the fluid flowing through it.

Every arena answers two questions for the simulator:
  contain(pos)  -> where do bots end up after hitting a wall?
  flow_at(pos)  -> how fast does the fluid push a bot at this position?

New shapes (Y-branch, a vessel loaded from a file) only need to implement
these two methods, so the rest of the project never changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Arena(ABC):
    @property
    @abstractmethod
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax), used for drawing and for starting positions."""

    @abstractmethod
    def contain(self, pos: np.ndarray) -> np.ndarray:
        """Return positions pushed back inside the walls. pos has shape (n, 2)."""

    @abstractmethod
    def flow_at(self, pos: np.ndarray) -> np.ndarray:
        """Fluid velocity at each position. Shape (n, 2)."""


class BoxArena(Arena):
    """Square box with still fluid. This is the Phase 1 arena."""

    def __init__(self, half_size: float = 5.0):
        self.half_size = half_size

    @property
    def bounds(self):
        h = self.half_size
        return (-h, h, -h, h)

    def contain(self, pos):
        return np.clip(pos, -self.half_size, self.half_size)

    def flow_at(self, pos):
        return np.zeros_like(pos)


class TubeArena(Arena):
    """Straight tube with parabolic (Poiseuille) flow.

    Flow is fastest in the middle and zero at the walls, like real fluid in a
    pipe. flow_direction = -1 pushes bots toward the left end, so a bot has to
    swim AGAINST the flow to reach a target at the right end.
    """

    def __init__(self, length: float = 10.0, half_width: float = 1.0,
                 flow_speed: float = 0.6, flow_direction: int = -1):
        self.length = length
        self.half_width = half_width
        self.flow_speed = flow_speed
        self.flow_direction = flow_direction

    @property
    def bounds(self):
        return (0.0, self.length, -self.half_width, self.half_width)

    def contain(self, pos):
        out = pos.copy()
        out[:, 0] = np.clip(out[:, 0], 0.0, self.length)
        out[:, 1] = np.clip(out[:, 1], -self.half_width, self.half_width)
        return out

    def flow_at(self, pos):
        r = pos[:, 1] / self.half_width
        u = self.flow_direction * self.flow_speed * (1.0 - r**2)
        flow = np.zeros_like(pos)
        flow[:, 0] = np.clip(u, -abs(self.flow_speed), abs(self.flow_speed))
        return flow
