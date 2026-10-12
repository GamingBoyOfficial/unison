"""The simulator: moves every bot one step under ONE global field."""

from __future__ import annotations

import numpy as np

from .arena import Arena
from .bots import Bots


class Simulator:
    def __init__(self, arena: Arena, bots: Bots, rng: np.random.Generator,
                 dt: float = 0.05, noise: float = 0.02):
        self.arena = arena
        self.bots = bots
        self.rng = rng
        self.dt = dt
        self.noise = noise
        self.bots.pos = self.arena.contain(self.bots.pos)

    @property
    def pos(self) -> np.ndarray:
        return self.bots.pos

    def step(self, field: np.ndarray) -> None:
        """Advance one time step.

        field: the single control signal, a 2D vector (length <= 1).
        Each bot moves by its own reaction to the field, plus the fluid flow,
        plus a little random jitter (Brownian motion).
        """
        field = np.asarray(field, dtype=float)
        vel = self.bots.speed_factor[:, None] * field[None, :]
        vel = vel + self.arena.flow_at(self.bots.pos)
        jitter = self.rng.normal(0.0, self.noise, self.bots.pos.shape)
        self.bots.pos = self.arena.contain(self.bots.pos + vel * self.dt + jitter)
