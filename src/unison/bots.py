"""Bots: every bot is just a few numbers. No two are identical."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Bots:
    pos: np.ndarray            # (n, 2) positions
    speed_factor: np.ndarray   # (n,) how strongly each bot reacts to the field
    radius: np.ndarray | None = None   # (n,) swarm radius (rolling-swarm physics only)
    v0: np.ndarray | None = None       # (n,) still-water rolling speed (rolling physics only)

    @property
    def n(self) -> int:
        return len(self.speed_factor)


def make_bots(rng: np.random.Generator, n: int = 300, spread: float = 0.15,
              start_center=(0.0, 0.0), start_radius: float = 0.3) -> Bots:
    """Create n bots with random size and magnetism.

    spread controls how different the bots are (0.01 = nearly identical).
    Bigger bots feel more drag; stronger magnetism means a stronger push.
    """
    size = rng.normal(1.0, spread, n).clip(0.5, 1.5)
    magnetism = rng.normal(1.0, spread, n).clip(0.5, 1.5)
    drag = 1.0 + 0.5 * (size - 1.0)
    mobility = 1.0 / drag
    speed_factor = mobility * magnetism
    pos = np.asarray(start_center) + rng.normal(0.0, start_radius, (n, 2))
    return Bots(pos=pos, speed_factor=speed_factor)


def make_rolling_bots(rng: np.random.Generator, n: int = 300, spread: float = 0.15,
                      start_center=(0.0, 0.0), start_radius: float = 0.3,
                      mean_radius: float = 0.27, size_sd: float = 0.25,
                      v_sat: float = 1.6, a_half: float = 0.04) -> Bots:
    """Finite-size rolling swarms (version 3 physics).

    Units: length 1 = about 75 micrometres, speed 1 = about 100 micrometres/s.
    Radius: lognormal around mean_radius (0.27 = about 20 micrometres).
    Still-water rolling speed rises with radius and saturates, v_sat * a / (a + a_half)
    (shape from Fig. 5c of Ahmed et al.), times a magnetic-response factor with relative
    spread `spread` (an assumption: the paper gives no such spread).
    """
    radius = np.clip(mean_radius * np.exp(rng.normal(0.0, size_sd, n)), 0.08, 0.9)
    response = rng.normal(1.0, spread, n).clip(0.5, 1.5)
    v0 = v_sat * radius / (radius + a_half) * response
    pos = np.asarray(start_center) + rng.normal(0.0, start_radius, (n, 2))
    return Bots(pos=pos, speed_factor=v0 / v0.mean(), radius=radius, v0=v0)
