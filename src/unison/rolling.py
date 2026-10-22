"""Version 3 physics: swarms that roll along the vessel wall.

Grounded in Ahmed et al. (arXiv 2005.12703), see docs/GROUNDING.md:

  * A swarm is a finite-size disc. It rolls only in contact with the wall; away from the
    wall it has no propulsion and is carried by the flow.
  * In contact, upstream speed falls roughly linearly with the imposed flow
    (V = V0 - kappa x flow), a weak coupling because wall friction resists the drag.
  * When the flow is so strong that rolling cannot hold position, traction is lost and
    the swarm is swept away.

Assumptions made here that the paper does not give (all listed in docs/GROUNDING.md):
the controller's lateral command (an acoustic-like push across the vessel, which the
paper's fixed transducer does not provide), the contact range, and the spread in magnetic
response. The always-on acoustic pull toward the nearest wall (3.0 = about 300 um/s) is
also a tuned assumption: in tests, 1.0 let swarms detach where the wall bends away at the
bifurcation (flow then flushed them back), 3.0 held them, 6.0 over-pinned them. The
paper says the acoustic force is what presses the swarm to the wall but gives no value
in the text read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bots import Bots
from .geometry import VesselArena


@dataclass(frozen=True)
class RollingParams:
    kappa_plot: float = 0.15       # slope of upstream speed vs centre-line flow in a reference
                                   # capillary (Fig. 5b fits: 0.09 for 15 um, ~0.3 for 40 um)
    contact_gap: float = 0.05      # gap (swarm edge to wall) within which rolling has full grip
    contact_soft: float = 0.10     # further gap over which grip fades to zero
    traction_margin: float = 1.1   # grip is lost beyond this multiple of the stalling flow
    traction_soft: float = 0.05    # softness of that cutoff (fraction of the stalling flow)
    lateral_speed: float = 3.0     # speed of the controller's lateral push (assumed)
    acoustic_drift: float = 3.0    # always-on pull toward the nearest wall (assumed; see below)
    wall_gap_min: float = 0.02     # closest the swarm edge can get to the wall


class RollingSimulator:
    """Moves finite-size rolling swarms under ONE global field.

    The field f = (fx, fy) acts in two ways: its component along the local WALL drives
    rolling (only effective in wall contact), and its component across the vessel axis
    pushes the swarm sideways, which lets the controller choose which wall to press against.
    """

    def __init__(self, arena: VesselArena, bots: Bots, rng: np.random.Generator,
                 dt: float = 0.05, noise: float = 0.01, params: RollingParams | None = None):
        if bots.radius is None or bots.v0 is None:
            raise ValueError("RollingSimulator needs bots made by make_rolling_bots")
        self.arena, self.bots, self.rng = arena, bots, rng
        self.dt, self.noise = dt, noise
        self.p = params or RollingParams()
        self.t = 0.0
        a = bots.radius
        self.kappa = self.p.kappa_plot / (a * (2.0 - np.minimum(a, 1.0)))   # coupling to local flow
        self._margin = a + self.p.wall_gap_min
        self.bots.pos = arena.contain_margin(bots.pos, self._margin)

    @property
    def pos(self) -> np.ndarray:
        return self.bots.pos

    def grip(self, pos=None):
        """How firmly each swarm is rolling on the wall (0 = free in the fluid, 1 = full grip)."""
        pos = self.bots.pos if pos is None else pos
        i, _, d, r, _, tau, _, _ = self.arena.local_frame(pos)
        gap = r - d - self.bots.radius
        p = self.p
        contact = np.clip(1.0 - (gap - p.contact_gap) / p.contact_soft, 0.0, 1.0)
        flow = self.arena.flow_at(pos, self.t)
        u_ax = np.linalg.norm(flow, axis=1)
        u_stop = self.bots.v0 / self.kappa
        traction = 1.0 / (1.0 + np.exp((u_ax - p.traction_margin * u_stop) / (p.traction_soft * u_stop)))
        return contact * traction

    def step(self, field: np.ndarray) -> None:
        f = np.asarray(field, dtype=float)
        p, pos, b = self.p, self.bots.pos, self.bots
        i, _, d, r, c, tau, nu, e = self.arena.local_frame(pos)
        flow = self.arena.flow_at(pos, self.t)
        gap = r - d - b.radius
        contact = np.clip(1.0 - (gap - p.contact_gap) / p.contact_soft, 0.0, 1.0)
        u_ax = np.linalg.norm(flow, axis=1)
        u_stop = b.v0 / self.kappa
        traction = 1.0 / (1.0 + np.exp((u_ax - p.traction_margin * u_stop) / (p.traction_soft * u_stop)))
        w = contact * traction

        f_lat = nu @ f                                   # push across the vessel
        wall_n = (pos - c) / np.maximum(d, 1e-9)[:, None]   # from the axis toward the wall
        along_wall = f[None, :] - (wall_n @ f)[:, None] * wall_n   # field's part along the wall
        v_roll = (w * b.v0)[:, None] * along_wall        # rolling follows the wall's own direction
        v_flow = (1.0 - w * (1.0 - self.kappa))[:, None] * flow
        drift = p.acoustic_drift * np.tanh(e / 0.05)     # toward the nearest wall
        v_lat = (p.lateral_speed * f_lat + drift)[:, None] * nu
        vel = v_roll + v_flow + v_lat
        jitter = self.rng.normal(0.0, self.noise, pos.shape)
        self.bots.pos = self.arena.contain_margin(pos + vel * self.dt + jitter, self._margin)
        self.t += self.dt
