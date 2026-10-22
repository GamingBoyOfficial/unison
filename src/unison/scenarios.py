"""Scenarios: an arena + a task + how the episode starts, bundled under one name."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .arena import Arena, TubeArena, y_branch
from .bots import make_bots, make_rolling_bots
from .geometry import bifurcation
from .rolling import RollingParams, RollingSimulator
from .simulator import Simulator
from .task import BifurcationTask, ReachTargetTask, SplitTask


@dataclass(frozen=True)
class Scenario:
    name: str
    make_arena: Callable[[], Arena]
    make_task: Callable[[Arena], object]
    start_center: tuple[float, float]
    steps: int
    kind: str = "point"        # "point" (versions 1-2) or "rolling" (version 3)
    noise: float = 0.02
    rolling: RollingParams | None = None   # physics settings for kind == "rolling"

    def new_sim(self, rng: np.random.Generator, arena: Arena, n_bots: int = 300,
                spread: float = 0.15, noise: float | None = None):
        """Create the bots and simulator for one episode (bots are drawn from rng first,
        so a seed reproduces the same swarm)."""
        noise = self.noise if noise is None else noise
        if self.kind == "rolling":
            bots = make_rolling_bots(rng, n=n_bots, spread=spread, start_center=self.start_center)
            return RollingSimulator(arena, bots, rng, noise=noise, params=self.rolling)
        bots = make_bots(rng, n=n_bots, spread=spread, start_center=self.start_center)
        return Simulator(arena, bots, rng, noise=noise)


def tube_scenario() -> Scenario:
    """Phase 2: reach the end of a straight tube against steady flow."""
    return Scenario(
        name="tube",
        make_arena=lambda: TubeArena(),
        make_task=lambda a: ReachTargetTask(target_x=a.length - 1.0),
        start_center=(0.5, 0.0),
        steps=600,
    )


def split_scenario(steps: int = 800, name: str = "split") -> Scenario:
    """Phase 3: split the swarm between two arms of a Y-branch, with pulsing
    flow and sticky walls. `steps` is the time budget: 800 is a tight deadline
    (the main benchmark), 1600 gives slow bots time to arrive."""
    return Scenario(
        name=name,
        make_arena=lambda: y_branch(),
        make_task=lambda a: SplitTask(target_x=11.0),
        start_center=(0.5, 0.0),
        steps=steps,
    )


def bifurcation_scenario(steps: int = 800, name: str = "bifurcation") -> Scenario:
    """Version 2 geometry: a parent vessel splitting into two curved, tapering
    daughters at a sharp carina, with Murray's-law flow, pulsing and sticky walls.
    Flow strength (0.8) was calibrated so the straight push scores about 0.6, leaving
    headroom for a learned controller; at 0.6 the realistic geometry is too easy."""
    return Scenario(
        name=name,
        make_arena=lambda: bifurcation(flow_speed=0.8),
        make_task=lambda a: BifurcationTask(a),
        start_center=(0.5, 0.0),
        steps=steps,
    )


def rolling_scenario(steps: int = 800, name: str = "rolling", flow_speed: float = 5.0,
                     params: RollingParams | None = None) -> Scenario:
    """Version 3: finite-size swarms that roll along the walls of a bifurcation, in the
    flow regime of the experiments (centre-line flow 0.5 mm/s with pulsing), with grip
    limited by traction. Physics from docs/GROUNDING.md; the field's lateral component
    (which wall to press against) is an assumption beyond the paper's hardware."""
    return Scenario(
        name=name,
        make_arena=lambda: bifurcation(flow_speed=flow_speed, wall_drag=0.0),
        make_task=lambda a: BifurcationTask(a),
        start_center=(0.5, 0.0),
        steps=steps,
        kind="rolling",
        noise=0.01,
        rolling=params,
    )


SCENARIOS = {
    "tube": tube_scenario,
    "split": split_scenario,
    "split_long": lambda: split_scenario(steps=1600, name="split_long"),
    "bifurcation": bifurcation_scenario,
    "rolling": rolling_scenario,
}
