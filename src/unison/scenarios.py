"""Scenarios: an arena + a task + how the episode starts, bundled under one name."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .arena import Arena, TubeArena, y_branch
from .geometry import bifurcation
from .task import BifurcationTask, ReachTargetTask, SplitTask


@dataclass(frozen=True)
class Scenario:
    name: str
    make_arena: Callable[[], Arena]
    make_task: Callable[[Arena], object]
    start_center: tuple[float, float]
    steps: int


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


SCENARIOS = {
    "tube": tube_scenario,
    "split": split_scenario,
    "split_long": lambda: split_scenario(steps=1600, name="split_long"),
    "bifurcation": bifurcation_scenario,
}
