"""Scenarios: an arena + a task + how the episode starts, bundled under one name."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .arena import Arena, TubeArena, y_branch
from .task import ReachTargetTask, SplitTask


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


def split_scenario() -> Scenario:
    """Phase 3: split the swarm between two arms of a Y-branch, with pulsing
    flow and sticky walls."""
    return Scenario(
        name="split",
        make_arena=lambda: y_branch(),
        make_task=lambda a: SplitTask(target_x=11.0),
        start_center=(0.5, 0.0),
        steps=800,
    )


SCENARIOS = {"tube": tube_scenario, "split": split_scenario}
