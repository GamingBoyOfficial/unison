"""Run a controller on many fresh random swarms and report honest statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arena import TubeArena
from .bots import make_bots
from .controllers import Controller
from .simulator import Simulator
from .task import ReachTargetTask


def run_episode(controller: Controller, seed: int, steps: int = 600,
                n_bots: int = 300, spread: float = 0.15, noise: float = 0.02,
                arena: TubeArena | None = None) -> float:
    """One episode on a freshly randomized swarm. Returns the score (0 to 1)."""
    rng = np.random.default_rng(seed)
    arena = arena or TubeArena()
    bots = make_bots(rng, n=n_bots, spread=spread, start_center=(0.5, 0.0))
    sim = Simulator(arena, bots, rng, noise=noise)
    task = ReachTargetTask(target_x=arena.length - 1.0)
    for _ in range(steps):
        field = controller.act(task.observe(sim.pos), rng)
        sim.step(field)
    return task.fraction_reached(sim.pos)


@dataclass
class Result:
    mean: float
    ci95: float
    scores: np.ndarray


def evaluate(controller: Controller, n_episodes: int = 50, base_seed: int = 1000,
             **kwargs) -> Result:
    """Mean score with a 95% confidence interval over n_episodes random swarms."""
    scores = np.array([run_episode(controller, base_seed + i, **kwargs)
                       for i in range(n_episodes)])
    ci = 1.96 * scores.std(ddof=1) / np.sqrt(n_episodes)
    return Result(mean=float(scores.mean()), ci95=float(ci), scores=scores)
