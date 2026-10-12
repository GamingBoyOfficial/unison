"""Run a controller on many fresh random swarms and report honest statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bots import make_bots
from .controllers import Controller
from .scenarios import Scenario
from .simulator import Simulator


def build_obs(task, sim: Simulator, progress: float) -> np.ndarray:
    """Task summary + [progress, sin(heartbeat), cos(heartbeat)]."""
    s, c = sim.arena.pulse_phase(sim.t)
    return np.concatenate([task.observe(sim.pos), [progress, s, c]])


def run_episode(controller: Controller, seed: int, scenario: Scenario,
                n_bots: int = 300, spread: float = 0.15, noise: float = 0.02,
                steps: int | None = None) -> float:
    """One episode on a freshly randomized swarm. Returns the score (0 to 1)."""
    rng = np.random.default_rng(seed)
    arena = scenario.make_arena()
    task = scenario.make_task(arena)
    bots = make_bots(rng, n=n_bots, spread=spread, start_center=scenario.start_center)
    sim = Simulator(arena, bots, rng, noise=noise)
    steps = steps or scenario.steps
    for i in range(steps):
        field = controller.act(build_obs(task, sim, i / steps), rng)
        sim.step(field)
    return task.score(sim.pos)


@dataclass
class Result:
    mean: float
    ci95: float
    scores: np.ndarray


def evaluate(controller: Controller, scenario: Scenario, n_episodes: int = 50,
             base_seed: int = 1000, **kwargs) -> Result:
    """Mean score with a 95% confidence interval over n_episodes random swarms."""
    scores = np.array([run_episode(controller, base_seed + i, scenario, **kwargs)
                       for i in range(n_episodes)])
    ci = 1.96 * scores.std(ddof=1) / np.sqrt(n_episodes)
    return Result(mean=float(scores.mean()), ci95=float(ci), scores=scores)
