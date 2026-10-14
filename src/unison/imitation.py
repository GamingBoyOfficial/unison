"""Learn from human demonstrations (behavior cloning).

play.py saves a human's key presses for each episode, plus the episode's seed.
Because the simulator is deterministic given a seed, replaying those key presses
rebuilds exactly what the swarm looked like at every step. That gives
(observation -> action) pairs a model can learn from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bots import make_bots
from .controllers import Controller
from .evaluate import build_obs
from .playlog import latest_per_seed, read_rows
from .scenarios import Scenario
from .simulator import Simulator


@dataclass
class Demo:
    seed: int
    obs: np.ndarray            # (T, D) what the controller would have seen
    actions: np.ndarray        # (T, 2) the field the human applied
    score: float               # score obtained by replaying the actions
    logged_score: float | None # score recorded when the human played


def replay_demo(scenario: Scenario, seed: int, actions: np.ndarray,
                n_bots: int = 300, spread: float = 0.15, noise: float = 0.02,
                logged_score: float | None = None) -> Demo:
    """Replay recorded actions on the same seed and record the observations."""
    rng = np.random.default_rng(seed)
    arena = scenario.make_arena()
    task = scenario.make_task(arena)
    bots = make_bots(rng, n=n_bots, spread=spread, start_center=scenario.start_center)
    sim = Simulator(arena, bots, rng, noise=noise)
    steps = len(actions)
    obs = []
    for i in range(steps):
        obs.append(build_obs(task, sim, i / steps))
        sim.step(actions[i])
    return Demo(seed=seed, obs=np.array(obs), actions=np.asarray(actions, dtype=float),
                score=task.score(sim.pos), logged_score=logged_score)


def load_demos(scenario_name: str, scenario: Scenario, results_dir: str = "results",
               driver: str = "Manual", tolerance: float = 2e-3):
    """Load every logged manual episode for a scenario and replay it.

    Returns (demos, skipped) where skipped lists (seed, reason) for episodes that
    could not be used, for example if the physics changed since they were recorded
    and the replay no longer reproduces the logged score.
    """
    results = Path(results_dir)
    log = results / "play_scores.csv"
    if not log.exists():
        raise FileNotFoundError(f"{log} not found. Play some episodes with play.py first.")
    demos, skipped = [], []
    for row in latest_per_seed(read_rows(log), scenario_name):
        if row["driver"] != driver:
            continue
        seed = int(row["seed"])
        path = results / f"actions_{scenario_name}_{seed}.csv"
        if not path.exists():
            skipped.append((seed, "no actions file"))
            continue
        actions = np.loadtxt(path, delimiter=",", skiprows=1)
        demo = replay_demo(scenario, seed, actions, logged_score=float(row["score"]))
        if abs(demo.score - demo.logged_score) > tolerance:
            skipped.append((seed, f"replay gave {demo.score:.3f}, logged {demo.logged_score:.3f}"))
            continue
        demos.append(demo)
    return demos, skipped


class LearnedController(Controller):
    """Wraps a trained model that maps an observation to a field vector."""

    def __init__(self, model):
        self.model = model

    def act(self, obs, rng):
        f = self.model.predict(obs[None, :])[0]
        n = np.linalg.norm(f)
        return f / n if n > 1.0 else f
