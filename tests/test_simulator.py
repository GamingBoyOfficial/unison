import numpy as np

from unison.arena import TubeArena
from unison.bots import make_bots
from unison.controllers import PushTowardTarget, RandomController
from unison.evaluate import run_episode
from unison.simulator import Simulator


def make_sim(seed=0, spread=0.15, noise=0.02):
    rng = np.random.default_rng(seed)
    arena = TubeArena()
    bots = make_bots(rng, n=100, spread=spread, start_center=(5.0, 0.0))
    return Simulator(arena, bots, rng, noise=noise)


def test_bots_stay_inside_walls():
    sim = make_sim()
    for _ in range(300):
        sim.step(np.array([0.0, 1.0]))
    xmin, xmax, ymin, ymax = sim.arena.bounds
    assert sim.pos[:, 0].min() >= xmin and sim.pos[:, 0].max() <= xmax
    assert sim.pos[:, 1].min() >= ymin and sim.pos[:, 1].max() <= ymax


def test_flow_is_zero_at_the_wall_and_fastest_in_the_middle():
    arena = TubeArena(half_width=1.0, flow_speed=0.6)
    flow = arena.flow_at(np.array([[5.0, 1.0], [5.0, 0.0]]))
    assert abs(flow[0, 0]) < 1e-12
    assert abs(flow[1, 0]) > abs(flow[0, 0])


def test_identical_bots_react_identically():
    rng = np.random.default_rng(0)
    bots = make_bots(rng, n=50, spread=0.0, start_radius=0.0, start_center=(5.0, 0.0))
    assert np.allclose(bots.speed_factor, bots.speed_factor[0])


def test_same_seed_gives_same_result():
    a = run_episode(PushTowardTarget(), seed=5, steps=100)
    b = run_episode(PushTowardTarget(), seed=5, steps=100)
    assert a == b


def test_pushing_toward_target_beats_random():
    push = run_episode(PushTowardTarget(), seed=3, steps=300)
    rand = run_episode(RandomController(), seed=3, steps=300)
    assert push > rand
