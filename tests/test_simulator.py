import numpy as np

from unison.arena import TubeArena, y_branch
from unison.bots import make_bots
from unison.controllers import PushTowardTarget, RandomController
from unison.evaluate import run_episode
from unison.scenarios import split_scenario, tube_scenario
from unison.simulator import Simulator
from unison.task import SplitTask


def make_sim(arena=None, seed=0, spread=0.15, noise=0.02, start=(5.0, 0.0)):
    rng = np.random.default_rng(seed)
    arena = arena or TubeArena()
    bots = make_bots(rng, n=100, spread=spread, start_center=start)
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
    s = tube_scenario()
    a = run_episode(PushTowardTarget(), seed=5, scenario=s, steps=100)
    b = run_episode(PushTowardTarget(), seed=5, scenario=s, steps=100)
    assert a == b


def test_pushing_toward_target_beats_random():
    s = tube_scenario()
    push = run_episode(PushTowardTarget(), seed=3, scenario=s, steps=300)
    rand = run_episode(RandomController(), seed=3, scenario=s, steps=300)
    assert push > rand


def test_y_branch_keeps_bots_inside_and_never_in_the_divider():
    arena = y_branch()
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(-1, 13, 2000), rng.uniform(-4, 4, 2000)])
    out = arena.contain(pts)
    in_main = ((out[:, 0] >= 0) & (out[:, 0] <= 6) & (abs(out[:, 1]) <= 1))
    in_up = ((out[:, 0] >= 6) & (out[:, 0] <= 12) & (out[:, 1] >= 0.5) & (out[:, 1] <= 2.5))
    in_low = ((out[:, 0] >= 6) & (out[:, 0] <= 12) & (out[:, 1] <= -0.5) & (out[:, 1] >= -2.5))
    assert np.all(in_main | in_up | in_low)


def test_pulsing_flow_changes_with_time():
    arena = TubeArena(pulse_amp=0.5, pulse_period=4.0)
    p = np.array([[5.0, 0.0]])
    peak = arena.flow_at(p, t=1.0)[0, 0]      # sin = +1
    trough = arena.flow_at(p, t=3.0)[0, 0]    # sin = -1
    assert abs(peak) > abs(trough)


def test_sticky_walls_reduce_push_near_the_wall():
    arena = TubeArena(half_width=1.0, wall_drag=0.6)
    d = arena.drag_factor_at(np.array([[5.0, 0.0], [5.0, 1.0]]))
    assert d[0] == 1.0 and abs(d[1] - 0.4) < 1e-12


def test_split_score_rewards_balance_and_punishes_one_sided():
    task = SplitTask(target_x=11.0)
    balanced = np.array([[11.5, 1.0], [11.5, -1.0]])
    one_sided = np.array([[11.5, 1.0], [11.5, 1.2]])
    assert task.score(balanced) == 1.0
    assert task.score(one_sided) == 0.0


def test_split_episode_runs_and_scores_between_0_and_1():
    s = split_scenario()
    score = run_episode(PushTowardTarget(), seed=2, scenario=s, steps=200)
    assert 0.0 <= score <= 1.0
