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


def test_committed_fractions_sum_to_one():
    task = SplitTask(target_x=11.0)
    pos = np.array([[1.0, 0.0], [7.0, 1.0], [8.0, -1.0], [9.0, 1.5]])
    main, up, low = task.committed(pos)
    assert abs(main + up + low - 1.0) < 1e-12
    assert (main, up, low) == (0.25, 0.5, 0.25)


def test_split_long_scenario_has_a_longer_time_budget():
    from unison.scenarios import SCENARIOS
    assert SCENARIOS["split_long"]().steps == 2 * SCENARIOS["split"]().steps


def test_replaying_recorded_actions_reproduces_the_episode():
    from unison.imitation import replay_demo
    s = split_scenario()
    steps = 150
    actions = np.tile(np.array([1.0, 0.0]), (steps, 1))
    demo = replay_demo(s, seed=11, actions=actions)
    direct = run_episode(PushTowardTarget(), seed=11, scenario=s, steps=steps)
    assert demo.score == direct
    assert demo.obs.shape == (steps, 12)


def test_playlog_keeps_latest_attempt_and_continues_seeds():
    from unison.playlog import latest_per_seed, next_seed_base
    rows = [
        {"scenario": "split", "seed": "5001", "driver": "Manual", "score": "0.74"},
        {"scenario": "split", "seed": "5002", "driver": "Manual", "score": "0.80"},
        {"scenario": "split", "seed": "5001", "driver": "Manual", "score": "0.81"},
    ]
    kept = latest_per_seed(rows)
    assert len(kept) == 2
    assert [r["score"] for r in kept if r["seed"] == "5001"] == ["0.81"]
    assert next_seed_base(rows, "split", 5000) == 5002
    assert next_seed_base(rows, "tube", 5000) == 5000


def test_numpy_policy_matches_the_sklearn_model_it_came_from():
    import pytest
    pytest.importorskip("sklearn")
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from unison.policy import NumpyPolicy
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 12))
    y = np.tanh(X[:, :2])
    model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(8, 8),
                                                         max_iter=200, random_state=0)).fit(X, y)
    policy = NumpyPolicy.from_sklearn(model)
    for row in X[:20]:
        want = model.predict(row[None, :])[0]
        n = np.linalg.norm(want)
        want = want / n if n > 1 else want
        assert np.allclose(policy.act(row, None), want, atol=1e-9)
    mutated = policy.with_params(policy.params + 0.1)
    assert not np.allclose(mutated.act(X[0], None), policy.act(X[0], None))


def test_small_sample_confidence_interval_uses_t_not_1_96():
    from unison.stats import mean_ci, t_crit
    assert t_crit(6) == 2.447 and t_crit(500) == 1.96
    m, half = mean_ci([0.8, 0.9, 0.85, 0.95, 0.75, 0.9, 0.8])
    sd = np.std([0.8, 0.9, 0.85, 0.95, 0.75, 0.9, 0.8], ddof=1)
    assert abs(half - 2.447 * sd / np.sqrt(7)) < 1e-12
    assert half > 1.96 * sd / np.sqrt(7)


def test_vessel_arena_keeps_bots_inside_and_flows_toward_the_root():
    from unison.geometry import bifurcation
    a = bifurcation()
    rng = np.random.default_rng(3)
    lo_x, hi_x, lo_y, hi_y = a.bounds
    pts = np.column_stack([rng.uniform(lo_x - 1, hi_x + 1, 3000), rng.uniform(lo_y - 1, hi_y + 1, 3000)])
    out = a.contain(pts)
    t, closest, d, r = a._locate(out)
    assert (d - r).min(axis=0).max() <= 1e-9              # everything is inside after contain
    assert np.allclose(a.contain(out), out)               # contain is idempotent
    f = a.flow_at(np.array([[3.0, 0.0], [3.0, 0.999]]))
    assert f[0, 0] < -0.5 and abs(f[1, 0]) < 0.01         # toward the root, ~zero at the wall


def test_murray_flow_shares_add_up_at_the_branch():
    from unison.geometry import bifurcation
    a = bifurcation()
    daughters = {tb.branch: tb.flow for tb in a.tubes if tb.branch > 0}
    assert abs(sum(daughters.values()) - 1.0) < 1e-12 and len(daughters) == 2


def test_bifurcation_task_scores_balance_like_the_split_task():
    from unison.geometry import bifurcation
    from unison.task import BifurcationTask
    a = bifurcation()
    task = BifurcationTask(a)
    up, low = task._centres
    both = np.array([up, low])
    one_sided = np.array([up, up + 0.05])
    assert task.score(both) == 1.0 and task.score(one_sided) == 0.0
    assert task.observe(both).shape == (9,)


def test_bifurcation_episode_runs_and_scores_between_0_and_1():
    from unison.scenarios import SCENARIOS
    score = run_episode(PushTowardTarget(), seed=4, scenario=SCENARIOS["bifurcation"](), steps=150)
    assert 0.0 <= score <= 1.0


# ---- version 3: rolling swarms ------------------------------------------------------------

def _rolling_bot(a, y=0.3, x=300.0):
    from unison.bots import Bots
    v0 = 1.6 * a / (a + 0.04)
    return Bots(pos=np.array([[x, y]]), speed_factor=np.ones(1), radius=np.array([a]), v0=np.array([v0]))


def test_contain_margin_keeps_swarm_edges_off_the_wall():
    from unison.geometry import bifurcation
    arena = bifurcation()
    rng = np.random.default_rng(5)
    lo_x, hi_x, lo_y, hi_y = arena.bounds
    pts = np.column_stack([rng.uniform(lo_x, hi_x, 2000), rng.uniform(lo_y, hi_y, 2000)])
    margin = np.full(2000, 0.3)
    out = arena.contain_margin(pts, margin)
    t, closest, d, r = arena._locate(out)
    assert (d - np.maximum(r - 0.3, 0.02)).min(axis=0).max() <= 1e-9


def test_rolling_upstream_speed_follows_v0_minus_kappa_times_local_flow():
    from unison.geometry import straight_vessel
    from unison.rolling import RollingSimulator
    arena = straight_vessel(length=400.0, flow_speed=3.0)
    sim = RollingSimulator(arena, _rolling_bot(0.27), np.random.default_rng(0), noise=0.0)
    xs = []
    for _ in range(200):
        sim.step(np.array([1.0, 0.0]))
        xs.append(sim.pos[0, 0])
    measured = (xs[-1] - xs[-50]) / (49 * sim.dt)
    u_local = np.linalg.norm(arena.flow_at(sim.pos, 0.0)[0])
    expected = sim.bots.v0[0] - sim.kappa[0] * u_local
    assert abs(measured - expected) < 0.03 * sim.bots.v0[0]
    assert sim.grip()[0] > 0.99


def test_rolling_swarm_is_swept_away_when_flow_exceeds_the_stalling_flow():
    from unison.geometry import straight_vessel
    from unison.rolling import RollingSimulator
    arena = straight_vessel(length=400.0, flow_speed=16.0)
    sim = RollingSimulator(arena, _rolling_bot(0.27), np.random.default_rng(0), noise=0.0)
    x0 = None
    for k in range(160):
        sim.step(np.array([1.0, 0.0]))
        if k == 60:
            x0 = sim.pos[0, 0]
    assert sim.pos[0, 0] < x0                      # carried downstream despite rolling forward


def test_free_swarm_without_grip_is_carried_by_the_flow():
    from unison.geometry import straight_vessel
    from unison.rolling import RollingParams, RollingSimulator
    arena = straight_vessel(length=400.0, flow_speed=2.0)
    sim = RollingSimulator(arena, _rolling_bot(0.27, y=0.0), np.random.default_rng(0), noise=0.0,
                           params=RollingParams(acoustic_drift=0.0))
    x0 = sim.pos[0, 0]
    for _ in range(40):
        sim.step(np.zeros(2))
    speed = (x0 - sim.pos[0, 0]) / (40 * sim.dt)
    assert abs(speed - 2.0) < 0.05                 # moves at the centre-line flow speed


def test_rolling_scenario_is_reproducible_and_scores_between_0_and_1():
    from unison.scenarios import SCENARIOS
    s = SCENARIOS["rolling"]()
    a = run_episode(PushTowardTarget(), seed=9, scenario=s, steps=120)
    b = run_episode(PushTowardTarget(), seed=9, scenario=s, steps=120)
    assert a == b and 0.0 <= a <= 1.0
