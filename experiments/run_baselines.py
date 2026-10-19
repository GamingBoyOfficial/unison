"""Score the baseline controllers on every scenario.

These are the bars a learned controller must beat. For the split tasks the
SwitchController's switching moment is TUNED on one set of random swarms and then
scored on a different set, separately for each time budget, so the baseline is a
fair opponent and not a straw man.

Run:  python experiments/run_baselines.py
"""

from unison.controllers import (PushTowardTarget, RandomController,
                                SwitchController, WallHugging)
from unison.evaluate import evaluate
from unison.scenarios import SCENARIOS


def row(name, r):
    print(f"{name:<26}{r.mean:>12.3f}{'  +/- ' + format(r.ci95, '.3f'):>14}")


def header(title, note):
    print(f"\n{title}\n{note}\n")
    print(f"{'Controller':<26}{'Mean score':>12}{'  95% CI':>14}")
    print("-" * 52)


def tune_switch(scenario):
    best_p, best = None, -1.0
    for p in [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]:
        s = evaluate(SwitchController(p), scenario, n_episodes=8, base_seed=0).mean
        if s > best:
            best_p, best = p, s
    return best_p


# ---- Phase 2: straight tube ------------------------------------------------
tube = SCENARIOS["tube"]()
header("TUBE: get bots past the target against the flow",
       "Score = fraction of bots past the target line (50 random swarms each).")
for name, ctrl in [("Random", RandomController()),
                   ("Push toward target", PushTowardTarget()),
                   ("Wall-hugging", WallHugging())]:
    row(name, evaluate(ctrl, tube, n_episodes=50))

# ---- Phase 3: Y-branch split, two time budgets -------------------------------
for key in ("split", "split_long", "bifurcation"):
    sc = SCENARIOS[key]()
    print(f"\nTuning the switch moment for '{key}' on separate swarms ...", flush=True)
    p = tune_switch(sc)
    header(f"SPLIT ({sc.steps} steps): deliver half the swarm to each arm of a Y-branch",
           "Score = 2 x min(upper fraction, lower fraction); 1.0 is a perfect split\n"
           "with every bot delivered (30 random swarms each, pulsing flow, sticky walls).")
    for name, ctrl in [("Random", RandomController()),
                       ("Push toward target", PushTowardTarget()),
                       ("Wall-hugging", WallHugging()),
                       (f"Switch (tuned, {p})", SwitchController(p))]:
        row(name, evaluate(ctrl, sc, n_episodes=30))

print("\nA learned controller has to beat the best row in each table.")
