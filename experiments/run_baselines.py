"""Score the baseline controllers on both scenarios.

These are the bars a learned controller must beat. For the split task the
SwitchController's switching moment is TUNED on one set of random swarms and then
scored on a different set, so the baseline is a fair opponent and not a straw man.

Run:  python experiments/run_baselines.py
"""

from unison.controllers import (PushTowardTarget, RandomController,
                                SwitchController, WallHugging)
from unison.evaluate import evaluate
from unison.scenarios import split_scenario, tube_scenario


def row(name, r):
    print(f"{name:<26}{r.mean:>12.3f}{'  +/- ' + format(r.ci95, '.3f'):>14}")


def header(title, note):
    print(f"\n{title}\n{note}\n")
    print(f"{'Controller':<26}{'Mean score':>12}{'  95% CI':>14}")
    print("-" * 52)


# ---- Phase 2: straight tube ------------------------------------------------
tube = tube_scenario()
header("TUBE: get bots past the target against the flow",
       "Score = fraction of bots past the target line (50 random swarms each).")
for name, ctrl in [("Random", RandomController()),
                   ("Push toward target", PushTowardTarget()),
                   ("Wall-hugging", WallHugging())]:
    row(name, evaluate(ctrl, tube, n_episodes=50))

# ---- Phase 3: Y-branch split -----------------------------------------------
split = split_scenario()
print("Tuning the switch moment on separate swarms ...", flush=True)
best_p, best_score = None, -1.0
for p in [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]:
    s = evaluate(SwitchController(p), split, n_episodes=8, base_seed=0).mean
    if s > best_score:
        best_p, best_score = p, s
print(f"Best switch moment: {best_p} of the way through the episode\n")

header("SPLIT: deliver half the swarm to each arm of a Y-branch",
       "Score = 2 x min(upper fraction, lower fraction); 1.0 is a perfect split\n"
       "with every bot delivered (30 random swarms each, pulsing flow, sticky walls).")

for name, ctrl in [("Random", RandomController()),
                   ("Push toward target", PushTowardTarget()),
                   ("Wall-hugging", WallHugging()),
                   (f"Switch (tuned, {best_p})", SwitchController(best_p))]:
    row(name, evaluate(ctrl, split, n_episodes=30))

print("\nA learned controller has to beat the best row in each table.")
