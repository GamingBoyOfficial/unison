# Unison

**Learning to steer imperfect microrobot swarms with one global field.**

Real microrobots have no batteries. They are powered and steered from outside the
body, so every robot in a swarm feels the *same* magnetic field. If all the robots
were identical, one signal could only move them as a blob. But real robots differ
slightly in size and magnetism, and those differences can be exploited with clever
timing to sort and split the swarm.

Unison is a simulator for that problem, plus controllers that learn to solve it:

> **How reliably can one global signal control a swarm of imperfect bots, and what breaks it?**

This is a research-style simulation, not a medical product. See [Limitations](#limitations).

## The task

300 bots with randomly varied traits start at the left end of a Y-shaped channel.
The fluid flows *against* them, pulsing like a heartbeat, and the walls are sticky.
A single controller picks one field direction at every step. The goal is to deliver
**half the swarm to each arm** within a fixed time budget.

Score = `2 x min(fraction in upper arm, fraction in lower arm)`. A score of 1.0 is a
perfect 50/50 split with every bot delivered. Sending everything to one arm scores 0.

## Results

All AI numbers are means over **100 random swarms never used in training**
(seeds 20000-20099), with 95% confidence intervals.

| | `split` (800 steps, main benchmark) | `split_long` (1600 steps) |
|---|---|---|
| Straight push (best simple rule) | 0.648 +/- 0.008 | 0.901 +/- 0.007 |
| Imitation AI (copies a human) | 0.853 +/- 0.012 | 0.756 +/- 0.068 |
| **Practiced AI (evolution strategies)** | **0.897 +/- 0.010** | **0.936 +/- 0.017** |
| Human operator (23 swarms) | 0.870 +/- 0.021 | not measured |

- On `split`, the practiced AI beat the straight push on **100 of 100** swarms
  (paired difference +0.249 +/- 0.013).
- On `split_long`, it beat the push on 72 of 100 swarms (paired difference
  +0.035 +/- 0.019), but its worst swarm scored 0.30, so it is not always reliable.

### Human vs AI, head to head

On the first 8 benchmark swarms (seeds 20000-20007, identical for everyone), compared
swarm by swarm:

| | Mean +/- 95% CI (t-based) |
|---|---|
| Straight push | 0.633 +/- 0.041 |
| Human | 0.872 +/- 0.037 |
| Practiced AI | 0.897 +/- 0.029 |

The human beat the straight push on 8 of 8 swarms (+0.238 +/- 0.057). The AI minus the
human is +0.026 +/- 0.053, with the AI higher on 6 of 8 swarms: **no clear difference
at this sample size**. The honest summary is "roughly human-level". More swarms were
not played; see the analysis below for why a larger gap is not expected.

### Where the remaining loss goes

`experiments/diagnose_loss.py` splits the loss (1 - score) on 100 unseen swarms:

| Controller | Mean score | Bots never delivered | Unevenness between arms |
|---|---|---|---|
| Straight push | 0.641 | 0.320 | 0.038 |
| Practiced AI | 0.897 | 0.055 | 0.048 |

- The AI's undelivered bots are the slow ones (average speed factor 0.685 vs 1.022 for
  delivered bots), which probably cannot beat the flow within 800 steps.
- The arm unevenness is close to **chance level**: if every delivered bot picked an arm
  by a fair coin flip, 300 bots would give about 0.045 (AI) and 0.038 (push). The
  observed values are 0.048 and 0.038. This suggests neither controller decides which
  *individual* bot goes to which arm, and that about 0.90 is near the best achievable
  without doing so. This is an inference from matching numbers, not a proof.

Tube task (no branch, steady flow), from `experiments/run_baselines.py`:
random 0.000, straight push 0.888, wall-hugging 1.000. This task is too easy to be
interesting, which is why the Y-branch split task exists.

## Version 2: a general vessel engine

The first Y-branch is two rectangles with a flat divider, so the arms sit apart with
a gap, unlike a real bifurcation. Version 2 (`src/unison/geometry.py`) describes any
vessel tree by centerlines and radii:

- Each piece is a **tapered tube**; the arena is the union of all tubes, so curves,
  tapering and angled bifurcations with a **sharp carina** come from one mechanism.
- Flow follows the geometry: each tube carries a share of the root flow rate, shares
  split at each branch by **Murray's law** (proportional to radius cubed), and the
  parabolic profile speeds up where a vessel narrows.
- The `bifurcation` scenario is a parent vessel splitting into two curved, tapering
  daughters at +/-35 degrees. Run it with `python experiments/play.py bifurcation`.
- **Calibration, stated plainly:** with realistic geometry and the version-1 flow
  strength, the straight push already scores 0.863, leaving no room to learn. The flow
  speed was therefore raised to 0.8 so the straight push scores about 0.61 (similar
  headroom to version 1). This is a design choice made for benchmark difficulty, not a
  measured physiological value.
- Version 1 results above are unchanged and still reproducible with `split` and
  `split_long`. **No AI or human results exist yet for `bifurcation`**; trained models
  and recordings from version 1 do not transfer automatically.

## How it works

1. **Simulator.** Bots move by `own push x field + fluid flow + Brownian jitter`.
   Each bot has its own size and magnetism. Flow is parabolic across each channel,
   pulses over time, and bots near walls lose part of their push (sticky walls).
   Arenas are swappable modules (box, tube, Y-branch).
2. **Baselines.** Random, straight push, wall-hugging, and a time-based switch rule
   whose timing is tuned on separate swarms, so the baselines are fair opponents.
3. **Human demonstrations.** `experiments/play.py` lets a person steer by hand.
   Each episode's key presses and seed are saved, and replaying them reproduces the
   episode exactly (checked against the logged score).
4. **Imitation learning.** A small neural network (12 inputs, two layers of 64) is
   trained to copy the human's key presses from 23 demonstrated swarms.
5. **Practice (evolution strategies).** Starting from the imitation model, 60
   generations of mutated copies play fresh swarms and the model moves toward the
   higher-scoring mutations. No new human data is used; the score is the only
   teacher. The best model (selected on 20 validation swarms) is saved.

Swarm seeds are kept disjoint: human play 5000s, practice 40000s, model selection
50000s, final testing 20000s.

## Run it

```
git clone https://github.com/GamingBoyOfficial/unison.git
cd unison
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell (macOS/Linux: source .venv/bin/activate)
python -m pip install -e ".[dev,learn]"
python -m pytest
```

```
python experiments/run_baselines.py                      # baseline table
python experiments/play.py                               # play the split task yourself
python experiments/summarize_play.py                     # your score, +/- 95% CI
python experiments/train_imitation.py                    # copy the human demos
python experiments/train_es.py                           # let the AI practice
python experiments/evaluate_model.py --model results/es_split.npz
python experiments/evaluate_model.py --model results/es_split.npz --scenario split_long
```

The human recordings used here are in `results/`.

## Project layout

```
src/unison/
  arena.py       swappable shapes + flow (box, tube, rectangle Y-branch), pulsing flow, wall drag
  geometry.py    general vessel engine: tapered tubes, bifurcations, Murray's-law flow
  bots.py        bots with random traits
  simulator.py   one time step under one global field
  task.py        what counts as success (reach target, split)
  scenarios.py   arena + task + time budget bundled by name
  controllers.py baseline controllers
  evaluate.py    run many fresh swarms, report mean and 95% CI
  imitation.py   replay human demos, train the copying model
  policy.py      small NumPy neural network (mutable weights)
  playlog.py     play-log helpers (each swarm counted once)
experiments/     runnable scripts
tests/           pytest checks (walls, flow, determinism, replay, scoring)
```

## Limitations

Please read these before citing any number.

- **It is a simplified simulation.** The model is 2D, with rectangular channels,
  simple drag, and no bot-to-bot or bot-to-cell interaction. Its parameters have
  **not** been validated against published microrobot experiments. The AI is good at
  this simulated game; how well it transfers to real robots is unknown.
- **No medical claims.** This is not medical software, and it must not be used for
  treatment decisions.
- **The human comparison is small.** The 0.870 figure in the main table comes from 23
  swarms (seven were second attempts at swarms the player had seen) and different
  swarms than the AI's. The head-to-head above uses identical swarms but only 8 of
  them, so it cannot separate a small difference. One person played.
- **One training run.** The practiced model comes from a single run; variation across
  training runs has not been measured. The best model was picked on a small
  validation set, so the 100-swarm test, not the validation score, is the result.
- **Trained on one time budget.** The practiced model was trained only on 800-step
  episodes. On 1600 steps it works on average but occasionally fails badly
  (worst case 0.30).
- **Observation design.** The controller sees 12 summary numbers (swarm mean and
  spread, arm counts, episode progress, heartbeat phase), not a full picture.

## Next steps

- Validate the simulator against a published microrobot experiment.
- Test the chance-level hypothesis directly, and try to beat it: a controller that
  uses bot differences to choose which bot goes to which arm would break the
  ~0.90 plateau. This is the open question the project was built around.
- Train across several time budgets; repeat training with different seeds.
- Train and evaluate on `bifurcation` (start from `results/es_split.npz` with
  `train_es.py --model results/es_split.npz --scenario bifurcation`).
- Load real vessel centerlines from open datasets into the vessel engine; extend to 3D.
- A realistic blood model (shear-thinning viscosity, realistic heartbeat, cell crowding).
