# Unison

![CI](https://github.com/GamingBoyOfficial/unison/actions/workflows/ci.yml/badge.svg)

**Learning to steer imperfect microrobot swarms with one global field.**

Real microrobots have no batteries. They are powered and steered from outside the body, so
every robot in a swarm feels the *same* signal. If all the robots were identical, one signal
could only move them as a blob. Real robots differ slightly in size and magnetism, and those
differences can be exploited with clever timing to sort and split a swarm.

Unison is a 2D simulator for that problem, plus controllers that learn to solve it:

> **How reliably can one global signal control a swarm of imperfect bots, and what breaks it?**

This is a research-style simulation, not a medical product. See [Limitations](#limitations).

## Summary of findings

The same task was studied at three levels of physical realism. Scores are **not comparable
across rows** (each row is a different game); the meaningful column is the **gain over the
straight push** (a simple rule: keep rolling forward).

| Scenario | Physics | Straight push | Learned controller | Gain |
|---|---|---|---|---|
| `split` (version 1) | point bots, free steering, rectangular Y | 0.648 | **0.897** | +0.25 |
| `bifurcation` (version 2) | point bots, realistic branching, sticky walls | 0.608 | **0.820** (mean of 5 runs) | +0.21 |
| `rolling` (version 3) | finite-size swarms that must grip the wall, flow near the published range | 0.586 | 0.586 | **0.000** |

1. **With idealized physics, learned control clearly beats simple rules** (versions 1 and 2),
   and in version 1 it matches a skilled human (a small test: no clear difference).
2. **With physics closer to a published experiment, the advantage disappears** (version 3).
   The loss there is dominated by physics: small swarms cannot hold the wall against the
   flow at each heartbeat peak, and no controller can change that. A perfect splitter could
   add at most 0.034.
3. Simulator realism therefore changes the conclusion. Results from idealized point-bot
   models should not be read as predictions for real swarms.
4. The simulator is **consistent with one published experiment in its main trend, but not
   validated against it**: of three checks fixed in advance, two passed largely by
   construction (the slope was set from the paper's plot) and the one independent check
   failed (see [Validation](#validation-against-published-work)).

## The task

300 bots with randomly varied traits start at the left end of a branching channel. Fluid
flows *against* them and pulses like a heartbeat. One controller picks one global field
(two numbers) at every step. The goal is to deliver **half the swarm to each of the two
arms** within a fixed time budget.

Score = `2 x min(fraction in upper arm, fraction in lower arm)`. A score of 1.0 is a
perfect 50/50 split with every bot delivered; sending everything to one arm scores 0.

All learned-controller numbers below are means over **100 random swarms never used in
training** (seeds 20000-20099) with 95% confidence intervals.

## Three levels of physics

| | Version 1 `split` | Version 2 `bifurcation` | Version 3 `rolling` |
|---|---|---|---|
| Geometry | two rectangles, flat divider | tapered, curved daughters, sharp carina, Murray's-law flow split | same as version 2 |
| Bots | points | points | finite-size swarms |
| Moves | anywhere, by the field | anywhere, by the field | **only while pressed against a wall** |
| Walls | slow bots ("sticky") | slow bots ("sticky") | enable rolling (as in the paper) |
| Flow (centre-line) | 0.6 (abstract units) | 0.8 (abstract units) | 0.6 mm/s, pulsing |

Flow strengths for versions 2 and 3 were **chosen so the straight push scores about 0.6**
(room to learn), not measured. With realistic geometry and version 1's flow, the straight
push scored 0.863 in version 2's geometry, leaving nothing to learn.

All physics is **2D**. The 3D viewer (`demo/index.html`) renders the 2D positions in 3D
tubes; the vertical placement of each bot there is a visual offset, not simulated motion.

## Results

### Version 1: `split`

| | `split` (800 steps) | `split_long` (1600 steps) |
|---|---|---|
| Straight push | 0.648 +/- 0.008 | 0.901 +/- 0.007 |
| Imitation AI (copies a human) | 0.853 +/- 0.012 | 0.756 +/- 0.068 |
| **Practiced AI (evolution strategies)** | **0.897 +/- 0.010** | **0.936 +/- 0.017** |
| Human operator (23 swarms) | 0.870 +/- 0.021 | not measured |

- On `split` the practiced AI beat the straight push on 100 of 100 swarms (paired
  difference +0.249 +/- 0.013). On `split_long` it beat it on 72 of 100 (+0.035 +/- 0.019)
  but its worst swarm scored 0.30.
- **Human vs AI, same 8 swarms** (seeds 20000-20007): push 0.633 +/- 0.041, human 0.872
  +/- 0.037, AI 0.897 +/- 0.029. AI minus human +0.026 +/- 0.053: no clear difference.
  One person played; 7 of the 23 swarms in the 0.870 figure were second attempts.
- **Where the loss goes** (`diagnose_loss.py`, 100 swarms): the push loses 0.320 to bots
  never delivered and 0.038 to unevenness between the arms; the AI loses 0.055 and 0.048.
  Arm unevenness is close to what fair coin flips would give (0.045 for the AI, 0.038 for
  the push), suggesting neither controller decides which *individual* bot goes where.
  This is an inference from matching numbers, not a proof.

### Version 2: `bifurcation`

| | Mean +/- 95% CI |
|---|---|
| Straight push | 0.608 +/- 0.007 |
| Practiced AI, five independent training runs (seeds 0-4) | 0.830, 0.810, 0.801, 0.829, 0.828 |

Mean across runs **0.820**, standard deviation 0.013; every run beat the push by at least
+0.194. The first run beat the push on 100 of 100 swarms (+0.222 +/- 0.008). The starting
model (from version 1) scored 0.413 here, below the push, so the gain comes from practice
on this geometry. The five runs share a starting model and practice swarms and differ only
in mutation seed, so the spread likely understates true variation. No human benchmark.

### Version 3: `rolling`

Controls: **right/left rolls along the wall** (only while pressed against it);
**up/down presses swarms toward the upper/lower wall** (separate actuators, each in
[-1, 1]). Wall selection is an extension beyond the paper's fixed transducer.

| Controller | Score (30 swarms) |
|---|---|
| Random | 0.000 |
| Straight push | 0.595 +/- 0.014 |
| Wall-hugging (always press up) | 0.000 (all bots go to one arm) |
| Switch (up, then down; timing tuned on other swarms) | 0.536 +/- 0.017 |

On 100 unseen swarms:

| | Mean +/- 95% CI | Note |
|---|---|---|
| Straight push | 0.586 +/- 0.009 | |
| Practiced AI, started from a version-2 model | 0.321 +/- 0.009 | beat the push on 0 of 100; a poor starting point |
| Practiced AI, started from the straight push | 0.586 +/- 0.009 | paired difference +0.000 +/- 0.001; validation 0.597 vs 0.596 at the start |

**Why learning has little room here** (`analyze_delivery.py`):
- A perfect 50/50 split of the swarms the push already delivers would score 0.627 against
  0.593: balancing can add **at most 0.034**.
- The rest is physical. Swarms under about 15 micrometres have a stalling flow of 2-3
  against about 5.4 in the parent's wall lane at the systolic peak, so they lose grip every
  heartbeat: 21-41% of them are delivered, against 84% of swarms over 30 micrometres.
  Doubling the time budget raises delivery only from 0.63 to 0.71.
- Caveat: the controller's 12 inputs do not say which wall each swarm is on, so "no gain"
  means no gain *with these inputs and this training budget*. The headroom is small anyway.

## Validation against published work

Physics for version 3 follows Ahmed et al. (arXiv 2005.12703): swarms roll along the
wall, upstream speed falls roughly linearly with flow, and traction is lost above a
critical flow. `docs/GROUNDING.md` lists every modelling assumption as **supported**,
**assumed** or **in conflict** with the sources, and every parameter that is tuned.

`python experiments/validate_rolling.py` compares the model with the paper's Fig. 5 using
criteria fixed before the first run:

| Criterion | Result |
|---|---|
| 1. Upstream speed falls with flow, roughly linearly | PASS (largely by construction: slope set from the plot) |
| 2. Stopping flow is 2.5-22 times the still-water speed | PASS, 5.6-5.9 (largely by construction) |
| 3. At 0.2 mm/s a 60 micrometre swarm is 2-6 times faster than a 15 micrometre one | **FAIL**, 1.21 |

The model stops at 0.74-0.90 mm/s against the paper's 1.2-1.3. The paper's own panels b and
c disagree on the size dependence, so criterion 3 was not "fixed" by fitting. Related
simulation work on micro-swarms in pulsatile flow exists (for example arXiv 2605.25025,
abstract read only) with a different setup (decentralized agents).

Version 3 was debugged through hand play, which exposed four bugs now fixed and covered
by tests (wrong wall direction at the bifurcation; wall-selection push equal to the
acoustic pull; shared drive budget; sideways push sliding swarms along the wall). Numbers
measured before those fixes are not used.

## How it works

1. **Simulator.** Version 1-2 bots move by `own push x field + fluid flow + jitter`; version
   3 swarms roll along walls (`src/unison/rolling.py`). Arenas are swappable modules (box,
   tube, rectangle Y, general vessel tree).
2. **Baselines.** Random, straight push, wall-hugging and a time-based switch rule whose
   timing is tuned on separate swarms, so baselines are fair opponents.
3. **Human play.** `experiments/play.py` lets a person steer by hand. Key presses and seeds
   are saved, and replaying them reproduces the episode exactly.
4. **Imitation learning.** A small network (12 inputs) copies the human's key presses.
5. **Practice (evolution strategies).** Mutated copies of a model play fresh swarms and the
   model moves toward the higher scorers. The score is the only teacher.
6. **Evaluation.** 100 swarms never used in training, paired comparisons, confidence
   intervals, and several independent training runs where stated.

Swarm seeds are kept disjoint: human play 5000s, practice 40000s, model selection 50000s,
final testing 20000s. Continuous integration (GitHub Actions) runs lint, 30 tests on three
Python versions, and smoke runs of the simulation scripts.

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
python experiments/run_baselines.py                    # baseline tables
python experiments/play.py rolling                     # steer swarms by hand (also split, bifurcation)
python experiments/validate_rolling.py                 # comparison with the published figure
python experiments/analyze_delivery.py                 # where the loss goes, and the headroom
python experiments/train_es.py --model push --scenario rolling      # practice
python experiments/evaluate_model.py --model results/es_rolling.npz --scenario rolling
python experiments/evaluate_seeds.py --scenario bifurcation         # spread across training runs
```

3D viewer: `python experiments/export_replay.py --scenario bifurcation --model
results/es_bifurcation.npz`, then in `demo/` run `python -m http.server 8000` and open
http://localhost:8000.

## Project layout

```
src/unison/
  arena.py       box, tube, rectangle Y-branch; pulsing flow; wall drag
  geometry.py    general vessel engine: tapered tubes, bifurcations, Murray's-law flow
  rolling.py     version 3: finite-size swarms that roll along walls, traction limit
  bots.py        bots with random traits
  simulator.py   one time step under one global field (versions 1-2)
  task.py        what counts as success (reach target, split)
  scenarios.py   arena + task + physics + time budget, by name
  controllers.py baseline controllers
  evaluate.py    run many fresh swarms, report mean and 95% CI
  imitation.py   replay human demos, train the copying model
  policy.py      small NumPy neural network (mutable weights)
  playlog.py, stats.py   play-log helpers; small-sample statistics
experiments/     runnable scripts       demo/   3D replay viewer       docs/   GROUNDING.md
tests/           pytest checks
```

## Limitations

Please read these before citing any number.

- **It is a simplified 2D simulation.** No blood cells, no bot-to-bot interaction, no
  acoustic or magnetic-field physics, no 3D motion. It studies the *control* problem for
  imperfect swarms, not the actuation physics of any specific device.
- **Versions 1 and 2 are idealized** (free steering, unsupported sticky walls). Their AI
  results hold inside the simulator and are not predictions for real swarms.
- **Version 3 is closer to one published mechanism but not validated** (one of three
  checks failed, two pass by construction), and rests on tuned assumptions listed in
  `docs/GROUNDING.md`: the acoustic pull toward the wall, the wall-selection push, the
  flow strength, and the spread in swarm size and response.
- **Flow strengths in versions 2 and 3 were set for benchmark difficulty**, so scores are
  not comparable across scenarios.
- **The human comparison is small:** one person, 8 identical swarms in the head-to-head.
  `bifurcation` and `rolling` have no human baseline.
- **Training variation:** version 1 reports one run; version 2 five runs that share a
  starting model and practice swarms.
- **Observation design:** the controller sees 12 summary numbers, not a full picture, and
  not which wall each swarm is on.
- **No medical claims.** This is not medical software and must not be used for treatment
  decisions.

## Next steps

- Obtain the paper's supplementary data to resolve its two panels and test against a
  second, independent data set.
- Give the version 3 controller the extra inputs it lacks (which wall swarms are on) and
  retry; expected gain is at most about 0.03.
- Measure human benchmarks on `bifurcation` and `rolling`; vary starting models across runs.
- Real vessel geometry from open datasets in the vessel engine; true 3D physics.
- A realistic blood model (shear-thinning viscosity, heartbeat waveform, cell crowding).

## License

MIT. See `LICENSE`.
