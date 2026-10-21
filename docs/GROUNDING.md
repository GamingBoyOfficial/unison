# Grounding Unison in published microrobot research

Unison's simulator uses abstract units. This document ties them to published
experiments where it can, and states plainly where it cannot. Put this file in `docs/`.

Status labels: **Supported** (a source backs it), **Assumed** (our choice, no source),
**Conflicts** (a source suggests the opposite), **Unchecked** (general knowledge not yet
verified against a source in this project).

**Revision note (second revision).** The first version mapped a bot speed of 1.0 to
100 micrometres/s and said the simulator's flow was about ten times gentler than the
experiments. A second version withdrew both after reading the text, because the paper
quotes 100 micrometres/s only inside a Brownian-motion estimate. Reading Fig. 5 itself
shows both were approximately right: measured upstream speeds are about 50-250
micrometres/s and motion stopped at about 1.2 mm/s, a ratio of roughly 5-12. Two other
things from the figure also corrected earlier statements: the fitted slope of upstream
speed against flow is about -0.1 to -0.3, not -1, and the paper's two panels do not
agree with each other on the zero-flow speed (see "Figure readings").

## Sources read

1. **Ahmed et al., "Bio-inspired acousto-magnetic microswarm robots with upstream
   motility"** (ETH Zurich; arXiv 2005.12703). **Full text read.** Has a rolling model and
   measurements of swarms moving against a flow. The main source for this project.
2. **Wang et al., Science Advances 2021, 7, eabe5914** (CUHK). Abstract and summaries
   only. Magnetic nanoparticle swarms navigated near vessel boundaries, upstream and
   downstream, in flowing blood; mean velocities up to 40.8 mm/s reported; pulsatile
   flow; porcine coronary artery ex vivo.
3. **Collective cell microrobots in blood under ultrasound Doppler imaging** (IEEE).
   Abstract only. Navigation against flow up to 4.5 mL/min (mean velocity 5.97 mm/s).
4. **Magnetic nanoparticle swarm with upstream motility** (Nanoscale, 2023). Excerpts
   only. Swarm stable in 1 mm/s flow, tends toward the vessel wall.
5. **Microrobot swarm embolization work** (news summary only). Junction experiments and an
   ex vivo vessel with a 30 degree branching angle.
6. **Berman and Gal, arXiv 2605.25025.** Abstract only. A simulation study (CFD plus
   decentralized reinforcement learning) of 16 magnetic micro-robots in a pulsatile 2 mm
   tube. Related work with a different setup from Unison's single global signal.

Only source 1 has been read closely enough to take numbers or equations from. Numbers
from sources 2-6 come from summaries and must be re-checked in the originals before use.

## How source 1's swarm moves (what the paper actually says)

- A swarm of superparamagnetic particles self-assembles in a rotating magnetic field into
  a **disc-shaped, spinning cluster**. An acoustic field pushes it against the capillary
  wall, where it **rolls**. The wall is essential: the swarm needs a normal reaction force
  from the wall (supplied by the acoustic field) to roll at all. Estimated minimum
  reaction forces are about 25-30 piconewtons for swarms of 15-20 micrometre radius.
- Overdamped motion: swarm Reynolds number about 4e-4.
- Setup: circular capillaries of 150 and 300 micrometre diameter; flow fitted well by a
  Poiseuille profile. The Fig. 5b data use a 6 MHz acoustic field at 20 V and a magnetic
  field of 20 mT rotating at 17.5 Hz. Fits are shown for swarms of 15 and 40 micrometres.
- Force balance along the wall: friction from the wall (friction coefficient times the
  reaction force) balances the Stokes drag, which depends on the swarm's velocity **plus
  the fluid velocity at the swarm's centre of mass**. The fluid–wall gap was estimated at
  1.8 micrometres.
- Their Eq. 5 gives upstream speed as a rotation-frequency term plus terms in the fluid
  velocity at three heights (bottom, centre, top of the swarm). The measured upstream speed
  falls **roughly linearly as the imposed flow rises**; the paper says the speed "scales as
  minus V2".
- Upstream rolling was seen against streams up to **1.2 mm/s**, after which the swarms
  **lost traction** and were swept away. At low flow (0-0.2 mm/s) the effect of flow was
  small.
- Under high shear the swarm deforms (by over 30% in a supplementary movie) to stay in
  contact with the wall.
- The model's V2 is the **local fluid velocity at specific heights above the wall**. The
  Fig. 5b axis is called the imposed flow rate or stream velocity; the text does not say
  whether that is the centre-line, the mean, or the local velocity at the swarm.
- The text does not state the still-water rolling speed; it is read from Fig. 5 below.

## Figure readings (Fig. 5 of source 1, read by eye from a low-resolution screenshot)

Approximate (about +/-15-20%). Treat as order-of-magnitude until checked against the
supplementary information.

Panel b (upstream velocity vs flow velocity, swarms of mixed size):
- About 110-155 micrometres/s at 0.05-0.25 mm/s; about 90 at 0.7-0.8 mm/s; about 50-70
  at 1.0 mm/s; about 30 at 1.2 mm/s (the last point with upstream motion).
- Fitted lines: 15 micrometre swarm about 112 micrometres/s at zero flow, reaching zero
  near 1.2-1.3 mm/s (slope about -0.09); 40 micrometre swarm steeper (about -0.3),
  also reaching zero near 1.25 mm/s.
- Error bars are large: roughly +/-20-50 micrometres/s on speed and +/-0.1-0.2 mm/s on
  flow.

Panel c (upstream velocity vs swarm radius):
- At 0.2 mm/s flow: about 45 micrometres/s at 14 micrometre radius, about 100-135 at
  25-35, about 105-140 at 40-48, about 170 at 56 and about 240 at 62.
- Model, no flow: about 140 micrometres/s at 4 micrometre radius, about 200 by 10, and
  saturating near 240-250 at large radius. Model with 0.2 mm/s flow: about 25 at 4
  micrometres rising to about 165 at 70.

Inconsistency: panel b's 15 micrometre fit gives about 110 micrometres/s at zero flow,
while panel c's no-flow model gives over 200 at a similar radius. The cause (reading error,
different conditions, or different definitions of "flow velocity") is unresolved and needs
the supplementary information.

## Proposed unit mapping (an assumption)

| Simulator | Real-world reading | Basis |
|---|---|---|
| 1 length unit | about 75 micrometres | parent vessel half-width 1.0 read as the radius of a 150 micrometre capillary |
| bot speed factor 1.0 | about 100 micrometres/s (range 50-250) | measured upstream speeds in Fig. 5b; order of magnitude only |
| 1 time unit | about 0.75 s | follows from the two lines above |

## Status of each modelling assumption

| Model assumption | Status | Note |
|---|---|---|
| No inertia (movement proportional to force) | Supported | Re about 4e-4 for a swarm. |
| Parabolic flow profile, zero at the wall | Supported | Source 1 fitted Poiseuille flow. (Our model is a 2D channel; theirs is circular, so the profile shape differs.) |
| Upstream travel is easier near the wall, where flow is slow | Supported | Sources 1, 2, 4 and the Doppler papers. |
| Brownian jitter of 0.02 | Assumed | Source 1 says thermal motion is negligible for swarms of this size. Only justified for much smaller single particles. |
| **Sticky walls (wall drag reduces a bot's push)** | **Conflicts** | In source 1 the wall *enables* propulsion: rolling needs wall contact and a reaction force. Our assumption is the reverse for this mechanism. |
| A bot is propelled anywhere in the channel | **Conflicts** | Source 1's swarm rolls only against the wall; away from it, it follows the flow. |
| Bots are points that feel the flow at their own position | **Conflicts** | A real swarm is 15-60 micrometres across in a 150-300 micrometre capillary and feels the flow at its centre, which sits at least a swarm radius from the wall, where the flow is not zero. |
| A bot loses its push only through wall drag | **Conflicts** | Source 1 reports a traction limit: above a critical flow the swarm is swept away. |
| Flow strength (centre-line flow 0.8 of a bot's free speed) | **Conflicts (order of magnitude)** | Fig. 5b shows upstream motion against plotted flows up to 1.2 mm/s with still-water speeds of about 0.1-0.25 mm/s, a ratio of roughly 5-12. The relation between the plotted flow and the flow at the swarm is unclear, so this is not a precise comparison. |
| Sine-wave pulsing, amplitude 0.3 | Assumed | Pulsatile flow appears in sources 2 and 6, but not with our waveform. |
| Flow splits at branches by Murray's law (radius cubed) | Unchecked | A standard physiological rule from general knowledge; cite a source before publishing. |
| Bifurcation angle of 35 degrees per daughter | Assumed | Source 5 mentions a 30 degree ex vivo branching angle. |
| Bot speed varies by about 15% | Assumed | Not sourced. |

## What this means for the results

- The qualitative picture (viscous dominance, slow flow near walls, upstream travel along
  the wall) matches the literature.
- The simulator's **wall drag** and **free-anywhere propulsion** assumptions contradict
  the rolling mechanism in source 1. Results that depend on them (including the
  benchmark calibration of flow speed 0.8) are valid inside the simulator but are not
  predictions about real swarms.
- "AI beats simple rules" comparisons are comparisons inside the simulator.

## Proposed physics for a version 3 (grounded in source 1)

- A swarm has a finite radius a. When it rolls on the wall, its centre is about a (plus a
  micrometre-scale gap) from the wall.
- Propulsion exists only in contact with the wall. Away from the wall a swarm is carried
  by the flow.
- Along the wall: speed falls with the imposed flow with a fitted slope of about -0.1 to
  -0.3 (Fig. 5b), not -1, so a swarm is only weakly coupled to the plotted flow. A model of
  the form speed = V0 - kappa x flow is supported by the plot, while traction holds.
- Traction limit: above a critical local flow the swarm is swept away.
- Remove the unsupported wall drag.
- Parameters (rolling speed, critical flow) to be set from the paper's Fig. 5 data, not
  tuned for benchmark difficulty.

This changes the physics, so trained models and numbers from earlier versions do not
carry over.

## Proposed validation test (to agree before running)

Reproduce **upstream speed versus imposed flow** (Fig. 5b of source 1) with the version 3
model.

- Calibrate free parameters on **one** data set (for example the 15 micrometre swarm).
- Test the prediction on a **different** one (for example the 40 micrometre swarm, or the
  fixed-flow dependence on swarm size in Fig. 5c) without refitting.
- Pass/fail criteria, written **before** running anything (coarse, because the paper's
  two panels disagree):
  1. Upstream speed falls with flow, roughly linearly.
  2. The flow at which upstream motion stops is about 5-12 times the still-water speed;
     accept within a factor of two either side (2.5-22).
  3. At fixed flow, larger swarms move faster; the paper shows roughly 4-5 times between
     15 and 60 micrometre radius, accept 2-6.
- The current simulator is expected to fail criterion 2 (its flow is much gentler and its
  walls slow bots down). Report the result whichever way it goes.

Still needed: the supplementary information (fit parameters) and clarity on what flow
velocity the plot's axes represent; without it the two panels cannot be used to calibrate
and test each other.

## Known limits

2D physics; no blood cells; no bot-bot interactions; no acoustic or magnetic-field
physics. Unison studies the *control* problem for imperfect swarms, not the actuation
physics of any specific device.
