# Grounding Unison in published microrobot research

Unison's simulator uses abstract units. This document ties them to published
experiments where it can, and states plainly where it cannot. Put this file in `docs/`.

Status labels: **Supported** (a source backs it), **Assumed** (our choice, no source),
**Conflicts** (a source suggests the opposite), **Unchecked** (general knowledge not yet
verified against a source in this project).

## Sources read

1. **Ahmed et al., "Bio-inspired acousto-magnetic microswarm robots with upstream
   motility"** (ETH Zurich; arXiv 2005.12703). Full text read. The most useful single
   source: it has a model and measurements for swarms moving *against* a flow.
2. **Wang et al., Science Advances 2021, 7, eabe5914** (CUHK). Abstract and summaries
   read, not the full paper. Magnetic nanoparticle swarms navigated near vessel
   boundaries, upstream and downstream, in flowing blood; reported mean velocities up to
   40.8 mm/s; pulsatile flow; porcine coronary artery ex vivo.
3. **Collective cell microrobots in blood under ultrasound Doppler imaging** (IEEE).
   Abstract only. Navigation against flow up to 4.5 mL/min (mean velocity 5.97 mm/s).
4. **Magnetic nanoparticle swarm with upstream motility and peritumor vessel crossing**
   (Nanoscale, 2023). Excerpts only. Swarm stable in 1 mm/s flow, tends toward the vessel
   wall.
5. **Microrobot swarm embolization work** (news summary only). Junction experiments in
   microfluidic channels and an ex vivo vessel with a 30 degree branching angle.
6. **Berman and Gal, arXiv 2605.25025.** Abstract only. A simulation study (CFD plus
   decentralized reinforcement learning) of 16 magnetic micro-robots in a pulsatile 2 mm
   tube. Related work, and a different setup from Unison's single global signal.

Only sources 1 and 2 have been examined closely enough to take numbers from. Everything
below that cites a number from 3-6 is from a summary or abstract and must be re-checked
in the original before it is used in a claim.

## What source 1 measured (the numbers we can use)

- Circular capillaries of **150 and 300 micrometre diameter**. Measured flow matched a
  Poiseuille profile.
- Swarms of roughly **15-40 micrometre** size in the fitted data (a swarm radius of about
  30 micrometres moving at about **100 micrometres/s** is quoted for the Brownian-motion
  estimate).
- Swarm Reynolds number about **4e-4**: viscous forces dominate and motion is overdamped.
- Upstream "rolling" along the wall. At low flow (0-0.2 mm/s) the upstream speed was only
  marginally affected; as flow rose the upstream speed fell roughly linearly; upstream
  motion was seen up to a stream velocity of about **1.2 mm/s**, after which swarms lost
  traction and were swept away.
- Propulsion **requires contact with the wall**: a reaction force from the wall (supplied
  acoustically) is what lets the swarm roll. A thin lubrication gap of about 1.8
  micrometres was estimated.
- Their model gives upstream speed as a function of rotation frequency, swarm size, and
  the flow speed at several heights across the swarm.

## Proposed unit mapping (an assumption, anchored to source 1)

| Simulator | Real-world reading | Basis |
|---|---|---|
| 1 length unit | about 75 micrometres | the parent vessel half-width (1.0) read as the radius of a 150 micrometre capillary |
| 1 speed unit (bot speed factor 1.0) | about 100 micrometres/s | the swarm speed quoted in source 1 |
| 1 time unit | about 0.75 s | follows from the two lines above (75 um / 100 um/s) |
| flow_speed 0.8 (centre-line) | about 80 micrometres/s | direct consequence of the mapping |

This mapping is a choice made for illustration. It has not been checked against any
measurement of our own.

## Status of each modelling assumption

| Model assumption | Status | Note |
|---|---|---|
| No inertia (movement proportional to force) | Supported | Source 1 reports Re about 4e-4 for a swarm. |
| Parabolic flow profile, zero at the wall | Supported | Source 1 fitted Poiseuille flow; other sources state flow is zero at the wall. (Our model is a 2D channel; their capillary is circular, so the profile shape differs.) |
| Upstream travel is easier near the wall, where flow is slow | Supported | Sources 1, 2, 4 and the Doppler papers describe exactly this. |
| Brownian jitter of 0.02 | Assumed | Source 1 says thermal motion is negligible for swarms of this size. Jitter is only physically justified for much smaller, single particles. |
| **Sticky walls (wall drag reduces a bot's push)** | **Conflicts** | In source 1 the wall is what *enables* propulsion (rolling needs wall contact). Our assumption is the reverse for this mechanism. |
| Flow strength: centre-line flow about 0.8 of a bot's free speed | **Conflicts** | Source 1 saw upstream motion against streams of about 1.2 mm/s, roughly ten times a swarm's quoted still-water speed of about 0.1 mm/s (order-of-magnitude comparison, using two numbers that may not refer to the same swarm). Our regime is far gentler. |
| Bots are points | Conflicts | Real swarms are 15-60 micrometres in a 150-300 micrometre capillary (a large fraction of the channel), so they average the flow over their body. |
| Sine-wave pulsing with amplitude 0.3 | Assumed | Pulsatile flow is used in sources 2 and 6, but not with our waveform. |
| Flow splits at branches by Murray's law (radius cubed) | Unchecked | A standard physiological rule from general knowledge; cite a source before publishing. |
| Bifurcation angle of 35 degrees per daughter | Assumed | Source 5 mentions an ex vivo vessel with a 30 degree branching angle. |
| Radii ratio 0.79 for daughters | Unchecked | Follows from Murray's law above. |
| Each bot's speed varies by about 15% | Assumed | Not sourced. |

## What this means for the results

- The qualitative picture (slow flow near walls, viscous dominance, upstream travel by
  hugging the wall) matches the literature.
- Two parameter choices, **wall drag** and **flow strength**, are not supported and may
  be wrong in direction. Results that depend on them, including the bifurcation
  scenario's calibration (flow speed 0.8 chosen for benchmark difficulty), should not be
  read as predictions about real swarms.
- The "AI vs simple rules" comparisons are comparisons *inside the simulator*. They say
  nothing yet about real robots.

## Proposed validation test (to agree before running)

Reproduce the **upstream speed vs imposed flow** relationship from source 1 (their
Fig. 5b), using a finite-size rolling swarm model with wall-contact propulsion.

- Calibrate free parameters (swarm size, rolling speed in still fluid) on **one** data
  set.
- Test the prediction on a **different** one (for example the flow limit where upstream
  motion stops), without refitting.
- Decide the pass/fail criteria **before** running, for example: the predicted decline of
  upstream speed with flow has the right sign and roughly linear shape, and the predicted
  flow at which motion stops is within a factor of two of the measured one.

What is still needed: the plotted data points of source 1's Fig. 5b (read from the
figure or the supplement), and the precise meaning of its flow velocity V2 (centre-line,
mean, or local), which the extracted text did not state. A second, independent data set
is also needed for the test; source 2 is a candidate but its details are not yet read.

## Known limits

2D physics; no blood cells; no bot-bot interactions; no magnetic-field or acoustic
physics. Unison models the *control* problem for imperfect swarms, not the actuation
physics of any specific device.
