# Roadmap

## START HERE — the next thing to build

### The world was broken, and it had been broken since M7 began (2026-07-26)

**Read this before anything below it.** Everything in the M8 section that
follows was measured on a world where the observation carried almost no
information about position, so re-run anything you intend to rely on.

Measured on the patch world every M7 and M8 run used:

| separation | patch correlation |
|---|---|
| 0 – 0.1 cells | **+0.99** |
| 0.1 – 0.25 | +0.42 |
| 0.25 – 0.5 | −0.12 |
| 0.5 – 1.0 | −0.04 |

**Observation correlation dies within 0.16 cells. The agent steps 0.5 cells.**
Consecutive observations are statistically independent, so the reverse read and
the drift gate — half the architecture — had no signal to work with. "Have I
seen this before?" was answerable only within a sixth of a cell, which in
continuous space essentially never recurs.

The bitter part: **Rung 5 was promoted specifically to fix this.** The M6 note
below says in as many words that "observation similarity carries no spatial
information at all, which specifically cripples the reverse read." The patch
world was built to solve that and *did not*, because nobody measured whether it
had. Building the intended mechanism is not evidence that the mechanism works.

The evidence was also already collected and written up backwards. The drift gate
sat at 0.05–0.34 across fourteen arms, near its 0.12 initialisation, and
CLAUDE.md records this as "corrections are small continuous nudges, never
teleports, which is the right response." It is not. **A gate that never leaves
its initialisation is evidence about the world, not a property of the
mechanism.**

**Cause and fix.** `motif_sigma` cannot fix it (measured at 4/8/16/32/64:
correlation length stays 0.06–0.18 cells) because motifs are drawn
**independently per tile**, so there is a discontinuity at every cell boundary
however smooth each side is. The cap is the tile, not the filter. The new
`motif_cells` widens the tile:

| motif_cells | sigma | corr length | spatial falloff | ambiguity |
|---|---|---|---|---|
| 1 (all runs so far) | 4 | 0.16 | −0.001 | 35 |
| **4** | **16** | **0.62** | **+0.136** | 72 |

At 4/16 the correlation length finally exceeds the step, and repeats still occur
~14x per motif so ambiguity survives — it *rises*, which is the direction the
architecture wants.

**Frozen encoders (`--encoder dct|pca|random|gabor`).** Zero trainable
parameters. The point is pressure, not speed: encoder and predictor share a
loss, so the loss can fall either by localising better or by reshaping the
observation space until patches separate more easily, and the second is much
cheaper. Freezing removes the option. Corrected DCT is near-lossless —
distance correlation 1.000, 99.1% of variance in 10 of 64 dims. Gabor is
unusable at 2% variance retained. Frozen encoders also make embedding-caching
gradient-exact (verified 4.8e-08 float32, 8.5e-17 float64), which is what makes
a precomputed `(velocity, embedding)` dataset viable.

**Run `scripts/m9_basis_check.py` before training on any frozen basis.** It
costs ten seconds and it caught three bad configurations of mine, including two
defaults I had argued for in prose: `drop_dc` discarded 90% of the variance
(on smooth textures a patch is nearly flat, so its mean brightness *is* the
observation) and whitening drove distance correlation from 1.000 to 0.237.
`m7_image.py` now refuses to start below 0.5 correlation, because a frozen basis
cannot be trained out of a bad start.

**Caveat when reading the gate.** A rising gate is not by itself proof that
landmarks became useful — the frozen-M1-transfer arm ran the highest gate of any
run (0.145) precisely because its position stream was bad and it had nothing
else to lean on. Confirm with retrieval distance, not the gate alone.

### M7 is DONE and M8 (grid cells) is half-done — read this first (2026-07-26)

**M7, the continuous image walker: PASSED at 51.75%** on images never trained
on, against 0.61% for appearance-addressed retrieval, 0.04% chance, and 4.65%
for retrieval *with perfect localisation*. Beating that last number 11x is the
result: the model is not copying a remembered patch, it interpolates several
into a view it has never had from that spot. `scripts/m7_image.py`.

**M8, grid cells: bands yes, hexagons no.** Eight arms overnight. The recipe
that works is **nonnegative units (`--activation relu`) + place-cell targets
(`--w-place`) + activity cost (`--l1-code`) + an arena big enough to resolve
several periods (`--grid 16`)**. See `figures/m8_control_vs_best.png`: the
control gives blobs and gradients, the recipe gives unmistakable bands, and in
arm F the orientations are *module-consistent* (module 1 all six units at 90
degrees, module 0 five of six at 0) where the control's are scattered.

No hexagons in any arm. Consistent with M5: a square world privileges 0 and 90
degrees, and hexagons need three bands interfering at 60.

**The decisive negative:** arm H switched the memory objective OFF entirely
(`--w-pred 0`), making a pure path integrator trained only on place targets --
which is what the published grid-cell models ARE. It produced no more structure
than the rest. So the memory objective is NOT what suppresses hexagons, and the
dot-product side-lobe argument does not explain their absence.

**What to do next, in order:**

1. **A much larger arena.** Every automated count tonight is unreliable and the
   reason is measurable: the learned wavelength came out at ~5.3 cells in every
   module, so only ~3 periods fit the arena. Three periods is too few for any
   spectral method to separate a band from a blob or a gradient -- four
   different discriminators were tried and none did. Go to 30+ cells so 6-10
   periods are visible, and the classifier stops being the bottleneck.
2. **Hex topology or a hex-symmetric world**, since bands now form reliably and
   the only thing missing for hexagons is a world whose symmetry rewards 60
   degrees.
3. Then resume the target below (adapter, walls, policy).

**Do not trust `--analyse`'s printed verdict.** It gates on `field_score < 0.5`,
which rejects genuine bands: a two-stripe unit can put over half its mass in one
component. `scripts/m8_grid_report.py` re-scores on FIXED-START walks (required
-- random starts average unrelated codes into mush) but inherits the same gate.
**Open `rate_maps_fixed_start.npy` and look.**



**Rung 5, the patch observation model.** Promoted above everything else,
because M6 showed continuous movement and continuous observation are ONE change
rather than two: continuous movement paired with the current piecewise-constant
symbol field creates a task demanding sub-cell precision while supplying no
sub-cell information, and observation similarity carries no spatial information
at all, which specifically cripples the reverse read. M6 reached 53% of ceiling
where discrete reached 98%, and this is the suspect.

What it needs:

1. Observations become an 8x8 patch centred on the continuous position, from a
   background image, replacing `ContinuousEnvironment.observe`'s cell lookup.
2. `to_value` (currently `Linear(n_observations -> obs_dim)` off a one-hot)
   becomes a small patch encoder. Note it feeds BOTH the forward read's values
   and the reverse read's keys.
3. The target stops being classification. Use contrastive (pick the true patch
   from distractors) over regression — MSE produces blurry averages.
4. Baselines need patch-space equivalents: nearest-neighbour retrieval in place
   of the node/edge tables.
5. Start on a synthetic image with controlled statistics before real
   photographs, so observation ambiguity stays a measured quantity rather than
   an accident of the data.

Then: adapter (Rung 6) → walls → policy → task-driven actions, per
"The target". Rung 7's composition test slots in after the adapter,
independently of the image work.

**Rungs 0, 0b, 0c, 0d are CLOSED** — they were the M5 investigation, which is
finished (see CLAUDE.md). Do not restart them. Rung 0c was never run and is not
worth running: M5's question is answered.

## Operating notes for a fresh session

- Interpreter: `C:/Users/JmgLi/anaconda3/envs/ToastEnv/python.exe` (not on PATH).
- **Rent cheap hosts with fast CPUs.** This workload is kernel-launch bound, so
  GPU class is nearly irrelevant and single-thread CPU is everything: a $0.088
  RTX A4000 + i7-13700 matched a $0.268 RTX 5090 and beat a $0.308 5090 on an
  EPYC by 5.5x. Use `--thresholds vast/thresholds_cheap.json`. Full table and
  gotchas in `vast/README.md`.
- **Never `destroy --all-remote`** — this account runs other projects'
  instances concurrently. `--all` is already scoped to this repo.
- **Never reuse an offer id.** Re-running `search` is mandatory; a stale id
  either creates a phantom billing contract or a wedged instance. Both happened.
- `vastai logs` is silent on some images. SSH and read
  `/workspace/onstart.log` before concluding an instance is dead.


Work after M0–M5, ordered by information gained per unit of cost.

**This plan is provisional and expected to change.** Every rung carries a
falsifiable prediction, and the point of the prediction is that a wrong one
should reorder what follows. Two hypotheses have already been refuted this way
(§"What has already changed"), and the ordering below assumes further
refutations. Treat it as the current best guess, not a commitment. §"When to
rewrite this plan" lists the specific results that would reorder it.

Each rung states a prediction that could fail, because the discipline of naming
one in advance is what has caught the real mistakes here: an ablation that
suppressed almost nothing, a ceiling computed against the wrong bound, and a
"curriculum effect" that was a diverging learning rate.

## Where things stand

| Milestone | State |
|---|---|
| M0 environments and walks | done |
| M1 position stream, linear probe | done — 100% decoding on held-out short walks |
| M2 forward read + readout | done — 88.9% vs 41.8% edge agent |
| M3 reverse read + drift gate | done — 97.4% at 300 steps, degradation inverted; gate worth 14.6 points |
| M4 multi-environment, zero-shot | done — **71.1% on unseen environments, 98% of the 72.5% ceiling** |
| M5 analysis harness | **periodic codes in 100% of units** — band modules sharing wavelength and orientation. No hexagons, correct for a square world |
| M6 continuous movement | passes — 17.96% vs 5.82% baseline, half the transition parameters. Only 53% of ceiling; the discrete observation model is the suspect |

The system works, and the codes are periodic. M5's original "no periodic
structure" verdict was a metric artefact: gridness asks only "is this
hexagonal?", so bands score ≈0 exactly like noise. Counting Fourier peaks shows
periodic structure in every configuration, rising to **100% of units** under
capacity pressure, organised into modules that share wavelength and
orientation — learned with **no positional supervision at all**.

Hexagons are genuinely absent, and that is the right answer: they need three
bands interfering at 60°, and the modules each settle on a single orientation.
On a hex topology the orientations moved to the hex axes (8°, 121°) but stayed
bands, so **orientation follows the world's symmetry while hexagonal
combination does not emerge**.

**Next change is the patch observation model.** M6 established that the
continuous position stream trains and beats baseline with half the parameters,
but it reached only 53% of ceiling because it pairs continuous movement with a
*piecewise-constant* symbol field: the task demands sub-cell precision while
giving zero information about sub-cell position, and observation similarity
carries no spatial information, which cripples the reverse read. Continuous
movement and continuous observation are **one change, not two** — they only
make sense together.

## Rung 0 — close out M5

The L2-on-position-code sweep and the 21×21 arena. Between them these are the
last two hypotheses with a specific mechanism behind them.

If both come back negative, stop hunting and record the honest result: **this
architecture, on this task, produces place codes.** That is a real finding, and
it is consistent with published scepticism that grid codes emerge robustly from
trained path integrators rather than from carefully chosen readouts,
regularisers and nonlinearities.

## Rung 0b — capacity pressure (added 2026-07-25, user hypothesis)

Grid codes may only appear when a place code becomes *unaffordable*. Measured
ratios so far:

| arena | locations | position dims | locations per unit | periodic units |
|---|---|---|---|---|
| 11×11 | 121 | 120 | 1.0 | 1/120 |
| 21×21 | 441 | 120 | 3.7 | 0/120 |

At 1.0 the model can nearly afford a dimension per location, so there is no
pressure toward an efficient code at all. 3.7 improved mean periodicity
slightly (−0.390 → −0.369) but passed nothing.

**Test it by shrinking the code, not by changing the world.** `--module-dims`
takes 21×21 to 14.7 or 22.1 locations per unit while holding everything else
fixed. Continuous space (Rung 4) is a *poor* test of this hypothesis despite
seeming like a natural one: it changes the action representation, arena size,
observation structure and loss simultaneously, so a positive result would not
be attributable.

**Prediction (FALSIFIED 2026-07-25): periodicity does NOT rise monotonically
with locations-per-unit. It peaks and reverses.**

| arena / dims | loc per unit | mean periodicity | field score | accuracy vs ceiling |
|---|---|---|---|---|
| 11×11 / 120 | 1.0 | −0.390 | 0.93 | 98% |
| 21×21 / 120 | 3.7 | −0.369 | 0.87 | 96% |
| **21×21 / 30** | **14.7** | **−0.054** | **0.50** | **98.9%** |
| 21×21 / 20 | 22.1 | −0.225 | 0.70 | 93% |
| 31×31 / 20 | 48.0 | −0.144 | 0.60 | 98.3% |

Complete sweep, five points. Peak at 14.7 locations per unit; no configuration
put a single unit over 0.30. A second prediction failed here too: 48 loc/unit
was expected to be *worse* than 22 on a starvation argument, and it is better
on both metrics (−0.144 vs −0.225, 98.3% vs 93% of ceiling). Capacity per unit
is evidently not the whole story — arena size interacts with it.

Capacity pressure is the only hypothesis to show a real effect — the field
score halving from 0.93 to 0.50 means position units stopped being single blobs
and became genuinely spread out. But it peaks near 15 locations per unit and
then regresses, and the accuracy column says why: at 22 the model is starved
enough to drop to 93% of ceiling, and a model that is not solving the task
cleanly has no clean structure of any kind. 4 dims per module is the structural
floor for a 2D phase, so there is no slack left.

**Revised reading: capacity moves codes AWAY from place-like without arriving
at periodic** — a distributed middle ground. Best observed is still −0.05
against a 0.30 threshold.

**The likely reason none of this worked, identified 2026-07-25.** Every
published demonstration of grid cells emerging in a trained network supervises
*space* directly: Banino et al. train against place-cell and head-direction
activations, Cueva & Wei against (x, y) with activity regularisation, and
Sorscher et al. show the *shape* of the place-cell target is what produces
hexagons. **This model has no positional target at all** — its only objective
is next-observation prediction, and position is learned instrumentally as a
memory address. We have been turning knobs on a model asked a different
question. See Rung 0c.

**Why continuous space (Rung 4) may matter more than capacity.** On a discrete
grid a "periodic" code barely differs from a lookup table: a module with a
3-cell cycle takes only 3 distinct values along that axis, and since the model
is only ever evaluated at integer positions it can treat that as an arbitrary
3-state categorical variable and be exactly as correct. Discreteness has been
letting the model *avoid* the property being tested. Continuous space closes
that escape — the code must interpolate between samples, and smooth plus
repeating is genuinely periodic. Continuous space also fixes the measurement:
rate maps here are 11×11 or 21×21, giving 21×21 or 41×41 autocorrelograms,
which is small for this metric; binning a continuous arena at 100×100 with many
cycles visible is the regime gridness was designed for.

Caveat from the non-monotonicity: "continuous space has effectively infinite
positions, therefore maximum pressure" would likely land *past* the sweet spot.
Expect failure-to-learn, not elegance, if the code is also squeezed.

## Rung 2b — hierarchical place (added 2026-07-25, user proposal)

The current memory is flat: one query, one pass over everything, retrieving
fine-grained *locations*. A second level would aggregate a **set** of retrieved
memories into a coarse region descriptor — "what kind of place is this" — and
use it to prime or gate the fine search.

Terminology note: a place cell in the literature is fine-grained, so our cache
entries are already place cells in the conventional sense. What this rung adds
is the *coarse* end of the scale — closer to the dorsal-to-ventral place-field
size gradient (fields growing from centimetres to metres) and to context coding
via global remapping.

Solves a real scaling problem: as memory grows, searching all of it is both
expensive and noisier, which is why β = log(n) exists at all. Narrowing the
search first is the standard fix, and it connects to the hierarchy idea of one
navigator within a room and another over the graph of rooms.

Sits naturally after the codebook (Rung 2), whose slots are the obvious thing
to aggregate over.

## Rung 0c — auxiliary spatial head — **THE LAST M5 EXPERIMENT**

Run this, then close M5 whatever the outcome. Six mechanisms have been refuted
(frequencies, drift gate, arena size, L2, capacity, nonlinear bottleneck) and
every one of them removed an obstacle rather than supplying a reward. This is
the only remaining test that supplies one.

**Do both ingredients together**, since the published recipe needs both and
either alone is expected to fail: place-cell-shaped targets (Gaussian bumps
whose similarity structure is what yields hexagons) AND nonnegative
activations in the position stream. Currently we have neither — units are tanh
then LayerNorm, so rate maps are of a *signed* quantity where grid cells are
nonnegative rates.

Add a head off the position stream predicting **place-cell-shaped targets**
(a population of Gaussian bumps over locations), weighted small, everything
else unchanged. A deliberate, labelled deviation from the rule that locations
are analysis-only, justified as a diagnostic rather than a change to the task.

Raw (x, y) targets are probably NOT enough: the minimal solution is a linear
code in two dimensions with no pressure to be periodic. The published recipe is
place-cell-shaped targets plus nonnegative activations plus an activity cost —
we currently have none of the three.

**Prediction, and it is a real one: this will be neutral-to-harmful for task
accuracy.** We already measured the relevant effect. M1's operators were
trained for linear decodability (99.6% probe accuracy) and, transferred into
the memory model, scored **76.5% against 95.8% from scratch** — worse than
random init as memory addresses. Decoding-optimal and retrieval-optimal appear
to conflict.

Outcomes:
- Grid cells appear → the *objective* determines the code, the architecture was
  always capable, and everything tested in Rung 0/0b was a knob on the wrong
  question.
- No grid cells → the architecture genuinely cannot produce them, which is a
  far stronger negative result than anything currently held.
- Accuracy drops → evidence that **a memory-addressing system does not want
  grid codes**, because what makes a code decodable is not what makes it a good
  address. That would make M5's outcome a finding rather than a failure.

## Rung 0d — nonlinear grid→place bottleneck — **REFUTED 2026-07-25**

**The leading hypothesis for M5, and the first that is not a knob.**

Why no grid codes form: the memory addresses by dot product, and a periodic
code gives periodic similarity — a main peak plus side lobes wherever modules
re-align, so a query partially retrieves memories a full period away. Unimodal
similarity is strictly better for retrieval, so the loss selects place codes.

Why the existing projection does not rescue it: `to_key` is
`Linear(120 → 64, bias=False)`. Key similarity is `e(x)ᵀWᵀW e(y)` — a reweighted
version of the same quadratic form. **A linear map cannot turn periodic
similarity into unimodal similarity**; it can rescale or discard modules but the
side lobes come from the periodic components themselves. So the demand for
clean unimodal similarity propagates back onto `e_t`, and the recurrent state
is forced to do the place-cell job because nothing downstream can do it for it.

The biological transform is nonlinear and that is the point: grid → place is
sum-several-periodic-inputs-and-threshold. Where all modules constructively
align the sum clears threshold; at side lobes, where only some align, it does
not. Thresholding is what kills the lobes. Our architecture skips this step
entirely.

**The change:** `Linear → ReLU → key`, with the ReLU load-bearing. Probably
*wider* than the position code, not narrower — grid codes are compact, place
codes are sparse and need more units, so 120 → 64 is the wrong direction.

**The catch, and why this pairs with Rung 0b.** A nonlinear head makes a
grid-like `e_t` *permissible*, not *preferred*. If the head can build good keys
from anything, the state has no reason to become periodic. The incentive has to
come from **capacity pressure on `e_t`** — the effect measured in Rung 0b,
peaking near 15 locations per unit, since grid codes pack position into fewer
dimensions than place codes.

Neither alone works, which explains both negative results: the capacity sweep
failed because periodicity was still punished at the key, and a nonlinear head
alone would fail because nothing rewards it. **Together is the first
configuration where a grid code is both allowed and advantageous.**

**Prediction:** a division of labour — `e_t` becomes grid-like while the head's
output becomes place-like.

**RESULT: neither happened.** Two arms, identical but for the ReLU on a
30 → 256 key:

| | position periodicity | key periodicity | key field | accuracy vs ceiling |
|---|---|---|---|---|
| linear key | −0.024 | +0.016 | 0.701 | 98.8% |
| nonlinear key | −0.068 | −0.038 | 0.436 | 98.0% |

The position code is no more periodic with the nonlinearity than without, and
the key did not become more place-like either — its field score *fell*.
Accuracy was unaffected, so the ReLU is not harmful, it simply changes nothing.
The linear projection was not the obstacle. The argument that a linear map
cannot convert periodic similarity to unimodal is still true; it was not the
binding constraint.

**The pattern across all six refutations:** every intervention tested so far
*removed an obstacle* to periodicity. Not one of them *rewarded* it. That is
what Rung 0c exists to fix, and it is the last M5 experiment worth running.

## Rung 1 — conjunctive reverse read

Concatenate the position estimate onto the reverse read's query and keys, with
a learnable λ *bounded so position breaks ties rather than filtering*:

    score = observation_similarity + λ · position_similarity

Because a dot product of concatenated vectors is the sum of the parts, and
softmax turns a sum into a product, this multiplies the two pieces of evidence.
Prior × likelihood — Bayesian localisation, obtained from concatenation alone
with no new mechanism.

**Constraint that decides whether it works:** `λ × (position similarity range)`
must stay below the gap between a matching and non-matching observation.
Otherwise a wrong-symbol memory at the believed position outranks a
right-symbol memory elsewhere, and the correction confirms its own error.

**Prediction:** the learned gate opens well past its current 0.12. The gate is
the model's own verdict on how much the correction is worth; a sharper
correction should earn more trust. If accuracy improves but the gate stays at
0.12, the mechanism story is wrong and something else caused the gain.

Do this before the codebook: a codebook inherits the same ambiguity, so sharpen
the key before discretising it.

## Rung 2 — codebook memory

Replace the unbounded KV cache with a fixed set of novelty-allocated slots.
Note the current write rule ("append only if no similar conjunction is stored")
is already a degenerate codebook — unbounded slots, no updating.

Buys three things the current design lacks: bounded capacity, **pattern
completion** (snap to one slot rather than averaging a blend, which is the CA3
function one-shot attention cannot do), and the substrate for Rung 3.

**Prediction:** retrieval distance drops sharply from the measured 1.20 cells
(chance 3.59). Standard VQ hazards apply: codebook collapse, dead slots,
non-differentiable lookup needing a straight-through estimator.

## Rung 3 — persistence across walks

Keep the codebook between episodes in the same environment.

**The first rung that breaks a ceiling rather than approaching one.** M4 is
bounded by the within-walk revisit rate because every walk starts amnesiac —
the model cannot know a place it has not visited *this episode*. Persistence
makes returning to a known environment categorically different from entering a
new one. It is also the missing hippocampal piece: consolidation.

**Prediction:** second-visit accuracy exceeds the single-walk revisit ceiling,
which nothing so far has done or could do.

## Rung 4 — continuous space

Replace the four-matrix action lookup with two velocity generators:

    W(v) = expm( v_x · G_x + v_y · G_y )        # exact
    e_next = e + (v_x G_x + v_y G_y) · e        # first-order, cheaper

The first-order form is Euler integration — it steps along the tangent rather
than the arc — and the per-step LayerNorm cancels most of the resulting radius
drift, which is why published continuous models get away with it. Start there.

Touches `position.py` and walk generation only. The memory streams, gate and
readout never knew the world was discrete; they only consume `e`.

**Prediction:** comparable accuracy with **half** the transition parameters
(5,760 vs 11,520). Also removes the quantisation floor that has hampered M5 —
arena size becomes free relative to grid spacing, which is the regime the
periodicity measure was designed for.

## Rung 5 — image world

An image as the backdrop, an 8×8 patch around the current position as the
observation. Depends on Rung 4.

The biggest single jump, because three things change at once: the target
becomes contrastive or regression rather than classification; `W_x` becomes a
learned patch encoder rather than a projection off a one-hot; and the node/edge
baselines need patch-space equivalents (nearest-neighbour retrieval).

Worth doing. Worth not starting while anything else is unsettled, since it
changes the measuring instruments at the same time as the model.

## Rung 6 — rotation adapter

Frozen operators plus a low-rank **on-manifold** correction:

    W_new = W_frozen · exp(A),   A skew-symmetric, low rank

`exp` of a skew-symmetric matrix is exactly a rotation, so eigenvalues cannot
leave the unit circle however the adapter trains. This matters because the
operator is applied ~300 times in a row: an additive LoRA-style bump nudges
eigenvalues off the circle and the error compounds. Geometry preservation is
structural here, not a hope.

**Prediction:** a few dozen adapter parameters recover most of the measured
19-point gap (frozen 76.5% vs scratch 95.8%). Independent of Rungs 1–5.

## The target (user, 2026-07-25) — an agent that walks an image

The rungs below are steps toward this, and it is worth stating plainly because
it changes which of them matter.

**An agent that classifies an image by walking it**: moving continuously,
seeing only a local patch, building a position-addressed memory of what it has
seen where, choosing where to look next, with a **frozen navigator reused
across visual domains**. The claim would be that the navigation primitive
transfers untouched to a new domain and only the appearance-specific parts
retrain — active perception plus the transfer story, which is stronger than
either half alone.

**Why continuous is a prerequisite, not a nicety.** Discrete operators are
*lattice-bound*: four matrices tied to four moves on one grid, and an adapter
can rescale or rotate them but they still only express that lattice. Continuous
operators are *geometry-bound*: two generators plus a velocity express any
translation, so the adapter tunes a **metric** — scale, aspect, shear,
orientation, three or four numbers — covering essentially all 2D geometries.
A reusable primitive has to be continuous; the discrete one never could be.

**Walls, without breaking the algebra.** The rule that must survive is "a
velocity means exactly this displacement". So let the world resolve collisions
however it likes and feed the position stream the **realized** velocity, not
the intended one. The agent integrates what actually happened, walls become
transparent to path integration, and the policy's problem (avoid walls) stays
separate from the navigator's problem (know where you are).

**The policy step is where difficulty concentrates.** A policy makes the
agent's own behaviour determine its data — non-stationary, with collapse modes
(stops moving, circles one region). Action selection through a
non-differentiable world normally means REINFORCE and its variance.

*But this architecture already contains a predictive world model.* Choose
actions by querying its own predictions — move where the predicted observation
is most uncertain, or where uncertainty about the task label falls fastest.
Model-based active perception, no RL stack, reusing machinery already built.

**Order:** continuous (Rung 4, running) → image patches (Rung 5) → adapter
(Rung 6) → walls with realized-velocity feedback → policy → task-driven actions.
Rung 7's composition test can slot in after the adapter, independently.

## Rung 7 — N=2 composition

Two frozen navigators, a product topology (grid × ring), a trained attention
router over their position codes, knobs on each. Freeze both cores; train only
routing and knobs.

**The north star's atomic test.** Direct sums of representations are
automatically representations, so the geometry should compose for free and the
router only has to learn the *binding*. Success means the thousand-copy version
has a green light; failure means the interface is the bottleneck, learned at
N=2 instead of N=1000.

**Prediction:** it path-integrates a product space neither core was trained on.

**Known obstruction, do not design around it:** position factorises, memory does
not. A landmark is a function on the *joint* space, and independent per-factor
memories can only represent separable functions. The copies must share one
conjunctive memory.

## Tracks

Three loosely independent lines:

- **memory quality** — 1 → 2 → 3
- **world richness** — 4 → 5
- **composability** — 6 → 7

Fastest path to the composability endgame: **1 → 6 → 7**, treating 2–5 as depth
added afterwards. Recommended path: **1 → 3 → 6 → 7**, because persistence
changes what the system fundamentally *is* rather than how well it does the
task it already does.

## When to rewrite this plan

Concrete results that should reorder the rungs:

- **Rung 1's gate does not open.** The conjunctive story is wrong. Do not build
  the codebook on top of it — reconsider whether the reverse read should be
  retrieval at all, or whether the ambiguity needs solving some other way.
- **Rung 0's L2 sweep produces periodic codes.** Then regularisation was the
  answer all along, M5 reopens, and it becomes worth sweeping properly before
  anything else — a passing M5 changes what "a good navigator" means for
  Rungs 6 and 7.
- **Rung 3 does not beat the revisit ceiling.** Persistence is not delivering
  what it promises; suspect the codebook's capacity or its allocation rule
  before concluding persistence does not help.
- **Rung 6 recovers the frozen gap cheaply.** Jump straight to Rung 7 — the
  knob question was the crux and it is answered.
- **Rung 6 fails.** Rung 7 is premature. Reconsider whether a pretrained
  navigator should be trained on an objective closer to how it will be *used*,
  since M1's operators were excellent for linear decoding and measurably worse
  than random init as memory addresses.
- **Anything makes M4 accuracy fall.** Check it against the *ceiling*, not
  against the old number. The ceiling moves with walk length and arena size.

## Standing lessons

Earned the hard way; they apply to every rung above.

1. **Verify an ablation by checking the mechanism's own statistic goes to
   zero.** The first `--no-gate` suppressed the correction only between
   truncation windows, left it running for 19 of every 20 steps, scored within
   a point of the full model, and read as "the gate is useless".
2. **Compute the ceiling, do not assume it.** The revisit rate is not the
   bound; guessing on unvisited cells is worth ~2 points. Reporting against the
   wrong bound flattered the model and made accuracy look like it could exceed
   a bound it never approached.
3. **A rising loss is divergence.** Do not theorise about the task until the
   learning rate is exonerated.
4. **Measure the thing you are actually claiming.** The step loop is
   kernel-launch bound and batch 16 → 128 is free — both correct, both measured
   on one machine. I then concluded renting would not help, which that
   measurement could not support, and repeated it twice with confidence. A
   rented host with a fast consumer CPU ran the identical job **~9x faster**
   (0.25 vs 2.46 s/iter). Launch issue rate is CPU-side, so the host CPU *is*
   the bottleneck. Confirmed twice over: a $0.088 RTX A4000 with an i7-13700
   matched a $0.268 RTX 5090 to within 7%, while a $0.308 RTX 5090 on an EPYC
   was **5.5x slower than both**. Shop for the CPU; buy the cheapest card.
5. **Validate a metric on synthetic data before trusting it on real.** The
   periodicity score was checked against a hexagonal lattice (+0.90), noise
   (−0.10) and a single blob (−0.44) first, which is what makes the negative
   result credible rather than a suspected bug.

## What has already changed

Kept as evidence that the plan moves:

- *M5 frequencies.* Four of five modules had cycles that could not fit an
  11-cell arena. Fixing it improved accuracy (95% → 98% of ceiling) and did
  nothing for periodicity. Hypothesis refuted.
- *M5 drift gate.* Suspected of anchoring codes to places. The ablated model
  scores worse. Hypothesis refuted.
- *M4 curriculum effect.* "A pool beats infinite fresh environments" was a
  diverging learning rate. Retracted.
