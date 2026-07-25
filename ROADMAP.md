# Roadmap

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
| M5 analysis harness | built and validated. Memory fields yes (0.80). **Periodic position codes no** (1/120 units) |

The system works. M5 asks whether it works *the elegant way*, and so far the
answer is no: it found place codes, not grid codes. Two mechanistic
explanations have been tested and refuted.

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
