# SmallCore

A small sequence model that learns the structure of a space by walking around it and
predicting what it will observe next.

## Why this model

An agent moving through a world has to know where it is in order to predict what it will
see. It has exactly two sources of information about that, and neither one is sufficient:

- **Its own movements.** Integrating a sequence of actions gives a position estimate that
  is accurate over the short term and *drifts* over the long term, because errors accumulate
  with every step.
- **What it can see.** Landmarks are drift-free but **ambiguous** — the same observation
  occurs in many different places, so recognising one narrows your position without fixing it.

Dead reckoning drifts; landmarks alias. The only way to stay located is to use each to
correct the other, continuously. **That correction loop is what SmallCore is.**

This is worth building for two reasons beyond navigation:

1. **Structure is reusable; appearance is not.** How movements compose — that north then
   south returns you where you started — is a property of a space that holds across every
   space with that shape. If a model represents that separately from what any particular
   place looks like, it learns a new environment almost immediately instead of from scratch.
   Separating the two is the entire mechanism behind that transfer, and it is the main claim
   this architecture makes.

2. **It is a memory addressed by position rather than by content.** Ordinary attention
   retrieves things that *look like* what you are asking about. Here retrieval is keyed on
   *where you are*. That is a genuinely different indexing scheme, and it is the right one
   whenever content is ambiguous but position is not.

**Success is not a falling loss.** It is: structured periodic codes appear in the position
stream, and the model predicts correctly at locations reached by action sequences it has
never executed. Build the analysis harness before tuning anything (§6).

## 1. Architecture

Three components. The unusual one is the memory read.

### Position stream — recurrent, action-conditioned

Position is not a function of sequence index. It is a state evolved by the action taken:

```
e_{t+1} = LayerNorm( σ( e_t · W_{a_t} ) )
```

`W_{a_t}` is a learnable transition matrix **per action**, block-diagonal over `M` modules
initialised at geometrically-spaced frequencies so they settle at different spatial scales.
Multi-scale structure is an architectural prior we supply; the question is whether the model
organises it into clean periodic codes.

### Memory stream — causal attention, split sources

Queries/keys and values come from **different places**, and this is the core design decision:

```
q_t = e_t · W_e            # query  <- POSITION only
K   = E_{<t} · W_e         # keys   <- POSITION only  (tied to the query projection)
V   = X_{<t} · W_x         # values <- OBSERVATION only

y_t = softmax( β · q_t Kᵀ / √d_k ) · V
```

Position alone decides **what to attend to**; observation alone decides **what comes back**.
Query with *where you are*, retrieve *what was there*.

The split is not stylistic. Observations are deliberately ambiguous, so if keys carried
observation information the query could be satisfied by appearance similarity — which is
exactly the failure mode this design exists to prevent. Keys must be addresses, values must
be contents.

There is **no separate memory matrix — the KV cache is the memory**, and it spans the whole
walk.

`β = log(n_memories)`: an adaptive softmax temperature. Without it, attention flattens toward
uniform as the cache grows and late steps in long walks retrieve nothing useful.

**Write rule:** append `(e_t, x_t)` only if no sufficiently similar conjunction is already
stored (dot-product similarity against existing keys). Don't store the same fact repeatedly.

### The reverse read — drift correction

Path integration drifts, so it must be re-anchored on what is actually visible. That needs a
second attention running the *other way*:

```
q'_t = x_t · W_x           # query  <- OBSERVATION
K'   = X_{<t} · W_x        # keys   <- OBSERVATION
V'   = E_{<t} · W_e        # values <- POSITION
e_t^ret = softmax( q'_t K'ᵀ / √d_k ) · V'
```

"I see this symbol; it was at position p; so I am probably near p." Then blend the
path-integrated estimate with the retrieved one through a learned gate:

```
e_t = e_t^PI + gate([e_t^ret, e_t^PI]) ⊙ (e_t^ret − e_t^PI)
```

The gate is how the model learns *how much to trust each source* — heavily weighting
integration right after a good fix, heavily weighting landmarks once drift has accumulated.
**Omit this and the model looks fine on short walks and falls apart on long ones.**

### Readout

`y_t`, and separately `e_t` alone, → small MLP → logits over the observation vocabulary.
Observations are one-hot, so prediction is plain classification.

## 2. Loss

1. `L_pred` — cross-entropy on the next observation from the full model **(main objective)**
2. `L_pred_pos` — cross-entropy on the observation from the position code alone
3. `L_drift` — squared error between the gated position and the pure path-integrated position
4. `L_reg_w` — L2 on weights
5. `L_reg_e` — **L2 on the position code**, starting weight ~`0.01`

Term 5 matters far more than its magnitude suggests: pressure toward an efficient position
code is a large part of what drives clean periodic structure. Without it, expect unstructured
mush. Anneal the loss weights over training rather than fixing them.

## 3. Layout

```
smallcore/
  graphs.py      # topologies, environments, walks          [done]
  plots.py       # visualisation                            [done]
  envs/          # generated environment JSON               [done]
  config.py      # hyperparameters                          [todo]
  position.py    # recurrent action-conditioned encoder     [todo]
  memory.py      # attention: forward read, reverse read, write rule   [todo]
  readout.py     # observation prediction head              [todo]
  model.py       # ties the streams together                [todo]
  losses.py      # the five terms                           [todo]
  train.py       # training loop                            [todo]
  analysis.py    # rate maps, periodicity, fields, zero-shot [todo]
scripts/
  m0_demo.py     # build envs, sample walks, render         [done]
figures/         # generated output
```

## 4. Running it

The interpreter is a conda env, **not** on PATH:

```
C:/Users/JmgLi/anaconda3/envs/ToastEnv/python.exe
```

Python 3.11.14 · numpy 2.4.6 · matplotlib 3.10.8 · scipy 1.17.1 · torch 2.10.0+cu130 (CUDA available).

```bash
"C:/Users/JmgLi/anaconda3/envs/ToastEnv/python.exe" scripts/m0_demo.py
```

`main.py` at the project root is an unused PyCharm stub and can be deleted.

## 5. Conventions

**Topology vs. Environment.** A `Topology` is the graph: locations and the actions
connecting them. An `Environment` is a topology plus an observation assignment. Calling
`assign_observations` repeatedly on one topology yields environments that share structure but
look nothing alike — the setting the whole architecture is built to exploit. Keep these
separate; collapsing them destroys the transfer story.

**An action means the same thing everywhere.** `transitions[a, i]` is deterministic, and an
action unavailable at a location is simply never sampled. A blocked move is *never* silently
converted to a no-op, because then "north" would mean *move* in the interior and *stay* at the
boundary, and the single learned operator `W_north` could not be coherent. If you add
topologies, preserve this.

**Walk indexing.** At step `t` the agent is at `locations[t]`, observes `observations[t]`, and
takes `actions[t]`, arriving at `locations[t+1]`. The last action is `NO_ACTION = -1`.

**`walk.locations` is for analysis only.** The model gets observations and actions; recovering
position is the task, not an input. Never feed it in.

**Observation ambiguity is a requirement.** 121 locations share 45 symbols (~2.7 locations
each). If observations were unique, next-observation prediction would be table lookup and
position would never need representing. Do not "fix" this by enlarging the vocabulary.

**Validate walks.** `walk.validate(env)` checks legality cheaply. `m0_demo.py` validates 96
walks per run; keep that habit whenever walk generation changes.

## 6. Analysis harness — write this before tuning

You cannot read success off a loss curve. Build these first:

- **Rate maps** — average each position-stream unit's activation per location, spatially
  smoothed, → 2D map.
- **Periodicity score** — autocorrelogram of the rate map, correlated against rotations at
  30/60/90/120/150°; score = `min(corr@60,120) − max(corr@30,90,150)`. **0.3–0.5** is the
  conventional threshold for calling a unit periodic. This is the headline acceptance test.
- **Field score** — for memory-stream units: connected components of the rate map
  (`scipy.ndimage.label`), scored as *firing mass in the largest component ÷ total*. High
  means it fires in one place.
- **Zero-shot accuracy** — prediction accuracy at a location reached by an action sequence
  never executed. The real generalisation test.
- **Baselines** — a *node agent* (remembers the observation per location) and an *edge agent*
  (remembers per state-action transition). **Beat both, or nothing has been learned.**

## 7. Build order

| # | Milestone | Done when |
|---|---|---|
| **M0** | Environments and walks | ✅ 96 walks validated; figures rendered |
| **M1** | Position stream alone; linear probe decoding true location from `e_t` | ✅ 100% decoding on held-out 50-step walks (chance 0.8%), 87.9% at step 200 |
| **M2** | Forward read + readout, single environment | ✅ 88.9% vs 41.8% edge / 18.9% node on 100-step walks; 81.2% at 300 steps |
| **M3** | Reverse read + drift gate | ✅ 97.4% at 300 steps vs 95.8% at 50 — degradation inverted (−1.6%); ablation without the gate degrades +4.8% |
| **M4** | Multi-environment training, fresh observations per env | ✅ 69.6% on unseen environments — 95% of the 73.5% ceiling, vs a 42.2% edge agent |
| **M5** | Analysis harness | ⚠️ Built and validated. Memory units DO show localised fields (mean 0.80). Position units do NOT become periodic: 1/120 reach 0.30, mean −0.41 |

M3 is where a sloppy implementation reveals itself. M5 is the actual result.

**Work after M5 is planned in [ROADMAP.md](ROADMAP.md)** — seven rungs, each
with a falsifiable prediction, plus the specific results that should reorder
them. It is provisional by design: two hypotheses have already been refuted and
one claim retracted, and the ordering assumes more of both.

**Where M5 landed (2026-07-25) — half passes, half does not.**

Memory-stream units **do** develop localised fields: mean field score 0.80,
best 1.00. That half of §6 holds.

Position units **do not** become periodic. Only 1 of 120 reaches the 0.30
threshold and the mean is −0.41 (M3's model: 2/120, mean −0.43). The rate maps
show single blobs, inverted blobs and edge gradients — **place-like and
boundary-like codes, not grid-like ones**. Position units also score 0.93 mean
*field*, i.e. strongly localised, which is the opposite of what a periodic code
looks like.

Before blaming the architecture, check the arithmetic: the rotation advances
`pi * freq` per step, so a module's full cycle is `2 / freq` cells.

| module | freq | cycle | resolvable in an 11-cell arena? |
|---|---|---|---|
| 0 | 0.99 | 2.0 | no — at Nyquist on a discrete grid |
| 1 | 0.30 | 6.7 | yes — ~1.6 periods fit |
| 2 | 0.09 | 22 | no — exceeds the arena |
| 3 | 0.03 | 67 | no — exceeds the arena |
| 4 | 0.01 | 200 | no — exceeds the arena |

**Four of five modules cannot express spatial periodicity in this arena at
all** — they can only look like gradients, which is exactly what the maps show.
The §8 frequencies were never matched to an 11x11 world. Before concluding the
architecture does not produce grid codes, retest with frequencies whose cycles
land in roughly 3–8 cells, or a much larger grid. The measured ~x3 spacing
between modules is also far wider than the biological 1.4–1.7.

**Two suspects tested and refuted (2026-07-25).**

*Frequencies.* Retrained with cycles 3.0 / 4.2 / 5.9 / 8.2 / 11.5 cells, all
fitting the arena, ratio 1.4. Accuracy improved — 71.1% against a 72.5%
ceiling, i.e. 98% of achievable, up from 95% — but periodicity did not move:
1/120 units, mean −0.390 against −0.405 before. The frequency mismatch was
real and worth fixing; it was not the reason.

*The drift gate.* Suspected of anchoring codes to specific remembered
positions and so favouring place-like solutions. The gate-ablated model scores
*worse* (mean −0.561 vs −0.390), so the gate is not suppressing periodicity and
may mildly help.

**All four mechanistic hypotheses are now refuted.**

| hypothesis | test | units >= 0.30 | mean | verdict |
|---|---|---|---|---|
| baseline | original config | 1/120 | −0.405 | — |
| frequencies too coarse | cycles 3.0–11.5 cells | 1/120 | −0.390 | refuted |
| drift gate anchors codes | gate ablated | 1/120 | −0.561 | refuted (worse) |
| arena too small | 21×21, 1.8–5.0 cycles/module | **0/120** | −0.369 | refuted |
| under-regularised code | L2 0.01 → 0.1 | 1/120 | −0.405 | refuted |

Each was worth testing and each was wrong. Notably the fixes were not neutral —
corrected frequencies took zero-shot accuracy from 95% to **98% of ceiling**,
and 10x L2 more than doubled the drift gate (0.12 → 0.26) at no accuracy cost,
which is the only thing found so far that moves the gate at all.

**Conclusion: this architecture, on this task, produces place codes rather than
grid codes.** That is a real result, not a failure. It is consistent with
published scepticism that grid codes emerge robustly from trained path
integrators rather than from carefully chosen readouts, regularisers and
nonlinearities — and the model reaches 98% of the information-theoretic ceiling
with the codes it does build, so nothing pressures it toward a more elegant
solution.

**MAJOR CORRECTION (2026-07-25): periodic codes DID form. They are bands.**

Every "no periodic structure" claim below was an artefact of the metric. The
gridness score asks one question — *is this hexagonal?* — and a 1-D striped
pattern answers "no" indistinguishably from noise answering "no". Both land at
≈0. Position units with obvious diagonal bands scored −0.01 to −0.02 and were
recorded as unstructured. Found by *looking at the rate-map figures*, after
hours of trusting the scalar over the images it was drawing.

Counting 2D Fourier peaks says which symmetry is present (2 = band, 4 = square,
6 = hexagonal). On the 21×21 / 20-dim run:

    band (2) 10   square (4) 3   hex (6) 0   none/other 7

**13 of 20 units are genuinely periodic**, and the scales match the design:

| module | designed cycle | measured wavelength | peak counts |
|---|---|---|---|
| 0 | 3.0 | 14.2 | 2,4,2,2 |
| 1 | 4.2 | 8.2 | 4,2,6,2 |
| 2 | 5.9 | 7.5 | 2,2,2,2 |
| **3** | **8.2** | **8.2** | 2,2,2,2 |
| 4 | 11.5 | 13.0 | 4,2,2,2 |

Module 3 lands exactly on its designed cycle. Modules 2 and 3 formed clean
bands across all four of their units. Module 0 sits at Nyquist and is
unresolvable, as flagged.

What is genuinely absent is **hexagonal** structure. A hexagonal grid requires
three bands at 60° interfering constructively, and nothing in this world
privileges 60°: it is a **square lattice with four cardinal actions**, whose
natural angles are 0° and 90°. Bands plus square units is the periodic code
this world's symmetry calls for. `analysis.spectral_structure` measures it;
`periodicity_score` alone cannot and must not be used on its own.

**Always look at the rate maps.** A scalar that answers one question will
report "nothing" for every structure that is not the one it asks about.

**Why place codes, mechanistically (2026-07-25).** Grid codes are optimal for
representing position at high resolution in few units *given a downstream
decoder that can disambiguate combinatorially*. This memory has no decoder — it
has a **dot product**. And the two codes have opposite similarity structure:

* a place-like code gives `e(x)·e(y)` a single clean peak decaying with
  distance, so a query retrieves memories near x and nothing else;
* a grid-like code is periodic per module, so the dot product has a main peak
  **plus side lobes** wherever modules re-align — a query at x also partially
  retrieves memories a full period away, at an unrelated location.

For attention those side lobes are spurious retrievals. **The model is not
failing to find the optimum; it found the optimum for its actual objective.**
The loss rewards retrieving the right memory, and unimodal similarity beats
periodic similarity at that.

Note "accurate position" here means *lands in the right neighbourhood*, not
*decodes to precise coordinates* — measured retrieval distance is 1.20 cells
against 3.59 at chance. The two senses of accuracy come apart and the
architecture only ever asked for one.

This also retro-explains the M1 transfer result: operators optimised for linear
decodability (99.6% probe) scored 76.5% against 95.8% from scratch as memory
addresses. Two independent measurements, one explanation.

**Consequence:** continuous space will not fix this either — a dot product
prefers unimodal similarity regardless of discreteness. Go continuous for its
own payoffs (closes the lookup-table escape, halves transition parameters,
fixes measurement resolution, prerequisite for an image world), not for grid
cells. And an auxiliary spatial head (ROADMAP Rung 0c) creates pressure toward
grid codes that directly *fights* the retrieval pressure, which is why it is
predicted to cost accuracy.

One structural suspect is still untested: units are **tanh then LayerNorm**, so
a rate map here is of a *signed* quantity where grid cells are nonnegative
rates, and the LayerNorm couples all units every step.

**Where M2 landed (2026-07-25).** Walks start at a *random* location from M2
onward. Under M1's fixed start the readout can memorise the map — position
alone reached 99% — which would have made the memory stream decorative. With an
unknown origin the position code carries only displacement, position-only
accuracy sits at 7% against a 6% floor, and the forward read has to do the
work. Nothing is pretrained; the position stream is learned from scratch with
no direct objective, purely because good codes make retrieval land well.

`scripts/m2_analyse.py` shows *how* it works, and it is not what the milestone
assumed. Retrieval is not exact-cell lookup — the top attended memory is the
same location only 0.2% of the time. It is **neighbourhood** retrieval:
attention lands 1.20 cells away on average against 3.59 at chance, and the
same-location ratio (2.5x on 300-step walks) exceeds the same-symbol ratio
(1.6x), confirming keys carry addresses rather than appearance. The model
retrieves a local patch of remembered symbols, uses it to localise on the map
it learned in training, and predicts from there. That is why accuracy exceeds
the revisit rate, which bounds pure lookup only and is *not* a ceiling.

The 100-step to 300-step drop (88.9% → 81.2%) is the drift M3 exists to remove.

**Where M3 landed (2026-07-25).** The reverse read and gate work, and a proper
ablation separates them from the training change that arrived alongside:

| 300-step accuracy | reverse read | backprop |
|---|---|---|
| M2, 81.2% | no | full-walk |
| M3 ablation, 82.8% | no | truncated |
| M3, **97.4%** | **yes** | truncated |

Truncated backprop was worth ~1.6 points; the drift gate ~14.6. Without the
gate, accuracy still *degrades* with length (+4.8%); with it, degradation
inverts (−1.6%) because accumulated memory outweighs accumulated error.

The mean gate settles near **0.12** — corrections are small continuous nudges,
never teleports, which is the right response to a reverse read whose answer is
a blend of the ~2.7 locations sharing the queried symbol.

**Ablations must disable the mechanism inside the loop.** The first `--no-gate`
only replaced state *between* truncation windows, leaving the gate running for
19 of every 20 steps; it scored within a point of the full model and looked
like evidence the gate was useless. Verify an ablation by checking the
mechanism's own statistic goes to zero.

**Where M4 landed (2026-07-25).** Trained on a pool of 8 environments sharing
the topology, evaluated on 4 never trained on:

| | accuracy | ceiling | revisit rate | edge agent |
|---|---|---|---|---|
| 300-step, unseen | **69.6%** | 73.5% | 71.6% | 42.2% |
| 100-step, unseen | **55.9%** | 60.0% | 57.2% | 41.9% |
| 300-step, seen pool | 72.9% | — | — | — |

The model reaches **95% of the ceiling** on layouts never seen, and the
seen-vs-unseen gap is 3.3 points, so almost nothing was memorised.

**The ceiling is NOT the revisit rate** — that mistake was made and corrected
here. A location never visited this walk is still worth guessing at: an oracle
knowing the environment's symbol histogram names the most common symbol and is
right 6.6% of the time. So the bound is `revisit + (1 - revisit) * 6.6%`, about
2 points above the revisit rate, and quoting the bare revisit rate both
flatters the model and makes accuracy look like it can exceed the bound when it
merely exceeds revisits. Use `baselines.memory_ceiling`, not
`oracle_memory_accuracy`.

**Environment diversity is fine; the learning rate was not.** Fresh-every-batch
first appeared to be *worse* than a fixed pool (22.5% vs 69.6%), which looked
like a curriculum effect. It was not. That run used lr 2e-3, raised for the
larger batch; at the original 9.4e-4 it reaches **67.7%**, within 2 points of
the pool. The stable pool objective tolerated the higher rate and the
fresh-every-batch objective — a different world in every batch, so far noisier
gradients — did not. Divergence, not difficulty:

| | lr | 300-step unseen |
|---|---|---|
| pool of 8 | 2e-3 | 69.6% |
| fresh every batch | 9.4e-4 | 67.7% |
| fresh every batch | 2e-3 | 22.5% (diverged) |

Read a rising loss as divergence before theorising about the task.

**Large batches are nearly free here** (measured): 16 → 128 leaves wall-clock
per iteration flat at ~1.37 s while throughput scales 8x, because the step loop
is bound by sequential kernel launches, not arithmetic. The same measurement
answers whether to rent GPUs: a faster card does not shorten one run, since the
limit is launch issue rate on the CPU. Rent for parallel configurations only.

**Transfer from M1 hurts (2026-07-25).** Seeding M3's operators from a trained
M1 and continuing: 82.9%/92.3% (50/300-step). Freezing them: 76.5%/87.4%.
From scratch: 95.8%/97.4%. Monotone, and the wrong way round. M1's operators
path-integrate beautifully (99.6% linear decoding at step 200) but M3 needs a
code whose *dot products* behave well, since that is what attention scores are
— a different objective, and optimising hard for the first lands in a worse
basin for the second. The frozen arm also ran the highest gate of any run
(0.145 vs ~0.10): unable to fix its position stream, it leaned on landmarks.

## 8. Hyperparameters (starting point)

| Parameter | Value |
|---|---|
| Position modules | 5 |
| Dims per module | [30, 30, 24, 18, 18] (120 total) |
| Module init frequencies | [0.99, 0.3, 0.09, 0.03, 0.01] |
| Observation vocab | 45, compressed to 10 |
| Batch size | 16 |
| Rollout / truncation window | 20 steps |
| Walk length | 25–300 |
| LR | 9.4e-4 → 8e-5, ×0.5 every 4000 steps |
| Optimiser | Adam, no gradient clipping |
| Training iterations | 20,000 |

Target **under 500k parameters** — hours on GPU, plausibly overnight on CPU. Start with
**one attention layer, one head**, and a small post-attention residual+LayerNorm block
(hidden ≈ 20–64). Add depth only if M2 stalls.

## 9. Gotchas

1. **Truncated backprop vs. persistent memory.** The KV cache must span the whole walk (it
   *is* the memory) while gradients flow only through the current ~20-step window. So the cache
   holds **detached** entries from earlier chunks and grad-attached entries for the current one.
   Getting this wrong either explodes memory or silently kills long-range credit assignment.
2. **`straight_bias` is weaker than it looks.** Measured mean run lengths: bias 1.0 → 1.30
   steps, 2.0 → 1.59, 5.0 → 2.42. The default of 2.0 is only mildly correlated. If M1 struggles
   to learn path integration, **raise this first** — straighter trajectories make the transition
   operators far easier to identify.
3. **Rescale the softmax as the cache grows** (`β = log(n_memories)`), or late steps in long
   walks attend almost uniformly.
4. **Periodic structure is regularisation-sensitive.** If maps look unstructured, suspect the
   L2 on the position code before suspecting the architecture.
5. **Coverage is partial.** 300-step walks visit ~72% of an 11×11 grid (53–84%). Rate maps
   from a single walk will have holes; aggregate across walks before scoring.
6. **Depth and head count are unconstrained by the design.** Only the split-source attention,
   causal masking, and recurrent position encoding are load-bearing. Everything else is yours
   to choose, so tune it empirically rather than looking for a canonical answer.
