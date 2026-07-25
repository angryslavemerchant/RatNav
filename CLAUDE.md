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
| **M4** | Multi-environment training, fresh observations per env | ✅ 69.6% on unseen environments against a 70.5% ceiling and a 42.2% edge agent |
| **M5** | Analysis harness | Position units pass the periodicity threshold; memory units show localised fields |

M3 is where a sloppy implementation reveals itself. M5 is the actual result.

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

| | accuracy | revisit ceiling | edge agent |
|---|---|---|---|
| 300-step, unseen | **69.6%** | 70.5% | 42.2% |
| 100-step, unseen | **55.9%** | 55.1% | 41.9% |
| 300-step, seen pool | 72.9% | — | — |

Within 0.9 points of the ceiling on layouts never seen, and the seen-vs-unseen
gap is 3.3 points, so almost nothing was memorised. Unlike M2/M3 the revisit
rate is a genuine bound here: with appearance redrawn, an unvisited location is
unknowable, so this is close to the most any architecture could do.

**Training on a POOL beats training on infinite fresh environments**, which is
backwards from the obvious expectation. Fresh-every-batch tracked the pool to
iteration 500 (47.5% vs 46.0%) and then diverged — loss climbing 2.30 → 3.04,
accuracy collapsing to 22.5%. Suspect the learning rate first (2e-3 was raised
for the larger batch, and the fresh objective is far noisier since every batch
is a different world); a pool may also act as a curriculum, giving the memory
machinery a stable signal to bootstrap on before it has to generalise.

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
