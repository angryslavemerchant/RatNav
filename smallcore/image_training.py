"""Training and evaluation for the image world.

Kept separate from ``training.py`` rather than folded into it with a flag: the
window loop is the same shape, but the objective genuinely differs -- a
contrastive choice among real patches instead of cross-entropy over a
vocabulary -- and the metrics and baselines differ with it. Two short honest
loops beat one loop with a mode switch threaded through every line.

What is measured, and against what
----------------------------------
Accuracy is top-1 over a pool of candidate patches, so it keeps the symbol
world's reading: the fraction of steps where the model named what was about to
appear. Chance is one over the pool size.

Two baselines, both of which the model has to beat for anything to have been
learned. They replace the node and edge agents, which were tables keyed on a
symbol and cannot exist here:

* **persistence** -- predict the patch you can currently see. Free, and strong
  whenever steps are small relative to the patch, so it calibrates how much of
  the score is smooth continuation rather than knowledge.
* **nearest neighbour** -- find the most similar patch seen earlier in this
  walk and predict whatever followed it. This is the content-addressed
  baseline, and it is the one that matters: it is exactly what a memory
  retrieves when it is keyed on *appearance* rather than on *position*. The
  model's whole claim is that addressing by position beats it.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from smallcore.config import Config
from smallcore.patches import blocked_mask, info_nce_grouped
from smallcore.recurrent import SmallCoreRecurrent


def to_tensors(walks, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """``(velocities (B, T-1, 2), patches (B, T, P, P))``."""
    velocities = torch.tensor(
        np.stack([w.velocities[:-1] for w in walks]), dtype=torch.float32,
        device=device,
    )
    patches = torch.tensor(
        np.stack([w.patches for w in walks]), dtype=torch.float32, device=device
    )
    return velocities, patches


def overlap_radius(patch: int, cell: int, speed: float) -> int:
    """How many steps apart two views still share pixels.

    Candidates this close to the target are partly the answer rather than
    negatives, so they are excluded from the pool. Derived from the geometry
    rather than picked: views at ``d`` cells apart overlap while
    ``d < patch / cell``, so the radius is ``ceil(patch_cells / speed) - 1``.
    Zero is a legitimate answer -- at a full patch per step nothing overlaps and
    nothing should be masked.
    """
    return max(0, math.ceil((patch / cell) / speed) - 1)


def _ids(batch: int, start: int, stop: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Flattened (walk, timestep) labels for a ``(B, L)`` block."""
    walk = torch.arange(batch, device=device).repeat_interleave(stop - start)
    time = torch.arange(start, stop, device=device).repeat(batch)
    return walk, time


def run_image_walk(
    model: SmallCoreRecurrent,
    velocities: torch.Tensor,
    patches: torch.Tensor,
    config: Config,
    window: int,
    use_gate: bool,
    train: bool,
    mask_radius: int,
    collect: bool = False,
) -> dict:
    """Run a batch of image walks in truncation windows.

    As in ``training.run_walk``, each window is backpropagated as it completes
    so memory stays bounded by the window rather than the walk, and gradients
    accumulate across windows for one optimiser step.

    During training the contrastive pool is the current window, which is what
    makes the loss cheap. ``collect`` additionally returns every prediction and
    target so evaluation can score against the *whole walk* -- a much larger and
    harder pool, and a fixed one, so the number means the same thing across
    runs of different batch size.
    """
    batch, steps = patches.shape[:2]
    state = model.initial_state(batch, patches.device)
    state = model.observe(state, patches[:, 0])  # step 0: remember, don't predict

    n_windows = max(1, (steps - 1 + window - 1) // window)
    hits = seen = 0
    hits_position = 0
    loss_sum = gate_sum = 0.0
    kept_pred, kept_target = [], []

    for start in range(1, steps, window):
        stop = min(start + window, steps)
        state = state.detach()

        output, state = model.run_chunk(
            state, velocities[:, start - 1 : stop - 1], patches[:, start:stop],
            use_gate=use_gate,
        )

        width = output.logits.shape[-1]
        # Group the batch so the contrastive pool stays a fixed size however
        # large the batch grows; see patches.info_nce_grouped.
        # Capped at the batch, because evaluation runs a handful of walks and
        # a group larger than the batch is simply "the whole batch". The
        # reported eval number is scored over the whole walk by
        # _pool_accuracy regardless, so grouping does not move it.
        group = min(config.contrastive_group or batch, batch)
        if batch % group:
            raise ValueError(
                f"batch {batch} must divide by contrastive_group {group}"
            )
        groups = batch // group
        predicted = F.normalize(
            output.logits.reshape(groups, -1, width), dim=-1
        )
        predicted_pos = F.normalize(
            output.logits_position.reshape(groups, -1, width), dim=-1
        )
        target = F.normalize(
            output.values.reshape(groups, -1, config.obs_dim), dim=-1
        )

        walk_id, time_id = _ids(group, start, stop, patches.device)
        blocked = blocked_mask(walk_id, time_id, mask_radius)

        pred_loss, pred_hits = info_nce_grouped(
            predicted, target, blocked, config.temperature
        )
        pos_loss, pos_hits = info_nce_grouped(
            predicted_pos, target, blocked, config.temperature
        )
        drift = (output.gated - output.integrated).pow(2).mean()
        reg_code = output.gated.pow(2).mean()

        loss = (
            pred_loss
            + config.w_pred_pos * pos_loss
            + config.w_drift * drift
            + config.l2_position_code * reg_code
        )
        if train:
            (loss / n_windows).backward()

        with torch.no_grad():
            hits += pred_hits.sum().item()
            hits_position += pos_hits.sum().item()
            seen += pred_hits.numel()
            loss_sum += pred_loss.item()
            gate_sum += output.gate.mean().item()
            if collect:
                kept_pred.append(predicted.detach().reshape(batch, -1, width))
                kept_target.append(
                    target.detach().reshape(batch, -1, config.obs_dim)
                )

    metrics = {
        "accuracy_window": hits / max(seen, 1),
        "accuracy_position_window": hits_position / max(seen, 1),
        "loss": loss_sum / n_windows,
        "gate": gate_sum / n_windows,
    }
    if collect:
        # Windows were laid out (walk-major within window); reassemble into one
        # (B, T-1, D) block so the pool below is the whole walk.
        metrics["predicted"] = torch.cat(kept_pred, dim=1)
        metrics["target"] = torch.cat(kept_target, dim=1)
    return metrics


def _flat_normalised(patches: torch.Tensor) -> torch.Tensor:
    """``(B, T, P, P)`` pixels -> ``(B, T, D)`` mean-removed unit vectors.

    Removing the mean before normalising makes the similarity a correlation, so
    two patches match on *structure* rather than on overall brightness.
    """
    flat = patches.reshape(*patches.shape[:2], -1)
    flat = flat - flat.mean(dim=-1, keepdim=True)
    return F.normalize(flat, dim=-1)


def _pool_accuracy(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask_radius: int,
    source: torch.Tensor | None = None,
) -> float:
    """Top-1 over the whole-walk pool. Inputs ``(B, N, D)``, unit-norm.

    ``source`` is the trap this metric sets for any baseline that predicts by
    *copying* a remembered patch. That copy is itself a member of the candidate
    pool, and it correlates 1.0 with itself, so without excluding it the
    baseline unfailingly retrieves its own prediction and scores exactly zero --
    which reads as "memory cannot help here" when it means nothing of the sort.
    Pass the ``(B, N)`` pool index each prediction was copied from, or -1 where
    the prediction is not a pool member. The model needs none of this: it
    predicts an embedding it never copied from anywhere.
    """
    batch, n, _ = predicted.shape
    walk_id = torch.arange(batch, device=predicted.device).repeat_interleave(n)
    time_id = torch.arange(n, device=predicted.device).repeat(batch)
    blocked = blocked_mask(walk_id, time_id, mask_radius)
    if source is not None:
        rows = torch.arange(batch * n, device=predicted.device)
        flat_source = source.reshape(-1)
        valid = flat_source >= 0
        columns = walk_id * n + flat_source.clamp(min=0)
        blocked[rows[valid], columns[valid]] = True
        blocked.fill_diagonal_(False)  # never block the answer
    logits = predicted.reshape(batch * n, -1) @ target.reshape(batch * n, -1).T
    logits = logits.masked_fill(blocked, -1e9)
    labels = torch.arange(batch * n, device=predicted.device)
    return (logits.argmax(dim=1) == labels).float().mean().item()


@torch.no_grad()
def patch_baselines(
    patches: torch.Tensor,
    velocities: torch.Tensor,
    mask_radius: int,
    direction_weight: float = 1.0,
) -> dict:
    """Content-addressed baselines and the ambiguity ceiling, on the same pool.

    Every entry predicts a *patch*, scored in pixel space against exactly the
    candidates the model faced, so the numbers are directly comparable.

    * **persistence** -- predict what you can see right now. Near chance
      whenever a step exceeds the patch width, since consecutive views are then
      disjoint; that is the point of reporting it. It says how much of a score
      is smooth continuation rather than knowledge, and here it should say
      almost none.
    * **nearest neighbour** -- the edge agent's patch-space equivalent, and the
      baseline that matters. At step t it searches everything seen before
      ``t - mask_radius`` for the closest match to the current view *taken
      together with the current heading*, and predicts whatever followed that
      match. Conditioning on heading is what makes it a fair opponent: without
      it the agent retrieves a place it has stood before but has no idea which
      way it was facing, so it predicts a neighbour in an arbitrary direction
      and scores near chance for reasons that have nothing to do with memory.
    * **oracle** -- score the *true* patch against the pool. Not a model at
      all: it is the ceiling that observation ambiguity imposes, because two
      pixel-identical views at different places cannot be told apart by any
      predictor of appearance. Compute the bound, never assume it.
    """
    batch, steps = patches.shape[:2]
    flat = _flat_normalised(patches)  # (B, T, D)
    targets = flat[:, 1:]  # what must be predicted at each step

    persistence = flat[:, :-1]  # "next looks like now"

    heading = F.normalize(velocities, dim=-1)  # (B, T-1, 2)
    patch_similarity = torch.bmm(flat, flat.transpose(1, 2))  # (B, T, T)
    heading_similarity = torch.bmm(heading, heading.transpose(1, 2))

    neighbour = torch.empty_like(persistence)
    source = torch.full((batch, steps - 1), -1, dtype=torch.long,
                        device=patches.device)
    for t in range(1, steps):
        limit = t - 1 - mask_radius  # newest admissible match
        if limit < 1:
            neighbour[:, t - 1] = flat[:, t - 1]  # nothing old enough yet
            source[:, t - 1] = t - 2
            continue
        score = (
            patch_similarity[:, t - 1, :limit]
            + direction_weight * heading_similarity[:, t - 1, :limit]
        )
        best = score.argmax(dim=1)  # (B,)
        neighbour[:, t - 1] = torch.gather(
            flat, 1, (best + 1)[:, None, None].expand(-1, 1, flat.shape[-1])
        ).squeeze(1)
        source[:, t - 1] = best  # pool row j holds the patch at step j + 1

    # Persistence copies step t - 1, which is pool row t - 2.
    persistence_source = (
        torch.arange(steps - 1, device=patches.device) - 1
    ).expand(batch, -1)

    pool = batch * (steps - 1)
    return {
        "persistence": _pool_accuracy(
            persistence, targets, mask_radius, persistence_source
        ),
        "nearest_neighbour": _pool_accuracy(
            neighbour, targets, mask_radius, source
        ),
        "oracle": _pool_accuracy(targets, targets, mask_radius),
        "chance": 1.0 / pool,
        "pool": float(pool),
    }


@torch.no_grad()
def retrieval_oracle(walks, patches: torch.Tensor, mask_radius: int) -> float:
    """What retrieval could achieve if localisation were **perfect**.

    At each step it consults the true position -- which the model never sees --
    finds the nearest place it has previously stood, and predicts the patch it
    saw there.

    This is the instrument the continuous world needs and the discrete world
    did not. There, revisiting a location returned *exactly* the remembered
    symbol, so retrieval was exact and the only question was whether the model
    knew where it was. Here a walk essentially never returns to a point, only
    near one, and a patch remembered even a fraction of a cell away is already
    a poor prediction. Without this number a low score is unattributable: it
    could mean the position stream has failed, or it could mean no amount of
    knowing where you are would have helped. Separating those is the whole
    reason to compute it.

    Positions are used here for analysis only, exactly as ``locations`` are in
    the discrete baselines. Nothing about this feeds the model.
    """
    positions = torch.tensor(
        np.stack([w.positions for w in walks]), dtype=torch.float32,
        device=patches.device,
    )
    batch, steps = patches.shape[:2]
    flat = _flat_normalised(patches)
    targets = flat[:, 1:]
    predicted = torch.empty_like(targets)

    distance = torch.cdist(positions, positions)  # (B, T, T)
    source = torch.full((batch, steps - 1), -1, dtype=torch.long,
                        device=patches.device)
    for t in range(1, steps):
        limit = t - mask_radius  # places visited long enough ago to count
        if limit < 1:
            predicted[:, t - 1] = flat[:, 0]
            continue
        best = distance[:, t, :limit].argmin(dim=1)
        predicted[:, t - 1] = torch.gather(
            flat, 1, best[:, None, None].expand(-1, 1, flat.shape[-1])
        ).squeeze(1)
        source[:, t - 1] = best - 1  # flat step j is pool row j - 1
    return _pool_accuracy(predicted, targets, mask_radius, source)


@torch.no_grad()
def revisit_rate(walks, env, radius: float, mask_radius: int) -> float:
    """Fraction of steps whose position was already visited.

    Ignores the recent window, matching the baselines. This is *not* a ceiling
    and must not be quoted as one -- an unvisited patch is still
    partly guessable from texture statistics, and the discrete world's ceiling
    mistake was made by treating the revisit rate as a bound. It is reported as
    context for how much of the walk memory could possibly help with.
    """
    hit = total = 0
    for walk in walks:
        positions = walk.positions
        for t in range(1, len(positions)):
            limit = t - 1 - mask_radius
            if limit < 1:
                total += 1
                continue
            past = positions[:limit]
            distance = np.linalg.norm(past - positions[t], axis=1).min()
            hit += int(distance <= radius)
            total += 1
    return hit / max(total, 1)
