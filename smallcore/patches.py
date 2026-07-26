"""The patch encoder -- what replaces the one-hot observation projection.

In the symbol world ``to_value`` is ``Linear(n_observations -> obs_dim)`` off a
one-hot: a lookup table with nothing to learn about *appearance*, because a
symbol has none. With an image the observation is a patch of pixels, and the
projection becomes a small convnet.

It is worth being explicit about how much this one module carries, because it
sits at the junction of both memory streams:

* forward read -- it produces the **values**, so it decides what "what was
  there" means;
* reverse read -- it produces the **keys**, so it decides which observations
  count as similar, and therefore how well "I have seen this before" works.

That second role is the one this whole rung exists for. In M6 the reverse read
compared one-hot symbols, so similarity was 1 for an exact match and 0
otherwise, carrying no information about *how far* two observations are apart.
Patch embeddings vary smoothly with position, so observation similarity now
carries distance -- which is what a landmark fix needs to be worth anything.

The output is LayerNormed. Both streams consume it through a dot product, and
an encoder free to grow its output norm would silently sharpen the reverse
read's softmax alongside the ``beta = log n`` schedule that is supposed to
control it.
"""

from __future__ import annotations

import torch
from torch import nn


class PatchEncoder(nn.Module):
    """``(..., P, P)`` pixels -> ``(..., obs_dim)`` embedding.

    Two strided convolutions and a projection. Deliberately small: the point of
    this rung is the *world*, not the vision model, and a large encoder would
    let appearance do work that should be falling to the memory.
    """

    def __init__(
        self, patch: int, obs_dim: int, channels: tuple[int, int] = (16, 32)
    ) -> None:
        super().__init__()
        self.patch = patch
        self.obs_dim = obs_dim
        c1, c2 = channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, c1, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            nn.GELU(),
        )
        reduced = -(-patch // 4)  # two stride-2 layers, ceil division
        self.project = nn.Linear(c2 * reduced * reduced, obs_dim)
        self.norm = nn.LayerNorm(obs_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        lead = patches.shape[:-2]
        flat = patches.reshape(-1, 1, self.patch, self.patch)
        if flat.shape[0] == 0:  # empty memory block
            return patches.new_zeros(*lead, self.obs_dim)
        hidden = self.conv(flat).flatten(1)
        return self.norm(self.project(hidden)).reshape(*lead, self.obs_dim)


def info_nce(
    predicted: torch.Tensor,
    target: torch.Tensor,
    blocked: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contrastive loss and top-1 hits over a pool of candidate patches.

    The symbol world scored prediction with cross-entropy over a 45-way
    vocabulary. A patch has no vocabulary, and the two obvious replacements
    fail differently: regression to pixels is minimised by the *average* of the
    plausible continuations, so it produces blur and rewards hedging, while a
    reconstruction loss spends capacity on texture detail that has nothing to
    do with knowing where you are. Picking the true patch out of a pool of real
    ones asks exactly the question the symbol task asked -- *which* of these is
    about to appear -- and inherits its interpretation, chance included.

    Args:
        predicted / target: ``(N, D)``, both L2-normalised.
        blocked: ``(N, N)`` bool, candidates excluded from row i's denominator.
            Patches close in time along one walk overlap in pixels, so they are
            near-duplicates of the answer rather than negatives; counting them
            would train the encoder to separate observations that genuinely are
            the same thing seen twice.
        temperature: Softmax temperature on cosine similarity.

    Returns:
        ``(loss, hits)`` -- hits is a bool ``(N,)`` of top-1 correctness.
    """
    logits = predicted @ target.T / temperature
    logits = logits.masked_fill(blocked, -1e9)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = nn.functional.cross_entropy(logits, labels)
    return loss, (logits.argmax(dim=1) == labels)


def info_nce_grouped(
    predicted: torch.Tensor,
    target: torch.Tensor,
    blocked: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """InfoNCE run independently inside each group of walks.

    Exists so that **batch size stops changing the objective**. Scored against
    the whole batch, the candidate pool is ``batch * window``, so raising the
    batch from 64 to 4096 turns a 1280-way choice into an 81920-way one -- a
    harder task with a different loss scale, which would make a large-batch run
    incomparable to every run already measured. Worse, it would confound the
    thing a bigger batch is supposed to buy: a lower-variance gradient on the
    *same* problem.

    That is not hypothetical here. At one patch per step the pool was already
    hard enough that the loss sat at exactly ``ln(pool)`` for 500 iterations
    before anything moved, so pool size is known to be load-bearing.

    Groups all share one ``blocked`` mask because every group has the same
    ``(walks, timesteps)`` layout.

    Args:
        predicted / target: ``(G, N, D)``, L2-normalised, N = group * window.
        blocked: ``(N, N)`` bool, shared across groups.
    """
    logits = torch.bmm(predicted, target.transpose(1, 2)) / temperature
    logits = logits.masked_fill(blocked.unsqueeze(0), -1e9)
    n = logits.shape[1]
    labels = torch.arange(n, device=logits.device).expand(logits.shape[0], n)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, n), labels.reshape(-1)
    )
    return loss, (logits.argmax(dim=-1) == labels)


def blocked_mask(
    walk_id: torch.Tensor, time_id: torch.Tensor, radius: int
) -> torch.Tensor:
    """``(N, N)`` mask of same-walk, temporally-adjacent candidates.

    The diagonal is never blocked -- it is the answer.
    """
    same = walk_id[:, None] == walk_id[None, :]
    near = (time_id[:, None] - time_id[None, :]).abs() <= radius
    mask = same & near
    mask.fill_diagonal_(False)
    return mask
