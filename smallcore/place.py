"""Place-cell-shaped supervision — the one thing never tried (ROADMAP Rung 0c).

M5 refuted six mechanisms in the hunt for grid codes: module frequencies, the
drift gate, arena size, the L2 on the position code, capacity pressure, and a
nonlinear key bottleneck. The pattern across all six is that **every one of
them removed an obstacle to periodicity, and not one of them rewarded it.**

Every published demonstration of grid cells emerging in a trained network
supplies a reward: they supervise *space* directly. Crucially it is not raw
coordinates — the minimal solution for predicting (x, y) is a linear code in
two dimensions with no pressure whatsoever to repeat. What produces hexagons is
the *shape* of the target: a population of localised bumps, whose similarity
structure falls off with distance, read out linearly from the recurrent state.

So the target here is a softmax over ``n_place_cells`` Gaussian bumps scattered
through the arena, and the loss is cross-entropy against it.

This is a deliberate, labelled exception to the rule that ``positions`` are
analysis-only, and it should be read as a diagnostic rather than a change to
the task: it asks whether the architecture *can* produce grid cells when
rewarded for it, which is a different question from whether the memory
objective alone ever will.

**Predicted cost, from the roadmap and worth holding onto:** this should be
neutral-to-harmful for task accuracy. M1's operators were trained for linear
decodability (99.6% probe accuracy) and, transferred into the memory model,
scored 76.5% against 95.8% from scratch — worse than random initialisation as
memory addresses. Decoding-optimal and retrieval-optimal appear to conflict, so
a spatial head creating pressure toward grid codes is expected to fight the
retrieval pressure directly. If accuracy drops while periodicity rises, that is
not a failure; it is the trade being measured.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class PlaceTargets:
    """A fixed population of Gaussian bumps tiling the arena.

    Centres are drawn once and held fixed for a run: the readout has to learn a
    stable map, and resampling centres every batch would make the target a
    moving goal.
    """

    def __init__(
        self,
        n_cells: int,
        bounds: tuple[float, float, float, float],
        sigma: float,
        rng: np.random.Generator,
        device,
    ) -> None:
        lo_x, hi_x, lo_y, hi_y = bounds
        centres = np.stack(
            [rng.uniform(lo_x, hi_x, n_cells), rng.uniform(lo_y, hi_y, n_cells)],
            axis=1,
        )
        self.centres = torch.as_tensor(
            centres, dtype=torch.float32, device=device
        )
        self.sigma = sigma
        self.n_cells = n_cells

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        """``(B, T, 2)`` positions -> ``(B, T, n_cells)`` target distribution.

        A softmax over negative squared distance, which is the normalised
        Gaussian population response. Softmax rather than independent bumps
        because the readout is trained with cross-entropy, and because
        normalising makes the target a *pattern* over the population rather
        than a set of independent magnitudes.
        """
        delta = positions[..., None, :] - self.centres  # (B, T, C, 2)
        squared = (delta * delta).sum(-1)
        return torch.softmax(-squared / (2.0 * self.sigma ** 2), dim=-1)


class PlaceHead(nn.Module):
    """Linear readout from the position code to the place-cell population.

    Linear on purpose. A deep head could manufacture place-like outputs from
    any code at all, which would remove the pressure on the recurrent state --
    and the pressure on the state is the entire mechanism being tested.
    """

    def __init__(self, position_dim: int, n_cells: int) -> None:
        super().__init__()
        self.out = nn.Linear(position_dim, n_cells)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.out(codes)


def place_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between predicted and target place-cell distributions."""
    return -(targets * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
