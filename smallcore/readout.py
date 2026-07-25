"""Observation prediction heads.

Two readouts, deliberately separate:

* the **full** head sees what memory returned together with where the model
  thinks it is, and produces the main prediction;
* the **position-only** head sees the position code alone.

The second exists to put pressure on the position code rather than to be
accurate. Its loss term is what stops the position stream from degenerating
into whatever arbitrary state happens to make the memory read convenient.

A small post-attention residual block sits in front of the full head, per the
brief's one-layer / one-head starting point.
"""

from __future__ import annotations

import torch
from torch import nn


class Readout(nn.Module):
    """Map (retrieved memory, position code) -> logits over observations."""

    def __init__(
        self,
        position_dim: int,
        obs_dim: int,
        n_observations: int,
        model_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.project = nn.Linear(obs_dim + position_dim, model_dim)
        self.block = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_dim),
        )
        self.norm = nn.LayerNorm(model_dim)
        self.out = nn.Linear(model_dim, n_observations)

        self.position_only = nn.Sequential(
            nn.Linear(position_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_observations),
        )

    def forward(
        self, retrieved: torch.Tensor, codes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(full_logits, position_only_logits)``, both ``(B, T, V)``."""
        hidden = self.project(torch.cat([retrieved, codes], dim=-1))
        hidden = self.norm(hidden + self.block(hidden))
        return self.out(hidden), self.position_only(codes)
