"""Analysis harness.

Success cannot be read off a loss curve, so measurement tooling comes first.
This module grows with the milestones:

    M1  linear probes -- can true location be decoded from the position code?
    M5  rate maps, periodicity scores, field scores, zero-shot accuracy  [todo]

Probes use walk ``locations``, which the model itself never sees. That is the
point: locations are ground truth for analysis, not input.
"""

from __future__ import annotations

import torch
from torch import nn


def fit_linear_probe(
    codes: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    iters: int = 400,
    lr: float = 0.05,
    weight_decay: float = 1e-4,
) -> nn.Linear:
    """Fit a multinomial logistic probe from ``codes`` ``(N, D)`` to ``labels``.

    Full-batch Adam; deliberately simple. The probe being *linear* is what
    makes the result meaningful -- it certifies the information is present in
    an easily-readable form, not merely recoverable by another network.
    """
    codes = codes.detach()
    probe = nn.Linear(codes.shape[1], n_classes).to(codes.device)
    optimiser = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(iters):
        optimiser.zero_grad()
        loss = nn.functional.cross_entropy(probe(codes), labels)
        loss.backward()
        optimiser.step()
    return probe


@torch.no_grad()
def probe_predictions(probe: nn.Linear, codes: torch.Tensor) -> torch.Tensor:
    """Predicted class per row of ``codes``."""
    return probe(codes.detach()).argmax(dim=-1)


@torch.no_grad()
def probe_accuracy(
    probe: nn.Linear, codes: torch.Tensor, labels: torch.Tensor
) -> float:
    """Fraction of ``labels`` the probe decodes correctly."""
    return (probe_predictions(probe, codes) == labels).float().mean().item()
