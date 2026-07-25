"""The loss terms.

1. ``L_pred``     -- cross-entropy on the observation from the full model (main)
2. ``L_pred_pos`` -- cross-entropy on the observation from the position code alone
3. ``L_drift``    -- squared error between the gated and path-integrated position
4. ``L_reg_w``    -- L2 on weights (applied as Adam weight decay, not here)
5. ``L_reg_e``    -- L2 on the position code

Term 5 matters far more than its magnitude suggests: pressure toward an
efficient position code is a large part of what drives clean periodic
structure. Without it, expect unstructured mush.

Term 3 needs the drift gate and so is inert until M3; it is defined here so the
loss signature does not change when the gate arrives.

Step 0 is excluded from every prediction term. Its position code is the learned
initial state and its memory is empty, so it is not a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from smallcore.model import ModelOutput


@dataclass
class LossTerms:
    """The individual terms, for logging, plus the scalar actually optimised."""

    total: torch.Tensor
    pred: torch.Tensor
    pred_position: torch.Tensor
    drift: torch.Tensor
    reg_code: torch.Tensor
    accuracy: float
    accuracy_position: float


def compute_losses(
    output: ModelOutput,
    targets: torch.Tensor,
    w_pred_pos: float,
    w_reg_code: float,
    w_drift: float = 0.0,
    path_integrated: torch.Tensor | None = None,
) -> LossTerms:
    """Evaluate every term.

    Args:
        output: What the model returned.
        targets: ``(B, T)`` integer observation indices.
        w_pred_pos: Weight on the position-only prediction term.
        w_reg_code: Weight on the L2 penalty on the position code.
        w_drift: Weight on the drift term (M3; zero until the gate exists).
        path_integrated: ``(B, T, D)`` pre-gate codes, required if
            ``w_drift`` is non-zero.
    """
    logits = output.logits[:, 1:]
    logits_position = output.logits_position[:, 1:]
    target = targets[:, 1:]
    flat_target = target.reshape(-1)

    pred = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), flat_target
    )
    pred_position = nn.functional.cross_entropy(
        logits_position.reshape(-1, logits_position.shape[-1]), flat_target
    )
    reg_code = output.codes.pow(2).mean()

    if w_drift and path_integrated is not None:
        drift = (output.codes - path_integrated).pow(2).mean()
    else:
        drift = torch.zeros((), device=logits.device)

    total = pred + w_pred_pos * pred_position + w_reg_code * reg_code
    if w_drift:
        total = total + w_drift * drift

    with torch.no_grad():
        accuracy = (logits.argmax(-1) == target).float().mean().item()
        accuracy_position = (
            (logits_position.argmax(-1) == target).float().mean().item()
        )

    return LossTerms(
        total=total,
        pred=pred,
        pred_position=pred_position,
        drift=drift,
        reg_code=reg_code,
        accuracy=accuracy,
        accuracy_position=accuracy_position,
    )
