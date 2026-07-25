"""The reverse read and the drift gate -- correcting path integration.

Path integration drifts: every step compounds a small error, so a position
estimate built from actions alone slowly loses registration with the world.
M2 shows this directly (88.9% at 100 steps, 81.2% at 300). Re-anchoring it on
what is actually visible needs a second attention running the *other way*:

    q'_t = x_t   . W_x       query  <- OBSERVATION
    K'   = X_<t  . W_x       keys   <- OBSERVATION
    V'   = E_<t              values <- POSITION

"I see this symbol; it was at position p; so I am probably near p."

Note the values are the raw position codes rather than a projection of them.
The retrieved estimate has to be blended with the path-integrated one, so it
must live in the same space; projecting through W_e would land it in the
key space instead.

The retrieved estimate is *ambiguous by construction* -- roughly 2.7 locations
share every symbol, so a symbol query returns a blend of several positions.
That is the point: landmarks are drift-free but ambiguous, dead reckoning is
unambiguous but drifts. Neither is sufficient, so the model learns how much to
trust each:

    e_t = e_t^PI + gate([e_t^ret, e_t^PI]) * (e_t^ret - e_t^PI)

A gate near 0 keeps the integrated estimate; near 1 it jumps to the landmark
fix. It is per-dimension, so the model can trust the retrieved value in some
parts of the code and the integrated one in others -- which is what you want
when the modules run at different spatial scales.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def attend(
    query: torch.Tensor,
    past_keys: torch.Tensor,
    past_values: torch.Tensor,
    recent_keys: torch.Tensor | None,
    recent_values: torch.Tensor | None,
    beta: float,
) -> torch.Tensor:
    """One attention step over a memory split into two pieces.

    The cache is kept as a large **detached** block from earlier truncation
    windows plus a small **grad-attached** block from the current one (see the
    truncated-backprop note in the model). Scoring each separately and
    softmaxing over the concatenation avoids rebuilding the whole cache tensor
    on every step, which would be quadratic in walk length.

    Args:
        query: ``(B, d_k)``.
        past_keys / past_values: ``(B, n_past, d_k)`` / ``(B, n_past, d_v)``.
        recent_keys / recent_values: same, for the current window, or None.
        beta: Softmax temperature scale (``log n_memories``).

    Returns:
        ``(B, d_v)``. All-zero where the memory is empty.
    """
    scale = beta / math.sqrt(query.shape[-1])
    scores, values = [], []
    if past_keys is not None and past_keys.shape[1] > 0:
        scores.append(torch.einsum("bd,bnd->bn", query, past_keys))
        values.append(past_values)
    if recent_keys is not None and recent_keys.shape[1] > 0:
        scores.append(torch.einsum("bd,bnd->bn", query, recent_keys))
        values.append(recent_values)
    if not scores:
        width = past_values.shape[-1] if past_values is not None else 0
        return query.new_zeros(query.shape[0], width)

    attention = torch.cat(scores, dim=-1).mul(scale).softmax(dim=-1)
    return torch.einsum("bn,bnd->bd", attention, torch.cat(values, dim=1))


class DriftGate(nn.Module):
    """Learned per-dimension blend of path-integrated and retrieved position.

    Initialised to open only slightly (bias -2, so roughly 0.12): starting
    near-closed means the model begins by trusting path integration, which is
    the estimate that is actually reliable early in training, and opens the
    gate as the reverse read becomes worth listening to.
    """

    def __init__(self, position_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * position_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, position_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(
        self,
        integrated: torch.Tensor,
        retrieved: torch.Tensor,
        has_memory: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(blended_position, gate)``.

        With an empty memory there is nothing to correct towards, so the
        integrated estimate passes through untouched rather than being pulled
        toward the zero vector the attention returns.
        """
        if not has_memory:
            return integrated, torch.zeros_like(integrated)
        gate = torch.sigmoid(self.net(torch.cat([retrieved, integrated], dim=-1)))
        return integrated + gate * (retrieved - integrated), gate
