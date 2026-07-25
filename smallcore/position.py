"""The position stream: a recurrent, action-conditioned encoder.

Position is not a function of sequence index; it is a state evolved by the
action taken:

    e_{t+1} = LayerNorm( tanh( e_t @ W_{a_t} ) )

with one learnable transition matrix per action, block-diagonal over modules.
Each module is initialised as a random rotation whose angle is set by that
module's frequency, so the modules start out cycling at geometrically-spaced
spatial scales. That multi-scale structure is a prior we supply; whether it
organises into clean periodic codes is what training has to show.

The initialisation is deliberately *not* given any group structure beyond the
per-module scale: opposite actions are initialised independently rather than as
inverses. Learning that north followed by south returns you where you started
is the task, not a gift.

Rotation angles are ``pi * frequency`` rather than ``2*pi * frequency``: a full
``2*pi`` would alias the fastest module (freq 0.99) back down to a ~100-step
period, collapsing the intended separation of scales.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _rotation(dim: int, angle: float, generator: torch.Generator) -> torch.Tensor:
    """A random ``dim x dim`` rotation with largest rotation angle ``angle``.

    Built as ``expm(angle * A)`` for a random antisymmetric ``A`` scaled to unit
    spectral norm, which is orthogonal with rotation planes chosen at random.
    """
    raw = torch.randn(dim, dim, generator=generator)
    antisym = raw - raw.T
    antisym = antisym / torch.linalg.matrix_norm(antisym, ord=2)
    return torch.matrix_exp(angle * antisym)


class PositionEncoder(nn.Module):
    """Integrates a sequence of actions into a position code.

    The encoder sees actions only. Given a known starting state, the code it
    produces is a position estimate; given an unknown one, it is a displacement
    estimate. Anchoring it to what is actually visible is the reverse read's
    job (M3), not this module's.
    """

    def __init__(
        self,
        n_actions: int,
        module_dims: tuple[int, ...],
        module_freqs: tuple[float, ...],
        seed: int = 0,
    ) -> None:
        super().__init__()
        if len(module_dims) != len(module_freqs):
            raise ValueError("module_dims and module_freqs must align")

        self.n_actions = n_actions
        self.module_dims = tuple(module_dims)
        self.dim = sum(module_dims)

        generator = torch.Generator().manual_seed(seed)
        blocks = [
            torch.stack(
                [_rotation(dim, math.pi * freq, generator) for _ in range(n_actions)]
            )
            for dim, freq in zip(module_dims, module_freqs)
        ]  # one (n_actions, dim, dim) block per module

        # Equal-width modules stack into a single tensor, which turns the whole
        # step into one einsum instead of a Python loop over modules -- roughly
        # 3x faster, and the step is run once per timestep of every walk. The
        # loop remains for unequal widths, which cannot be stacked.
        self.uniform = len(set(module_dims)) == 1
        if self.uniform:
            self.module_dim = module_dims[0]
            self.transitions = nn.Parameter(torch.stack(blocks))
        else:
            self.transitions = nn.ParameterList(nn.Parameter(b) for b in blocks)

        self.initial = nn.Parameter(torch.randn(self.dim, generator=generator))
        self.norm = nn.LayerNorm(self.dim)

    def initial_state(self, batch_size: int) -> torch.Tensor:
        """The learned starting code, ``(batch_size, dim)``."""
        return self.norm(self.initial).expand(batch_size, -1)

    def step(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Advance ``state`` ``(B, dim)`` by one action per batch element."""
        if self.uniform:
            chunks = state.view(state.shape[0], -1, self.module_dim)
            moved = torch.einsum(
                "bnd,nbde->bne", chunks, self.transitions[:, actions]
            )
            return self.norm(torch.tanh(moved.reshape(state.shape[0], self.dim)))

        parts = []
        offset = 0
        for blocks in self.transitions:
            dim = blocks.shape[-1]
            chunk = state[:, offset : offset + dim]
            parts.append(
                torch.bmm(chunk.unsqueeze(1), blocks[actions]).squeeze(1)
            )
            offset += dim
        return self.norm(torch.tanh(torch.cat(parts, dim=-1)))

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """Integrate ``actions`` ``(B, S)`` into codes ``(B, S + 1, dim)``.

        ``output[:, t]`` is the code after the first ``t`` actions, so it pairs
        with ``locations[t]`` under the walk indexing convention. ``actions``
        must not contain ``NO_ACTION``; pass ``walk.actions[:-1]``.
        """
        state = self.initial_state(actions.shape[0])
        codes = [state]
        for t in range(actions.shape[1]):
            state = self.step(state, actions[:, t])
            codes.append(state)
        return torch.stack(codes, dim=1)
