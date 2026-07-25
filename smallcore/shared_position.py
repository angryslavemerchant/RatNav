"""One shared navigator core, instantiated N times, differentiated only by knobs.

The block-diagonal encoder in :mod:`position` is already N independent
navigators running side by side -- the modules never exchange information. This
module asks the question that observation invites: do they need to be
*independent*, or can they be N copies of a single shared core, told apart by a
handful of cheap per-copy parameters?

The modules differ by design only in their spatial scale, and scale is an
*angle*: a module is a rotation, and a coarser module is the same rotation
through a smaller angle. So the natural knob is one scalar per copy, and the
whole multi-scale stack collapses to

    W_{copy i, action a} = expm( s_i * G_a )

with ``G_a`` a single generator per action shared by every copy, and ``s_i``
that copy's scale knob. Five copies of one core, five numbers to tell them
apart.

Initialisation is deliberately identical to :class:`~smallcore.position.PositionEncoder`:
``G_a`` is ``pi`` times a random unit-norm antisymmetric matrix, so ``expm(s_i *
G_a)`` at ``s_i = freq_i`` is a rotation through ``pi * freq_i`` -- exactly the
baseline's starting operators. The only thing that changes is that the copies
now share ``G_a`` instead of each owning a private matrix. That makes the
comparison a controlled one: sharing is the single variable.

Knobs are stored in log space. They span two orders of magnitude (0.99 down to
0.01), and a learning rate that moves the fastest knob sensibly would be a
enormous relative step for the slowest one.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _generator(dim: int, generator: torch.Generator) -> torch.Tensor:
    """``pi`` x a random antisymmetric matrix of unit spectral norm.

    ``expm(s * G)`` is then a rotation whose largest rotation angle is
    ``pi * s``, so the scale knob reads directly as a frequency.
    """
    raw = torch.randn(dim, dim, generator=generator)
    antisym = raw - raw.T
    return math.pi * antisym / torch.linalg.matrix_norm(antisym, ord=2)


class SharedPositionEncoder(nn.Module):
    """N copies of one navigator core, differentiated by a scale knob each.

    Interface-compatible with :class:`~smallcore.position.PositionEncoder`, so
    the M1 training and probing code can drive either one.

    Args:
        n_actions: Number of distinct movements.
        n_copies: How many instances of the shared core to run side by side.
        copy_dim: Dimensions per copy. Total code width is ``n_copies * copy_dim``.
        knob_init: Starting scale per copy -- the frequencies the baseline's
            modules are initialised at.
        knob_per_action: If ``True``, each copy gets one knob per action rather
            than a single knob shared across actions. Loosens the prior that a
            copy has *one* spatial scale regardless of which way you move,
            which is true on a uniform grid and may not be elsewhere.
        seed: Seed for generator initialisation.
    """

    def __init__(
        self,
        n_actions: int,
        n_copies: int,
        copy_dim: int,
        knob_init: tuple[float, ...],
        knob_per_action: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if len(knob_init) != n_copies:
            raise ValueError("knob_init must have one entry per copy")
        if min(knob_init) <= 0:
            raise ValueError("knob_init must be positive (scales live in log space)")

        self.n_actions = n_actions
        self.n_copies = n_copies
        self.copy_dim = copy_dim
        self.dim = n_copies * copy_dim

        gen = torch.Generator().manual_seed(seed)

        # The shared core: one generator per action, common to every copy.
        self.generators = nn.Parameter(
            torch.stack([_generator(copy_dim, gen) for _ in range(n_actions)])
        )  # (n_actions, copy_dim, copy_dim)

        # The knobs: what makes copy i different from copy j.
        log_knobs = torch.log(torch.tensor(knob_init, dtype=torch.float32))
        if knob_per_action:
            log_knobs = log_knobs.unsqueeze(1).repeat(1, n_actions)
        self.log_knobs = nn.Parameter(log_knobs)

        self.initial = nn.Parameter(torch.randn(self.dim, generator=gen))
        self.norm = nn.LayerNorm(self.dim)

    @property
    def knobs(self) -> torch.Tensor:
        """Scale per copy, ``(n_copies,)`` or ``(n_copies, n_actions)``."""
        return self.log_knobs.exp()

    def operators(self) -> torch.Tensor:
        """Materialise the transition operators, ``(n_copies, n_actions, d, d)``.

        Cheap enough to recompute once per forward pass -- the operators do not
        depend on the step, only on the shared core and the knobs.
        """
        scales = self.knobs
        if scales.dim() == 1:
            scales = scales.unsqueeze(1).expand(-1, self.n_actions)
        scaled = scales[..., None, None] * self.generators  # broadcast over copies
        flat = torch.matrix_exp(scaled.reshape(-1, self.copy_dim, self.copy_dim))
        return flat.reshape(self.n_copies, self.n_actions, self.copy_dim, self.copy_dim)

    def initial_state(self, batch_size: int) -> torch.Tensor:
        """The learned starting code, ``(batch_size, dim)``."""
        return self.norm(self.initial).expand(batch_size, -1)

    def step(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        operators: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance ``state`` ``(B, dim)`` by one action per batch element."""
        if operators is None:
            operators = self.operators()
        chunks = state.view(state.shape[0], self.n_copies, self.copy_dim)
        selected = operators[:, actions]  # (n_copies, B, d, d)
        moved = torch.einsum("bnd,nbde->bne", chunks, selected)
        return self.norm(torch.tanh(moved.reshape(state.shape[0], self.dim)))

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """Integrate ``actions`` ``(B, S)`` into codes ``(B, S + 1, dim)``."""
        operators = self.operators()  # once per pass, not once per step
        state = self.initial_state(actions.shape[0])
        codes = [state]
        for t in range(actions.shape[1]):
            state = self.step(state, actions[:, t], operators)
            codes.append(state)
        return torch.stack(codes, dim=1)
