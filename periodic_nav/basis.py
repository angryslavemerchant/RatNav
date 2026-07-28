"""Periodic position basis with frozen structure and learned usage.

The position representation is a set of periodic basis functions at
multiple spatial frequencies.  Each function's phase advances by a
fixed amount per action — the periodic structure is given, not learned.
The model learns how to *use* these codes (via attention and readout),
not how to produce them.

The guarantee: sin/cos output is periodic by construction, and phase
advancement is exact (no drift).  Going north then south returns to
exactly the same code.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PeriodicBasis(nn.Module):
    """Fixed periodic basis with action-conditioned phase increments.

    Each module has a spatial frequency and ``orient_per_module`` basis
    functions (default 2: x- and y-direction for a square grid).  The
    output code is ``[sin(φ_1), …, sin(φ_N), cos(φ_1), …, cos(φ_N)]``,
    with ``N = n_modules * orient_per_module``.

    Phase increments are frozen by default (``learn_increments=False``).
    """

    def __init__(
        self,
        n_modules: int,
        n_actions: int,
        module_cycles: list[float] | None = None,
        orient_per_module: int = 2,
        learn_increments: bool = False,
    ):
        super().__init__()
        self.n_modules = n_modules
        self.n_actions = n_actions
        self.orient_per_module = orient_per_module
        self.n_basis = n_modules * orient_per_module
        self.code_dim = 2 * self.n_basis

        if module_cycles is None:
            module_cycles = [3.0 * 1.4 ** i for i in range(n_modules)]

        increments = torch.zeros(n_actions, self.n_basis)
        for m in range(n_modules):
            freq = 2 * math.pi / module_cycles[m]
            base = m * orient_per_module
            if orient_per_module >= 2 and n_actions >= 4:
                # Square grid: N=0, E=1, S=2, W=3
                increments[1, base + 0] = freq    # east → +x
                increments[3, base + 0] = -freq   # west → −x
                increments[0, base + 1] = freq    # north → +y
                increments[2, base + 1] = -freq   # south → −y
            else:
                for o in range(orient_per_module):
                    angle = math.pi * o / orient_per_module
                    for a in range(n_actions):
                        action_angle = 2 * math.pi * a / n_actions
                        increments[a, base + o] = freq * math.cos(
                            action_angle - angle
                        )

        if learn_increments:
            self.phase_increments = nn.Parameter(increments)
        else:
            self.register_buffer("phase_increments", increments)

        # Continuous: velocity (v_x, v_y) → phase increments via freq_matrix
        freq_matrix = torch.zeros(2, self.n_basis)
        for m in range(n_modules):
            freq = 2 * math.pi / (module_cycles[m] if isinstance(module_cycles, list)
                                   else module_cycles[m])
            base = m * orient_per_module
            if orient_per_module >= 2:
                freq_matrix[0, base + 0] = freq  # v_x → x-basis
                freq_matrix[1, base + 1] = freq  # v_y → y-basis
        self.register_buffer("freq_matrix", freq_matrix)

        self.register_buffer("_zero_phases", torch.zeros(self.n_basis))

    def initial_state(self, batch_size: int) -> torch.Tensor:
        """(B, n_basis) initial phases — all zeros."""
        return self._zero_phases.unsqueeze(0).expand(batch_size, -1)

    def step(self, phases: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Advance phases by the increment for each action.

        Args:
            phases: (B, n_basis)
            actions: (B,) int action indices

        Returns:
            (B, n_basis) new phases
        """
        return phases + self.phase_increments[actions]

    def step_continuous(
        self, phases: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Advance phases by velocity-conditioned increments.

        Args:
            phases: (B, n_basis)
            velocity: (B, 2) — (v_x, v_y) in cells
        """
        return phases + velocity @ self.freq_matrix

    def encode(self, phases: torch.Tensor) -> torch.Tensor:
        """Phases → position code: [sin(φ_1)…sin(φ_N), cos(φ_1)…cos(φ_N)].

        Works on any leading shape: (B, n_basis) or (B, T, n_basis).
        """
        return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)

    def phases_from_code(self, code: torch.Tensor) -> torch.Tensor:
        """Recover phases from a position code via atan2."""
        sin_part = code[..., : self.n_basis]
        cos_part = code[..., self.n_basis :]
        return torch.atan2(sin_part, cos_part)
