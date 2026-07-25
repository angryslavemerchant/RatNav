"""Hyperparameters.

Values are the starting points from the project brief (CLAUDE.md §8). They are
a single flat dataclass so a run can be described by one object and overridden
with ``dataclasses.replace``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- position stream --------------------------------------------------- #
    # Block-diagonal modules of the transition operators. Each module is
    # initialised as a rotation at its own frequency, so the modules settle at
    # geometrically-spaced spatial scales.
    # Equal widths, measured 2026-07-25 to match the brief's (30,30,24,18,18)
    # exactly on M1 (99.0% observation accuracy, 100% short-walk decoding,
    # 87.9% vs 81.3% on 200-step walks) while letting the position step run as
    # a single einsum instead of a loop over modules.
    module_dims: tuple[int, ...] = (24, 24, 24, 24, 24)
    module_freqs: tuple[float, ...] = (0.99, 0.3, 0.09, 0.03, 0.01)

    # --- observations ------------------------------------------------------ #
    n_observations: int = 45
    # Observations are compressed before entering memory as values: the memory
    # returns *contents*, and 45 one-hot dims is a wasteful way to carry them.
    obs_dim: int = 10

    # --- memory stream ----------------------------------------------------- #
    # Queries and keys are both position, through one tied projection: they
    # have to live in the same space for a dot product between them to mean
    # "am I near where I was?".
    key_dim: int = 64
    # Post-attention residual block, per the brief's one-layer/one-head start.
    model_dim: int = 64
    hidden_dim: int = 64

    # --- loss weights ------------------------------------------------------ #
    # L2 on the position code. Small number, large effect: this is the pressure
    # that pushes the code toward clean periodic structure.
    l2_position_code: float = 0.01
    # L2 on the weights (applied as Adam weight decay).
    weight_decay: float = 1e-5
    # Weight on predicting the observation from the position code alone. With
    # random walk starts the position code carries displacement rather than
    # absolute location, so this term is genuinely unsatisfiable and is kept
    # small -- it is a pressure on the code, not an objective to reach.
    w_pred_pos: float = 0.1

    # --- training ---------------------------------------------------------- #
    batch_size: int = 16
    lr: float = 9.4e-4
    lr_min: float = 8e-5
    lr_decay: float = 0.5
    lr_decay_every: int = 4000
    straight_bias: float = 2.0

    @property
    def position_dim(self) -> int:
        return sum(self.module_dims)

    def __post_init__(self) -> None:
        if len(self.module_dims) != len(self.module_freqs):
            raise ValueError("module_dims and module_freqs must align")
