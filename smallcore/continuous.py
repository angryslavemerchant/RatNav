"""Continuous space: movement by velocity vector rather than by action index.

The discrete model stores one transition matrix per action and selects by
lookup. That cannot extend to continuous movement -- there are infinitely many
velocities -- so the lookup becomes a *function* of the velocity:

    W(v) = expm( v_x * G_x + v_y * G_y )                    exact
    e'   = LayerNorm( tanh( e + (v_x G_x + v_y G_y) e ) )   first order

Two learned generators replace four learned matrices. `G_x` is "how phase turns
per unit of eastward motion", `G_y` the same for north. Any velocity mixes them,
so infinitely many movements come out of two objects -- and the parameter count
*halves*, since 2 generators beat 4 action matrices.

The first-order form is Euler integration: it steps along the tangent rather
than the arc, so the state's magnitude drifts. The per-step LayerNorm cancels
most of that, which is why published continuous grid-cell models use it. Exact
`expm` is available for large steps.

Why this matters beyond generality (M5, 2026-07-25): on a discrete grid a
"periodic" code barely differs from a lookup table. A module with a 3-cell
cycle takes only 3 distinct values along that axis, and since the model is only
ever scored at integer positions it can treat that as an arbitrary 3-state
categorical variable and be exactly as correct. Discreteness lets the model
avoid the property being tested. Continuity closes that escape: the code must
interpolate between sampled positions, and smooth-plus-repeating is genuinely
periodic. It also removes the quantisation floor that made rate maps tiny.

Deliberately kept from changing here: observations stay discrete, assigned per
cell of a background lattice that the continuous position is binned into. That
isolates *continuous movement* as the single variable. Continuous observations
(image patches) are a separate step, and stacking both at once would repeat the
confound that has already cost this project two wrong conclusions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ContinuousEnvironment:
    """A rectangle of continuous positions with symbols painted on a lattice.

    Attributes:
        width / height: Extent in cells. Position is real-valued within.
        observations: ``(height * width,)`` symbol per lattice cell.
        n_observations: Vocabulary size.
    """

    width: int
    height: int
    observations: np.ndarray
    n_observations: int

    @property
    def n_cells(self) -> int:
        return self.width * self.height

    def observe(self, positions: np.ndarray) -> np.ndarray:
        """Symbol at each ``(N, 2)`` continuous position, by binning."""
        col = np.clip(np.floor(positions[:, 0]), 0, self.width - 1).astype(int)
        row = np.clip(np.floor(positions[:, 1]), 0, self.height - 1).astype(int)
        return self.observations[row * self.width + col]


@dataclass(frozen=True)
class ContinuousWalk:
    """A trajectory through continuous space.

    At step ``t`` the agent is at ``positions[t]``, observes ``observations[t]``
    and moves by ``velocities[t]``, arriving at ``positions[t + 1]``. The final
    velocity is zero, mirroring ``NO_ACTION`` in the discrete walk.

    ``positions`` is for analysis only, exactly as ``locations`` is in the
    discrete world. The model never sees it.
    """

    positions: np.ndarray  # (T, 2) float
    observations: np.ndarray  # (T,) int
    velocities: np.ndarray  # (T, 2) float

    def __len__(self) -> int:
        return len(self.positions)

    def validate(self, env: ContinuousEnvironment) -> None:
        """Raise if this walk is not a legal path through ``env``."""
        if not (len(self.positions) == len(self.observations) == len(self.velocities)):
            raise ValueError("walk arrays have inconsistent lengths")
        if np.any(self.velocities[-1] != 0):
            raise ValueError("final velocity must be zero")
        expected = env.observe(self.positions)
        if not np.array_equal(expected, self.observations):
            raise ValueError("observations do not match the environment")
        moved = self.positions[:-1] + self.velocities[:-1]
        if not np.allclose(moved, self.positions[1:], atol=1e-6):
            raise ValueError("velocities do not lead to the next position")
        inside = (
            (self.positions[:, 0] >= 0) & (self.positions[:, 0] <= env.width)
            & (self.positions[:, 1] >= 0) & (self.positions[:, 1] <= env.height)
        )
        if not inside.all():
            raise ValueError("walk leaves the arena")


def generate_continuous_walks(
    env: ContinuousEnvironment,
    batch_size: int,
    length: int,
    rng: np.random.Generator,
    speed: float = 1.0,
    turn_sigma: float = 0.6,
) -> list[ContinuousWalk]:
    """Sample smooth walks that stay inside the arena.

    Heading follows a random walk (``turn_sigma`` radians per step), which is
    the continuous analogue of ``straight_bias``: small values give long
    straight runs, large values give jitter. A step that would leave the arena
    is not taken -- the heading is reflected and resampled instead, so a
    velocity always means "move by exactly this much", never "move unless
    blocked". Silently truncating a step would break the one-action-one-meaning
    rule the operator algebra depends on, exactly as converting a blocked grid
    move into a no-op would.
    """
    positions = np.empty((batch_size, length, 2))
    velocities = np.zeros((batch_size, length, 2))

    current = np.stack(
        [rng.uniform(0, env.width, batch_size), rng.uniform(0, env.height, batch_size)],
        axis=1,
    )
    heading = rng.uniform(0, 2 * np.pi, batch_size)

    for step in range(length):
        positions[:, step] = current
        if step == length - 1:
            break

        for _ in range(8):  # reflect and retry until the step lands inside
            proposal = current + speed * np.stack(
                [np.cos(heading), np.sin(heading)], axis=1
            )
            outside = (
                (proposal[:, 0] < 0) | (proposal[:, 0] > env.width)
                | (proposal[:, 1] < 0) | (proposal[:, 1] > env.height)
            )
            if not outside.any():
                break
            heading[outside] = rng.uniform(0, 2 * np.pi, int(outside.sum()))
        proposal = np.clip(
            proposal, [1e-6, 1e-6], [env.width - 1e-6, env.height - 1e-6]
        )

        velocities[:, step] = proposal - current
        current = proposal
        heading = heading + rng.normal(0, turn_sigma, batch_size)

    return [
        ContinuousWalk(
            positions=positions[i],
            observations=env.observe(positions[i]),
            velocities=velocities[i],
        )
        for i in range(batch_size)
    ]


def assign_continuous_observations(
    width: int, height: int, n_observations: int, rng: np.random.Generator
) -> ContinuousEnvironment:
    """Paint a fresh random symbol assignment onto the background lattice."""
    return ContinuousEnvironment(
        width=width,
        height=height,
        observations=rng.integers(0, n_observations, size=width * height).astype(int),
        n_observations=n_observations,
    )


class ContinuousPositionEncoder(nn.Module):
    """Velocity-conditioned path integration.

    Holds two generators per module instead of one matrix per action, so the
    parameter count halves while the set of expressible movements becomes
    infinite. Generators are initialised antisymmetric, so at small velocity the
    update is a rotation -- the same starting geometry as the discrete encoder's
    rotation initialisation, expressed as a rate rather than a fixed step.
    """

    def __init__(
        self,
        module_dims: tuple[int, ...],
        module_freqs: tuple[float, ...],
        seed: int = 0,
        exact: bool = False,
    ) -> None:
        super().__init__()
        if len(module_dims) != len(module_freqs):
            raise ValueError("module_dims and module_freqs must align")
        if len(set(module_dims)) != 1:
            raise ValueError("continuous encoder requires equal module widths")

        self.module_dims = tuple(module_dims)
        self.module_dim = module_dims[0]
        self.n_modules = len(module_dims)
        self.dim = sum(module_dims)
        self.exact = exact

        gen = torch.Generator().manual_seed(seed)
        blocks = []
        for freq in module_freqs:
            pair = []
            for _ in range(2):  # G_x and G_y
                raw = torch.randn(self.module_dim, self.module_dim, generator=gen)
                skew = raw - raw.T
                skew = skew / torch.linalg.matrix_norm(skew, ord=2)
                # pi * freq radians per cell, matching the discrete encoder's
                # per-step rotation angle so the two are directly comparable.
                pair.append(math.pi * freq * skew)
            blocks.append(torch.stack(pair))
        self.generators = nn.Parameter(torch.stack(blocks))  # (M, 2, d, d)

        self.initial = nn.Parameter(torch.randn(self.dim, generator=gen))
        self.norm = nn.LayerNorm(self.dim)

    def initial_state(self, batch_size: int) -> torch.Tensor:
        return self.norm(self.initial).expand(batch_size, -1)

    def step(self, state: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """Advance ``state`` ``(B, dim)`` by ``velocity`` ``(B, 2)``."""
        batch = state.shape[0]
        # (B, M, d, d): the per-module operator this velocity calls for.
        operator = torch.einsum("bc,mcde->bmde", velocity, self.generators)
        chunks = state.view(batch, self.n_modules, self.module_dim)
        if self.exact:
            moved = torch.einsum(
                "bmd,bmde->bme", chunks, torch.matrix_exp(operator)
            )
        else:
            moved = chunks + torch.einsum("bmd,bmde->bme", chunks, operator)
        return self.norm(torch.tanh(moved.reshape(batch, self.dim)))

    def forward(self, velocities: torch.Tensor) -> torch.Tensor:
        """Integrate ``velocities`` ``(B, S, 2)`` into codes ``(B, S + 1, dim)``."""
        state = self.initial_state(velocities.shape[0])
        codes = [state]
        for t in range(velocities.shape[1]):
            state = self.step(state, velocities[:, t])
            codes.append(state)
        return torch.stack(codes, dim=1)
