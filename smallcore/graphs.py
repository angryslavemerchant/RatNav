"""Graph environments and walk generation.

The model learns by walking around a graph and predicting what it will observe
next. This module produces those graphs and those walks.

Three ideas drive the design:

1.  **Structure and appearance are separate things.** A `Topology` is a graph:
    locations and the actions that connect them. An `Environment` is a topology
    plus an assignment of observations to locations. The same topology can be
    dressed in fresh observations indefinitely. This split exists because the
    thing we want the model to learn -- how movement composes -- is a property
    of the topology alone, and is worth learning once and reusing. A model that
    cannot separate the two has to relearn space in every new room.

2.  **An action must mean the same thing everywhere.** `transitions[a, i]` is
    deterministic, and an action unavailable at a location is simply not
    sampled. We never silently convert a blocked move into a no-op, because
    that would make "north" mean *move north* in the middle of the graph and
    *stay put* at its edge. The model learns one transition operator per
    action; that operator can only be coherent if the action's effect is
    consistent.

3.  **Walks are correlated, not random.** Real trajectories go somewhere.
    `straight_bias` upweights repeating the previous action, producing runs
    rather than jitter. Uncorrelated walks make position tracking dramatically
    harder to learn, and they are not what the model will face in use.

Observations are indices into a small vocabulary and are deliberately
**ambiguous**: there are far more locations than symbols, so the same symbol
appears in many places. This is the point. If every location looked unique,
predicting the next observation would be pure lookup and the model would never
need to represent position at all. Ambiguity is what forces it to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Marks the final step of a walk, where no action is taken.
NO_ACTION = -1

# Marks an action that is not available at a location.
NO_EDGE = -1


@dataclass(frozen=True)
class Topology:
    """A graph of locations connected by actions.

    Attributes:
        name: Identifier, used for filenames and plot titles.
        n_locations: Number of nodes.
        n_actions: Number of distinct movements.
        transitions: ``(n_actions, n_locations)`` int array. ``transitions[a, i]``
            is the location reached by taking action ``a`` at location ``i``, or
            ``NO_EDGE`` if that action is unavailable there.
        coords: ``(n_locations, 2)`` float array of xy positions. Used only for
            plotting -- the model never sees these.
        action_names: Human-readable label per action, for plots and debugging.
    """

    name: str
    n_locations: int
    n_actions: int
    transitions: np.ndarray
    coords: np.ndarray
    action_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.transitions.shape != (self.n_actions, self.n_locations):
            raise ValueError(
                f"transitions must be {(self.n_actions, self.n_locations)}, "
                f"got {self.transitions.shape}"
            )
        if self.coords.shape != (self.n_locations, 2):
            raise ValueError(
                f"coords must be {(self.n_locations, 2)}, got {self.coords.shape}"
            )
        if len(self.action_names) != self.n_actions:
            raise ValueError("action_names must have one entry per action")

    def valid_actions(self, location: int) -> np.ndarray:
        """Return the actions that lead somewhere from ``location``."""
        return np.flatnonzero(self.transitions[:, location] != NO_EDGE)

    def edges(self) -> list[tuple[int, int]]:
        """Return unique undirected ``(i, j)`` pairs, for plotting."""
        seen = set()
        for a in range(self.n_actions):
            for i, j in enumerate(self.transitions[a]):
                if j != NO_EDGE:
                    seen.add((min(i, int(j)), max(i, int(j))))
        return sorted(seen)


@dataclass(frozen=True)
class Environment:
    """A topology with observations attached to its locations."""

    topology: Topology
    observations: np.ndarray  # (n_locations,) int in [0, n_observations)
    n_observations: int

    def __post_init__(self) -> None:
        if self.observations.shape != (self.topology.n_locations,):
            raise ValueError("observations must have one entry per location")
        if self.observations.min() < 0 or self.observations.max() >= self.n_observations:
            raise ValueError("observation indices out of range")

    @property
    def name(self) -> str:
        return self.topology.name

    def to_json(self, path: str | Path) -> None:
        """Write the environment to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.topology.name,
            "n_locations": self.topology.n_locations,
            "n_actions": self.topology.n_actions,
            "n_observations": self.n_observations,
            "action_names": list(self.topology.action_names),
            "transitions": self.topology.transitions.tolist(),
            "coords": self.topology.coords.tolist(),
            "observations": self.observations.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "Environment":
        """Read an environment written by :meth:`to_json`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        topology = Topology(
            name=payload["name"],
            n_locations=payload["n_locations"],
            n_actions=payload["n_actions"],
            transitions=np.asarray(payload["transitions"], dtype=int),
            coords=np.asarray(payload["coords"], dtype=float),
            action_names=tuple(payload["action_names"]),
        )
        return cls(
            topology=topology,
            observations=np.asarray(payload["observations"], dtype=int),
            n_observations=payload["n_observations"],
        )


@dataclass(frozen=True)
class Walk:
    """A trajectory through an environment.

    At step ``t`` the agent is at ``locations[t]``, observes ``observations[t]``,
    and takes ``actions[t]``, which moves it to ``locations[t + 1]``. The final
    entry of ``actions`` is :data:`NO_ACTION`, since nothing follows it.

    ``locations`` is retained for analysis and debugging only. The model is given
    observations and actions; recovering position is its job, not its input.
    """

    locations: np.ndarray
    observations: np.ndarray
    actions: np.ndarray

    def __len__(self) -> int:
        return len(self.locations)

    def one_hot(self, n_observations: int) -> np.ndarray:
        """Observations as a ``(T, n_observations)`` float array."""
        return np.eye(n_observations, dtype=np.float32)[self.observations]

    def validate(self, env: Environment) -> None:
        """Raise if this walk is not a legal path through ``env``.

        Cheap enough to call in tests and after any change to walk generation.
        """
        topology = env.topology
        if not (len(self.locations) == len(self.observations) == len(self.actions)):
            raise ValueError("walk arrays have inconsistent lengths")
        if self.actions[-1] != NO_ACTION:
            raise ValueError("final action must be NO_ACTION")
        expected = env.observations[self.locations]
        if not np.array_equal(expected, self.observations):
            raise ValueError("observations do not match the environment")
        for t, action in enumerate(self.actions[:-1]):
            if action == NO_ACTION:
                raise ValueError(f"unexpected NO_ACTION at step {t}")
            destination = topology.transitions[action, self.locations[t]]
            if destination == NO_EDGE:
                raise ValueError(f"step {t} takes an unavailable action")
            if destination != self.locations[t + 1]:
                raise ValueError(f"step {t} does not lead to the next location")


# --------------------------------------------------------------------------- #
# Topologies
# --------------------------------------------------------------------------- #


def square_grid(width: int, height: int, name: str | None = None) -> Topology:
    """A 4-connected rectangular grid.

    Actions are north, east, south, west. Edges have no wraparound, so boundary
    locations genuinely have fewer available actions -- the model has to cope
    with a bounded world rather than a torus.
    """
    n_locations = width * height
    action_names = ("north", "east", "south", "west")
    deltas = ((0, 1), (1, 0), (0, -1), (-1, 0))

    transitions = np.full((len(deltas), n_locations), NO_EDGE, dtype=int)
    coords = np.zeros((n_locations, 2), dtype=float)

    for index in range(n_locations):
        col, row = index % width, index // width
        coords[index] = (col, row)
        for action, (dcol, drow) in enumerate(deltas):
            ncol, nrow = col + dcol, row + drow
            if 0 <= ncol < width and 0 <= nrow < height:
                transitions[action, index] = nrow * width + ncol

    return Topology(
        name=name or f"square_{width}x{height}",
        n_locations=n_locations,
        n_actions=len(deltas),
        transitions=transitions,
        coords=coords,
        action_names=action_names,
    )


def hex_grid(width: int, height: int, name: str | None = None) -> Topology:
    """A 6-connected hexagonal grid, in odd-row-offset layout.

    Included as a control. Nothing in the model is specific to four actions or
    to square geometry, and a topology with different connectivity is the
    cleanest way to demonstrate that.
    """
    n_locations = width * height
    action_names = ("east", "northeast", "northwest", "west", "southwest", "southeast")

    # Neighbour offsets differ by row parity in an offset layout.
    even_row = ((1, 0), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1))
    odd_row = ((1, 0), (1, -1), (0, -1), (-1, 0), (0, 1), (1, 1))

    transitions = np.full((len(action_names), n_locations), NO_EDGE, dtype=int)
    coords = np.zeros((n_locations, 2), dtype=float)

    for index in range(n_locations):
        col, row = index % width, index // width
        coords[index] = (col + 0.5 * (row % 2), row * np.sqrt(3) / 2)
        offsets = odd_row if row % 2 else even_row
        for action, (dcol, drow) in enumerate(offsets):
            ncol, nrow = col + dcol, row + drow
            if 0 <= ncol < width and 0 <= nrow < height:
                transitions[action, index] = nrow * width + ncol

    return Topology(
        name=name or f"hex_{width}x{height}",
        n_locations=n_locations,
        n_actions=len(action_names),
        transitions=transitions,
        coords=coords,
        action_names=action_names,
    )


# --------------------------------------------------------------------------- #
# Environments and walks
# --------------------------------------------------------------------------- #


def assign_observations(
    topology: Topology,
    n_observations: int,
    rng: np.random.Generator,
) -> Environment:
    """Dress ``topology`` in a fresh random assignment of observations.

    Call this repeatedly on one topology to produce many environments that share
    a structure but look nothing alike. That is the setting the model is meant
    to exploit: structure transfers, appearances do not.
    """
    observations = rng.integers(0, n_observations, size=topology.n_locations)
    return Environment(
        topology=topology,
        observations=observations.astype(int),
        n_observations=n_observations,
    )


def generate_walk(
    env: Environment,
    length: int,
    rng: np.random.Generator,
    straight_bias: float = 2.0,
    start: int | None = None,
) -> Walk:
    """Sample a walk of ``length`` steps through ``env``.

    Args:
        env: The environment to walk in.
        length: Number of steps, including the starting location.
        rng: Source of randomness.
        straight_bias: Multiplier on the probability of repeating the previous
            action. ``1.0`` gives an unbiased walk; the default of ``2.0``
            produces the roughly-straight runs characteristic of real
            trajectories. Values below 1 make the walk actively jittery.
        start: Optional starting location. Random if omitted.

    Returns:
        A :class:`Walk` of exactly ``length`` steps.
    """
    if length < 1:
        raise ValueError("length must be at least 1")
    if straight_bias <= 0:
        raise ValueError("straight_bias must be positive")

    topology = env.topology
    locations = np.empty(length, dtype=int)
    actions = np.full(length, NO_ACTION, dtype=int)

    current = int(rng.integers(topology.n_locations)) if start is None else int(start)
    previous_action = NO_ACTION

    for step in range(length):
        locations[step] = current
        if step == length - 1:
            break

        available = topology.valid_actions(current)
        if available.size == 0:
            raise ValueError(f"location {current} is isolated")

        weights = np.ones(available.size, dtype=float)
        if previous_action != NO_ACTION:
            weights[available == previous_action] *= straight_bias
        weights /= weights.sum()

        action = int(rng.choice(available, p=weights))
        actions[step] = action
        previous_action = action
        current = int(topology.transitions[action, current])

    return Walk(
        locations=locations,
        observations=env.observations[locations],
        actions=actions,
    )


def generate_batch(
    env: Environment,
    batch_size: int,
    length: int,
    rng: np.random.Generator,
    straight_bias: float = 2.0,
) -> list[Walk]:
    """Sample ``batch_size`` independent walks through one environment."""
    return [generate_walk(env, length, rng, straight_bias) for _ in range(batch_size)]


def generate_batch_fast(
    env: Environment,
    batch_size: int,
    length: int,
    rng: np.random.Generator,
    straight_bias: float = 2.0,
    start: int | None = None,
) -> list[Walk]:
    """Vectorised :func:`generate_batch`: step every walk in the batch at once.

    Identical in distribution to calling :func:`generate_walk` repeatedly, but
    it advances the whole batch per Python iteration instead of one walk, which
    matters once walk generation shares a training loop with the model. The
    per-walk version stays as the readable reference; this is what training
    calls.

    ``start`` is shared by every walk when given, and drawn independently per
    walk when omitted.
    """
    if length < 1:
        raise ValueError("length must be at least 1")
    if straight_bias <= 0:
        raise ValueError("straight_bias must be positive")

    topology = env.topology
    n_actions = topology.n_actions
    transitions = topology.transitions

    locations = np.empty((batch_size, length), dtype=int)
    actions = np.full((batch_size, length), NO_ACTION, dtype=int)
    rows = np.arange(batch_size)

    if start is None:
        current = rng.integers(0, topology.n_locations, size=batch_size)
    else:
        current = np.full(batch_size, int(start), dtype=int)
    previous = np.full(batch_size, NO_ACTION, dtype=int)

    for step in range(length):
        locations[:, step] = current
        if step == length - 1:
            break

        destinations = transitions[:, current]  # (n_actions, batch)
        weights = (destinations != NO_EDGE).astype(float)

        has_previous = previous != NO_ACTION
        if has_previous.any():
            columns = np.flatnonzero(has_previous)
            weights[previous[columns], columns] *= straight_bias

        totals = weights.sum(axis=0)
        if not np.all(totals > 0):
            raise ValueError(
                f"location {int(current[np.argmin(totals)])} is isolated"
            )

        # Inverse-CDF sampling down the action axis. Actions with zero weight
        # leave the cumulative sum flat and so can never be selected.
        cumulative = np.cumsum(weights, axis=0)
        draw = rng.random(batch_size) * totals
        chosen = (cumulative < draw).sum(axis=0)
        np.minimum(chosen, n_actions - 1, out=chosen)

        actions[:, step] = chosen
        current = destinations[chosen, rows]
        previous = chosen

    return [
        Walk(
            locations=locations[i],
            observations=env.observations[locations[i]],
            actions=actions[i],
        )
        for i in range(batch_size)
    ]
