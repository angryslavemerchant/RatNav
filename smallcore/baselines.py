"""Baselines the model has to beat, plus the ceiling it is chasing.

Both agents predict the next observation by remembering what followed what.
Neither has any notion of position -- that is the point. They are
**content-addressed**: they retrieve on what things look like.

* **node agent** -- remembers, for each observation, what usually comes next.
* **edge agent** -- remembers, for each (observation, action) pair, what
  usually comes next. Strictly stronger than the node agent.

Beating them is the whole of M2. It should be possible precisely because
observations are ambiguous: roughly 2.7 locations share every symbol, so
"symbol 12 then north" leads to different places on different occasions and no
appearance-keyed table can resolve it. A model that tracks *where it is* can.
If the model cannot beat these, its memory is doing nothing a lookup table
does not already do.

:func:`oracle_memory_accuracy` is the other end of the scale: what a model
could achieve with a *perfect* position code and this exact memory design.
Since the forward read can only return observations from places already
visited, its ceiling is the revisit rate of the walk. Reporting it stops us
mistaking "memory is working" for "memory has run out of things to know".
"""

from __future__ import annotations

import numpy as np

from smallcore.graphs import Walk


class NodeAgent:
    """Predicts the next observation from the current observation alone."""

    def __init__(self, n_observations: int) -> None:
        self.n_observations = n_observations
        self.counts = np.zeros((n_observations, n_observations), dtype=np.int64)

    def fit(self, walks: list[Walk]) -> "NodeAgent":
        for walk in walks:
            for t in range(len(walk) - 1):
                self.counts[walk.observations[t], walk.observations[t + 1]] += 1
        return self

    def _table(self) -> np.ndarray:
        fallback = self.counts.sum(axis=0).argmax() if self.counts.any() else 0
        table = np.full(self.n_observations, fallback, dtype=np.int64)
        seen = self.counts.sum(axis=1) > 0
        table[seen] = self.counts[seen].argmax(axis=1)
        return table

    def accuracy(self, walks: list[Walk]) -> float:
        table = self._table()
        correct = total = 0
        for walk in walks:
            predicted = table[walk.observations[:-1]]
            correct += int((predicted == walk.observations[1:]).sum())
            total += len(walk) - 1
        return correct / max(total, 1)


class EdgeAgent:
    """Predicts the next observation from (current observation, action)."""

    def __init__(self, n_observations: int, n_actions: int) -> None:
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.counts = np.zeros(
            (n_observations, n_actions, n_observations), dtype=np.int64
        )

    def fit(self, walks: list[Walk]) -> "EdgeAgent":
        for walk in walks:
            for t in range(len(walk) - 1):
                self.counts[
                    walk.observations[t], walk.actions[t], walk.observations[t + 1]
                ] += 1
        return self

    def _table(self) -> np.ndarray:
        flat = self.counts.reshape(-1, self.n_observations)
        fallback = flat.sum(axis=0).argmax() if flat.any() else 0
        table = np.full(flat.shape[0], fallback, dtype=np.int64)
        seen = flat.sum(axis=1) > 0
        table[seen] = flat[seen].argmax(axis=1)
        return table.reshape(self.n_observations, self.n_actions)

    def accuracy(self, walks: list[Walk]) -> float:
        table = self._table()
        correct = total = 0
        for walk in walks:
            predicted = table[walk.observations[:-1], walk.actions[:-1]]
            correct += int((predicted == walk.observations[1:]).sum())
            total += len(walk) - 1
        return correct / max(total, 1)


def oracle_memory_accuracy(walks: list[Walk]) -> float:
    """Accuracy of a perfect position code driving this exact memory design.

    At step ``t`` the memory holds observations from steps before ``t``. With a
    flawless position code the model retrieves the right one exactly when the
    current location has been visited earlier in the walk, and cannot possibly
    know otherwise. So this is the revisit rate -- the ceiling for any
    within-episode positional memory, and the number the model should be
    compared against rather than 100%.
    """
    correct = total = 0
    for walk in walks:
        seen: set[int] = {int(walk.locations[0])}
        for t in range(1, len(walk)):
            location = int(walk.locations[t])
            correct += location in seen
            seen.add(location)
            total += 1
    return correct / max(total, 1)


def chance_accuracy(walks: list[Walk], n_observations: int) -> float:
    """Accuracy of always guessing the most common observation."""
    counts = np.zeros(n_observations, dtype=np.int64)
    for walk in walks:
        counts += np.bincount(walk.observations, minlength=n_observations)
    return counts.max() / max(counts.sum(), 1)
