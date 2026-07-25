"""Visualisation for environments and walks.

These plots exist to answer one question at this stage: are the walks legal,
and do they look like trajectories rather than noise? Everything downstream is
easier to debug against an environment you have already seen and trust.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from .graphs import NO_ACTION, Environment, Walk

# Observations are categorical, so the colour map only needs to make neighbours
# distinguishable -- it carries no ordering.
OBSERVATION_CMAP = "hsv"

# The trajectory is coloured by time, which does have an order.
TIME_CMAP = "viridis"


def plot_environment(
    env: Environment,
    ax: Axes | None = None,
    node_size: float = 60.0,
    show_edges: bool = True,
) -> Axes:
    """Draw the graph, colouring each location by its observation."""
    ax = ax or plt.gca()
    coords = env.topology.coords

    if show_edges:
        segments = [(coords[i], coords[j]) for i, j in env.topology.edges()]
        ax.add_collection(
            LineCollection(segments, colors="0.85", linewidths=0.8, zorder=1)
        )

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=env.observations,
        cmap=OBSERVATION_CMAP,
        vmin=0,
        vmax=env.n_observations - 1,
        s=node_size,
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
    )

    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def plot_walk(
    env: Environment,
    walk: Walk,
    ax: Axes | None = None,
    node_size: float = 40.0,
    linewidth: float = 1.8,
) -> Axes:
    """Draw a walk over its environment, coloured from start to finish.

    A small perpendicular offset is applied to each step so that repeated
    traversals of the same edge remain visible instead of overdrawing.
    """
    ax = ax or plt.gca()
    plot_environment(env, ax=ax, node_size=node_size, show_edges=True)

    path = env.topology.coords[walk.locations]
    steps = np.stack([path[:-1], path[1:]], axis=1)

    # Offset each segment sideways by a jitter that is stable per step, so
    # overlapping traversals fan out rather than stack.
    deltas = steps[:, 1] - steps[:, 0]
    lengths = np.linalg.norm(deltas, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    normals = np.stack([-deltas[:, 1], deltas[:, 0]], axis=1) / lengths
    jitter = 0.06 * np.linspace(-1.0, 1.0, len(steps))[:, None]
    steps = steps + (normals * jitter)[:, None, :]

    ax.add_collection(
        LineCollection(
            steps,
            array=np.arange(len(steps)),
            cmap=TIME_CMAP,
            linewidths=linewidth,
            alpha=0.9,
            zorder=3,
        )
    )

    ax.scatter(*path[0], marker="o", s=110, facecolors="none",
               edgecolors="black", linewidths=1.6, zorder=4, label="start")
    ax.scatter(*path[-1], marker="X", s=110, c="black", zorder=4, label="end")
    return ax


def run_lengths(walk: Walk) -> np.ndarray:
    """Lengths of consecutive runs of the same action.

    The direct quantitative read on ``straight_bias``: an unbiased walk gives
    short runs, a biased one gives long ones.
    """
    actions = walk.actions[walk.actions != NO_ACTION]
    if actions.size == 0:
        return np.zeros(0, dtype=int)
    boundaries = np.flatnonzero(np.diff(actions)) + 1
    return np.diff([0, *boundaries, actions.size])


def plot_run_lengths(
    walks_by_bias: dict[float, list[Walk]],
    ax: Axes | None = None,
) -> Axes:
    """Compare action run-length distributions across ``straight_bias`` values."""
    ax = ax or plt.gca()
    max_run = 1

    for bias, walks in sorted(walks_by_bias.items()):
        runs = np.concatenate([run_lengths(w) for w in walks])
        max_run = max(max_run, int(runs.max()))
        counts = np.bincount(runs)[1:]
        ax.plot(
            np.arange(1, len(counts) + 1),
            counts / counts.sum(),
            marker="o",
            markersize=4,
            label=f"bias = {bias:g}  (mean {runs.mean():.2f})",
        )

    ax.set_xlim(0.5, min(max_run, 12) + 0.5)
    ax.set_xlabel("consecutive steps in the same direction")
    ax.set_ylabel("fraction of runs")
    ax.set_title("Straight-line bias")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_observation_ambiguity(env: Environment, ax: Axes | None = None) -> Axes:
    """Show how many locations share each observation.

    Ambiguity is a design requirement, not a defect: if observations were
    unique, next-observation prediction would be lookup and position would
    never need to be represented. This plot confirms the ambiguity is present.
    """
    ax = ax or plt.gca()
    counts = np.bincount(env.observations, minlength=env.n_observations)
    shared = np.bincount(counts)

    ax.bar(np.arange(len(shared)), shared, color="0.4")
    ax.set_xlabel("locations sharing an observation")
    ax.set_ylabel("number of observations")
    ax.set_title(
        f"{env.topology.n_locations} locations, {env.n_observations} symbols"
    )
    ax.spines[["top", "right"]].set_visible(False)
    return ax
