"""M0: build the environments, sample walks, and look at them.

Run from the project root:

    python scripts/m0_demo.py

Writes environment files to ``smallcore/envs/`` and figures to ``figures/``.
Every generated walk is checked against its environment, so a clean run is
evidence that walk generation is correct, not just that it produced output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcore.graphs import (
    Environment,
    assign_observations,
    generate_batch,
    generate_walk,
    hex_grid,
    square_grid,
)
from smallcore.plots import (
    plot_observation_ambiguity,
    plot_run_lengths,
    plot_walk,
    run_lengths,
)

ROOT = Path(__file__).resolve().parent.parent
ENV_DIR = ROOT / "smallcore" / "envs"
FIGURE_DIR = ROOT / "figures"

N_OBSERVATIONS = 45
SEED = 0


def main() -> int:
    rng = np.random.default_rng(SEED)
    FIGURE_DIR.mkdir(exist_ok=True)

    # Two topologies with different connectivity. Nothing in the model is
    # specific to four actions, and the hex grid is how we keep ourselves
    # honest about that.
    topologies = [square_grid(11, 11), hex_grid(9, 9)]

    environments: dict[str, Environment] = {}
    for topology in topologies:
        env = assign_observations(topology, N_OBSERVATIONS, rng)
        env.to_json(ENV_DIR / f"{topology.name}.json")
        environments[topology.name] = env
        print(
            f"{topology.name:>14}: {topology.n_locations:>4} locations, "
            f"{topology.n_actions} actions, "
            f"{N_OBSERVATIONS} symbols "
            f"({topology.n_locations / N_OBSERVATIONS:.1f} locations per symbol)"
        )

    # Round-trip through disk, so the saved files are known to be usable.
    for name in environments:
        reloaded = Environment.from_json(ENV_DIR / f"{name}.json")
        assert np.array_equal(reloaded.observations, environments[name].observations)
        assert np.array_equal(
            reloaded.topology.transitions, environments[name].topology.transitions
        )
    print(f"\nwrote and reloaded {len(environments)} environments -> {ENV_DIR}")

    square = environments["square_11x11"]
    hexagonal = environments["hex_9x9"]

    # --- validate a decent sample of walks ------------------------------- #
    checked = 0
    for env in (square, hexagonal):
        for length in (25, 80, 300):
            for walk in generate_batch(env, 16, length, rng):
                walk.validate(env)
                checked += 1
    print(f"validated {checked} walks (lengths 25-300, both topologies)")

    # --- figure 1: what walks look like ---------------------------------- #
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    demos = [
        (square, 40, "square grid, 40 steps"),
        (square, 300, "square grid, 300 steps"),
        (hexagonal, 120, "hex grid, 120 steps"),
    ]
    for ax, (env, length, title) in zip(axes, demos):
        walk = generate_walk(env, length, rng)
        walk.validate(env)
        plot_walk(env, walk, ax=ax)
        ax.set_title(title, fontsize=11)

    fig.suptitle(
        "Walks are legal paths; colour runs dark to light with time",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURE_DIR / "m0_walks.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- figure 2: the two properties the walks must have ---------------- #
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    walks_by_bias = {
        bias: generate_batch(square, 40, 200, rng, straight_bias=bias)
        for bias in (1.0, 2.0, 5.0)
    }
    plot_run_lengths(walks_by_bias, ax=axes[0])
    plot_observation_ambiguity(square, ax=axes[1])

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "m0_properties.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote figures -> {FIGURE_DIR}")

    # --- the numbers behind figure 2 ------------------------------------- #
    print("\nmean run length by straight_bias:")
    for bias, walks in sorted(walks_by_bias.items()):
        runs = np.concatenate([run_lengths(w) for w in walks])
        print(f"  bias {bias:>4g}: {runs.mean():.2f} steps")

    coverage = [
        len(np.unique(w.locations)) / square.topology.n_locations
        for w in generate_batch(square, 32, 300, rng)
    ]
    print(
        f"\n300-step walks cover {np.mean(coverage):.0%} of the grid on average "
        f"(min {np.min(coverage):.0%}, max {np.max(coverage):.0%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
