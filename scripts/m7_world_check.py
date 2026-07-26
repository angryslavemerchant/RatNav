"""Measure the image world before training anything on it.

The discrete world could be specified by fiat -- revisiting a location returned
exactly the remembered symbol, so memory was obviously useful and the only open
question was whether the model could localise. Continuous space removes that
guarantee: a walk returns *near* a place, never to it, and a patch remembered
even a fraction of a cell away may already be a useless prediction. Whether
memory can help at all is now a property of the numbers -- arena size, walk
length, patch width, step length -- and it has to be measured.

So this sweeps candidate worlds and reports, for each, what a **perfectly
localised** agent would score by retrieval. That is the number that decides
whether the world is worth training on. If it sits at chance, no position
stream however good could do better, and a low training score would be
unattributable -- exactly the confound that made M6's 53% hard to read.

    python scripts/m7_world_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcore.image_training import patch_baselines, retrieval_oracle, to_tensors
from smallcore.image_world import (
    generate_image_walks,
    make_image_environment,
    measure_ambiguity,
)

CONFIGS = [
    # grid, cell, patch, speed, motifs, steps
    (15, 16, 8, 0.5, 45, 300),   # the first guess: patch = step = half a cell
    (10, 16, 16, 1.0, 16, 300),  # smaller arena, patch = cell, step = patch
    (10, 16, 16, 1.0, 45, 300),
    (10, 16, 24, 1.0, 16, 300),  # patch wider than the step: views overlap
    (10, 16, 24, 1.5, 16, 300),
    (8, 16, 24, 1.5, 16, 300),
    (8, 16, 16, 1.0, 16, 300),
    (10, 16, 16, 1.0, 16, 600),  # denser coverage by walking longer
]


def main() -> int:
    device = torch.device("cpu")
    print(
        f"{'grid':>5} {'patch':>6} {'step':>5} {'motifs':>7} {'steps':>6} | "
        f"{'amb':>5} {'return':>7} | {'retr-oracle':>11} {'nn':>7} "
        f"{'persist':>8} {'ceiling':>8} {'chance':>7}"
    )
    for grid, cell, patch, speed, motifs, steps in CONFIGS:
        rng = np.random.default_rng(0)
        env = make_image_environment(grid, grid, cell, patch, motifs, rng)
        walks = generate_image_walks(env, 8, steps, rng, speed)
        velocities, patches = to_tensors(walks, device)
        mask = max(1, int(round((patch / cell) / (2.0 * speed))))

        base = patch_baselines(patches, velocities, mask)
        retr = retrieval_oracle(walks, patches, mask)
        amb = measure_ambiguity(env, rng, n_queries=128)

        positions = np.stack([w.positions for w in walks])
        distance = np.linalg.norm(
            positions[:, :, None, :] - positions[:, None, :, :], axis=-1
        )
        old = np.abs(np.arange(steps)[:, None] - np.arange(steps)[None, :]) > mask
        nearest = np.where(old[None], distance, 1e9).min(axis=2)

        print(
            f"{grid:>5} {patch:>6} {speed:>5.1f} {motifs:>7} {steps:>6} | "
            f"{amb['cells_per_patch']:>5.1f} {np.median(nearest):>7.3f} | "
            f"{retr:>10.1%} {base['nearest_neighbour']:>7.1%} "
            f"{base['persistence']:>8.1%} {base['oracle']:>8.1%} "
            f"{base['chance']:>7.2%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
