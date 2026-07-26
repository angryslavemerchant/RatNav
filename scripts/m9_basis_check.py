"""Price every fixed feature basis before training on any of them.

    python scripts/m9_basis_check.py

A frozen encoder is only usable if it keeps the two properties the memory
streams actually consume, and both are measurable in seconds against real
patches -- no training required:

* **distance** -- the reverse read asks "have I seen something like this?", and
  the answer is worth having only if embedding distance tracks pixel distance.
  A basis that warps distance breaks the landmark fix.
* **discriminability** -- the forward read has to tell the true next patch from
  a pool of real ones, so the basis has to keep patches from different places
  apart. Reported as top-1 accuracy of nearest-neighbour retrieval in embedding
  space against the pixel-space answer.

Run this before any GPU is rented. If a basis cannot separate patches at all it
cannot be trained into doing so -- that is what frozen means -- and the run
would have been a wasted hour telling us so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcore.image_world import (  # noqa: E402
    TorchPatchSampler,
    generate_trajectories,
    make_image_environment,
)
from smallcore.patches import FixedFeatureEncoder  # noqa: E402

BASES = ("dct", "pca", "random", "gabor")


def neighbour_recall(
    embedded: torch.Tensor, pixels: torch.Tensor, k: int = 10
) -> float:
    """Fraction of each patch's k pixel-space neighbours the basis keeps close.

    Measured against PIXEL neighbours rather than nearby positions, and the
    first version of this script got that wrong. Scoring "does the nearest
    embedding come from the nearest place?" gave 0.033 for raw pixels
    themselves -- the ceiling, since pixels discard nothing -- because the world
    is *built* ambiguous: 16 motifs tile 900 cells, so the patch most similar to
    this one usually is somewhere else entirely. That number measured the
    world's designed ambiguity and said nothing about any basis.

    What a frozen basis can be blamed for is only whether it preserves the
    similarity structure the raw pixels already have, which is what both memory
    streams consume. Hence: same question, asked of the embedding.
    """
    def neighbours(x: torch.Tensor) -> torch.Tensor:
        distance = torch.cdist(x, x)
        distance.fill_diagonal_(float("inf"))
        return distance.topk(k, largest=False).indices

    reference = neighbours(pixels)
    found = neighbours(embedded)
    overlap = (reference.unsqueeze(2) == found.unsqueeze(1)).any(dim=2)
    return float(overlap.float().mean())


def spatial_falloff(
    embedded: torch.Tensor, positions: torch.Tensor, limit: float = 4.0
) -> float:
    """Does embedding similarity report *how far I have moved*?

    The tie-breaker between bases, and the one metric that is about the model
    rather than about faithfulness to pixels. `neighbour_recall` rewards a basis
    for agreeing with pixel space -- but pixel space is dominated by the lowest
    spatial frequencies, so a basis can score well there by keeping only coarse
    brightness, which is smooth everywhere and therefore says almost nothing
    about position. The reverse read needs the opposite: similarity that *falls*
    with separation.

    Scored as the correlation between cosine similarity and negative spatial
    distance, over pairs closer than `limit` cells. Beyond that the world's
    designed ambiguity takes over -- 16 motifs across 900 cells -- and any
    honest embedding stops carrying distance.
    """
    normalised = torch.nn.functional.normalize(embedded, dim=-1)
    similarity = (normalised @ normalised.T).reshape(-1)
    distance = torch.cdist(positions, positions).reshape(-1)
    keep = (distance > 0) & (distance < limit)
    return float(torch.corrcoef(
        torch.stack([similarity[keep], -distance[keep]])
    )[0, 1])


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    patch, obs_dim = 8, 10
    # Both worlds, because the right basis is a property of the world and the
    # answer flips between them: whitening is merely unhelpful on the narrow
    # tiled world and catastrophic on the widened one.
    worlds = (("tiled  ", 1, 4.0), ("widened", 4, 16.0))

    for label, motif_cells, motif_sigma in worlds:
        env = make_image_environment(
            30, 30, 16, patch, 16, np.random.default_rng(0),
            motif_sigma=motif_sigma, motif_cells=motif_cells,
        )
        sampler = TorchPatchSampler(env, device)
        positions, _ = generate_trajectories(env, 64, 32, rng, 0.5)
        pos = torch.as_tensor(positions, dtype=torch.float32, device=device)
        patches = sampler(pos).reshape(-1, patch, patch)
        flat_pos = pos.reshape(-1, 2)
        # Sub-sample for the O(N^2) neighbour comparison; 1024 is plenty stable.
        subset = patches.reshape(len(patches), -1)[:1024]
        subset_pos = flat_pos[:1024]

        print(f"\n=== {label} world  (motif_cells={motif_cells}, "
              f"sigma={motif_sigma:.0f})  {len(patches)} patches, "
              f"{patch}x{patch} -> {obs_dim} dims")
        print(f"{'basis':<8} {'dropDC':>7} {'whiten':>7} {'dist corr':>10} "
              f"{'var kept':>9} {'nbr recall':>11} {'falloff':>8}")
        print(f"{'pixels':<8} {'-':>7} {'-':>7} {1.000:>10.3f} {1.000:>9.3f} "
              f"{1.000:>11.3f} {spatial_falloff(subset, subset_pos):>8.3f}")

        for basis in BASES:
            for drop_dc in (False, True):
                for whiten in (False, True):
                    encoder = FixedFeatureEncoder(
                        patch, obs_dim, basis=basis,
                        drop_dc=drop_dc, seed=0,
                    ).to(device)
                    encoder.fit(patches, whiten=whiten)
                    metrics = encoder.verify_isometry(patches[:512])
                    with torch.no_grad():
                        embedded = encoder(
                            subset.reshape(-1, patch, patch)
                        )
                    print(f"{basis:<8} {str(drop_dc):>7} {str(whiten):>7} "
                          f"{metrics['distance_corr']:>10.3f} "
                          f"{metrics['variance_retained']:>9.3f} "
                          f"{neighbour_recall(embedded, subset):>11.3f} "
                          f"{spatial_falloff(embedded, subset_pos):>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
