"""Look at the image world and at what the model built inside it.

    python scripts/m7_analyse.py --run m7_pilot

Two panels of world figures -- the backdrop with a walk drawn on it, and the
strip of patches that walk actually produced -- plus, if a checkpoint exists,
position-stream rate maps binned in real coordinates.

The rate maps are not decoration. Every "no periodic structure" claim in this
project was an artefact of trusting a scalar that could only ask one question,
and it was found by opening a figure. Continuous position also makes these maps
worth more than they were: binning is no longer stuck at one sample per cell,
so a spatial periodicity measure finally has the resolution it was designed for.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcore.analysis import (  # noqa: E402
    field_score,
    periodicity_score,
    rate_maps_coords,
    spectral_structure,
)
from smallcore.image_training import to_tensors  # noqa: E402
from smallcore.image_world import (  # noqa: E402
    generate_image_walks,
    make_image_environment,
    measure_ambiguity,
)
from smallcore.recurrent import SmallCoreRecurrent  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"


def draw_world(env, walk, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), width_ratios=[1.15, 1])
    axes[0].imshow(
        env.image, cmap="gray", origin="upper",
        extent=(0, env.grid_w, env.grid_h, 0),
    )
    axes[0].plot(walk.positions[:, 0], walk.positions[:, 1], lw=0.8, color="#ff5c5c")
    axes[0].scatter(*walk.positions[0], s=40, color="#4cd964", zorder=3, label="start")
    half = env.patch / (2.0 * env.cell)
    for t in (0, len(walk) // 2, len(walk) - 1):
        x, y = walk.positions[t]
        axes[0].add_patch(
            plt.Rectangle((x - half, y - half), 2 * half, 2 * half,
                          fill=False, color="#4da6ff", lw=1.5)
        )
    axes[0].set_title(
        f"{env.grid_w}x{env.grid_h} cells of {env.cell}px, {env.n_motifs} motifs\n"
        f"{len(walk)}-step walk, {env.patch}px patch"
    )
    axes[0].legend(loc="upper right", fontsize=8)

    grid = 8
    strip = walk.patches[: grid * grid]
    canvas = np.zeros((grid * env.patch, grid * env.patch))
    for i, patch in enumerate(strip):
        r, c = divmod(i, grid)
        canvas[
            r * env.patch : (r + 1) * env.patch, c * env.patch : (c + 1) * env.patch
        ] = patch
    axes[1].imshow(canvas, cmap="gray")
    axes[1].set_title("first 64 observations, in order")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def collect_codes(model, env, config, rng, n_walks, length, window):
    """Gated position codes and the true positions they were built at."""
    walks = generate_image_walks(env, n_walks, length, rng, config.speed)
    velocities, patches = to_tensors(walks, next(model.parameters()).device)
    state = model.initial_state(len(walks), patches.device)
    state = model.observe(state, patches[:, 0])
    codes = []
    for start in range(1, length, window):
        stop = min(start + window, length)
        state = state.detach()
        output, state = model.run_chunk(
            state, velocities[:, start - 1 : stop - 1], patches[:, start:stop]
        )
        codes.append(output.gated.cpu().numpy())
    codes = np.concatenate(codes, axis=1)  # (B, length - 1, D)
    positions = np.stack([w.positions[1:] for w in walks])
    return codes.reshape(-1, codes.shape[-1]), positions.reshape(-1, 2)


def draw_rate_maps(maps, extent, path: Path, title: str) -> None:
    n = min(len(maps), 30)
    cols = 6
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.2 * rows))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        if i >= n:
            ax.axis("off")
            continue
        structure = spectral_structure(maps[i])
        ax.imshow(maps[i], cmap="viridis", extent=extent, origin="lower")
        ax.set_title(
            f"u{i}  {structure['symmetry']}  "
            f"g={periodicity_score(maps[i]):+.2f}",
            fontsize=7,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=str, default=None, help="run directory under runs/")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--grid", type=int, default=10)
    p.add_argument("--cell", type=int, default=16)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--n-motifs", type=int, default=16, dest="n_motifs")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--walks", type=int, default=24)
    p.add_argument("--length", type=int, default=300)
    p.add_argument("--bin-size", type=float, default=0.25, dest="bin_size")
    args = p.parse_args()

    FIGURES.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    checkpoint = None
    if args.run:
        path = ROOT / "runs" / args.run / "best.pt"
        if path.exists():
            checkpoint = torch.load(path, weights_only=False, map_location="cpu")

    if checkpoint is not None:
        config = checkpoint["config"]
        grid, cell = checkpoint["grid"], checkpoint["cell"]
        patch, motifs = config.patch_size, args.n_motifs
    else:
        config = None
        grid, cell, patch, motifs = args.grid, args.cell, args.patch, args.n_motifs

    env = make_image_environment(grid, grid, cell, patch, motifs, rng)
    speed = config.speed if config is not None else args.speed
    walk = generate_image_walks(env, 1, args.length, rng, speed)[0]
    walk.validate(env)
    ambiguity = measure_ambiguity(env, rng)
    print(
        f"world: {grid}x{grid} cells of {cell}px, {motifs} motifs, {patch}px patch\n"
        f"ambiguity: {ambiguity['cells_per_patch']:.2f} cells per patch "
        f"(max {ambiguity['max_cells_per_patch']}, "
        f"{ambiguity['unique_fraction']:.0%} unique)"
    )
    draw_world(env, walk, FIGURES / "m7_world.png")
    print(f"wrote {FIGURES / 'm7_world.png'}")

    if checkpoint is None:
        print("no checkpoint given; skipping rate maps")
        return 0

    model = SmallCoreRecurrent(0, config, seed=0)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    codes, positions = collect_codes(
        model, env, config, rng, args.walks, args.length, config.tbptt_window
    )
    maps, extent = rate_maps_coords(
        codes, np.arange(len(positions)), positions,
        bin_size=args.bin_size, smooth=1.5,
    )
    name = args.run.replace("/", "_")
    draw_rate_maps(
        maps, extent, FIGURES / f"m7_position_{name}.png",
        f"position stream, {args.run} -- {len(positions):,} samples",
    )

    structures = [spectral_structure(m) for m in maps]
    counts: dict[str, int] = {}
    for s in structures:
        counts[s["symmetry"]] = counts.get(s["symmetry"], 0) + 1
    periodic = sum(v for k, v in counts.items() if k != "none")
    print(
        f"\nposition units: {len(maps)}  "
        f"periodic {periodic}/{len(maps)} ({periodic / len(maps):.0%})\n"
        f"  symmetry: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + "\n"
        f"  mean gridness {np.mean([periodicity_score(m) for m in maps]):+.3f}  "
        f"mean field {np.mean([field_score(m) for m in maps]):.3f}"
    )
    print(f"wrote {FIGURES / f'm7_position_{name}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
