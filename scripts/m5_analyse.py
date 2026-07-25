"""M5: look inside a trained model. This is the actual result.

    python scripts/m5_analyse.py --run runs/m4_pool8
    python scripts/m5_analyse.py --run runs/m3_gate --tag m3

Accuracy says the model works. M5 asks whether it works *the way the theory
says*: do the position units organise into periodic spatial codes, and do the
memory units develop localised fields?

    periodicity score = min(corr at 60, 120) - max(corr at 30, 90, 150)

on the autocorrelogram of each unit's rate map. **0.3-0.5 is the conventional
threshold** for calling a unit periodic, and that is the headline acceptance
test for the whole project.

Walks here start at a FIXED location, unlike training. With random starts the
position code is displacement from an unknown origin, so the same cell carries a
different code in every walk and a rate map averaged across walks would be
meaningless. A fixed start makes location and displacement one-to-one. The
transition operators do not depend on where a walk began, so this measures the
code the model genuinely learned.

Caveat worth keeping in view: 11x11 is a *small* arena for this measure, which
was designed for spaces holding several grid periods. Scores are indicative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcore.analysis import field_score, periodicity_score, rate_maps
from smallcore.config import Config
from smallcore.graphs import assign_observations, generate_batch_fast, square_grid
from smallcore.recurrent import SmallCoreRecurrent
from smallcore.training import to_tensors

ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = ROOT / "figures"


@torch.no_grad()
def collect(model, env, config, device, n_walks, length, window, start):
    """Position codes, memory activations and true locations, pooled over steps."""
    rng = np.random.default_rng(11)
    walks = generate_batch_fast(
        env, n_walks, length, rng, config.straight_bias, start=start
    )
    for walk in walks[:4]:
        walk.validate(env)
    actions, one_hot, indices = to_tensors(walks, env.n_observations, device)
    locations = np.stack([w.locations for w in walks])

    state = model.initial_state(n_walks, device)
    state = model.observe(state, one_hot[:, 0])
    codes, memories = [], []
    for begin in range(1, length, window):
        stop = min(begin + window, length)
        output, state = model.run_chunk(
            state, actions[:, begin - 1 : stop - 1], one_hot[:, begin:stop]
        )
        codes.append(output.gated.cpu().numpy())
        memories.append(output.retrieved.cpu().numpy())

    codes = np.concatenate(codes, axis=1)  # (B, T-1, D)
    memories = np.concatenate(memories, axis=1)
    flat_locations = locations[:, 1:].reshape(-1)
    return (
        codes.reshape(-1, codes.shape[-1]),
        memories.reshape(-1, memories.shape[-1]),
        flat_locations,
    )


def panel(maps, scores, title, path, label, n_show=12):
    """Render the best-scoring units as rate maps."""
    order = np.argsort(scores)[::-1][:n_show]
    cols = 6
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.1 * rows))
    for ax, unit in zip(np.ravel(axes), order):
        ax.imshow(maps[unit], origin="lower", cmap="viridis", interpolation="nearest")
        ax.set_title(f"u{unit}  {label}={scores[unit]:+.2f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in np.ravel(axes)[len(order):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=str, default="runs/m4_pool8")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--walks", type=int, default=64)
    parser.add_argument("--length", type=int, default=300)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--smooth", type=float, default=1.0)
    parser.add_argument("--grid", type=int, default=11)
    args = parser.parse_args()

    checkpoint_path = ROOT / args.run / "best.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"{checkpoint_path} missing -- train first")
    tag = args.tag or Path(args.run).name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", Config())

    topology = square_grid(args.grid, args.grid)
    env = assign_observations(
        topology, config.n_observations, np.random.default_rng(4242)
    )
    model = SmallCoreRecurrent(topology.n_actions, config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    start = topology.n_locations // 2
    codes, memories, locations = collect(
        model, env, config, device, args.walks, args.length, args.window, start
    )
    coverage = len(np.unique(locations)) / topology.n_locations
    print(
        f"M5 on {args.run} (iteration {checkpoint.get('iteration', '?')})\n"
        f"{args.walks} fixed-start walks of {args.length} steps, "
        f"covering {coverage:.0%} of the grid\n"
    )

    position_maps = rate_maps(
        codes, locations, args.grid, args.grid, smooth=args.smooth
    )
    memory_maps = rate_maps(
        memories, locations, args.grid, args.grid, smooth=args.smooth
    )

    periodicity = np.array([periodicity_score(m) for m in position_maps])
    fields = np.array([field_score(m) for m in memory_maps])
    position_fields = np.array([field_score(m) for m in position_maps])

    valid = periodicity[~np.isnan(periodicity)]
    passing = (valid >= 0.3).sum()

    # Per-module breakdown: the modules are initialised at different spatial
    # scales, so if the prior is doing anything they should not score alike.
    # A module cannot show spatial periodicity the arena is too small to hold.
    # The rotation advances pi*freq per step, so a full cycle takes 2/freq
    # cells; below ~3 cells it is at the Nyquist limit of a discrete grid, and
    # above the arena width it can only look like a gradient.
    print("periodicity by module (position stream):")
    offset = 0
    module_rows = []
    for index, (dim, freq) in enumerate(
        zip(config.module_dims, config.module_freqs)
    ):
        block = periodicity[offset : offset + dim]
        block = block[~np.isnan(block)]
        module_rows.append(
            {"module": index, "freq": freq, "mean": float(block.mean()),
             "max": float(block.max()), "n_passing": int((block >= 0.3).sum()),
             "n_units": int(block.size)}
        )
        cycle = 2.0 / freq
        if cycle < 3:
            note = f"cycle {cycle:.1f} cells - at Nyquist, unresolvable"
        elif cycle > args.grid:
            note = f"cycle {cycle:.0f} cells - EXCEEDS the {args.grid}-cell arena"
        else:
            note = f"cycle {cycle:.1f} cells - {args.grid / cycle:.1f} repeats fit"
        print(
            f"  module {index} (freq {freq:>5}): mean {block.mean():+.3f}  "
            f"best {block.max():+.3f}  "
            f"{(block >= 0.3).sum():>2}/{block.size} >= 0.30   {note}"
        )
        offset += dim

    print(
        f"\nposition stream, all {valid.size} units:\n"
        f"  mean periodicity   {valid.mean():+.3f}\n"
        f"  best periodicity   {valid.max():+.3f}\n"
        f"  units >= 0.30      {passing} ({passing / valid.size:.0%})\n"
        f"  mean field score   {np.nanmean(position_fields):.3f}  "
        f"(low = spatially spread, as a periodic code should be)\n"
        f"\nmemory stream, {fields.size} units:\n"
        f"  mean field score   {np.nanmean(fields):.3f}\n"
        f"  best field score   {np.nanmax(fields):.3f}  "
        f"(high = fires in one place)"
    )

    FIGURE_DIR.mkdir(exist_ok=True)
    panel(
        position_maps, np.nan_to_num(periodicity, nan=-9),
        f"Position stream rate maps, best periodicity ({tag})",
        FIGURE_DIR / f"m5_position_{tag}.png", "per",
    )
    panel(
        memory_maps, np.nan_to_num(fields, nan=-9),
        f"Memory stream rate maps, most localised ({tag})",
        FIGURE_DIR / f"m5_memory_{tag}.png", "field", n_show=min(12, fields.size),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(valid, bins=24, color="tab:blue", alpha=0.85)
    axes[0].axvline(0.3, color="crimson", linestyle="--", label="0.30 threshold")
    axes[0].axvline(0.5, color="crimson", linestyle=":", label="0.50")
    axes[0].set_xlabel("periodicity score")
    axes[0].set_ylabel("position units")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Periodicity across the position stream")

    axes[1].hist(fields[~np.isnan(fields)], bins=16, color="tab:green", alpha=0.85)
    axes[1].set_xlabel("field score (mass in largest component)")
    axes[1].set_ylabel("memory units")
    axes[1].set_title("Locality of memory-stream units")
    fig.suptitle(f"M5 acceptance measures ({tag})")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURE_DIR / f"m5_scores_{tag}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    (ROOT / args.run / "m5.json").write_text(
        json.dumps(
            {"modules": module_rows,
             "mean_periodicity": float(valid.mean()),
             "max_periodicity": float(valid.max()),
             "units_passing": int(passing),
             "n_units": int(valid.size),
             "memory_field_mean": float(np.nanmean(fields)),
             "memory_field_max": float(np.nanmax(fields))},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote figures -> {FIGURE_DIR}  and {args.run}/m5.json")

    # A handful of units clearing the bar is not the milestone. The claim is
    # that the position stream ORGANISES into periodic codes, so require a
    # meaningful fraction of it; and the memory stream must show localised
    # fields, which is the other half of the acceptance test.
    fraction = passing / max(valid.size, 1)
    periodic_ok = fraction >= 0.10
    fields_ok = float(np.nanmean(fields)) >= 0.5
    reasons = []
    if not periodic_ok:
        reasons.append(
            f"only {passing}/{valid.size} ({fraction:.0%}) position units reach "
            f"0.30 periodicity, mean {valid.mean():+.3f}"
        )
    if not fields_ok:
        reasons.append(
            f"memory fields not localised (mean {np.nanmean(fields):.2f})"
        )
    verdict = "PASSED" if (periodic_ok and fields_ok) else "NOT PASSED"
    print(
        f"\nM5 {verdict}" + ("" if not reasons else ": " + "; ".join(reasons))
    )
    print(
        f"  periodic position codes: {'yes' if periodic_ok else 'NO'}\n"
        f"  localised memory fields: {'yes' if fields_ok else 'NO'}"
    )
    return 0 if (periodic_ok and fields_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
