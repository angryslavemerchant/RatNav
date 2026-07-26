"""Score saved checkpoints for grid structure, on FIXED-START walks.

    python scripts/m8_grid_report.py runs/m8_A runs/m8_B ...

Separate from the in-run ``--analyse`` because that one got it wrong, and the
mistake is worth naming: it built rate maps from **random-start** walks. The
position code carries displacement from wherever a walk began, so the same
location receives a different code in every walk, and averaging activation per
location averages unrelated things. The result is a blob no matter what the
model learned -- which is exactly what every map looked like. The project's own
analysis.py docstring warns about this; the warning was not heeded.

Everything else is the same two-gate test: a peak ratio above what matched
structureless noise reaches, AND a field score low enough that the unit fires
in more than one place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcore.analysis import (  # noqa: E402
    field_score,
    noise_peak_ratio,
    periodicity_score,
    rate_maps_coords,
    spectral_structure,
)
from smallcore.image_world import (  # noqa: E402
    TorchPatchSampler,
    generate_trajectories,
    make_image_environment,
)
from smallcore.recurrent import SmallCoreRecurrent  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NAMES = {2: "band", 4: "square", 6: "hex"}


@torch.no_grad()
def codes_for(model, env, config, device, walks, length, start, seed):
    sampler = TorchPatchSampler(env, device)
    rng = np.random.default_rng(seed)
    positions, velocities = generate_trajectories(
        env, walks, length, rng, config.speed, start=start
    )
    pos = torch.as_tensor(positions, dtype=torch.float32, device=device)
    patches = sampler(pos)
    vel = torch.as_tensor(velocities[:, :-1], dtype=torch.float32, device=device)
    state = model.initial_state(walks, device)
    state = model.observe(state, patches[:, 0])
    collected = []
    for begin in range(1, length, config.tbptt_window):
        stop = min(begin + config.tbptt_window, length)
        state = state.detach()
        out, state = model.run_chunk(
            state, vel[:, begin - 1 : stop - 1], patches[:, begin:stop]
        )
        collected.append(out.gated.cpu().numpy())
    codes = np.concatenate(collected, axis=1)
    return codes.reshape(-1, codes.shape[-1]), positions[:, 1:].reshape(-1, 2)


def score(run_dir: Path, args, device) -> dict | None:
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    config = checkpoint["config"]
    grid, cell = checkpoint["grid"], checkpoint["cell"]

    model = SmallCoreRecurrent(0, config, seed=0).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rng = np.random.default_rng(11)
    env = make_image_environment(
        grid, grid, cell, config.patch_size, 16, rng
    )
    # Centre start: maximises how much of the arena the walks can reach.
    start = (grid / 2.0, grid / 2.0)

    all_codes, all_positions = [], []
    for seed in range(args.repeats):
        codes, positions = codes_for(
            model, env, config, device, args.walks, args.length, start, 500 + seed
        )
        all_codes.append(codes)
        all_positions.append(positions)
    codes = np.concatenate(all_codes)
    positions = np.concatenate(all_positions)

    maps, _ = rate_maps_coords(
        codes, np.arange(len(positions)), positions,
        bin_size=args.bin_size, smooth=1.5,
    )
    null = noise_peak_ratio(
        maps[0].shape, 1.5, np.random.default_rng(0), n_samples=200
    )
    fields = [field_score(m) for m in maps]
    counts: dict[str, int] = {}
    for m, field in zip(maps, fields):
        structure = spectral_structure(m, min_peak_ratio=null)
        key = NAMES.get(structure["n_peaks"], "none")
        if field >= 0.5:
            key = "none"
        counts[key] = counts.get(key, 0) + 1
    np.save(run_dir / "rate_maps_fixed_start.npy", maps)

    periodic = sum(v for k, v in counts.items() if k != "none")
    return {
        "run": run_dir.name,
        "accuracy": json.loads((run_dir / "metrics.json").read_text())["accuracy"],
        "units": len(maps),
        "counts": counts,
        "periodic": periodic,
        "hex": counts.get("hex", 0),
        "mean_field": float(np.mean(fields)),
        "mean_gridness": float(np.nanmean([periodicity_score(m) for m in maps])),
        "samples": int(len(positions)),
        "map_shape": list(maps[0].shape),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+")
    p.add_argument("--walks", type=int, default=64)
    p.add_argument("--length", type=int, default=300)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--bin-size", type=float, default=0.25, dest="bin_size")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'run':<10} {'acc':>7} {'units':>6} {'periodic':>9} {'hex':>4} "
          f"{'field':>6} {'gridness':>9}  counts")
    results = []
    for name in args.runs:
        row = score(Path(name), args, device)
        if row is None:
            print(f"{name:<10} no checkpoint")
            continue
        results.append(row)
        print(f"{row['run']:<10} {100 * row['accuracy']:>6.2f}% {row['units']:>6} "
              f"{row['periodic']:>4}/{row['units']:<4} {row['hex']:>4} "
              f"{row['mean_field']:>6.3f} {row['mean_gridness']:>+9.3f}  "
              f"{row['counts']}")
    out = ROOT / "runs" / "grid_report.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}   (fixed-start walks, "
          f"{results[0]['samples']:,} samples, "
          f"{results[0]['map_shape']} maps)" if results else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
