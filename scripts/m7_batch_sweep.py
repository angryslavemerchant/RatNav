"""Find the largest USEFUL batch on this card, which is not the largest that fits.

    python scripts/m7_batch_sweep.py --max-gb 96

The step loop is kernel-launch bound: measured locally, batch 16 -> 64 is four
times the arithmetic for 5% more wall clock, because the GPU sits idle while
Python issues ~50 small kernels per step. Under those conditions extra batch is
very nearly free -- right up until the GPU saturates, after which it is fully
priced. Two limits therefore exist and they are different numbers:

* the **VRAM ceiling**, roughly 10 MiB per 300-step walk, so ~8,000 walks on a
  96 GB card;
* the **throughput knee**, where seconds per walk stops falling.

Past the knee the extra memory buys nothing, so the knee is the answer and the
ceiling is a guardrail. This sweep measures both, and reports walks/second so
the two can be compared directly rather than argued about.

Walk length is set to the longest the training run samples, because the KV
cache spans the whole walk and a batch tuned on the average length will die
partway through the first long one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcore.config import Config
from smallcore.image_training import run_image_walk, to_tensors
from smallcore.image_world import generate_image_walks, make_image_environment
from smallcore.recurrent import SmallCoreRecurrent


def trial(batch, walk, config, env, rng, device, repeats=3):
    """(seconds per iteration, peak MiB) or None if it does not fit."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = SmallCoreRecurrent(0, config, seed=0).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        walks = generate_image_walks(env, batch, walk, rng, config.speed)
        velocities, patches = to_tensors(walks, device)
        # One untimed iteration: the allocator's first pass at a new size is
        # not representative of the steady state.
        optimiser.zero_grad()
        run_image_walk(model, velocities, patches, config,
                       config.tbptt_window, True, True, 1)
        optimiser.step()
        torch.cuda.synchronize()

        began = time.perf_counter()
        for _ in range(repeats):
            optimiser.zero_grad()
            run_image_walk(model, velocities, patches, config,
                           config.tbptt_window, True, True, 1)
            optimiser.step()
        torch.cuda.synchronize()
        each = (time.perf_counter() - began) / repeats
        peak = torch.cuda.max_memory_allocated() / 2**20
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    finally:
        for name in ("model", "optimiser", "velocities", "patches"):
            if name in dir():
                pass
    del model, optimiser, velocities, patches
    torch.cuda.empty_cache()
    return each, peak


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--walk", type=int, default=300)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--group", type=int, default=64,
                   help="contrastive group, so the objective is batch-independent")
    p.add_argument("--batches", type=str,
                   default="32,64,128,256,512,1024,2048,4096")
    p.add_argument("--out", type=str, default="runs/batch_sweep.json")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)
    print(
        f"{properties.name}  {properties.total_memory / 2**30:.0f} GB  "
        f"{properties.multi_processor_count} SMs\n"
        f"walk {args.walk} steps, patch {args.patch}px, "
        f"contrastive group {args.group}\n"
    )

    config = replace(
        Config(), module_dims=(6,) * 5,
        module_freqs=(0.667, 0.5, 0.377, 0.282, 0.213),
        observation_mode="patch", patch_size=args.patch, obs_dim=32,
        continuous=True, speed=0.5, contrastive_group=args.group,
    )
    rng = np.random.default_rng(0)
    env = make_image_environment(10, 10, 16, args.patch, 16, rng)

    print(f"{'batch':>6} {'s/iter':>9} {'walks/s':>10} {'peak MiB':>10} "
          f"{'MiB/walk':>9} {'vs prev':>8}")
    rows, best = [], None
    previous = None
    for batch in [int(b) for b in args.batches.split(",")]:
        if batch % args.group:
            print(f"{batch:>6}   skipped: not a multiple of group {args.group}")
            continue
        result = trial(batch, args.walk, config, env, rng, device)
        if result is None:
            print(f"{batch:>6}   out of memory -- ceiling reached")
            break
        seconds, peak = result
        rate = batch / seconds
        gain = "-" if previous is None else f"{rate / previous:.2f}x"
        print(f"{batch:>6} {seconds:>9.3f} {rate:>10.1f} {peak:>10.0f} "
              f"{peak / batch:>9.2f} {gain:>8}")
        rows.append({"batch": batch, "seconds": seconds, "walks_per_s": rate,
                     "peak_mib": peak})
        if best is None or rate > best["walks_per_s"]:
            best = rows[-1]
        previous = rate

    # The knee: the largest batch still buying at least 10% more throughput
    # than the one below it. Beyond that the GPU is saturated and the extra
    # memory is spent for nothing.
    knee = rows[0] if rows else None
    for earlier, later in zip(rows, rows[1:]):
        if later["walks_per_s"] > 1.10 * earlier["walks_per_s"]:
            knee = later
    if rows:
        print(
            f"\nfastest      batch {best['batch']:>5}  "
            f"{best['walks_per_s']:.1f} walks/s\n"
            f"knee         batch {knee['batch']:>5}  "
            f"{knee['walks_per_s']:.1f} walks/s  "
            f"({knee['peak_mib'] / 1024:.1f} GB)\n"
            f"speedup over batch {rows[0]['batch']}: "
            f"{best['walks_per_s'] / rows[0]['walks_per_s']:.1f}x throughput"
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"device": properties.name, "rows": rows,
             "knee": knee, "fastest": best}, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
