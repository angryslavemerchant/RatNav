"""How much VRAM a training iteration costs, and therefore the largest batch.

    python scripts/m7_vram.py --walk 300 --target-gb 96

Fits ``bytes = fixed + per_sample * batch`` to measured peaks and reports the
batch that fills a given card. Two details make the naive estimate wrong:

* **Walk length dominates, not batch alone.** The KV cache spans the whole walk
  by design, so memory scales with ``batch * walk_length``. Walks are sampled
  100-300 steps, so the *longest* walk sets the peak and a batch tuned on the
  average will die partway through a run.
* **Peak is not the sum of the parts.** Autograd frees each truncation window's
  graph as it is backpropagated, so the peak is one window's activations plus
  the whole detached cache -- much less than a naive whole-walk estimate.

Measured with ``torch.cuda.max_memory_allocated``, which is per-process and so
remains valid while other jobs share the card. It excludes allocator
fragmentation and the CUDA context (a few hundred MB), so the headroom below is
deliberate rather than timid.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcore.config import Config
from smallcore.image_training import run_image_walk, to_tensors
from smallcore.image_world import generate_image_walks, make_image_environment
from smallcore.recurrent import SmallCoreRecurrent


def measure(batch: int, walk: int, device, config, env, rng) -> float:
    """Peak allocated bytes for one full training iteration."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = SmallCoreRecurrent(0, config, seed=0).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    walks = generate_image_walks(env, batch, walk, rng, config.speed)
    velocities, patches = to_tensors(walks, device)
    optimiser.zero_grad()
    run_image_walk(model, velocities, patches, config, config.tbptt_window,
                   True, True, 1)
    optimiser.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del model, optimiser, velocities, patches
    torch.cuda.empty_cache()
    return float(peak)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--walk", type=int, default=300, help="longest walk sampled")
    p.add_argument("--target-gb", type=float, default=96.0, dest="target_gb")
    p.add_argument("--headroom", type=float, default=0.85,
                   help="fraction of the card to plan for")
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--batches", type=str, default="8,16,32,64,128")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device; nothing to measure")
        return 1
    device = torch.device("cuda")
    config = replace(
        Config(), module_dims=(6,) * 5,
        module_freqs=(0.667, 0.5, 0.377, 0.282, 0.213),
        observation_mode="patch", patch_size=args.patch, obs_dim=32,
        continuous=True, speed=0.5,
    )
    rng = np.random.default_rng(0)
    env = make_image_environment(10, 10, 16, args.patch, 16, rng)

    sizes = [int(b) for b in args.batches.split(",")]
    peaks = []
    print(f"walk {args.walk} steps, window {config.tbptt_window}, "
          f"patch {args.patch}px\n")
    print(f"{'batch':>6} {'peak MiB':>10} {'MiB/sample':>12}")
    for batch in sizes:
        peak = measure(batch, args.walk, device, config, env, rng)
        peaks.append(peak)
        print(f"{batch:>6} {peak / 2**20:>10.1f} {peak / 2**20 / batch:>12.2f}")

    # Least squares on (batch, peak): the intercept is weights, optimiser state
    # and the CUDA workspace; the slope is what one more walk actually costs.
    slope, fixed = np.polyfit(np.array(sizes, dtype=float), np.array(peaks), 1)
    budget = args.target_gb * 2**30 * args.headroom
    max_batch = int((budget - fixed) / slope)

    print(
        f"\nfit: {fixed / 2**20:.0f} MiB fixed + {slope / 2**20:.2f} MiB per walk\n"
        f"target {args.target_gb:.0f} GB at {args.headroom:.0%} headroom "
        f"= {budget / 2**30:.1f} GB usable\n"
        f"  -> max batch ~= {max_batch:,} walks of {args.walk} steps"
    )
    print(
        "\nThe cap that matters is probably not this one. The loop is\n"
        "launch-bound, so throughput stops improving once the GPU is saturated;\n"
        "measure s/iter against batch and take the knee, not the ceiling."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
