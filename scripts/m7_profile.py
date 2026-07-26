"""Where the wall-clock actually goes, and how much of it the GPU could take.

    python scripts/m7_profile.py

Standing lesson 4 of the roadmap exists because this exact question was
answered twice, confidently, from a measurement that could not support the
answer. So this measures rather than reasons:

* how long a training iteration spends generating walks in numpy on the CPU,
  copying them to the device, and running the model;
* how the model's step loop scales with batch size, which is the tell for
  launch-bound work -- if 4x the arithmetic costs nothing, the GPU was idle and
  the limit is how fast Python can issue kernels;
* what a single step costs, against how many kernels it launches.
"""

from __future__ import annotations

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


def build(device):
    config = replace(
        Config(), module_dims=(6,) * 5,
        module_freqs=(0.667, 0.5, 0.377, 0.282, 0.213),
        observation_mode="patch", patch_size=16, obs_dim=32,
        continuous=True, speed=0.5,
    )
    model = SmallCoreRecurrent(0, config, seed=0).to(device)
    return config, model


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, model = build(device)
    rng = np.random.default_rng(0)
    env = make_image_environment(10, 10, 16, 16, 16, rng)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps = 200

    print(f"device {device}   walk {steps} steps   window {config.tbptt_window}\n")

    print("--- one training iteration, broken down (batch 64) ---")
    generate = copy = compute = 0.0
    for _ in range(3):
        t0 = time.perf_counter()
        walks = generate_image_walks(env, 64, steps, rng, 0.5)
        t1 = time.perf_counter()
        velocities, patches = to_tensors(walks, device)
        sync(device)
        t2 = time.perf_counter()
        optimiser.zero_grad()
        run_image_walk(model, velocities, patches, config, 20, True, True, 1)
        optimiser.step()
        sync(device)
        t3 = time.perf_counter()
        generate += t1 - t0
        copy += t2 - t1
        compute += t3 - t2
    total = generate + copy + compute
    for name, value in (
        ("walk generation (numpy, CPU)", generate),
        ("host->device copy", copy),
        ("model + backward", compute),
    ):
        print(f"  {name:<32} {value / 3:6.3f} s  {100 * value / total:5.1f}%")

    print("\n--- model time vs batch size (launch-bound if flat) ---")
    base = None
    for batch in (16, 64, 256):
        walks = generate_image_walks(env, batch, steps, rng, 0.5)
        velocities, patches = to_tensors(walks, device)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(3):
            optimiser.zero_grad()
            run_image_walk(model, velocities, patches, config, 20, True, True, 1)
            optimiser.step()
        sync(device)
        each = (time.perf_counter() - t0) / 3
        base = each if base is None else base
        print(
            f"  batch {batch:>4}  {each:6.3f} s/iter  "
            f"{each / base:4.2f}x the batch-16 cost for {batch // 16}x the work"
        )

    print("\n--- one step of the recurrent loop ---")
    walks = generate_image_walks(env, 64, 21, rng, 0.5)
    velocities, patches = to_tensors(walks, device)
    with torch.no_grad():
        state = model.initial_state(64, device)
        state = model.observe(state, patches[:, 0])
        sync(device)
        t0 = time.perf_counter()
        for _ in range(10):
            model.run_chunk(state, velocities[:, :20], patches[:, 1:21])
        sync(device)
        each = (time.perf_counter() - t0) / 10 / 20
    print(f"  {1e6 * each:7.0f} us per step, forward only, batch 64")
    print(
        "  a step issues ~40-60 kernels (position einsum, two attentions, gate\n"
        "  MLP, readout MLP, the concatenations); at a few microseconds of\n"
        "  launch overhead each that is most of the number above."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
