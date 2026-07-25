"""Diagnostic: how much learning signal reaches each module/copy?

The shared-core encoder trains far more slowly than the independent-module
baseline. One candidate explanation is a gradient imbalance across scales: a
copy's operator is ``expm(s_i * G)``, so its influence on the shared core
scales with its knob ``s_i``, and those span two orders of magnitude
(0.99 down to 0.01). If true, the fine copy shapes the shared core and the
coarse copies are dragged along wherever it goes.

This measures the gradient reaching each module's slice of the position code at
initialisation, for both encoders. The independent encoder is the control: if
*it* shows the same imbalance and still trains fine, imbalance is not the
explanation and the problem is expressiveness, not optimisation.

    python scripts/m1_diag_gradients.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcore.config import Config
from smallcore.graphs import Environment, generate_walk
from smallcore.position import PositionEncoder
from smallcore.shared_position import SharedPositionEncoder

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / "smallcore" / "envs" / "square_11x11.json"

N_BATCHES = 20
LENGTH = 50


def module_gradients(
    encoder: nn.Module,
    boundaries: list[tuple[int, int]],
    env: Environment,
    config: Config,
    rng: np.random.Generator,
    device: torch.device,
    start: int,
) -> np.ndarray:
    """Mean gradient norm per position-code slice, normalised per dimension.

    Returns one number per module: the RMS gradient arriving at that module's
    dimensions of the code. Per-dimension so modules of unequal width compare
    fairly.
    """
    head = nn.Linear(encoder.dim, env.n_observations).to(device)
    totals = np.zeros(len(boundaries))

    for _ in range(N_BATCHES):
        walks = [
            generate_walk(env, LENGTH, rng, config.straight_bias, start=start)
            for _ in range(config.batch_size)
        ]
        actions = torch.tensor(
            np.stack([w.actions[:-1] for w in walks]), dtype=torch.long, device=device
        )
        observations = torch.tensor(
            np.stack([w.observations for w in walks]), dtype=torch.long, device=device
        )

        codes = encoder(actions)
        codes.retain_grad()
        logits = head(codes[:, 1:])
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), observations[:, 1:].reshape(-1)
        )
        encoder.zero_grad()
        head.zero_grad()
        loss.backward()

        grad = codes.grad  # (B, T, dim)
        for index, (lo, hi) in enumerate(boundaries):
            slice_grad = grad[:, :, lo:hi]
            totals[index] += slice_grad.pow(2).mean().sqrt().item()

    return totals / N_BATCHES


def main() -> int:
    if not ENV_PATH.exists():
        raise SystemExit(f"{ENV_PATH} missing -- run scripts/m0_demo.py first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()
    env = Environment.from_json(ENV_PATH)
    start = env.topology.n_locations // 2

    print(
        f"gradient reaching each module's slice of the position code, at init\n"
        f"({N_BATCHES} batches of {config.batch_size} x {LENGTH} steps, "
        f"RMS per dimension)\n"
    )

    # --- independent baseline (the control) -------------------------------- #
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    independent = PositionEncoder(
        env.topology.n_actions, config.module_dims, config.module_freqs, seed=0
    ).to(device)
    bounds, offset = [], 0
    for dim in config.module_dims:
        bounds.append((offset, offset + dim))
        offset += dim
    ind_grads = module_gradients(
        independent, bounds, env, config, rng, device, start
    )

    # --- shared core ------------------------------------------------------- #
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n_copies = len(config.module_dims)
    copy_dim = config.position_dim // n_copies
    shared = SharedPositionEncoder(
        env.topology.n_actions,
        n_copies=n_copies,
        copy_dim=copy_dim,
        knob_init=config.module_freqs,
        seed=0,
    ).to(device)
    shared_bounds = [(i * copy_dim, (i + 1) * copy_dim) for i in range(n_copies)]
    shared_grads = module_gradients(
        shared, shared_bounds, env, config, rng, device, start
    )

    print(f"{'module':>8} {'freq':>7} {'independent':>13} {'shared':>10}")
    for index, freq in enumerate(config.module_freqs):
        print(
            f"{index:>8} {freq:>7} {ind_grads[index]:>13.3e} "
            f"{shared_grads[index]:>10.3e}"
        )

    print(
        f"\nspread (max / min across modules):\n"
        f"  independent: {ind_grads.max() / ind_grads.min():>8.1f}x\n"
        f"  shared:      {shared_grads.max() / shared_grads.min():>8.1f}x"
    )
    print(
        "\nIf the two spreads are similar, imbalance across scales is normal and\n"
        "does not explain the shared encoder's slow training."
    )

    # --- knob gradients ---------------------------------------------------- #
    head = nn.Linear(shared.dim, env.n_observations).to(device)
    walks = [
        generate_walk(env, LENGTH, rng, config.straight_bias, start=start)
        for _ in range(config.batch_size)
    ]
    actions = torch.tensor(
        np.stack([w.actions[:-1] for w in walks]), dtype=torch.long, device=device
    )
    observations = torch.tensor(
        np.stack([w.observations for w in walks]), dtype=torch.long, device=device
    )
    codes = shared(actions)
    loss = nn.functional.cross_entropy(
        head(codes[:, 1:]).reshape(-1, env.n_observations),
        observations[:, 1:].reshape(-1),
    )
    shared.zero_grad()
    loss.backward()
    print(
        f"\nknob gradients (log-space, one backward pass):\n"
        f"  {shared.log_knobs.grad.abs().cpu().numpy().round(5).tolist()}"
    )
    print(
        f"shared-core generator gradient norm: "
        f"{shared.generators.grad.norm().item():.4e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
