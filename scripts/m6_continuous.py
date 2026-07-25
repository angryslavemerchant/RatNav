"""M6: continuous movement — velocity vectors instead of action indices.

    python scripts/m6_continuous.py --iters 3000 --batch-size 128

Same model, same memory, same loss. The only change is that the position stream
integrates a velocity rather than looking up one of four matrices, so two
learned generators replace four action matrices and the set of expressible
movements becomes infinite.

Observations stay discrete, painted on a background lattice the continuous
position is binned into. That keeps continuous MOVEMENT as the single changed
variable — continuous observations are a separate step, and changing both at
once is the confound that has already produced wrong conclusions here twice.

Why this matters for M5. On a discrete grid a "periodic" code barely differs
from a lookup table: a module with a 3-cell cycle takes only 3 values along an
axis, and since the model is scored only at integer positions it can treat that
as an arbitrary 3-state categorical variable and be exactly as correct.
Continuity closes that escape — the code must interpolate between sampled
positions, and smooth-plus-repeating is genuinely periodic. It also removes the
quantisation floor that kept rate maps small.
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
from smallcore.continuous import (
    assign_continuous_observations,
    generate_continuous_walks,
)
from smallcore.recurrent import SmallCoreRecurrent
from smallcore.training import run_walk

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"


def to_tensors(walks, n_observations, device):
    """(velocities, one-hot observations, observation indices)."""
    velocities = torch.tensor(
        np.stack([w.velocities[:-1] for w in walks]), dtype=torch.float32,
        device=device,
    )
    indices = torch.tensor(
        np.stack([w.observations for w in walks]), dtype=torch.long, device=device
    )
    one_hot = torch.nn.functional.one_hot(indices, n_observations).float()
    return velocities, one_hot, indices


def cell_indices(walk, env):
    """Which lattice cell each continuous position falls in (analysis only)."""
    col = np.clip(np.floor(walk.positions[:, 0]), 0, env.width - 1).astype(int)
    row = np.clip(np.floor(walk.positions[:, 1]), 0, env.height - 1).astype(int)
    return row * env.width + col


def ceiling_and_baselines(walks, env):
    """Revisit-based ceiling plus a content-addressed baseline, on binned cells.

    The ceiling is the revisit rate PLUS what guessing earns on cells never
    visited, exactly as in the discrete world -- quoting the bare revisit rate
    flatters the model by a couple of points.
    """
    counts = np.bincount(env.observations, minlength=env.n_observations)
    p_guess = counts.max() / env.n_cells
    hit = total = 0.0
    steps = 0
    node_correct = 0
    for walk in walks:
        cells = cell_indices(walk, env)
        seen = {int(cells[0])}
        for t in range(1, len(walk)):
            c = int(cells[t])
            hit += 1.0 if c in seen else 0.0
            total += 1.0 if c in seen else p_guess
            seen.add(c)
            steps += 1
        # Node baseline: predict the symbol you just saw (positions are close
        # together, so persistence is the natural content-addressed guess).
        node_correct += int((walk.observations[:-1] == walk.observations[1:]).sum())
    n = max(steps, 1)
    return {
        "revisit": hit / n,
        "ceiling": total / n,
        "node": node_correct / max(sum(len(w) - 1 for w in walks), 1),
    }


@torch.no_grad()
def evaluate(model, env, config, rng, device, length, window, n_walks=32):
    model.eval()
    walks = generate_continuous_walks(env, n_walks, length, rng, config.speed)
    for w in walks[:4]:
        w.validate(env)
    velocities, one_hot, indices = to_tensors(walks, env.n_observations, device)
    metrics = run_walk(
        model, velocities, one_hot, indices, config, window, True, train=False
    )
    model.train()
    metrics.update(ceiling_and_baselines(walks, env))
    return metrics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--grid", type=int, default=21)
    p.add_argument("--walk-min", type=int, default=100, dest="walk_min")
    p.add_argument("--walk-max", type=int, default=300, dest="walk_max")
    p.add_argument("--n-envs", type=int, default=8, dest="n_envs")
    p.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    p.add_argument("--lr", type=float, default=9.4e-4)
    p.add_argument("--module-dims", type=str, default="6,6,6,6,6", dest="module_dims")
    p.add_argument(
        "--module-freqs", type=str,
        default="0.667,0.476,0.34,0.243,0.174", dest="module_freqs",
    )
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--run-name", type=str, default=None, dest="run_name")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dims = tuple(int(d) for d in args.module_dims.split(","))
    freqs = tuple(float(f) for f in args.module_freqs.split(","))
    n_obs = max(2, round(args.grid * args.grid / 2.7))
    config = replace(
        Config(), module_dims=dims, module_freqs=freqs, n_observations=n_obs,
        batch_size=args.batch_size, lr=args.lr, continuous=True,
        speed=args.speed,
    )

    pool = [
        assign_continuous_observations(args.grid, args.grid, n_obs, rng)
        for _ in range(args.n_envs)
    ]
    held_rng = np.random.default_rng(args.seed + 9999)
    held_out = [
        assign_continuous_observations(args.grid, args.grid, n_obs, held_rng)
        for _ in range(4)
    ]

    run_name = args.run_name or time.strftime("m6_%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "LATEST").write_text(str(run_dir), encoding="utf-8")

    model = SmallCoreRecurrent(0, config, seed=args.seed).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(
        f"M6 CONTINUOUS on a {args.grid}x{args.grid} arena, {n_obs} symbols\n"
        "modules: "
        + ", ".join(f"cycle {2.0 / f:.1f} cells" for f in freqs)
        + f"\nposition stream: {config.position_dim} dims, 2 velocity generators "
        f"({model.position.generators.numel():,} transition params)\n"
        f"parameters: {n_params:,}  device {device}  run {run_dir}\n"
    )

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="smallcore", name=run_name,
                         config={**vars(args), "parameters": n_params})

    optimiser = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    best = -1.0
    began = time.time()

    for iteration in range(1, args.iters + 1):
        lr = max(config.lr_min,
                 config.lr * config.lr_decay ** (iteration // config.lr_decay_every))
        for g in optimiser.param_groups:
            g["lr"] = lr

        env = pool[int(rng.integers(len(pool)))]
        length = int(rng.integers(args.walk_min, args.walk_max + 1))
        walks = generate_continuous_walks(
            env, config.batch_size, length, rng, args.speed
        )
        velocities, one_hot, indices = to_tensors(walks, n_obs, device)

        optimiser.zero_grad()
        stats = run_walk(
            model, velocities, one_hot, indices, config, args.window, True, train=True
        )
        optimiser.step()

        if run is not None:
            run.log({f"train/{k}": v for k, v in stats.items()}, step=iteration)

        if iteration % 250 == 0 or iteration == 1:
            runs_ = [evaluate(model, e, config, rng, device, 300, args.window)
                     for e in held_out]
            unseen = {k: float(np.mean([r[k] for r in runs_])) for k in runs_[0]}
            print(
                f"  iter {iteration:>5}  loss {stats['loss']:.3f}  "
                f"unseen 300-step {unseen['accuracy']:.2%}  "
                f"(ceiling {unseen['ceiling']:.2%}, node {unseen['node']:.2%})  "
                f"gate {stats['gate']:.3f}  ({time.time() - began:.0f}s)"
            )
            if run is not None:
                run.log({f"unseen/{k}": v for k, v in unseen.items()}, step=iteration)
            if unseen["accuracy"] > best:
                best = unseen["accuracy"]
                torch.save({"model": model.state_dict(), "config": config,
                            "iteration": iteration, "metrics": unseen,
                            "grid": args.grid}, run_dir / "best.pt")

    runs_ = [evaluate(model, e, config, rng, device, 300, args.window)
             for e in held_out]
    final = {k: float(np.mean([r[k] for r in runs_])) for k in runs_[0]}
    (run_dir / "metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(
        f"\nfinal, environments never trained on:\n"
        f"  300-step accuracy  {final['accuracy']:.2%}  "
        f"(ceiling {final['ceiling']:.2%} = revisit {final['revisit']:.2%} + guessing)\n"
        f"  node baseline      {final['node']:.2%}\n"
        f"  mean gate          {final['gate']:.3f}"
    )
    passed = final["accuracy"] > final["node"]
    print(f"\nM6 {'PASSED' if passed else 'NOT PASSED'}: continuous path "
          f"integration {final['accuracy']:.2%} vs node baseline {final['node']:.2%}")
    if run is not None:
        run.finish()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
