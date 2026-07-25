"""M2: forward read + readout on a single environment.

    python scripts/m2_train.py                 # the real test (random starts)
    python scripts/m2_train.py --fixed-start    # control, see below
    python scripts/m2_train.py --wandb --run-name m2_baseline

Passes when accuracy is well above the node and edge agents (see
``smallcore.baselines``).

Why walks start at a RANDOM location
------------------------------------
M1 used a fixed start, which made the position code an *absolute* position and
let a linear head reach 99% observation accuracy from ``e_t`` alone. If M2 kept
that, the memory stream would be decorative: the readout could simply memorise
the map, pass the milestone, and teach us nothing about whether the forward
read works.

With a random start the position code carries displacement from an unknown
origin. Absolute location is unrecoverable in principle, so no amount of
memorising the map helps -- the only way to know what is here is to remember
having been here *in this walk*. That makes the forward read load-bearing,
which is the thing M2 exists to test. ``--fixed-start`` runs the degenerate
version for comparison.

``revisit rate`` is reported alongside: the ceiling for a model that does
nothing but look up what it saw last time it stood here. The model can exceed
it, and does -- training on one environment lets it learn that environment's
layout, so retrieval can serve to establish *where it is* and the map supplies
the rest. Beating the revisit rate is therefore evidence of localisation rather
than lookup.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcore.baselines import (
    EdgeAgent,
    NodeAgent,
    chance_accuracy,
    oracle_memory_accuracy,
)
from smallcore.config import Config
from smallcore.graphs import Environment, Walk, generate_batch_fast
from smallcore.losses import compute_losses
from smallcore.model import SmallCore

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / "smallcore" / "envs" / "square_11x11.json"
RUNS_DIR = ROOT / "runs"


def sample_walks(
    env: Environment,
    batch_size: int,
    length: int,
    rng: np.random.Generator,
    straight_bias: float,
    start: int | None,
) -> list[Walk]:
    return generate_batch_fast(env, batch_size, length, rng, straight_bias, start)


def to_tensors(
    walks: list[Walk], n_observations: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(actions, one-hot observations, observation indices)."""
    actions = torch.tensor(
        np.stack([w.actions[:-1] for w in walks]), dtype=torch.long, device=device
    )
    indices = torch.tensor(
        np.stack([w.observations for w in walks]), dtype=torch.long, device=device
    )
    one_hot = torch.nn.functional.one_hot(indices, n_observations).float()
    return actions, one_hot, indices


@torch.no_grad()
def evaluate(
    model: SmallCore,
    env: Environment,
    config: Config,
    rng: np.random.Generator,
    device: torch.device,
    length: int,
    start: int | None,
    n_walks: int = 64,
) -> dict:
    """Accuracy on fresh walks, alongside the baselines and the ceiling."""
    model.eval()
    walks = sample_walks(env, n_walks, length, rng, config.straight_bias, start)
    for walk in walks[:8]:
        walk.validate(env)
    actions, one_hot, indices = to_tensors(walks, env.n_observations, device)
    output = model(actions, one_hot)
    terms = compute_losses(
        output, indices, config.w_pred_pos, config.l2_position_code
    )

    # Train the tabular baselines on independent walks, score on the same ones.
    fitting = sample_walks(env, 256, length, rng, config.straight_bias, start)
    model.train()
    return {
        "accuracy": terms.accuracy,
        "accuracy_position": terms.accuracy_position,
        "loss": terms.pred.item(),
        "node": NodeAgent(env.n_observations).fit(fitting).accuracy(walks),
        "edge": EdgeAgent(env.n_observations, env.topology.n_actions)
        .fit(fitting)
        .accuracy(walks),
        "oracle_memory": oracle_memory_accuracy(walks),
        "chance": chance_accuracy(walks, env.n_observations),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--walk-min", type=int, default=50, dest="walk_min")
    parser.add_argument("--walk-max", type=int, default=150, dest="walk_max")
    parser.add_argument("--eval-length", type=int, default=100, dest="eval_length")
    parser.add_argument(
        "--fixed-start",
        action="store_true",
        dest="fixed_start",
        help="degenerate control: absolute position becomes recoverable and the "
        "readout can memorise the map without using memory at all",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name", type=str, default=None, dest="run_name")
    parser.add_argument(
        "--checkpoint-every", type=int, default=1000, dest="checkpoint_every"
    )
    args = parser.parse_args()

    if not ENV_PATH.exists():
        raise SystemExit(f"{ENV_PATH} missing -- run scripts/m0_demo.py first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    config = Config()
    env = Environment.from_json(ENV_PATH)
    start = env.topology.n_locations // 2 if args.fixed_start else None

    run_name = args.run_name or time.strftime("m2_%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "LATEST").write_text(str(run_dir), encoding="utf-8")

    model = SmallCore(env.topology.n_actions, config, seed=args.seed).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            project="smallcore",
            name=run_name,
            config={**vars(args), "parameters": n_params},
        )

    print(
        f"M2 on {env.name}: {env.topology.n_locations} locations, "
        f"{env.n_observations} symbols, "
        f"start {'fixed' if args.fixed_start else 'RANDOM'}\n"
        f"parameters: {n_params:,}  device {device}  run {run_dir}\n"
    )

    optimiser = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    best = -1.0
    began = time.time()

    for iteration in range(1, args.iters + 1):
        lr = max(
            config.lr_min,
            config.lr * config.lr_decay ** (iteration // config.lr_decay_every),
        )
        for group in optimiser.param_groups:
            group["lr"] = lr

        length = int(rng.integers(args.walk_min, args.walk_max + 1))
        walks = sample_walks(
            env, config.batch_size, length, rng, config.straight_bias, start
        )
        actions, one_hot, indices = to_tensors(walks, env.n_observations, device)

        output = model(actions, one_hot)
        terms = compute_losses(
            output, indices, config.w_pred_pos, config.l2_position_code
        )

        optimiser.zero_grad()
        terms.total.backward()
        optimiser.step()

        if run is not None:
            run.log(
                {
                    "loss/total": terms.total.item(),
                    "loss/pred": terms.pred.item(),
                    "loss/pred_position": terms.pred_position.item(),
                    "loss/reg_code": terms.reg_code.item(),
                    "train/accuracy": terms.accuracy,
                    "train/accuracy_position": terms.accuracy_position,
                    "lr": lr,
                },
                step=iteration,
            )

        if iteration % 500 == 0 or iteration == 1:
            metrics = evaluate(
                model, env, config, rng, device, args.eval_length, start
            )
            print(
                f"  iter {iteration:>5}  loss {terms.pred.item():.3f}  "
                f"acc {metrics['accuracy']:.2%}  "
                f"(pos-only {metrics['accuracy_position']:.2%})  "
                f"node {metrics['node']:.2%}  edge {metrics['edge']:.2%}  "
                f"revisit {metrics['oracle_memory']:.2%}  "
                f"({time.time() - began:.0f}s)"
            )
            if run is not None:
                run.log({f"eval/{k}": v for k, v in metrics.items()}, step=iteration)
            if metrics["accuracy"] > best:
                best = metrics["accuracy"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": config,
                        "iteration": iteration,
                        "metrics": metrics,
                    },
                    run_dir / "best.pt",
                )

        if iteration % args.checkpoint_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "iteration": iteration,
                },
                run_dir / "latest.pt",
            )

    # --- final verdict ------------------------------------------------------ #
    final = evaluate(model, env, config, rng, device, args.eval_length, start)
    long_walk = evaluate(model, env, config, rng, device, 300, start)
    (run_dir / "metrics.json").write_text(
        json.dumps({"final": final, "length_300": long_walk}, indent=2),
        encoding="utf-8",
    )

    print(
        f"\nfinal, {args.eval_length}-step held-out walks:\n"
        f"  full model      {final['accuracy']:.2%}\n"
        f"  position only   {final['accuracy_position']:.2%}\n"
        f"  edge agent      {final['edge']:.2%}\n"
        f"  node agent      {final['node']:.2%}\n"
        f"  most common     {final['chance']:.2%}\n"
        f"  oracle memory   {final['oracle_memory']:.2%}  (revisit-rate ceiling)\n"
        f"\n300-step walks: {long_walk['accuracy']:.2%} "
        f"(edge {long_walk['edge']:.2%}, revisit {long_walk['oracle_memory']:.2%})"
    )

    beat_baselines = final["accuracy"] > max(final["edge"], final["node"])
    print(
        f"\nM2 {'PASSED' if beat_baselines else 'NOT PASSED'}: full model "
        f"{final['accuracy']:.2%} vs best baseline "
        f"{max(final['edge'], final['node']):.2%}"
    )
    if run is not None:
        run.finish()
    return 0 if beat_baselines else 1


if __name__ == "__main__":
    raise SystemExit(main())
