"""M4: many environments, fresh observations each -- does structure transfer?

    python scripts/m4_train.py                  # a new environment every batch
    python scripts/m4_train.py --n-envs 8       # a fixed pool, to measure memorisation
    python scripts/m4_train.py --wandb --run-name m4_baseline

Same topology throughout, so the *movement structure* is constant; only the
observation assignment changes. That is the whole claim under test: how actions
compose is a property of the space and worth learning once, while what any
particular place looks like is not.

What changes from M3, and why it should be harder
-------------------------------------------------
M2 and M3 trained on ONE environment, so the model could and did learn that
environment's layout into its weights -- measurably, since accuracy exceeded
the within-walk revisit rate. Redrawing the observations every batch closes
that route completely. A symbol seen at a location in one batch means nothing
about that location in the next, so the only usable knowledge is (a) how
movements compose and (b) what has been observed *in the current walk*.

So expect accuracy to fall relative to M3's 97.4%, and expect the revisit rate
to become a genuine ceiling rather than the reference point it was in M2/M3.
That is not a regression: it is the task finally being the one the architecture
was designed for. The number that matters is how close to the ceiling the model
gets on an environment it has never seen -- and the ceiling is the revisit rate
plus what guessing earns on locations never visited, not the revisit rate
alone (see ``smallcore.baselines.memory_ceiling``).

Baselines are refitted per evaluation environment, since a node/edge table from
a different observation assignment is meaningless in this one.
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

from smallcore.baselines import (
    EdgeAgent,
    NodeAgent,
    chance_accuracy,
    memory_ceiling,
    oracle_memory_accuracy,
)
from smallcore.config import Config
from smallcore.graphs import (
    Environment,
    assign_observations,
    generate_batch_fast,
    square_grid,
)
from smallcore.recurrent import SmallCoreRecurrent
from smallcore.training import run_walk, to_tensors

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"


@torch.no_grad()
def evaluate(model, env, config, rng, device, length, window, n_walks=32) -> dict:
    """Accuracy on fresh walks through ``env``, with baselines refitted here."""
    model.eval()
    walks = generate_batch_fast(env, n_walks, length, rng, config.straight_bias)
    for walk in walks[:4]:
        walk.validate(env)
    actions, one_hot, indices = to_tensors(walks, env.n_observations, device)
    metrics = run_walk(
        model, actions, one_hot, indices, config, window, True, train=False
    )
    fitting = generate_batch_fast(env, 256, length, rng, config.straight_bias)
    model.train()
    metrics.update(
        node=NodeAgent(env.n_observations).fit(fitting).accuracy(walks),
        edge=EdgeAgent(env.n_observations, env.topology.n_actions)
        .fit(fitting)
        .accuracy(walks),
        revisit=oracle_memory_accuracy(walks),
        ceiling=memory_ceiling(walks, env),
        chance=chance_accuracy(walks, env.n_observations),
    )
    return metrics


def averaged(model, envs, config, rng, device, length, window) -> dict:
    """Mean of :func:`evaluate` across several environments."""
    runs = [evaluate(model, e, config, rng, device, length, window) for e in envs]
    return {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--walk-min", type=int, default=50, dest="walk_min")
    parser.add_argument("--walk-max", type=int, default=150, dest="walk_max")
    parser.add_argument(
        "--n-envs", type=int, default=0, dest="n_envs",
        help="0 (default) draws a fresh environment every batch, so nothing "
        "about appearance can be memorised. A positive value trains on a fixed "
        "pool, which lets held-out accuracy be compared against seen-pool "
        "accuracy to measure how much was memorised.",
    )
    parser.add_argument(
        "--init-from", type=str, default=None, dest="init_from",
        help="seed the transition operators from an earlier run's checkpoint",
    )
    parser.add_argument(
        "--freeze-position", action="store_true", dest="freeze_position",
        help="hold the transferred operators fixed",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, dest="batch_size",
        help="override Config.batch_size. Measured 2026-07-25: the step loop is "
        "kernel-launch bound, so batch 128 costs the same wall-clock per "
        "iteration as batch 16 while seeing 8x the walks. Larger batches here "
        "are close to free.",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--grid", type=int, default=11,
        help="arena width/height. Periodicity is only detectable when several "
        "cycles fit inside the arena, so 11 is cramped for M5 even with well "
        "chosen frequencies; 21 gives roughly twice the repeats.",
    )
    parser.add_argument(
        "--module-freqs", type=str, default=None, dest="module_freqs",
        help="comma-separated per-module frequencies. A module's full cycle is "
        "2/freq CELLS, so a frequency whose cycle exceeds the arena cannot "
        "express spatial periodicity at all -- which is what the shipped "
        "defaults (cycles 2, 6.7, 22, 67, 200) do on an 11-cell grid.",
    )
    parser.add_argument(
        "--n-observations", type=int, default=None, dest="n_observations",
        help="observation vocabulary. Defaults to keeping ~2.7 locations per "
        "symbol, matching the ambiguity the 11x11 world was designed with.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name", type=str, default=None, dest="run_name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    config = Config()
    if args.batch_size:
        config = replace(config, batch_size=args.batch_size)
    if args.lr:
        config = replace(config, lr=args.lr)
    if args.module_freqs:
        freqs = tuple(float(f) for f in args.module_freqs.split(","))
        if len(freqs) != len(config.module_dims):
            raise SystemExit(
                f"{len(freqs)} frequencies for {len(config.module_dims)} modules"
            )
        config = replace(config, module_freqs=freqs)
    n_locations = args.grid * args.grid
    n_obs = args.n_observations or max(2, round(n_locations / 2.7))
    config = replace(config, n_observations=n_obs)

    # One topology, many appearances. Built directly rather than loaded so the
    # observation assignment is clearly ours to redraw.
    topology = square_grid(args.grid, args.grid)
    pool = [
        assign_observations(topology, config.n_observations, rng)
        for _ in range(args.n_envs)
    ]
    # Held-out environments, never trained on, fixed across the run so the
    # evaluation curve is comparable between checkpoints.
    held_out_rng = np.random.default_rng(args.seed + 9999)
    held_out = [
        assign_observations(topology, config.n_observations, held_out_rng)
        for _ in range(4)
    ]

    run_name = args.run_name or time.strftime("m4_%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "LATEST").write_text(str(run_dir), encoding="utf-8")

    model = SmallCoreRecurrent(topology.n_actions, config, seed=args.seed).to(device)
    origin = "scratch"
    if args.init_from:
        checkpoint = torch.load(
            ROOT / args.init_from, map_location=device, weights_only=False
        )
        source = checkpoint.get("position") or checkpoint["model"]
        if "position" in checkpoint:
            model.position.load_state_dict(source)
        else:
            model.load_state_dict(source)
        origin = args.init_from
    if args.freeze_position:
        for parameter in model.position.parameters():
            parameter.requires_grad_(False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in model.parameters())

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            project="smallcore", name=run_name,
            config={**vars(args), "parameters": n_params},
        )

    print(
        "modules: "
        + ", ".join(
            f"cycle {2.0 / f:.1f} cells" for f in config.module_freqs
        )
        + f"  (arena {args.grid} cells wide)"
    )
    print(
        f"M4 on {topology.name}: "
        + ("a fresh environment every batch" if not pool
           else f"a fixed pool of {len(pool)} environments")
        + f", {len(held_out)} held out\n"
        f"position stream: {origin}"
        f"{', FROZEN' if args.freeze_position else ''}\n"
        f"parameters: {n_params:,} ({sum(p.numel() for p in trainable):,} trainable)"
        f"  device {device}  run {run_dir}\n"
    )

    optimiser = torch.optim.Adam(
        trainable, lr=config.lr, weight_decay=config.weight_decay
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

        env = (
            pool[int(rng.integers(len(pool)))]
            if pool
            else assign_observations(topology, config.n_observations, rng)
        )
        length = int(rng.integers(args.walk_min, args.walk_max + 1))
        walks = generate_batch_fast(
            env, config.batch_size, length, rng, config.straight_bias
        )
        actions, one_hot, indices = to_tensors(walks, env.n_observations, device)

        optimiser.zero_grad()
        stats = run_walk(
            model, actions, one_hot, indices, config, args.window, True, train=True
        )
        optimiser.step()

        if run is not None:
            run.log({f"train/{k}": v for k, v in stats.items()}, step=iteration)

        if iteration % 250 == 0 or iteration == 1:
            unseen = averaged(
                model, held_out, config, rng, device, 300, args.window
            )
            headroom = unseen["accuracy"] - unseen["ceiling"]
            print(
                f"  iter {iteration:>5}  loss {stats['loss']:.3f}  "
                f"unseen-env 300-step {unseen['accuracy']:.2%}  "
                f"(ceiling {unseen['ceiling']:.2%}, "
                f"vs ceiling {headroom:+.2%})  "
                f"edge {unseen['edge']:.2%}  gate {stats['gate']:.3f}  "
                f"({time.time() - began:.0f}s)"
            )
            if run is not None:
                run.log(
                    {**{f"unseen/{k}": v for k, v in unseen.items()},
                     "unseen/vs_ceiling": headroom},
                    step=iteration,
                )
            if unseen["accuracy"] > best:
                best = unseen["accuracy"]
                torch.save(
                    {"model": model.state_dict(), "config": config,
                     "iteration": iteration, "metrics": unseen},
                    run_dir / "best.pt",
                )

    unseen_300 = averaged(model, held_out, config, rng, device, 300, args.window)
    unseen_100 = averaged(model, held_out, config, rng, device, 100, args.window)
    results = {"unseen_300": unseen_300, "unseen_100": unseen_100}
    if pool:
        results["seen_300"] = averaged(
            model, pool[:4], config, rng, device, 300, args.window
        )
    (run_dir / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print(
        f"\nfinal, environments never trained on:\n"
        f"  300-step accuracy  {unseen_300['accuracy']:.2%}  "
        f"(ceiling {unseen_300['ceiling']:.2%} = revisit "
        f"{unseen_300['revisit']:.2%} + guessing, edge {unseen_300['edge']:.2%})\n"
        f"  100-step accuracy  {unseen_100['accuracy']:.2%}  "
        f"(ceiling {unseen_100['ceiling']:.2%} = revisit "
        f"{unseen_100['revisit']:.2%} + guessing, edge {unseen_100['edge']:.2%})"
    )
    if pool:
        print(
            f"  seen-pool 300-step {results['seen_300']['accuracy']:.2%}  "
            f"<- gap to unseen is what was memorised"
        )

    beats_baseline = unseen_300["accuracy"] > unseen_300["edge"]
    print(
        f"\nM4 {'PASSED' if beats_baseline else 'NOT PASSED'}: zero-shot "
        f"{unseen_300['accuracy']:.2%} vs edge agent {unseen_300['edge']:.2%} "
        f"on environments whose layout the model has never seen"
    )
    if run is not None:
        run.finish()
    return 0 if beats_baseline else 1


if __name__ == "__main__":
    raise SystemExit(main())
