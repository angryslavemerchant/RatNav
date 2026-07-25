"""M3: reverse read + drift gate.

    python scripts/m3_train.py
    python scripts/m3_train.py --no-gate      # ablation: M2's forward read only
    python scripts/m3_train.py --wandb --run-name m3_baseline

Passes when 300-step accuracy stops degrading relative to short walks. M2's
measured drop was 88.9% at 100 steps to 81.2% at 300, with nothing to
re-anchor a position estimate that accumulates error every step. Closing that
gap is the whole milestone -- a model that is merely accurate on short walks
has not demonstrated the correction loop.

Trained with truncated backprop: the memory cache spans the entire walk while
gradients flow only through the most recent window (``--window``, default 20).
See ``smallcore.recurrent`` for how the two blocks are kept apart.

``--no-gate`` holds everything else fixed and disables the correction, which
isolates what the reverse read actually buys.
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
from smallcore.graphs import Environment, generate_batch_fast
from smallcore.recurrent import SmallCoreRecurrent
from smallcore.training import run_walk, to_tensors

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / "smallcore" / "envs" / "square_11x11.json"
RUNS_DIR = ROOT / "runs"


@torch.no_grad()
def evaluate(model, env, config, rng, device, length, window, use_gate, n_walks=32):
    model.eval()
    walks = generate_batch_fast(env, n_walks, length, rng, config.straight_bias)
    for walk in walks[:4]:
        walk.validate(env)
    actions, one_hot, indices = to_tensors(walks, env.n_observations, device)
    metrics = run_walk(
        model, actions, one_hot, indices, config, window, use_gate, train=False
    )
    fitting = generate_batch_fast(env, 256, length, rng, config.straight_bias)
    model.train()
    metrics.update(
        node=NodeAgent(env.n_observations).fit(fitting).accuracy(walks),
        edge=EdgeAgent(env.n_observations, env.topology.n_actions)
        .fit(fitting)
        .accuracy(walks),
        revisit=oracle_memory_accuracy(walks),
        chance=chance_accuracy(walks, env.n_observations),
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--walk-min", type=int, default=50, dest="walk_min")
    parser.add_argument("--walk-max", type=int, default=150, dest="walk_max")
    parser.add_argument(
        "--no-gate", action="store_true", dest="no_gate",
        help="ablation: disable the drift correction, keeping everything else",
    )
    parser.add_argument(
        "--init-from", type=str, default=None, dest="init_from",
        help="path to an M1 position.pt whose transition operators seed this "
        "run. What M1 learned is how movements compose, which is a property of "
        "the topology and not of where a walk began -- so it transfers into "
        "the random-start setting even though M1's own objective cannot train "
        "there.",
    )
    parser.add_argument(
        "--freeze-position", action="store_true", dest="freeze_position",
        help="hold the transferred operators fixed and train only the memory, "
        "gate and readout around them. The N=1 case of reusing a frozen "
        "navigation primitive.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name", type=str, default=None, dest="run_name")
    args = parser.parse_args()

    if not ENV_PATH.exists():
        raise SystemExit(f"{ENV_PATH} missing -- run scripts/m0_demo.py first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    config = Config()
    env = Environment.from_json(ENV_PATH)
    use_gate = not args.no_gate

    run_name = args.run_name or time.strftime("m3_%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "LATEST").write_text(str(run_dir), encoding="utf-8")

    model = SmallCoreRecurrent(env.topology.n_actions, config, seed=args.seed).to(device)

    origin = "scratch"
    if args.init_from:
        checkpoint = torch.load(
            ROOT / args.init_from, map_location=device, weights_only=False
        )
        if tuple(checkpoint["module_dims"]) != tuple(config.module_dims):
            raise SystemExit(
                f"checkpoint module_dims {checkpoint['module_dims']} != "
                f"config {config.module_dims}; the operators would not align"
            )
        model.position.load_state_dict(checkpoint["position"])
        origin = f"{args.init_from} (M1 probe {checkpoint['probe_accuracy_50']:.1%})"
    if args.freeze_position:
        if not args.init_from:
            raise SystemExit("--freeze-position without --init-from freezes noise")
        for parameter in model.position.parameters():
            parameter.requires_grad_(False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in trainable)

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            project="smallcore", name=run_name,
            config={**vars(args), "parameters": n_params,
                    "trainable": n_trainable},
        )

    frozen_note = ", FROZEN" if args.freeze_position else ""
    print(
        f"M3 on {env.name}: drift gate {'ON' if use_gate else 'OFF (ablation)'}, "
        f"tbptt window {args.window}\n"
        f"position stream: {origin}{frozen_note}\n"
        f"parameters: {n_params:,} ({n_trainable:,} trainable)  "
        f"device {device}  run {run_dir}\n"
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

        length = int(rng.integers(args.walk_min, args.walk_max + 1))
        walks = generate_batch_fast(
            env, config.batch_size, length, rng, config.straight_bias
        )
        actions, one_hot, indices = to_tensors(walks, env.n_observations, device)

        optimiser.zero_grad()
        stats = run_walk(
            model, actions, one_hot, indices, config, args.window, use_gate, train=True
        )
        optimiser.step()

        if run is not None:
            run.log({f"train/{k}": v for k, v in stats.items()}, step=iteration)

        if iteration % 250 == 0 or iteration == 1:
            short = evaluate(model, env, config, rng, device, 50, args.window, use_gate)
            long = evaluate(model, env, config, rng, device, 300, args.window, use_gate)
            gap = short["accuracy"] - long["accuracy"]
            print(
                f"  iter {iteration:>5}  loss {stats['loss']:.3f}  "
                f"50-step {short['accuracy']:.2%}  300-step {long['accuracy']:.2%}  "
                f"gap {gap:+.2%}  gate {stats['gate']:.3f}  "
                f"({time.time() - began:.0f}s)"
            )
            if run is not None:
                run.log(
                    {
                        **{f"eval50/{k}": v for k, v in short.items()},
                        **{f"eval300/{k}": v for k, v in long.items()},
                        "eval/gap": gap,
                    },
                    step=iteration,
                )
            if long["accuracy"] > best:
                best = long["accuracy"]
                torch.save(
                    {"model": model.state_dict(), "config": config,
                     "iteration": iteration, "metrics": {"short": short, "long": long}},
                    run_dir / "best.pt",
                )

    short = evaluate(model, env, config, rng, device, 50, args.window, use_gate)
    long = evaluate(model, env, config, rng, device, 300, args.window, use_gate)
    (run_dir / "metrics.json").write_text(
        json.dumps({"short_50": short, "long_300": long}, indent=2), encoding="utf-8"
    )

    gap = short["accuracy"] - long["accuracy"]
    print(
        f"\nfinal:\n"
        f"   50-step accuracy  {short['accuracy']:.2%}  (edge {short['edge']:.2%})\n"
        f"  300-step accuracy  {long['accuracy']:.2%}  (edge {long['edge']:.2%})\n"
        f"  degradation        {gap:+.2%}   <- M2 measured +7.75%\n"
        f"  mean gate          {long['gate']:.3f}\n"
    )
    # Both conditions matter: a gap near zero is trivially satisfied by a model
    # that is uniformly bad, so accuracy has to clear the baseline as well.
    holds = gap < 0.03
    useful = long["accuracy"] > long["edge"]
    reasons = []
    if not holds:
        reasons.append(f"300-step still degrades by {gap:.2%}")
    if not useful:
        reasons.append(
            f"300-step accuracy {long['accuracy']:.2%} does not beat the "
            f"edge agent at {long['edge']:.2%}"
        )
    print(
        f"M3 {'PASSED' if holds and useful else 'NOT PASSED'}"
        + ("" if holds and useful else ": " + "; ".join(reasons))
    )
    passed = holds and useful
    if run is not None:
        run.finish()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
