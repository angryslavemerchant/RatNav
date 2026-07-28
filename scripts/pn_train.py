"""Train the periodic-basis navigation model.

Usage:
    python scripts/pn_train.py [--level standard|1|2|3|4] [--n-iter 10000]

The default environment is ``standard`` (11×11 grid, 45 aliased
observations), matching SmallCore M3 for direct comparison.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from periodic_nav.basis import PeriodicBasis
from periodic_nav.config import Config
from periodic_nav.model import PeriodicNav, State
from periodic_nav.envs import (
    make_standard,
    make_level1,
    make_level2,
    make_level3,
    make_level4,
)
from smallcore.graphs import generate_batch_fast
from smallcore.baselines import NodeAgent, EdgeAgent


# ── helpers ────────────────────────────────────────────────────────


def walks_to_tensors(walks, device):
    actions = torch.tensor(
        np.stack([w.actions[:-1] for w in walks]), dtype=torch.long, device=device,
    )
    obs = torch.tensor(
        np.stack([w.observations for w in walks]), dtype=torch.long, device=device,
    )
    return actions, obs


def run_walk(model, actions, obs, cfg, use_gate=True, train=True):
    """Process one batch of walks through truncation windows."""
    B, steps_plus_1 = obs.shape
    steps = actions.shape[1]
    window = cfg.tbptt_window
    device = actions.device

    state = model.initial_state(B, device)
    state = model.observe(state, obs[:, 0])

    correct = correct_pos = total = 0
    loss_sum = gate_sum = 0.0
    n_windows = max(1, (steps + window - 1) // window)

    for start in range(0, steps, window):
        stop = min(start + window, steps)
        state = state.detach()

        chunk, state = model.run_chunk(
            state,
            actions[:, start:stop],
            obs[:, start + 1 : stop + 1],
            use_gate=use_gate,
        )

        target = obs[:, start + 1 : stop + 1]
        flat_target = target.reshape(-1)
        n_obs = chunk.logits.shape[-1]

        loss_pred = F.cross_entropy(chunk.logits.reshape(-1, n_obs), flat_target)
        loss_pos = F.cross_entropy(chunk.logits_pos.reshape(-1, n_obs), flat_target)
        loss_drift = (chunk.codes_corrected - chunk.codes_pi).pow(2).mean()

        loss = loss_pred + cfg.w_pred_pos * loss_pos + cfg.w_drift * loss_drift
        if train:
            (loss / n_windows).backward()

        with torch.no_grad():
            correct += (chunk.logits.argmax(-1) == target).sum().item()
            correct_pos += (chunk.logits_pos.argmax(-1) == target).sum().item()
            total += target.numel()
            loss_sum += loss_pred.item()
            gate_sum += chunk.gate.mean().item()

    return {
        "accuracy": correct / max(total, 1),
        "accuracy_pos": correct_pos / max(total, 1),
        "loss": loss_sum / n_windows,
        "gate": gate_sum / n_windows,
    }


def evaluate(model, env, cfg, rng, device):
    model.eval()
    results = {}
    with torch.no_grad():
        for length in cfg.eval_lengths:
            walks = generate_batch_fast(
                env, cfg.eval_walks, length, rng, straight_bias=cfg.straight_bias,
            )
            actions, obs = walks_to_tensors(walks, device)
            r = run_walk(model, actions, obs, cfg, use_gate=True, train=False)
            results[length] = r
    model.train()
    return results


def run_baselines(env, rng, n_walks=64, length=300, straight_bias=2.0):
    walks = generate_batch_fast(env, n_walks, length, rng, straight_bias=straight_bias)
    node = NodeAgent(env.n_observations).fit(walks)
    edge = EdgeAgent(env.n_observations, env.topology.n_actions).fit(walks)
    return {
        "node": node.accuracy(walks),
        "edge": edge.accuracy(walks),
    }


# ── main ───────────────────────────────────────────────────────────


def build_env(level, cfg, rng):
    if level == "standard":
        return make_standard(cfg.grid_width, cfg.grid_height, cfg.n_observations, rng)
    elif level == "1":
        return make_level1(cfg.grid_width, cfg.grid_height)
    elif level == "2":
        return make_level2(rng=rng)
    elif level == "3":
        return make_level3(rng=rng)
    elif level == "4":
        return make_level4(rng=rng)
    else:
        raise ValueError(f"unknown level: {level}")


def main():
    parser = argparse.ArgumentParser(description="Train periodic-basis nav model")
    parser.add_argument("--level", default="standard",
                        choices=["standard", "1", "2", "3", "4"])
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--learn-increments", action="store_true")
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Adjust config for environment level
    overrides = {}
    if args.n_iter is not None:
        overrides["n_iterations"] = args.n_iter
    if args.eval_every is not None:
        overrides["eval_every"] = args.eval_every
    if args.learn_increments:
        overrides["learn_increments"] = True

    if args.level == "1":
        n_loc = 11 * 11
        overrides["n_observations"] = n_loc
    elif args.level in ("2", "3", "4"):
        overrides["n_observations"] = 8

    cfg = Config(**overrides)

    # Build environment
    env = build_env(args.level, cfg, rng)
    print(f"Environment: level={args.level}  "
          f"locations={env.topology.n_locations}  "
          f"observations={env.n_observations}")

    # Build model
    model = PeriodicNav(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params} total, {n_trainable} trainable")
    print(f"Position code: {model.basis.code_dim}d "
          f"({cfg.n_modules} modules × {cfg.orient_per_module} orientations × 2)")
    print(f"Gate: {'enabled' if not args.no_gate else 'disabled'}")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.lr_decay_every, gamma=cfg.lr_decay_factor,
    )

    # Baselines
    print("Computing baselines...")
    bl = run_baselines(env, np.random.default_rng(0), n_walks=64, length=300,
                       straight_bias=cfg.straight_bias)
    print(f"  node agent: {bl['node']:.1%}   edge agent: {bl['edge']:.1%}")
    print()

    # Training loop
    use_gate = not args.no_gate
    best_acc = 0.0
    t0 = time.time()

    for iteration in range(cfg.n_iterations):
        walks = generate_batch_fast(
            env, cfg.batch_size, cfg.walk_length, rng,
            straight_bias=cfg.straight_bias,
        )
        actions, obs = walks_to_tensors(walks, device)

        optimizer.zero_grad()
        r = run_walk(model, actions, obs, cfg, use_gate=use_gate, train=True)
        optimizer.step()
        scheduler.step()

        if iteration % 100 == 0:
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            print(
                f"[{iteration:5d}] loss={r['loss']:.4f}  "
                f"acc={r['accuracy']:.1%}  pos={r['accuracy_pos']:.1%}  "
                f"gate={r['gate']:.4f}  lr={lr:.1e}  "
                f"({elapsed:.0f}s)"
            )

        if iteration > 0 and iteration % cfg.eval_every == 0:
            print("\n--- evaluation ---")
            results = evaluate(model, env, cfg, np.random.default_rng(99), device)
            for length, er in sorted(results.items()):
                tag = ""
                if er["accuracy"] > best_acc:
                    best_acc = er["accuracy"]
                    tag = " *best*"
                print(
                    f"  {length:3d}-step: acc={er['accuracy']:.1%}  "
                    f"pos={er['accuracy_pos']:.1%}  gate={er['gate']:.4f}{tag}"
                )
            print(f"  baselines: node={bl['node']:.1%}  edge={bl['edge']:.1%}")
            print()

    # Final evaluation
    print("\n=== final evaluation ===")
    results = evaluate(model, env, cfg, np.random.default_rng(99), device)
    for length, er in sorted(results.items()):
        print(
            f"  {length:3d}-step: acc={er['accuracy']:.1%}  "
            f"pos={er['accuracy_pos']:.1%}  gate={er['gate']:.4f}"
        )
    print(f"  baselines: node={bl['node']:.1%}  edge={bl['edge']:.1%}")


if __name__ == "__main__":
    main()
