"""Periodic-basis model on the continuous image world.

The symbol world capped this architecture at its revisit rate (~73%),
and that was not a training-time artefact: observations there are
i.i.d. across locations, so at a location never visited there is
genuinely nothing to infer from. Patches are different -- nearby views
are correlated -- so neighbourhood retrieval carries information at
first-visit locations too. That is the reason to run this world.

Scoring follows M7 exactly so the numbers are comparable to its 51.75%:

* candidates within ``overlap_radius`` steps are EXCLUDED from the pool.
  At patch 16px / cell 16px / speed 0.5 a view one step away still shares
  pixels with the answer, so leaving it in scores partial credit as if it
  were knowledge -- and, in training, teaches the model to separate two
  views that are nearly the same image.
* evaluation scores against the WHOLE WALK pool at a fixed 32 walks, so
  the number does not move when batch size does.
* baselines are computed on the same pool: persistence, nearest
  neighbour (appearance-addressed memory -- the one that matters), and
  ``retrieval_oracle``, what retrieval would score with PERFECT
  localisation. Without that last one a low score is unattributable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from periodic_nav.config import Config
from periodic_nav.model import PeriodicNav
from smallcore.image_world import (
    generate_image_walks,
    make_image_environment,
    measure_ambiguity,
)
from smallcore.image_training import (
    _pool_accuracy,
    overlap_radius,
    patch_baselines,
    retrieval_oracle,
)
from smallcore.patches import FixedFeatureEncoder, blocked_mask, info_nce_grouped

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def walks_to_tensors(walks, device):
    velocities = torch.tensor(
        np.stack([w.velocities[:-1] for w in walks]),
        dtype=torch.float32, device=device,
    )
    patches = torch.tensor(
        np.stack([w.patches for w in walks]),
        dtype=torch.float32, device=device,
    )
    return velocities, patches


def embed(encoder, patches):
    """(B, T, P, P) pixels -> (B, T, obs_dim), no gradient (frozen basis)."""
    with torch.no_grad():
        b, t = patches.shape[:2]
        flat = patches.reshape(b * t, *patches.shape[2:])
        return encoder(flat).reshape(b, t, -1)


def run_image_walk(
    model, encoder, velocities, patches, cfg, mask_radius,
    use_gate=True, train=True, group=None, collect=False,
):
    """One batch of walks, in truncation windows.

    Returns train stats; with ``collect`` also every (prediction, target)
    so evaluation can score against the whole-walk pool.
    """
    batch = patches.shape[0]
    steps = velocities.shape[1]
    window = cfg.tbptt_window
    device = velocities.device
    group = group or batch

    all_emb = embed(encoder, patches)

    state = model.initial_state(batch, device)
    state = model.observe(state, all_emb[:, 0])

    loss_sum = gate_sum = 0.0
    hits = seen = 0
    n_windows = max(1, (steps + window - 1) // window)
    kept_pred, kept_target = [], []

    for start in range(0, steps, window):
        stop = min(start + window, steps)
        state = state.detach()

        chunk, state = model.run_chunk(
            state, velocities[:, start:stop],
            all_emb[:, start + 1 : stop + 1], use_gate=use_gate,
        )

        predicted = F.normalize(chunk.logits, dim=-1)
        target = F.normalize(all_emb[:, start + 1 : stop + 1], dim=-1)

        # Group the window so the candidate pool -- and therefore the
        # objective -- does not change with batch size.
        length = stop - start
        n_groups = max(1, batch // group)
        rows = n_groups * group
        pg = predicted[:rows].reshape(n_groups, group * length, -1)
        tg = target[:rows].reshape(n_groups, group * length, -1)

        walk_id = torch.arange(group, device=device).repeat_interleave(length)
        time_id = torch.arange(start, stop, device=device).repeat(group)
        blocked = blocked_mask(walk_id, time_id, mask_radius)

        loss, correct = info_nce_grouped(pg, tg, blocked, 0.07)

        if train:
            (loss / n_windows).backward()

        with torch.no_grad():
            loss_sum += loss.item()
            gate_sum += chunk.gate.mean().item()
            hits += correct.sum().item()
            seen += correct.numel()
            if collect:
                kept_pred.append(predicted.detach())
                kept_target.append(target.detach())

    out = {
        "loss": loss_sum / n_windows,
        "accuracy": hits / max(seen, 1),
        "gate": gate_sum / n_windows,
        "pool": float(group * min(window, steps)),
    }
    if collect:
        out["predicted"] = torch.cat(kept_pred, dim=1)
        out["target"] = torch.cat(kept_target, dim=1)
    return out


@torch.no_grad()
def evaluate(model, encoder, env, cfg, rng, device, length, speed, mask_radius,
             n_walks=32, use_gate=True):
    """Whole-walk pool, fixed size, with the baselines on the same pool."""
    walks = generate_image_walks(env, n_walks, length, rng, speed=speed)
    velocities, patches = walks_to_tensors(walks, device)

    out = run_image_walk(
        model, encoder, velocities, patches, cfg, mask_radius,
        use_gate=use_gate, train=False, group=n_walks, collect=True,
    )
    accuracy = _pool_accuracy(out["predicted"], out["target"], mask_radius)

    base = patch_baselines(patches, velocities, mask_radius)
    result = {
        "accuracy": accuracy,
        "gate": out["gate"],
        "loss": out["loss"],
        "retrieval_oracle": retrieval_oracle(walks, patches, mask_radius),
        "pool": float(n_walks * (length - 1)),
        "chance": 1.0 / (n_walks * (length - 1)),
    }
    result.update({k: float(v) for k, v in base.items()})
    return result


def main():
    p = argparse.ArgumentParser(description="Periodic basis on the image world")
    # world
    p.add_argument("--grid", type=int, default=10)
    p.add_argument("--cell", type=int, default=16)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--n-motifs", type=int, default=16, dest="n_motifs")
    p.add_argument("--motif-sigma", type=float, default=4.0, dest="motif_sigma")
    p.add_argument("--motif-cells", type=int, default=1, dest="motif_cells",
                   help="cells per motif. 1 is the original world, where patch "
                        "correlation dies within 0.16 cells against a 0.5-cell "
                        "step. 4 with --motif-sigma 16 puts it at 0.62 cells")
    p.add_argument("--speed", type=float, default=0.5)
    # model
    p.add_argument("--obs-dim", type=int, default=32, dest="obs_dim")
    p.add_argument("--encoder", default="dct", choices=("dct", "random", "gabor"))
    p.add_argument("--modules", type=int, default=5)
    p.add_argument("--cycles", type=str, default="3.0,4.2,5.9,8.2,11.5",
                   help="module cycles in cells, comma-separated")
    p.add_argument("--key-dim", type=int, default=64, dest="key_dim")
    p.add_argument("--hidden-dim", type=int, default=64, dest="hidden_dim")
    p.add_argument("--learn-increments", action="store_true",
                   dest="learn_increments",
                   help="unfreeze the phase increments (ablation)")
    p.add_argument("--no-gate", action="store_true", dest="no_gate")
    # training
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p.add_argument("--contrastive-group", type=int, default=None,
                   dest="contrastive_group",
                   help="walks per contrastive pool (default: batch size)")
    p.add_argument("--walk-length", type=int, default=200, dest="walk_length")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=500, dest="eval_every")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", type=str, default=None, dest="run_name")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cycles = tuple(float(c) for c in args.cycles.split(","))
    if len(cycles) != args.modules:
        raise ValueError(f"{args.modules} modules but {len(cycles)} cycles")

    env = make_image_environment(
        args.grid, args.grid, args.cell, args.patch, args.n_motifs, rng,
        motif_sigma=args.motif_sigma, motif_cells=args.motif_cells,
    )
    # Held out: same construction, never trained on. The M4 lesson --
    # structure transfers, appearance does not -- is only testable if the
    # evaluation world was never seen.
    held_rng = np.random.default_rng(args.seed + 1000)
    held_out = make_image_environment(
        args.grid, args.grid, args.cell, args.patch, args.n_motifs, held_rng,
        motif_sigma=args.motif_sigma, motif_cells=args.motif_cells,
    )
    ambiguity = measure_ambiguity(held_out, held_rng)
    mask_radius = overlap_radius(args.patch, args.cell, args.speed)

    encoder = FixedFeatureEncoder(args.patch, args.obs_dim, basis=args.encoder)
    encoder.to(device)

    cfg = Config(
        continuous=True, obs_dim=args.obs_dim, patch_size=args.patch,
        n_modules=args.modules, module_cycles=cycles,
        key_dim=args.key_dim, hidden_dim=args.hidden_dim,
        learn_increments=args.learn_increments,
        batch_size=args.batch_size, walk_length=args.walk_length,
        tbptt_window=args.window, lr=args.lr, n_iterations=args.iters,
    )
    model = PeriodicNav(cfg).to(device)
    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    group = args.contrastive_group or args.batch_size

    run_name = args.run_name or time.strftime("pn_image_%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "LATEST").write_text(str(run_dir), encoding="utf-8")

    print(
        f"PERIODIC BASIS on {args.grid}x{args.grid} cells of {args.cell}px, "
        f"{args.n_motifs} motifs, motif_cells={args.motif_cells}\n"
        f"observation: {args.patch}px patch, step {args.speed} cells, "
        f"{args.encoder} basis -> {args.obs_dim}d (frozen)\n"
        f"position: {model.basis.code_dim}d, cycles "
        + ", ".join(f"{c:.1f}" for c in cycles)
        + f" cells ({'LEARNED' if args.learn_increments else 'frozen'})\n"
        f"ambiguity: {ambiguity['cells_per_patch']:.1f} cells per patch\n"
        f"mask radius: {mask_radius} steps  "
        f"train pool: {group * args.window} candidates\n"
        f"parameters: {n_params:,}  device {device}  run {run_dir}\n"
    )

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="smallcore", name=run_name,
                         config={**vars(args), "parameters": n_params})

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=cfg.lr_decay_every, gamma=cfg.lr_decay_factor,
    )
    use_gate = not args.no_gate
    best = -1.0
    began = time.time()

    for iteration in range(1, args.iters + 1):
        walks = generate_image_walks(
            env, args.batch_size, args.walk_length, rng, speed=args.speed,
        )
        velocities, patches = walks_to_tensors(walks, device)

        optimiser.zero_grad()
        stats = run_image_walk(
            model, encoder, velocities, patches, cfg, mask_radius,
            use_gate=use_gate, train=True, group=group,
        )
        optimiser.step()
        scheduler.step()

        if iteration % 50 == 0 or iteration == 1:
            print(
                f"[{iteration:5d}] loss={stats['loss']:.3f}  "
                f"acc={stats['accuracy']:.1%}  gate={stats['gate']:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.1e}  "
                f"({time.time() - began:.0f}s)"
            )
            if run is not None:
                run.log({f"train/{k}": v for k, v in stats.items()
                         if isinstance(v, float)}, step=iteration)

        if iteration % args.eval_every == 0 or iteration == args.iters:
            unseen = evaluate(
                model, encoder, held_out, cfg, np.random.default_rng(99),
                device, 300, args.speed, mask_radius, use_gate=use_gate,
            )
            print(
                f"  iter {iteration:>5}  unseen 300-step "
                f"{unseen['accuracy']:.2%}  "
                f"(retrieval oracle {unseen['retrieval_oracle']:.2%}, "
                f"nn {unseen['nearest_neighbour']:.2%}, "
                f"persistence {unseen['persistence']:.2%}, "
                f"chance {unseen['chance']:.2%})  "
                f"gate {unseen['gate']:.3f}"
            )
            if run is not None:
                run.log({f"unseen/{k}": v for k, v in unseen.items()},
                        step=iteration)
            if unseen["accuracy"] > best:
                best = unseen["accuracy"]
                torch.save(
                    {"model": model.state_dict(), "config": cfg,
                     "iteration": iteration, "metrics": unseen,
                     "args": vars(args)},
                    run_dir / "best.pt",
                )

    torch.save({"model": model.state_dict(), "config": cfg,
                "iteration": args.iters, "args": vars(args)},
               run_dir / "latest.pt")

    final = evaluate(
        model, encoder, held_out, cfg, np.random.default_rng(99),
        device, 300, args.speed, mask_radius, use_gate=use_gate,
    )
    final["ambiguity"] = {k: float(v) for k, v in ambiguity.items()
                          if isinstance(v, (int, float))}
    final["args"] = vars(args)
    final["parameters"] = n_params
    (run_dir / "metrics.json").write_text(json.dumps(final, indent=2),
                                          encoding="utf-8")

    print(
        f"\nfinal, image never trained on "
        f"(pool of {final['pool']:.0f} candidates):\n"
        f"  300-step accuracy   {final['accuracy']:.2%}\n"
        f"  retrieval oracle    {final['retrieval_oracle']:.2%}   "
        f"(nearest remembered patch, PERFECT localisation)\n"
        f"  nearest neighbour   {final['nearest_neighbour']:.2%}   "
        f"(memory addressed by appearance)\n"
        f"  persistence         {final['persistence']:.2%}   "
        f"(predict what you see now)\n"
        f"  chance              {final['chance']:.2%}\n"
        f"  gate                {final['gate']:.3f}\n"
    )
    if run is not None:
        run.log({f"final/{k}": v for k, v in final.items()
                 if isinstance(v, float)})
        run.finish()


if __name__ == "__main__":
    main()
