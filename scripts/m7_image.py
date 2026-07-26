"""M7 / ROADMAP Rung 5: the continuous image walker.

    python scripts/m7_image.py --iters 3000 --batch-size 64

The agent moves continuously across an image and sees only an 8x8 patch around
wherever it is. Everything load-bearing is unchanged -- the recurrent position
stream, the split-source forward read, the reverse read and the drift gate all
run exactly as they did on symbols. What changes is what an observation *is*:

* ``to_value`` becomes a small convnet rather than a lookup off a one-hot, and
  it still serves twice, as forward-read values and reverse-read keys;
* the target becomes contrastive -- pick the true next patch out of a pool of
  real ones -- because a patch has no vocabulary to be cross-entropied over;
* the node and edge agents are replaced by patch-space baselines, persistence
  and nearest-neighbour retrieval.

Why this and not more continuous movement on symbols. M6 reached 53% of ceiling
against the discrete world's 98%, and the suspect was the pairing: continuous
position over a piecewise-constant symbol field demands sub-cell precision while
supplying no sub-cell information, and one-hot similarity is 1 or 0, so the
reverse read cannot tell "nearly the same place" from "somewhere else
entirely". Patches fix both at once. Continuous movement and continuous
observation are one change, not two.

What would count as success: beating **nearest neighbour** clearly. That
baseline is a memory addressed by appearance, which is the alternative this
architecture exists to argue against, and persistence is there to say how much
of any score is merely smooth continuation.
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
from smallcore.image_training import (
    _pool_accuracy,
    patch_baselines,
    overlap_radius,
    retrieval_oracle,
    revisit_rate,
    run_image_walk,
    to_tensors,
)
from smallcore.image_world import (
    crop_environments,
    generate_image_walks,
    load_backdrop,
    make_image_environment,
    measure_ambiguity,
    split_backdrop,
)
from smallcore.recurrent import SmallCoreRecurrent

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"


@torch.no_grad()
def evaluate(model, env, config, rng, device, length, window, speed, n_walks=8):
    """Whole-walk pool accuracy for the model and both patch baselines."""
    model.eval()
    walks = generate_image_walks(env, n_walks, length, rng, speed)
    for w in walks[:2]:
        w.validate(env)
    velocities, patches = to_tensors(walks, device)

    mask_radius = overlap_radius(env.patch, env.cell, speed)

    metrics = run_image_walk(
        model, velocities, patches, config, window, True, train=False,
        mask_radius=mask_radius, collect=True,
    )
    accuracy = _pool_accuracy(
        metrics.pop("predicted"), metrics.pop("target"), mask_radius
    )
    model.train()
    metrics["accuracy"] = accuracy
    metrics.update(patch_baselines(patches, velocities, mask_radius))
    metrics["retrieval_oracle"] = retrieval_oracle(walks, patches, mask_radius)
    metrics["revisit"] = revisit_rate(
        walks, env, radius=env.patch / (2.0 * env.cell), mask_radius=mask_radius
    )
    metrics["mask_radius"] = float(mask_radius)
    return metrics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int, default=20)
    # Chosen by scripts/m7_world_check.py, which measures what a perfectly
    # localised agent would score by retrieval. The first guess (15 cells, an
    # 8px patch, half-cell steps) left that at 1.7%, i.e. memory barely helping
    # however good the position stream -- a world in which a poor result would
    # have been unattributable. These settings put it at 5.5%.
    p.add_argument("--grid", type=int, default=10, help="arena width/height in cells")
    p.add_argument("--cell", type=int, default=16, help="pixels per cell")
    p.add_argument("--patch", type=int, default=16, help="observation window, pixels")
    p.add_argument("--n-motifs", type=int, default=16, dest="n_motifs")
    p.add_argument("--motif-sigma", type=float, default=4.0, dest="motif_sigma")
    # A photograph instead of the synthetic tiling. Worth trying precisely
    # because natural images carry structure at every scale, and the synthetic
    # texture's short correlation length is what pins perfect-localisation
    # retrieval near 5%: on a terrain elevation map two views a third of a cell
    # apart still correlate 0.42, against 0.14 for the synthetic texture. What
    # is given up is control -- ambiguity becomes a property of the picture, so
    # read it off the header rather than assuming it.
    p.add_argument("--image", type=str, default=None, help="backdrop photograph")
    # Half a patch per step, so consecutive views overlap. At a full patch per
    # step they share no pixels at all, and the contrastive loss then sat at
    # exactly ln(pool) for 500 iterations -- with nothing learnable from the
    # current view, the only signal is memory, and memory cannot bootstrap
    # before path integration works. Overlap is what breaks that deadlock.
    p.add_argument("--speed", type=float, default=0.5, help="cells per step")
    p.add_argument("--walk-min", type=int, default=100, dest="walk_min")
    p.add_argument("--walk-max", type=int, default=300, dest="walk_max")
    p.add_argument("--n-envs", type=int, default=8, dest="n_envs")
    p.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p.add_argument("--lr", type=float, default=9.4e-4)
    p.add_argument("--obs-dim", type=int, default=32, dest="obs_dim")
    p.add_argument("--module-dims", type=str, default="6,6,6,6,6", dest="module_dims")
    # Cycles 3.0 / 4.0 / 5.3 / 7.1 / 9.4 cells, all fitting a 10-cell arena and
    # none aliasing against a one-cell step -- the two failures M5 diagnosed.
    p.add_argument(
        "--module-freqs", type=str,
        default="0.667,0.5,0.377,0.282,0.213", dest="module_freqs",
    )
    p.add_argument("--no-gate", action="store_true", dest="no_gate")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--run-name", type=str, default=None, dest="run_name")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dims = tuple(int(d) for d in args.module_dims.split(","))
    freqs = tuple(float(f) for f in args.module_freqs.split(","))
    config = replace(
        Config(), module_dims=dims, module_freqs=freqs,
        observation_mode="patch", patch_size=args.patch, obs_dim=args.obs_dim,
        batch_size=args.batch_size, lr=args.lr, continuous=True, speed=args.speed,
    )

    def build(generator):
        return make_image_environment(
            args.grid, args.grid, args.cell, args.patch, args.n_motifs, generator,
            motif_sigma=args.motif_sigma,
        )

    held_rng = np.random.default_rng(args.seed + 9999)
    if args.image:
        # Disjoint halves of the picture, so a held-out arena shares its
        # statistics with training but not one pixel of its content.
        left, right = split_backdrop(load_backdrop(args.image))
        pool = crop_environments(
            left, args.n_envs, args.grid, args.cell, args.patch, rng
        )
        held_out = crop_environments(
            right, 4, args.grid, args.cell, args.patch, held_rng
        )
    else:
        pool = [build(rng) for _ in range(args.n_envs)]
        held_out = [build(held_rng) for _ in range(4)]
    ambiguity = measure_ambiguity(held_out[0], held_rng)

    run_name = args.run_name or time.strftime("m7_%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "LATEST").write_text(str(run_dir), encoding="utf-8")

    model = SmallCoreRecurrent(0, config, seed=args.seed).to(device)
    n_params = sum(t.numel() for t in model.parameters())
    encoder_params = sum(t.numel() for t in model.to_value.parameters())
    print(
        f"M7 IMAGE WALKER on {args.grid}x{args.grid} cells of {args.cell}px "
        f"({args.grid * args.cell}px square), "
        + (f"backdrop {Path(args.image).name}" if args.image
           else f"{args.n_motifs} synthetic motifs") + "\n"
        f"observation: {args.patch}x{args.patch} patch, bilinear, "
        f"step {args.speed} cells\n"
        "modules: " + ", ".join(f"cycle {2.0 / f:.1f} cells" for f in freqs) + "\n"
        f"ambiguity: {ambiguity['cells_per_patch']:.1f} cells per patch above "
        f"r={ambiguity['threshold']}\n"
        f"parameters: {n_params:,} ({encoder_params:,} in the patch encoder)  "
        f"device {device}  run {run_dir}\n"
    )

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="smallcore", name=run_name,
                         config={**vars(args), "parameters": n_params})

    optimiser = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    use_gate = not args.no_gate
    best = -1.0
    began = time.time()
    train_mask = overlap_radius(args.patch, args.cell, args.speed)

    for iteration in range(1, args.iters + 1):
        lr = max(config.lr_min,
                 config.lr * config.lr_decay ** (iteration // config.lr_decay_every))
        for g in optimiser.param_groups:
            g["lr"] = lr

        env = pool[int(rng.integers(len(pool)))]
        length = int(rng.integers(args.walk_min, args.walk_max + 1))
        walks = generate_image_walks(
            env, config.batch_size, length, rng, args.speed
        )
        velocities, patches = to_tensors(walks, device)

        optimiser.zero_grad()
        stats = run_image_walk(
            model, velocities, patches, config, args.window, use_gate,
            train=True, mask_radius=train_mask,
        )
        optimiser.step()

        if run is not None:
            run.log({f"train/{k}": v for k, v in stats.items()}, step=iteration)

        if iteration % 250 == 0 or iteration == 1:
            results = [
                evaluate(model, e, config, rng, device, 300, args.window, args.speed)
                for e in held_out
            ]
            unseen = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
            print(
                f"  iter {iteration:>5}  loss {stats['loss']:.3f}  "
                f"unseen 300-step {unseen['accuracy']:.2%}  "
                f"(retrieval oracle {unseen['retrieval_oracle']:.2%}, "
                f"nn {unseen['nearest_neighbour']:.2%}, "
                f"chance {unseen['chance']:.2%})  "
                f"gate {stats['gate']:.3f}  ({time.time() - began:.0f}s)"
            )
            if run is not None:
                run.log({f"unseen/{k}": v for k, v in unseen.items()}, step=iteration)
            if unseen["accuracy"] > best:
                best = unseen["accuracy"]
                torch.save({"model": model.state_dict(), "config": config,
                            "iteration": iteration, "metrics": unseen,
                            "grid": args.grid, "cell": args.cell},
                           run_dir / "best.pt")

    results = [
        evaluate(model, e, config, rng, device, 300, args.window, args.speed)
        for e in held_out
    ]
    final = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
    final["ambiguity"] = ambiguity
    final["args"] = vars(args)
    (run_dir / "metrics.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    print(
        f"\nfinal, images never trained on (pool of {final['pool']:.0f} candidates):\n"
        f"  300-step accuracy   {final['accuracy']:.2%}\n"
        f"  retrieval oracle    {final['retrieval_oracle']:.2%}   "
        f"(nearest remembered patch, PERFECT localisation)\n"
        f"  nearest neighbour   {final['nearest_neighbour']:.2%}   "
        f"(memory addressed by appearance + heading)\n"
        f"  persistence         {final['persistence']:.2%}   "
        f"(predict what you see now)\n"
        f"  chance              {final['chance']:.2%}\n"
        f"  ambiguity ceiling   {final['oracle']:.2%}   "
        f"(bound set by patch aliasing)\n"
        f"  revisit rate        {final['revisit']:.2%}   (context, NOT a ceiling)\n"
        f"  mean gate           {final['gate']:.3f}"
    )
    passed = final["accuracy"] > final["nearest_neighbour"]
    print(
        f"\nM7 {'PASSED' if passed else 'NOT PASSED'}: {final['accuracy']:.2%} vs "
        f"nearest neighbour {final['nearest_neighbour']:.2%}. "
        f"Position addressing is worth "
        f"{final['accuracy'] / max(final['nearest_neighbour'], 1e-9):.0f}x "
        f"appearance addressing."
    )
    if run is not None:
        run.finish()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
