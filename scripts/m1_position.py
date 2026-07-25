"""M1: the position stream alone -- does path integration work?

Trains the recurrent action-conditioned encoder, with a linear head predicting
the current observation from the position code and nothing else. Then freezes
it and asks the real question with a linear probe: can the agent's true
location be decoded from ``e_t``?

Run from the project root:

    python scripts/m1_position.py

Design note -- walks here start from a FIXED location (the grid centre). The
position stream sees only actions, so from a random unknown start ``e_t`` could
only ever encode displacement; absolute location is unrecoverable in principle
until the reverse read exists (M3). A fixed start makes absolute position equal
to integrated displacement, which is exactly the capability M1 tests. The
observation-prediction loss then forces the code to carry location, because the
same action sequence always lands in the same place.

Acceptance (CLAUDE.md §7): probe decoding well above chance (1/121) on short
walks. The script also reports decoding on 200-step walks -- longer than
anything trained on -- and a modal-location baseline that captures how
predictable location is from the step index alone, since early steps of
fixed-start walks are concentrated near the centre.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcore.analysis import fit_linear_probe, probe_accuracy, probe_predictions
from smallcore.config import Config
from smallcore.graphs import Environment, Walk, generate_walk
from smallcore.position import PositionEncoder
from smallcore.shared_position import SharedPositionEncoder

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / "smallcore" / "envs" / "square_11x11.json"
FIGURE_DIR = ROOT / "figures"

TRAIN_LENGTHS = (25, 80)  # sampled uniformly per batch; short walks are the M1 regime
EVAL_LENGTH_SHORT = 50
EVAL_LENGTH_LONG = 200  # longer than anything trained on


def walk_batch(
    env: Environment,
    batch_size: int,
    length: int,
    rng: np.random.Generator,
    straight_bias: float,
    start: int,
) -> list[Walk]:
    return [
        generate_walk(env, length, rng, straight_bias, start=start)
        for _ in range(batch_size)
    ]


def to_tensors(
    walks: list[Walk], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack same-length walks into (actions, observations, locations)."""
    actions = torch.tensor(
        np.stack([w.actions[:-1] for w in walks]), dtype=torch.long, device=device
    )
    observations = torch.tensor(
        np.stack([w.observations for w in walks]), dtype=torch.long, device=device
    )
    locations = torch.tensor(
        np.stack([w.locations for w in walks]), dtype=torch.long, device=device
    )
    return actions, observations, locations


def train(
    encoder: PositionEncoder,
    head: nn.Linear,
    env: Environment,
    config: Config,
    iters: int,
    rng: np.random.Generator,
    device: torch.device,
    start: int,
) -> None:
    parameters = list(encoder.parameters()) + list(head.parameters())
    optimiser = torch.optim.Adam(
        parameters, lr=config.lr, weight_decay=config.weight_decay
    )

    began = time.time()
    for iteration in range(1, iters + 1):
        lr = max(
            config.lr_min,
            config.lr * config.lr_decay ** (iteration // config.lr_decay_every),
        )
        for group in optimiser.param_groups:
            group["lr"] = lr

        length = int(rng.integers(TRAIN_LENGTHS[0], TRAIN_LENGTHS[1] + 1))
        walks = walk_batch(
            env, config.batch_size, length, rng, config.straight_bias, start
        )
        actions, observations, _ = to_tensors(walks, device)

        codes = encoder(actions)  # (B, T, D)
        logits = head(codes[:, 1:])  # predict obs at steps 1..T-1
        loss_pred = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            observations[:, 1:].reshape(-1),
        )
        loss_code = codes.pow(2).mean()
        loss = loss_pred + config.l2_position_code * loss_code

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if iteration % 250 == 0 or iteration == 1:
            with torch.no_grad():
                accuracy = (
                    (logits.argmax(-1) == observations[:, 1:]).float().mean().item()
                )
            print(
                f"  iter {iteration:>5}  loss {loss_pred.item():.3f}  "
                f"obs acc {accuracy:.2%}  lr {lr:.1e}  "
                f"({time.time() - began:.0f}s)"
            )


@torch.no_grad()
def collect_codes(
    encoder: PositionEncoder,
    env: Environment,
    n_walks: int,
    length: int,
    rng: np.random.Generator,
    straight_bias: float,
    device: torch.device,
    start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Position codes and true locations for steps 1..length-1 of fresh walks.

    Step 0 is excluded: with a fixed start it is a constant. Returns codes
    ``(n_walks, length - 1, D)`` and locations ``(n_walks, length - 1)``.
    """
    walks = walk_batch(env, n_walks, length, rng, straight_bias, start)
    for walk in walks[:8]:
        walk.validate(env)
    actions, _, locations = to_tensors(walks, device)
    codes = encoder(actions)
    return codes[:, 1:], locations[:, 1:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--encoder",
        choices=("independent", "shared"),
        default="independent",
        help="independent: a private transition matrix per module (the M1 baseline). "
        "shared: one core shared by all copies, differentiated only by a scale knob.",
    )
    parser.add_argument(
        "--knob-per-action",
        action="store_true",
        help="shared encoder only: give each copy one knob per action instead of one.",
    )
    parser.add_argument(
        "--uniform-dims",
        action="store_true",
        help="split the code width evenly across modules. The shared encoder is "
        "necessarily uniform, so this is the control that isolates sharing from "
        "the change in module widths.",
    )
    parser.add_argument("--run-name", type=str, default=None, dest="run_name")
    args = parser.parse_args()

    if not ENV_PATH.exists():
        raise SystemExit(f"{ENV_PATH} missing -- run scripts/m0_demo.py first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    config = Config()
    if args.uniform_dims:
        n_modules = len(config.module_dims)
        width = sum(config.module_dims) // n_modules
        config = replace(config, module_dims=(width,) * n_modules)

    env = Environment.from_json(ENV_PATH)
    topology = env.topology
    width = int(topology.coords[:, 0].max()) + 1
    start = topology.n_locations // 2  # grid centre
    chance = 1.0 / topology.n_locations

    if args.encoder == "independent":
        encoder = PositionEncoder(
            topology.n_actions, config.module_dims, config.module_freqs, seed=args.seed
        ).to(device)
    else:
        # Same total width and same starting operators as the baseline; the one
        # variable is that the copies now share a core.
        n_copies = len(config.module_dims)
        encoder = SharedPositionEncoder(
            topology.n_actions,
            n_copies=n_copies,
            copy_dim=config.position_dim // n_copies,
            knob_init=config.module_freqs,
            knob_per_action=args.knob_per_action,
            seed=args.seed,
        ).to(device)
    if encoder.dim != config.position_dim:
        raise SystemExit(
            f"encoder width {encoder.dim} != {config.position_dim}; "
            "module_dims must divide evenly across copies for a fair comparison"
        )

    head = nn.Linear(config.position_dim, env.n_observations).to(device)

    transition_params = sum(
        p.numel()
        for name, p in encoder.named_parameters()
        if name.startswith(("transitions", "generators", "log_knobs"))
    )
    n_params = sum(p.numel() for p in encoder.parameters()) + sum(
        p.numel() for p in head.parameters()
    )
    print(
        f"M1 [{args.encoder}] on {env.name}: {topology.n_locations} locations, "
        f"{topology.n_actions} actions, start fixed at {start}\n"
        f"transition operators: {transition_params:,} parameters\n"
        f"encoder + head:       {n_params:,} parameters, device {device}\n"
    )

    print(f"training {args.iters} iterations (walk lengths {TRAIN_LENGTHS})")
    train(encoder, head, env, config, args.iters, rng, device, start)
    encoder.eval()

    # --- the actual M1 question: decode location from the frozen code ------ #
    probe_codes, probe_locations = collect_codes(
        encoder, env, 64, EVAL_LENGTH_SHORT, rng, config.straight_bias, device, start
    )
    probe = fit_linear_probe(
        probe_codes.reshape(-1, config.position_dim),
        probe_locations.reshape(-1),
        topology.n_locations,
    )

    short_codes, short_locations = collect_codes(
        encoder, env, 64, EVAL_LENGTH_SHORT, rng, config.straight_bias, device, start
    )
    long_codes, long_locations = collect_codes(
        encoder, env, 64, EVAL_LENGTH_LONG, rng, config.straight_bias, device, start
    )

    short_accuracy = probe_accuracy(
        probe, short_codes.reshape(-1, config.position_dim), short_locations.reshape(-1)
    )

    # Per-step accuracy and spatial error on the long walks.
    predictions = probe_predictions(
        probe, long_codes.reshape(-1, config.position_dim)
    ).reshape(long_locations.shape)
    per_step = (predictions == long_locations).float().mean(dim=0).cpu().numpy()

    coords = torch.tensor(topology.coords, dtype=torch.float32, device=device)
    error = (
        (coords[predictions] - coords[long_locations])
        .norm(dim=-1)
        .mean(dim=0)
        .cpu()
        .numpy()
    )

    # Baseline: how well does the step index alone predict location? Early
    # steps of fixed-start walks are concentrated near the centre, so this is
    # the honest floor, not 1/121.
    baseline_codes, baseline_locations = collect_codes(
        encoder, env, 64, EVAL_LENGTH_LONG, rng, config.straight_bias, device, start
    )
    del baseline_codes
    modal = baseline_locations.mode(dim=0).values  # most common location per step
    modal_per_step = (
        (long_locations == modal.unsqueeze(0)).float().mean(dim=0).cpu().numpy()
    )

    late = slice(EVAL_LENGTH_LONG - 51, EVAL_LENGTH_LONG - 1)
    print(
        f"\nlinear probe, held-out walks:\n"
        f"  {EVAL_LENGTH_SHORT}-step walks:          {short_accuracy:.1%}  "
        f"(chance {chance:.1%})\n"
        f"  {EVAL_LENGTH_LONG}-step walks, last 50:  {per_step[late].mean():.1%}  "
        f"(modal-location baseline there {modal_per_step[late].mean():.1%})\n"
        f"  mean decode error, last 50 steps: {error[late].mean():.2f} grid units"
    )

    # --- figure ------------------------------------------------------------ #
    FIGURE_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    steps = np.arange(1, EVAL_LENGTH_LONG)

    axes[0].plot(steps, per_step, label="linear probe", color="tab:blue")
    axes[0].plot(
        steps, modal_per_step, label="modal location", color="tab:orange", alpha=0.8
    )
    axes[0].axhline(chance, color="grey", linestyle=":", label="chance")
    axes[0].axvline(TRAIN_LENGTHS[1], color="grey", linestyle="--", alpha=0.5)
    axes[0].text(
        TRAIN_LENGTHS[1] + 3, 0.9, "longest\ntraining walk", fontsize=8, color="grey"
    )
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("location decoding accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="center right", fontsize=9)
    axes[0].set_title("Decoding true location from the position code")

    axes[1].plot(steps, error, color="tab:blue")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("mean decode error (grid units)")
    axes[1].axvline(TRAIN_LENGTHS[1], color="grey", linestyle="--", alpha=0.5)
    axes[1].set_ylim(bottom=0)
    axes[1].set_title(f"Drift on {EVAL_LENGTH_LONG}-step walks ({width}x{width} grid)")

    fig.suptitle(
        f"M1 [{args.encoder}]: path integration -- position decoded from actions alone"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    tag = f"{args.encoder}{'_uniform' if args.uniform_dims else ''}"

    # Save the encoder. The transition operators are the *structural* part of
    # what this run learned -- how movements compose -- and that is independent
    # of where a walk started, so they transfer to settings where M1's own
    # objective would not train at all (see scripts/m3_train.py --init-from).
    run_dir = ROOT / "runs" / (args.run_name or f"m1_{tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "position": encoder.state_dict(),
            "module_dims": config.module_dims,
            "module_freqs": config.module_freqs,
            "encoder": args.encoder,
            "iterations": args.iters,
            "probe_accuracy_50": short_accuracy,
        },
        run_dir / "position.pt",
    )
    print(f"\nwrote {run_dir / 'position.pt'}")

    figure_path = FIGURE_DIR / f"m1_probe_{tag}.png"
    fig.savefig(figure_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {figure_path}")

    # What the knobs settled on is the whole point of the shared run: if the
    # copies stayed at distinct scales, one core did genuinely specialise.
    if isinstance(encoder, SharedPositionEncoder):
        with torch.no_grad():
            learned = encoder.knobs.cpu().numpy()
        print(
            f"\nscale knobs: init {np.array(config.module_freqs).round(3).tolist()}\n"
            f"             learned {learned.round(3).tolist()}"
        )

    passed = short_accuracy > 10 * chance
    print(
        f"\nM1 [{args.encoder}] {'PASSED' if passed else 'NOT PASSED'}: short-walk "
        f"decoding {short_accuracy:.1%} vs chance {chance:.1%}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
