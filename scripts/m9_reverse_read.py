"""Does the reverse read carry any information about position at all?

    python scripts/m9_reverse_read.py runs/gateA_widened runs/gateB_tiled

The drift gate has sat inside 0.05-0.34 across every arm ever run -- sixteen
of them, spanning encoder, activation, arena size, capacity, place-cell
targets, activity cost, the memory objective switched off entirely, a world
with four times the observation correlation length, and the drift loss term
itself removed. Nothing moves it.

At that point the question stops being "what is suppressing the gate?" and
becomes "is there anything for the gate to listen to?". A gate near zero is the
*correct* answer to a reverse read whose output is noise, and no change to a
loss term can or should override that.

So this measures the reverse read directly rather than inferring it from the
gate. `e_ret` is the position the model retrieves on the strength of what it
can see. If it were informative, the position it decodes to would sit closer to
where the agent actually is than a randomly chosen remembered position does.
That is a property of a trained checkpoint, needing no further training.

Reported as the distance from the true location to the location implied by
`e_ret`, against the same figure for `e_PI` (path integration, the thing it is
supposed to correct) and against chance. This is the M2 retrieval measurement
-- "attention lands 1.20 cells away against 3.59 at chance" -- asked of the
reverse read rather than the forward one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcore.image_world import (  # noqa: E402
    TorchPatchSampler,
    generate_trajectories,
    make_image_environment,
)
from smallcore.recurrent import SmallCoreRecurrent  # noqa: E402


@torch.no_grad()
def collect(model, env, config, device, walks, length, start, seed):
    """Run a walk, capturing what the gate was shown at every step."""
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def hook(_module, inputs, _output):
        # DriftGate.forward(integrated, retrieved, have_memories)
        captured.append((inputs[0].detach(), inputs[1].detach()))

    handle = model.gate.register_forward_hook(hook)
    try:
        sampler = TorchPatchSampler(env, device)
        rng = np.random.default_rng(seed)
        positions, velocities = generate_trajectories(
            env, walks, length, rng, config.speed, start=start
        )
        pos = torch.as_tensor(positions, dtype=torch.float32, device=device)
        patches = sampler(pos)
        vel = torch.as_tensor(
            velocities[:, :-1], dtype=torch.float32, device=device
        )
        state = model.initial_state(walks, device)
        state = model.observe(state, patches[:, 0])
        for begin in range(1, length, config.tbptt_window):
            stop = min(begin + config.tbptt_window, length)
            state = state.detach()
            _, state = model.run_chunk(
                state, vel[:, begin - 1 : stop - 1], patches[:, begin:stop]
            )
    finally:
        handle.remove()

    integrated = torch.stack([c[0] for c in captured], dim=1)
    retrieved = torch.stack([c[1] for c in captured], dim=1)
    steps = integrated.shape[1]
    truth = torch.as_tensor(
        positions[:, 1 : steps + 1], dtype=torch.float32, device=device
    )
    return (
        integrated.reshape(-1, integrated.shape[-1]),
        retrieved.reshape(-1, retrieved.shape[-1]),
        truth.reshape(-1, 2),
    )


def decode_error(codes: torch.Tensor, truth: torch.Tensor) -> float:
    """Median distance, in cells, from true position to a linear decode of it.

    A linear probe is the friendliest reading available: it asks only whether
    the information is *present* in the code, not whether anything downstream
    could use it. If a probe cannot find position here, the gate certainly
    cannot. Fitted on half and scored on the other half, since a probe with
    enough dimensions will otherwise fit noise.
    """
    n = len(codes) // 2
    design = torch.cat([codes, torch.ones(len(codes), 1, device=codes.device)], 1)
    solution = torch.linalg.lstsq(
        design[:n].double(), truth[:n].double()
    ).solution
    predicted = (design[n:].double() @ solution).float()
    return float((predicted - truth[n:]).norm(dim=-1).median())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+")
    p.add_argument("--walks", type=int, default=32)
    p.add_argument("--length", type=int, default=300)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'run':<16} {'e_PI':>8} {'e_ret':>8} {'chance':>8} {'optimal':>9} {'measured':>9}")
    for name in args.runs:
        path = Path(name) / "best.pt"
        if not path.exists():
            print(f"{Path(name).name:<16} no checkpoint")
            continue
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        config = checkpoint["config"]
        model = SmallCoreRecurrent(0, config, seed=0).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        grid, cell = checkpoint["grid"], checkpoint["cell"]
        env = make_image_environment(
            grid, grid, cell, config.patch_size, 16,
            np.random.default_rng(11),
            motif_sigma=getattr(config, "motif_sigma", 4.0),
            motif_cells=getattr(config, "motif_cells", 1),
        )
        integrated, retrieved, truth = collect(
            model, env, config, device, args.walks, args.length,
            (grid / 2.0, grid / 2.0), 500,
        )
        pi_error = decode_error(integrated, truth)
        ret_error = decode_error(retrieved, truth)
        # Chance: the typical distance between two positions in this arena,
        # which is what a code carrying nothing would achieve.
        shuffled = truth[torch.randperm(len(truth), device=device)]
        chance = float((truth - shuffled).norm(dim=-1).median())
        # The weight an optimal linear blend of two independent estimates puts
        # on the noisier one -- the Kalman gain. If the learned gate matches
        # this, the gate is not suppressed, pinned or starved: it is correct,
        # and its smallness is a statement about the reverse read's accuracy
        # rather than a symptom of anything.
        optimal = pi_error**2 / (pi_error**2 + ret_error**2)
        measured = json.loads(
            (Path(name) / "metrics.json").read_text()
        ).get("gate", float("nan"))
        print(f"{Path(name).name:<16} {pi_error:>8.2f} {ret_error:>8.2f} "
              f"{chance:>8.2f} {optimal:>9.3f} {measured:>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
