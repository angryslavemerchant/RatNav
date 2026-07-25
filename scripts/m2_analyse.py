"""Is the memory actually addressed by position?

Accuracy says the model works. It does not say *why*, and the architecture's
central claim is specifically about the addressing scheme: keys are positions,
so a query should retrieve the steps where the agent stood in the same place --
not steps that merely looked similar.

That is directly measurable, because ``walk.locations`` is available to
analysis even though the model never sees it. For every step, this asks where
the attention mass went:

* **same-location mass** -- fraction landing on earlier steps at the *same*
  location, against the chance rate for that step (how much of the accessible
  past was at that location anyway). A ratio near 1 means the memory is not
  position-addressed at all; well above 1 means it is.
* **retrieval distance** -- attention-weighted grid distance between where the
  agent is and where the retrieved memories were laid down. Near zero means
  retrieval is local.
* **same-observation mass** -- the control. Observations are ambiguous, so a
  content-addressed memory would concentrate here instead. If same-location
  mass greatly exceeds same-observation mass, keys are carrying addresses
  rather than appearance, which is exactly the design's load-bearing claim.

    python scripts/m2_analyse.py --run runs/m2_random_start
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcore.config import Config
from smallcore.graphs import Environment, generate_batch_fast
from smallcore.model import SmallCore

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / "smallcore" / "envs" / "square_11x11.json"


@torch.no_grad()
def measure(
    model: SmallCore,
    env: Environment,
    config: Config,
    device: torch.device,
    length: int,
    n_walks: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    walks = generate_batch_fast(env, n_walks, length, rng, config.straight_bias)
    for walk in walks[:8]:
        walk.validate(env)

    actions = torch.tensor(
        np.stack([w.actions[:-1] for w in walks]), dtype=torch.long, device=device
    )
    indices = torch.tensor(
        np.stack([w.observations for w in walks]), dtype=torch.long, device=device
    )
    locations = torch.tensor(
        np.stack([w.locations for w in walks]), dtype=torch.long, device=device
    )
    one_hot = torch.nn.functional.one_hot(indices, env.n_observations).float()

    output = model(actions, one_hot)
    attention = output.attention  # (B, T, T), row t supported on s < t

    # [b, t, s] = "was step s in the same place as step t"
    same_place = locations[:, :, None] == locations[:, None, :]
    same_symbol = indices[:, :, None] == indices[:, None, :]

    steps = attention.shape[1]
    causal = torch.ones(steps, steps, dtype=torch.bool, device=device).tril(-1)
    accessible = causal.view(1, steps, steps).expand_as(same_place)
    n_accessible = accessible.sum(-1).clamp(min=1).float()

    place_mass = (attention * (same_place & accessible)).sum(-1)
    symbol_mass = (attention * (same_symbol & accessible)).sum(-1)
    place_chance = (same_place & accessible).sum(-1).float() / n_accessible
    symbol_chance = (same_symbol & accessible).sum(-1).float() / n_accessible

    coords = torch.tensor(env.topology.coords, dtype=torch.float32, device=device)
    here = coords[locations]  # (B, T, 2)
    there = coords[locations]  # attended-to positions, same lookup
    distance = torch.cdist(here, there)  # (B, T, T) grid distance
    retrieval_distance = (attention * distance).sum(-1)
    chance_distance = (
        distance * accessible
    ).sum(-1) / n_accessible

    # Step 0 has no memory; its attention row is all zeros by construction.
    valid = slice(1, steps)
    top1 = attention[:, valid].argmax(-1)
    top1_same_place = torch.gather(
        same_place[:, valid], 2, top1.unsqueeze(-1)
    ).float().mean()

    return {
        "same_location_mass": place_mass[:, valid].mean().item(),
        "same_location_chance": place_chance[:, valid].mean().item(),
        "same_symbol_mass": symbol_mass[:, valid].mean().item(),
        "same_symbol_chance": symbol_chance[:, valid].mean().item(),
        "retrieval_distance": retrieval_distance[:, valid].mean().item(),
        "chance_distance": chance_distance[:, valid].mean().item(),
        "top1_same_location": top1_same_place.item(),
        "revisit_rate": place_chance[:, valid].gt(0).float().mean().item(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=str, default="runs/m2_random_start")
    parser.add_argument("--lengths", type=int, nargs="+", default=[100, 300])
    parser.add_argument("--walks", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    checkpoint_path = ROOT / args.run / "best.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"{checkpoint_path} missing -- train first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Environment.from_json(ENV_PATH)
    # Our own checkpoint, containing a Config dataclass alongside the weights.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", Config())

    model = SmallCore(env.topology.n_actions, config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(
        f"memory addressing, {args.run} "
        f"(checkpoint from iteration {checkpoint.get('iteration', '?')})\n"
    )
    for length in args.lengths:
        m = measure(model, env, config, device, length, args.walks, args.seed)
        place_ratio = m["same_location_mass"] / max(m["same_location_chance"], 1e-9)
        symbol_ratio = m["same_symbol_mass"] / max(m["same_symbol_chance"], 1e-9)
        print(
            f"{length}-step walks:\n"
            f"  attention on same LOCATION   {m['same_location_mass']:.1%}  "
            f"(chance {m['same_location_chance']:.1%}, "
            f"{place_ratio:.1f}x)\n"
            f"  attention on same SYMBOL     {m['same_symbol_mass']:.1%}  "
            f"(chance {m['same_symbol_chance']:.1%}, "
            f"{symbol_ratio:.1f}x)\n"
            f"  top-1 memory is same place   {m['top1_same_location']:.1%}\n"
            f"  retrieval distance           {m['retrieval_distance']:.2f} cells  "
            f"(chance {m['chance_distance']:.2f})\n"
        )
    print(
        "Position-addressed memory shows a large same-LOCATION ratio with the\n"
        "same-SYMBOL ratio near 1: retrieval keyed on where, not on what."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
