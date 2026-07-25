"""Shared training machinery for the recurrent model (M3 onwards).

Kept out of the milestone scripts so M3 and M4 exercise the *same* loop --
otherwise a difference in results could be a difference in the harness.
"""

from __future__ import annotations

import numpy as np
import torch

from smallcore.config import Config
from smallcore.graphs import Walk
from smallcore.recurrent import SmallCoreRecurrent


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


def run_walk(
    model: SmallCoreRecurrent,
    actions: torch.Tensor,
    one_hot: torch.Tensor,
    indices: torch.Tensor,
    config: Config,
    window: int,
    use_gate: bool,
    train: bool,
) -> dict:
    """Run a batch of walks in truncation windows.

    When ``train`` is set, each window is backpropagated as it completes and its
    graph released, so memory stays bounded by the window rather than the walk.
    Gradients accumulate across windows and the caller steps once.
    """
    batch, steps = indices.shape
    state = model.initial_state(batch, indices.device)
    state = model.observe(state, one_hot[:, 0])  # step 0: remember, don't predict

    correct = correct_position = total = 0
    loss_sum = gate_sum = 0.0
    n_windows = max(1, (steps - 1 + window - 1) // window)

    for start in range(1, steps, window):
        stop = min(start + window, steps)
        state = state.detach()

        output, state = model.run_chunk(
            state, actions[:, start - 1 : stop - 1], one_hot[:, start:stop],
            use_gate=use_gate,
        )

        target = indices[:, start:stop]
        flat = target.reshape(-1)
        pred = torch.nn.functional.cross_entropy(
            output.logits.reshape(-1, output.logits.shape[-1]), flat
        )
        pred_position = torch.nn.functional.cross_entropy(
            output.logits_position.reshape(-1, output.logits_position.shape[-1]), flat
        )
        drift = (output.gated - output.integrated).pow(2).mean()
        reg_code = output.gated.pow(2).mean()

        loss = (
            pred
            + config.w_pred_pos * pred_position
            + config.w_drift * drift
            + config.l2_position_code * reg_code
        )
        if train:
            (loss / n_windows).backward()

        with torch.no_grad():
            correct += (output.logits.argmax(-1) == target).sum().item()
            correct_position += (
                (output.logits_position.argmax(-1) == target).sum().item()
            )
            total += target.numel()
            loss_sum += pred.item()
            gate_sum += output.gate.mean().item()

    return {
        "accuracy": correct / max(total, 1),
        "accuracy_position": correct_position / max(total, 1),
        "loss": loss_sum / n_windows,
        "gate": gate_sum / n_windows,
    }
