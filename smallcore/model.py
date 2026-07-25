"""SmallCore: the position stream, the memory stream, and the readout.

At M2 the model is the forward half of the loop:

    actions --> position stream --> e_t --> query memory --> y_t --> logits
                                                  ^
                                    observations -+ (as values only)

The reverse read and the drift gate (M3) attach to the same position codes and
are what stop `e_t` drifting on long walks; until then the position estimate is
pure path integration and will degrade with walk length by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from smallcore.config import Config
from smallcore.memory import ForwardRead
from smallcore.position import PositionEncoder
from smallcore.readout import Readout


@dataclass
class ModelOutput:
    """Everything a loss or an analysis pass might want from one forward."""

    logits: torch.Tensor  # (B, T, V) full-model observation prediction
    logits_position: torch.Tensor  # (B, T, V) from the position code alone
    codes: torch.Tensor  # (B, T, D) position codes
    attention: torch.Tensor  # (B, T, T) memory read weights


class SmallCore(nn.Module):
    """Ties the streams together.

    Args:
        n_actions: Number of movements in the topology.
        config: Hyperparameters.
        seed: Seed for the position stream's operator initialisation.
    """

    def __init__(self, n_actions: int, config: Config, seed: int = 0) -> None:
        super().__init__()
        self.config = config
        self.position = PositionEncoder(
            n_actions, config.module_dims, config.module_freqs, seed=seed
        )
        self.memory = ForwardRead(
            position_dim=config.position_dim,
            n_observations=config.n_observations,
            key_dim=config.key_dim,
            obs_dim=config.obs_dim,
        )
        self.readout = Readout(
            position_dim=config.position_dim,
            obs_dim=config.obs_dim,
            n_observations=config.n_observations,
            model_dim=config.model_dim,
            hidden_dim=config.hidden_dim,
        )

    def forward(
        self, actions: torch.Tensor, observations: torch.Tensor
    ) -> ModelOutput:
        """Run the model over a batch of walks.

        Args:
            actions: ``(B, T - 1)`` actions actually taken. Pass
                ``walk.actions[:-1]``; ``NO_ACTION`` must not appear.
            observations: ``(B, T, V)`` one-hot observations.

        Returns:
            A :class:`ModelOutput`. Prediction at step ``t`` uses only
            information from steps before ``t``, so every returned logit is a
            genuine forecast.
        """
        codes = self.position(actions)
        retrieved, attention = self.memory(codes, observations)
        logits, logits_position = self.readout(retrieved, codes)
        return ModelOutput(
            logits=logits,
            logits_position=logits_position,
            codes=codes,
            attention=attention,
        )
