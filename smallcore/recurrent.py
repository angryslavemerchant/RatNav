"""M3: the full correction loop, run step by step.

M2's forward read could be one big masked matrix product because every position
code was known before any memory was read. M3 cannot: the corrected position at
step t depends on a memory read, and that corrected code is what later steps
read *from*. The loop is genuinely sequential, and each step goes:

    1. path-integrate            e_PI = step(e_{t-1}, a_{t-1})
    2. read memory, predict      query with e_PI  ->  what symbol is here?
    3. observe the symbol
    4. reverse-read and gate     e_t = blend(e_PI, retrieved position)
    5. write (e_t, x_t)

The ordering is the causal contract. Prediction happens *before* the symbol is
revealed; the symbol is only used afterwards, to correct position for the steps
that follow. So the forward read is always queried with the uncorrected
estimate, and the correction never leaks into its own prediction.

Truncated backprop (the brief's gotcha 1)
-----------------------------------------
The cache *is* the memory and must span the whole walk, but gradients should
only flow through a short recent window. So the state carries two blocks:

* ``past_*``  -- **detached** entries from earlier windows. Still fully
  readable, contribute no gradient.
* the current window's entries -- grad-attached, built inside the chunk.

Getting this wrong either explodes memory (graph spans 300 steps) or silently
kills long-range credit assignment (cache truncated with the graph). Keeping
them as separate blocks makes the distinction structural rather than a
convention someone has to remember.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from smallcore.config import Config
from smallcore.continuous import ContinuousPositionEncoder
from smallcore.drift import DriftGate, attend
from smallcore.patches import (
    FixedFeatureEncoder,
    LinearPatchEncoder,
    PatchEncoder,
)
from smallcore.place import PlaceHead
from smallcore.position import PositionEncoder
from smallcore.readout import Readout


@dataclass
class RecurrentState:
    """Everything carried between truncation windows."""

    code: torch.Tensor  # (B, D) current gated position
    past_codes: torch.Tensor  # (B, n, D) memory keys source, detached
    # Raw observations, or their embeddings when config.cache_projected_values
    # is set. Both are detached; they differ in whether the encoder still gets
    # gradient from re-projecting history (see the config note).
    past_obs: torch.Tensor

    def detach(self) -> "RecurrentState":
        return RecurrentState(
            self.code.detach(), self.past_codes.detach(), self.past_obs.detach()
        )

    @property
    def size(self) -> int:
        return self.past_codes.shape[1]


@dataclass
class ChunkOutput:
    # In the symbol world these two are logits over the vocabulary. In the
    # patch world they are predicted *embeddings*, scored against ``values``.
    logits: torch.Tensor  # (B, L, out_dim)
    logits_position: torch.Tensor  # (B, L, out_dim)
    integrated: torch.Tensor  # (B, L, D) pre-gate codes
    gated: torch.Tensor  # (B, L, D) post-gate codes
    gate: torch.Tensor  # (B, L, D) gate values, for analysis
    retrieved: torch.Tensor  # (B, L, obs_dim) memory-stream output, for M5
    # Encoded observations, i.e. what was written into memory as values. The
    # contrastive target, and free here -- the loop already computed them.
    values: torch.Tensor  # (B, L, obs_dim)


class SmallCoreRecurrent(nn.Module):
    """Position stream + forward read + reverse read + drift gate."""

    def __init__(self, n_actions: int, config: Config, seed: int = 0) -> None:
        super().__init__()
        self.config = config
        # Velocity-conditioned vs action-indexed path integration. Both expose
        # the same step(state, movement) interface, so nothing downstream --
        # memory, gate, readout -- needs to know which is in use.
        if config.continuous:
            self.position = ContinuousPositionEncoder(
                config.module_dims, config.module_freqs, seed=seed,
                activation=config.position_activation,
            )
        else:
            self.position = PositionEncoder(
                n_actions, config.module_dims, config.module_freqs, seed=seed
            )
        # W_e and W_x, shared by both reads. The forward read uses W_e for
        # queries/keys and W_x for values; the reverse read uses W_x for
        # queries/keys and the raw codes as values.
        # With nonlinear_key the head can convert a periodic position code into
        # place-like keys, which frees the recurrent state to be periodic. Note
        # this wants key_dim >= position_dim: grid codes are compact, place
        # codes are sparse and need more units, so compressing is the wrong
        # direction for this transform.
        if config.nonlinear_key:
            self.to_key = nn.Sequential(
                nn.Linear(config.position_dim, config.key_dim),
                nn.ReLU(),
            )
        else:
            self.to_key = nn.Linear(
                config.position_dim, config.key_dim, bias=False
            )
        # W_x. A lookup table off a one-hot in the symbol world, a convnet over
        # pixels in the patch world -- and in both cases it serves twice, as the
        # forward read's values and the reverse read's keys, so nothing else in
        # the loop needs to know which world it is in.
        if config.observation_mode == "patch":
            self.obs_shape: tuple[int, ...] = (config.patch_size, config.patch_size)
            if config.patch_encoder in ("dct", "random", "gabor", "pca"):
                # Frozen: no trainable parameters, so the contrastive loss can
                # only be lowered by localising better. Call fit_whitening once
                # with real patches before training (train.py does).
                self.to_value: nn.Module = FixedFeatureEncoder(
                    config.patch_size, config.obs_dim,
                    basis=config.patch_encoder, seed=seed,
                )
            elif config.patch_encoder == "linear":
                self.to_value = LinearPatchEncoder(
                    config.patch_size, config.obs_dim
                )
            else:
                self.to_value = PatchEncoder(config.patch_size, config.obs_dim)
        else:
            self.obs_shape = (config.n_observations,)
            self.to_value = nn.Linear(
                config.n_observations, config.obs_dim, bias=False
            )
        # Optional spatial supervision (ROADMAP Rung 0c). Linear by design --
        # a deep head would relieve the recurrent state of the pressure that is
        # the whole point of the experiment.
        self.place = (
            PlaceHead(config.position_dim, config.n_place_cells)
            if config.w_place > 0 else None
        )
        self.gate = DriftGate(config.position_dim, config.hidden_dim)
        self.readout = Readout(
            position_dim=config.position_dim,
            obs_dim=config.obs_dim,
            n_observations=config.n_observations,
            model_dim=config.model_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.readout_dim,
        )

    def initial_state(
        self, batch_size: int, device: torch.device
    ) -> RecurrentState:
        code = self.position.initial_state(batch_size)
        empty_codes = code.new_zeros(batch_size, 0, self.config.position_dim)
        shape = (
            (self.config.obs_dim,) if self.config.cache_projected_values
            else self.obs_shape
        )
        empty_obs = code.new_zeros(batch_size, 0, *shape)
        return RecurrentState(code, empty_codes, empty_obs)

    def observe(
        self, state: RecurrentState, observation: torch.Tensor
    ) -> RecurrentState:
        """Commit ``(current code, observation)`` without predicting.

        Used for step 0, which has no memory to predict from but whose
        observation is the first thing worth remembering.
        """
        return RecurrentState(
            state.code,
            torch.cat([state.past_codes, state.code.unsqueeze(1)], dim=1),
            torch.cat([state.past_obs, self._store(observation).unsqueeze(1)],
                      dim=1),
        )

    def _store(self, observation: torch.Tensor) -> torch.Tensor:
        """What goes into the cache: the observation, or its embedding."""
        if self.config.cache_projected_values:
            return self.to_value(observation).detach()
        return observation

    def run_chunk(
        self,
        state: RecurrentState,
        actions: torch.Tensor,
        observations: torch.Tensor,
        use_gate: bool = True,
    ) -> tuple[ChunkOutput, RecurrentState]:
        """Advance one truncation window.

        Args:
            state: Carried state; its ``past_*`` blocks should already be
                detached by the caller.
            actions: ``(B, L)`` action leading *into* each step of the chunk.
            observations: ``(B, L, *obs_shape)`` observation *at* each step --
                a one-hot row, or a patch of pixels.
            use_gate: When False, the reverse read is skipped entirely and the
                position stays purely path-integrated. This has to be handled
                *inside* the loop -- suppressing the correction only between
                windows leaves it running for every step within one, which
                ablates almost nothing.
        """
        batch, length = observations.shape[:2]

        # Project the detached block once: it is fixed for the whole window.
        past_keys = self.to_key(state.past_codes)
        # W_x serves twice -- forward-read values and reverse-read keys -- and
        # these were two separate calls on the same tensor, so the projection
        # ran twice per window and half of it was thrown away. Free in the
        # symbol world, where W_x is one small matmul; not free in the patch
        # world, where it is a convnet over every patch in the cache.
        # Cached: the block already holds embeddings, so no encoder work and
        # no gradient into it from history. Uncached: re-project every window,
        # which costs 45% of a conv iteration and feeds the encoder gradient
        # from the whole past. See config.cache_projected_values.
        past_values = (
            state.past_obs if self.config.cache_projected_values
            else self.to_value(state.past_obs)
        )
        past_obs_keys = past_values  # reverse read keys: the same projection
        past_code_values = state.past_codes  # reverse read values

        code = state.code
        # Entries written during this window, kept ALREADY PROJECTED. Growing
        # these by concatenation and projecting each entry once, at write time,
        # avoids re-projecting the whole window on every step -- which is
        # quadratic in the window length and was the loop's dominant cost.
        # Note W_x serves twice: as forward-read values and reverse-read keys.
        recent_codes = state.past_codes.new_zeros(batch, 0, self.config.position_dim)
        recent_keys = state.past_codes.new_zeros(batch, 0, self.config.key_dim)
        recent_obs_proj = state.past_codes.new_zeros(batch, 0, self.config.obs_dim)
        recent_obs: list[torch.Tensor] = []

        logits, logits_position, retrievals = [], [], []
        integrated, gated, gates = [], [], []

        for i in range(length):
            n_memories = state.size + i
            beta = max(1.0, math.log(max(n_memories, 2)))

            # 1. path integration
            code_pi = self.position.step(code, actions[:, i])

            # 2. forward read, queried with the UNCORRECTED estimate, and predict
            retrieved_obs = attend(
                self.to_key(code_pi), past_keys, past_values,
                recent_keys, recent_obs_proj, beta,
            )
            step_logits, step_logits_pos = self.readout(
                retrieved_obs.unsqueeze(1), code_pi.unsqueeze(1)
            )
            logits.append(step_logits.squeeze(1))
            logits_position.append(step_logits_pos.squeeze(1))
            retrievals.append(retrieved_obs)

            # 3-4. the symbol is now revealed: reverse-read and correct.
            observation = observations[:, i]
            observation_proj = self.to_value(observation)
            if use_gate:
                retrieved_code = attend(
                    observation_proj,
                    past_obs_keys, past_code_values,
                    recent_obs_proj, recent_codes, beta,
                )
                code, gate = self.gate(code_pi, retrieved_code, n_memories > 0)
            else:
                code, gate = code_pi, torch.zeros_like(code_pi)

            integrated.append(code_pi)
            gated.append(code)
            gates.append(gate)

            # 5. write, projecting once
            recent_codes = torch.cat([recent_codes, code.unsqueeze(1)], dim=1)
            recent_keys = torch.cat(
                [recent_keys, self.to_key(code).unsqueeze(1)], dim=1
            )
            recent_obs_proj = torch.cat(
                [recent_obs_proj, observation_proj.unsqueeze(1)], dim=1
            )
            recent_obs.append(observation)

        output = ChunkOutput(
            logits=torch.stack(logits, dim=1),
            logits_position=torch.stack(logits_position, dim=1),
            integrated=torch.stack(integrated, dim=1),
            gated=torch.stack(gated, dim=1),
            gate=torch.stack(gates, dim=1),
            retrieved=torch.stack(retrievals, dim=1),
            values=recent_obs_proj,
        )
        new_state = RecurrentState(
            code,
            torch.cat([state.past_codes, recent_codes], dim=1),
            torch.cat(
                [state.past_obs,
                 recent_obs_proj.detach() if self.config.cache_projected_values
                 else torch.stack(recent_obs, dim=1)],
                dim=1,
            ),
        )
        return output, new_state
