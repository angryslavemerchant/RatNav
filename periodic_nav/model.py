"""Periodic navigation model.

Fixed periodic basis for position tracking, split-source attention for
memory retrieval, reverse read + drift gate for re-localisation.
All periodic structure is frozen; only the attention projections, gate,
and readout MLP are learned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basis import PeriodicBasis
from .config import Config


@dataclass
class State:
    phases: torch.Tensor       # (B, n_basis)
    past_codes: torch.Tensor   # (B, T, code_dim)
    past_obs: torch.Tensor     # (B, T, obs_dim)

    def detach(self) -> "State":
        return State(
            self.phases.detach(),
            self.past_codes.detach(),
            self.past_obs.detach(),
        )


@dataclass
class ChunkOutput:
    logits: torch.Tensor          # (B, W, n_obs)
    logits_pos: torch.Tensor      # (B, W, n_obs)
    gate: torch.Tensor            # (B, W, n_basis)
    codes_pi: torch.Tensor        # (B, W, code_dim)
    codes_corrected: torch.Tensor # (B, W, code_dim)


def _attend(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scale: float,
    n_memories: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention with adaptive β = log(n) temperature.

    query: (B, d), keys: (B, T, d), values: (B, T, d_v)
    Returns: (retrieved (B, d_v), attention (B, T))
    """
    scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)  # (B, T)
    beta = math.log(max(n_memories, 2))
    scores = scores * beta / scale
    attn = F.softmax(scores, dim=-1)
    retrieved = torch.bmm(attn.unsqueeze(1), values).squeeze(1)
    return retrieved, attn


class PeriodicNav(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        n_actions = 4

        self.basis = PeriodicBasis(
            cfg.n_modules,
            n_actions,
            module_cycles=list(cfg.module_cycles),
            orient_per_module=cfg.orient_per_module,
            learn_increments=cfg.learn_increments,
        )
        code_dim = self.basis.code_dim

        self.to_key = nn.Linear(code_dim, cfg.key_dim, bias=False)

        if not cfg.continuous:
            self.obs_embed = nn.Embedding(cfg.n_observations, cfg.obs_dim)

        self.gate_net = nn.Sequential(
            nn.Linear(2 * code_dim, cfg.gate_hidden),
            nn.GELU(),
            nn.Linear(cfg.gate_hidden, self.basis.n_basis),
        )
        self.gate_net[-1].bias.data.fill_(cfg.gate_bias_init)

        out_dim = cfg.obs_dim if cfg.continuous else cfg.n_observations
        self.readout = nn.Sequential(
            nn.Linear(cfg.obs_dim + code_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, out_dim),
        )
        self.readout_pos = nn.Sequential(
            nn.Linear(code_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, out_dim),
        )

    def initial_state(
        self, batch_size: int, device: torch.device
    ) -> State:
        phases = self.basis.initial_state(batch_size).to(device)
        code = self.basis.encode(phases)
        return State(
            phases=phases,
            past_codes=code.new_zeros(batch_size, 0, self.basis.code_dim),
            past_obs=code.new_zeros(batch_size, 0, self.cfg.obs_dim),
        )

    def observe(self, state: State, obs: torch.Tensor) -> State:
        """Store the initial observation in memory without predicting.

        obs: (B,) int indices  OR  (B, obs_dim) float embeddings.
        """
        code = self.basis.encode(state.phases)
        obs_emb = self.obs_embed(obs) if obs.dtype in (torch.long, torch.int) else obs
        return State(
            state.phases,
            torch.cat([state.past_codes, code.unsqueeze(1)], dim=1),
            torch.cat([state.past_obs, obs_emb.unsqueeze(1)], dim=1),
        )

    def run_chunk(
        self,
        state: State,
        step_input: torch.Tensor,
        obs_input: torch.Tensor,
        use_gate: bool = True,
    ) -> tuple[ChunkOutput, State]:
        """Process one truncation window.

        Args:
            state: carried from previous chunk (or initial_state + observe)
            step_input: (B, W) int actions  OR  (B, W, 2) float velocities
            obs_input:  (B, W) int obs indices  OR  (B, W, obs_dim) float embeddings
            use_gate: apply reverse read + drift correction

        Returns:
            (ChunkOutput, new State)
        """
        continuous = step_input.dim() == 3
        B = step_input.shape[0]
        W = step_input.shape[1]
        device = step_input.device
        code_dim = self.basis.code_dim
        n_basis = self.basis.n_basis
        key_dim = self.cfg.key_dim
        obs_dim = self.cfg.obs_dim

        phases = state.phases
        past_codes = state.past_codes    # detached
        past_obs = state.past_obs        # detached

        past_keys = self.to_key(past_codes)  # (B, T_past, key_dim)

        recent_codes_list: list[torch.Tensor] = []
        recent_obs_list: list[torch.Tensor] = []
        recent_keys_list: list[torch.Tensor] = []
        all_logits: list[torch.Tensor] = []
        all_logits_pos: list[torch.Tensor] = []
        all_gates: list[torch.Tensor] = []
        all_pi: list[torch.Tensor] = []
        all_corrected: list[torch.Tensor] = []

        scale = math.sqrt(key_dim)
        obs_scale = math.sqrt(obs_dim)
        zero_gate = phases.new_zeros(B, n_basis)

        for i in range(W):
            if continuous:
                new_phases = self.basis.step_continuous(phases, step_input[:, i])
            else:
                new_phases = self.basis.step(phases, step_input[:, i])
            code_pi = self.basis.encode(new_phases)
            all_pi.append(code_pi)

            # --- forward read ---
            if recent_keys_list:
                rk = torch.stack(recent_keys_list, dim=1)
                ro = torch.stack(recent_obs_list, dim=1)
                rc = torch.stack(recent_codes_list, dim=1)
                fwd_keys = torch.cat([past_keys, rk], dim=1)
                fwd_obs_vals = torch.cat([past_obs, ro], dim=1)
                fwd_code_vals = torch.cat([past_codes, rc], dim=1)
            else:
                fwd_keys = past_keys
                fwd_obs_vals = past_obs
                fwd_code_vals = past_codes

            n_mem = fwd_keys.shape[1]
            if n_mem > 0:
                q = self.to_key(code_pi)
                retrieved_obs, _ = _attend(q, fwd_keys, fwd_obs_vals, scale, n_mem)
            else:
                retrieved_obs = code_pi.new_zeros(B, obs_dim)

            # --- predict ---
            all_logits.append(
                self.readout(torch.cat([retrieved_obs, code_pi], dim=-1))
            )
            all_logits_pos.append(self.readout_pos(code_pi))

            # --- observe ---
            if obs_input.dim() == 2:
                obs_emb = self.obs_embed(obs_input[:, i])
            else:
                obs_emb = obs_input[:, i]

            # --- reverse read + gate ---
            if use_gate and n_mem > 0:
                retrieved_code, _ = _attend(
                    obs_emb, fwd_obs_vals, fwd_code_vals, obs_scale, n_mem
                )
                gate = torch.sigmoid(
                    self.gate_net(torch.cat([code_pi, retrieved_code], dim=-1))
                )
                gate_exp = torch.cat([gate, gate], dim=-1)
                code_corrected = code_pi + gate_exp * (retrieved_code - code_pi)
                all_gates.append(gate)
                phases = self.basis.phases_from_code(code_corrected)
            else:
                code_corrected = code_pi
                phases = new_phases
                all_gates.append(zero_gate)

            all_corrected.append(code_corrected)
            recent_codes_list.append(code_corrected)
            recent_keys_list.append(self.to_key(code_corrected))
            recent_obs_list.append(obs_emb)

        rc = torch.stack(recent_codes_list, dim=1)
        ro = torch.stack(recent_obs_list, dim=1)

        return (
            ChunkOutput(
                logits=torch.stack(all_logits, dim=1),
                logits_pos=torch.stack(all_logits_pos, dim=1),
                gate=torch.stack(all_gates, dim=1),
                codes_pi=torch.stack(all_pi, dim=1),
                codes_corrected=torch.stack(all_corrected, dim=1),
            ),
            State(
                phases=phases.detach(),
                past_codes=torch.cat([past_codes, rc.detach()], dim=1),
                past_obs=torch.cat([past_obs, ro.detach()], dim=1),
            ),
        )
