"""The memory stream: causal attention with queries, keys and values split
across sources.

The forward read is the architecture's central claim:

    q_t = e_t  . W_e        query  <- POSITION only
    K   = E_<t . W_e        keys   <- POSITION only, tied to the query projection
    V   = X_<t . W_x        values <- OBSERVATION only

    y_t = softmax( beta * q_t K^T / sqrt(d_k) ) . V

Position alone decides *what to attend to*; observation alone decides *what
comes back*. Query with where you are, retrieve what was there.

The split is load-bearing rather than stylistic. Observations in this world are
deliberately ambiguous -- roughly 2.7 locations share every symbol -- so if the
keys carried any observation information the query could be satisfied by
appearance similarity, which is exactly the failure this design exists to
prevent. Keys are addresses. Values are contents.

There is no separate memory matrix: the KV cache *is* the memory, and it spans
the whole walk.

``beta = log(n_memories)`` is an adaptive softmax temperature. Without it,
attention flattens toward uniform as the cache grows and late steps of long
walks retrieve nothing useful.

Implementation note: the position stream is unavoidably sequential, but the
memory read is not. Once the codes exist for every step, the whole read is one
masked matrix product -- vastly cheaper than stepping through time again.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# Masked scores use a large negative number rather than -inf: a query with no
# valid memories (step 0) would otherwise softmax to NaN and poison the graph.
# Those rows are zeroed explicitly afterwards.
_MASK_VALUE = -1e9


class ForwardRead(nn.Module):
    """Retrieve past observations by querying with the current position code.

    Args:
        position_dim: Width of the position code.
        n_observations: Observation vocabulary size.
        key_dim: Width of the shared query/key space.
        obs_dim: Width the observations are compressed to as values.
    """

    def __init__(
        self,
        position_dim: int,
        n_observations: int,
        key_dim: int,
        obs_dim: int,
    ) -> None:
        super().__init__()
        self.key_dim = key_dim
        self.obs_dim = obs_dim
        # One projection producing BOTH queries and keys. Tying them is what
        # makes the dot product a comparison between two positions rather than
        # a learned association between two unrelated spaces.
        self.to_key = nn.Linear(position_dim, key_dim, bias=False)
        self.to_value = nn.Linear(n_observations, obs_dim, bias=False)

    def forward(
        self, codes: torch.Tensor, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read memory at every step.

        Args:
            codes: ``(B, T, position_dim)`` position codes.
            observations: ``(B, T, n_observations)`` one-hot observations.

        Returns:
            ``(retrieved, attention)`` with retrieved ``(B, T, obs_dim)`` and
            attention weights ``(B, T, T)``. Step ``t`` attends strictly to
            steps before it, so ``retrieved[:, t]`` never contains
            ``observations[:, t]`` -- the thing being predicted.
        """
        batch, steps, _ = codes.shape
        queries = self.to_key(codes)
        keys = self.to_key(codes)
        values = self.to_value(observations)

        scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.key_dim)

        # Adaptive temperature: query t sees t memories. Clamped at 2 so the
        # earliest steps keep a sane scale (log 1 = 0 would erase the scores).
        counts = torch.arange(steps, device=codes.device).clamp(min=2).float()
        scores = scores * counts.log().view(1, steps, 1)

        causal = torch.ones(steps, steps, dtype=torch.bool, device=codes.device)
        causal = causal.tril(diagonal=-1)  # strictly earlier steps only
        scores = scores.masked_fill(~causal.view(1, steps, steps), _MASK_VALUE)

        attention = scores.softmax(dim=-1)
        # Step 0 has no memories at all; its softmax is meaningless uniform.
        has_memory = causal.any(dim=-1).view(1, steps, 1).float()
        attention = attention * has_memory
        return attention @ values, attention


def write_mask(
    codes: torch.Tensor, threshold: float, chunk: int = 256
) -> torch.Tensor:
    """Which steps are worth committing to memory.

    The write rule from the brief: append ``(e_t, x_t)`` only if no
    sufficiently similar conjunction is already stored. Don't store the same
    fact repeatedly.

    Returns a ``(B, T)`` boolean mask. Genuinely sequential -- whether step t
    is stored depends on what was stored before it -- so this is a loop over
    time and costs real wall-clock. It is off by default at M2, where walks are
    short enough that duplicates are harmless; it earns its keep on the long
    walks of M3 where a revisited corridor would otherwise be written dozens of
    times.

    Args:
        codes: ``(B, T, D)`` position codes, assumed unit-ish scale (they come
            out of a LayerNorm, so cosine and dot similarity roughly agree).
        threshold: Cosine similarity above which a memory counts as already
            stored.
        chunk: Unused placeholder for a batched variant; kept so callers do not
            need changing if this is optimised later.
    """
    batch, steps, _ = codes.shape
    normed = torch.nn.functional.normalize(codes, dim=-1)
    keep = torch.zeros(batch, steps, dtype=torch.bool, device=codes.device)
    keep[:, 0] = True  # the first step has nothing to be redundant with
    for t in range(1, steps):
        stored = keep[:, :t].float()  # (B, t)
        similarity = torch.einsum("bd,btd->bt", normed[:, t], normed[:, :t])
        # Only compare against entries actually written.
        similarity = similarity.masked_fill(stored == 0, -1.0)
        keep[:, t] = similarity.max(dim=-1).values < threshold
    return keep
