"""The patch encoder -- what replaces the one-hot observation projection.

In the symbol world ``to_value`` is ``Linear(n_observations -> obs_dim)`` off a
one-hot: a lookup table with nothing to learn about *appearance*, because a
symbol has none. With an image the observation is a patch of pixels, and the
projection becomes a small convnet.

It is worth being explicit about how much this one module carries, because it
sits at the junction of both memory streams:

* forward read -- it produces the **values**, so it decides what "what was
  there" means;
* reverse read -- it produces the **keys**, so it decides which observations
  count as similar, and therefore how well "I have seen this before" works.

That second role is the one this whole rung exists for. In M6 the reverse read
compared one-hot symbols, so similarity was 1 for an exact match and 0
otherwise, carrying no information about *how far* two observations are apart.
Patch embeddings vary smoothly with position, so observation similarity now
carries distance -- which is what a landmark fix needs to be worth anything.

The output is LayerNormed. Both streams consume it through a dot product, and
an encoder free to grow its output norm would silently sharpen the reverse
read's softmax alongside the ``beta = log n`` schedule that is supposed to
control it.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class PatchEncoder(nn.Module):
    """``(..., P, P)`` pixels -> ``(..., obs_dim)`` embedding.

    Two strided convolutions and a projection. Deliberately small: the point of
    this rung is the *world*, not the vision model, and a large encoder would
    let appearance do work that should be falling to the memory.
    """

    def __init__(
        self, patch: int, obs_dim: int, channels: tuple[int, int] = (16, 32)
    ) -> None:
        super().__init__()
        self.patch = patch
        self.obs_dim = obs_dim
        c1, c2 = channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, c1, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            nn.GELU(),
        )
        reduced = -(-patch // 4)  # two stride-2 layers, ceil division
        self.project = nn.Linear(c2 * reduced * reduced, obs_dim)
        self.norm = nn.LayerNorm(obs_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        lead = patches.shape[:-2]
        flat = patches.reshape(-1, 1, self.patch, self.patch)
        if flat.shape[0] == 0:  # empty memory block
            return patches.new_zeros(*lead, self.obs_dim)
        hidden = self.conv(flat).flatten(1)
        return self.norm(self.project(hidden)).reshape(*lead, self.obs_dim)


class LinearPatchEncoder(nn.Module):
    """The whole patch as one vector, projected once. No convolution.

    Worth testing against `PatchEncoder` because a convnet may be the wrong
    inductive bias for this particular job, for two reasons.

    **It warps distance.** The reverse read is only useful if observation
    similarity carries *how far apart* two views are, and the correction gate
    has sat at 0.15-0.27 for every run so far, which is close to its 0.12
    initialisation. A linear map preserves pixel-space geometry up to that map,
    so two patches a fraction of a cell apart stay close in embedding space. A
    stack of convolutions and GELUs is free to fold that geometry however the
    contrastive loss finds convenient.

    **It is its own target.** The encoder produces the embeddings the model is
    trained to predict, so extra capacity there is extra freedom to co-adapt
    with the predictor rather than to describe the patch. A one-matrix encoder
    has far less room to make the task easy in ways that do not correspond to
    seeing better.

    Weight sharing is also a strange thing to want here: convolution is built
    to answer "what is present" while being relaxed about "where", and this
    task is precisely about where.
    """

    def __init__(self, patch: int, obs_dim: int) -> None:
        super().__init__()
        self.patch = patch
        self.obs_dim = obs_dim
        self.project = nn.Linear(patch * patch, obs_dim)
        self.norm = nn.LayerNorm(obs_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        lead = patches.shape[:-2]
        flat = patches.reshape(-1, self.patch * self.patch)
        if flat.shape[0] == 0:
            return patches.new_zeros(*lead, self.obs_dim)
        return self.norm(self.project(flat)).reshape(*lead, self.obs_dim)


def _dct_basis(patch: int, n: int, drop_dc: bool) -> torch.Tensor:
    """The ``n`` lowest-frequency 2D DCT-II basis vectors, orthonormal."""
    i = torch.arange(patch, dtype=torch.float64)
    u = torch.arange(patch, dtype=torch.float64)
    # cos(pi (2i+1) u / 2P), scaled so the 1D basis is orthonormal.
    one_d = torch.cos(torch.pi * (2 * i[None, :] + 1) * u[:, None] / (2 * patch))
    one_d *= torch.sqrt(torch.where(u == 0, 1.0 / patch, 2.0 / patch))[:, None]
    basis = torch.einsum("ui,vj->uvij", one_d, one_d).reshape(patch * patch, -1)
    freq = torch.sqrt(
        (u[:, None] ** 2 + u[None, :] ** 2).reshape(-1)
    )
    order = torch.argsort(freq, stable=True)
    if drop_dc:
        order = order[1:]  # the DC term is mean brightness, and it dominates
    return basis[order[:n]].to(torch.float32)


def _random_basis(patch: int, n: int, seed: int) -> torch.Tensor:
    """``n`` orthonormal directions drawn uniformly at random.

    Orthonormalised on purpose. An unconstrained Gaussian matrix would differ
    from the DCT in *two* ways at once -- which subspace it keeps and how badly
    conditioned it is -- and only the first is the question being asked.
    """
    generator = torch.Generator().manual_seed(seed)
    gaussian = torch.randn(
        patch * patch, n, generator=generator, dtype=torch.float64
    )
    q, _ = torch.linalg.qr(gaussian)
    return q.T.to(torch.float32)


def _gabor_basis(patch: int, n: int) -> torch.Tensor:
    """Oriented bandpass filters -- fixed conv kernels, in the V1 sense.

    Included because it is the classic answer to "fixed visual features", but
    it is not the default, and the reason is measurable rather than aesthetic.
    A Gabor bank is a *non-orthogonal, redundant* family: neighbouring
    orientations and scales overlap heavily, so the map from pixels to features
    is ill-conditioned and stretches some directions in pixel space far more
    than others. This model needs observation similarity to track *distance*
    -- that is the reverse read's entire job -- so a basis that warps distance
    is working against the mechanism. `verify_isometry` prices exactly this.
    """
    coords = torch.arange(patch, dtype=torch.float64) - (patch - 1) / 2.0
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    sigma = patch / 4.0
    envelope = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    filters = []
    # Sweep orientation fastest, then scale, then phase, so a truncated bank is
    # still orientation-complete rather than being all one angle.
    for phase in (0.0, math.pi / 2):
        for wavelength in (patch / 2.0, patch / 3.0, patch / 5.0):
            for k in range(4):
                theta = k * math.pi / 4
                rotated = x * math.cos(theta) + y * math.sin(theta)
                wave = torch.cos(2 * math.pi * rotated / wavelength + phase)
                filters.append((envelope * wave).reshape(-1))
    bank = torch.stack(filters[:n])
    bank = bank - bank.mean(dim=1, keepdim=True)
    return (bank / bank.norm(dim=1, keepdim=True)).to(torch.float32)


class FixedFeatureEncoder(nn.Module):
    """A patch encoder with **no trainable parameters at all**.

    Every learned encoder in this project shares one problem, and it is not
    speed. The encoder and the predictor are trained on the same loss, so there
    are two ways to lower it: build a better position code, or reshape the
    observation space until the patches are easier to tell apart. The second is
    far easier and the encoder is the thing that gets to do it -- so the
    contrastive objective can be satisfied without the recurrent state ever
    improving. Freezing the features removes that option. The only remaining
    way to predict the next patch is to know where you are going to be.

    That matters here specifically because *pressure* is the only thing that
    has ever moved periodicity in this model: M5 went 57% -> 93% -> 100%
    periodic purely by making a place code unaffordable. Six mechanisms that
    merely removed obstacles all failed. This one adds a constraint.

    Three consequences beyond the experiment:

    * **Distances stop moving.** With `dct` or `random` the map is orthonormal,
      so patch-space L2 distance is preserved exactly on the retained subspace
      -- the reverse read's "how far apart are these two views" is then a fact
      about the world rather than about the current state of training.
    * **`cache_projected_values` becomes exactly gradient-equivalent**, since
      there are no encoder weights for the re-projection path to feed. The
      45%-of-an-iteration caveat attached to that flag does not apply here.
    * **Walks become cacheable.** A frozen encoder means the embedding of a
      patch is a function of position alone, so a whole dataset of
      ``(velocity, embedding)`` can be precomputed once -- `obs_dim` floats a
      step instead of ``patch**2`` pixels, and no encoder forward or backward.

    Args:
        basis: ``dct`` (lowest spatial frequencies), ``random`` (a random
            orthonormal subspace -- the control that says whether *which*
            features matter or only that they are frozen), ``gabor`` (oriented
            bandpass filters), or ``pca`` (fitted by `fit_whitening`).
    """

    def __init__(
        self,
        patch: int,
        obs_dim: int,
        basis: str = "dct",
        drop_dc: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.patch = patch
        self.obs_dim = obs_dim
        self.basis = basis
        if obs_dim > patch * patch:
            raise ValueError(
                f"obs_dim {obs_dim} exceeds the {patch * patch} dimensions a "
                f"{patch}x{patch} patch has; a fixed basis cannot invent any"
            )
        if basis == "dct":
            weight = _dct_basis(patch, obs_dim, drop_dc)
        elif basis == "random":
            weight = _random_basis(patch, obs_dim, seed)
        elif basis == "gabor":
            weight = _gabor_basis(patch, obs_dim)
        elif basis == "pca":
            # Identity until fitted, so an unfitted `pca` run is obviously
            # wrong rather than quietly mediocre.
            weight = _dct_basis(patch, obs_dim, drop_dc)
        else:
            raise ValueError(f"unknown fixed basis {basis!r}")
        # Buffers, not Parameters: they move with .to(device) and land in the
        # checkpoint, but no optimiser will ever touch them.
        self.register_buffer("weight", weight)
        self.register_buffer("scale", torch.ones(obs_dim))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))
        # Non-affine: an affine LayerNorm would hand back two learnable vectors
        # per dimension and reopen a narrow version of the escape route.
        self.norm = nn.LayerNorm(obs_dim, elementwise_affine=False)

    @torch.no_grad()
    def fit(self, patches: torch.Tensor, whiten: bool = False) -> None:
        """Fit the basis (``pca`` only) and, optionally, per-feature scales.

        **Whitening defaults off, and the first version of this had it on.**
        The argument for it was that image spectra fall off steeply, so raw DCT
        coefficients differ in scale by orders of magnitude, and since both
        memory streams consume the embedding through a dot product the two or
        three lowest frequencies would decide every retrieval while the rest
        were decoration. All of that is true. None of it is a problem: those low
        frequencies are *where the signal is*, and equalising the variance
        promotes the near-empty high-frequency dimensions into noise of equal
        weight. Measured on the widened world, whitening took the correlation
        between embedding distance and pixel distance from **1.000 to 0.237**.
        Variance concentration was the structure of the data, not a pathology.

        Kept as an option because it is the right move on a world whose spectrum
        really is flat -- but it must be justified by `verify_isometry` on the
        world in hand, never assumed.
        """
        flat = patches.reshape(-1, self.patch * self.patch).to(torch.float32)
        flat = flat - flat.mean(dim=0, keepdim=True)
        if self.basis == "pca":
            # Components of the actual patch distribution, largest variance
            # first. DCT is the analytic limit of this for a stationary
            # texture, so agreement between the two is a useful check.
            _, _, v = torch.linalg.svd(flat.double(), full_matrices=False)
            self.weight.copy_(v[: self.obs_dim].to(torch.float32))
        if whiten:
            projected = flat @ self.weight.T
            self.scale.copy_(1.0 / projected.std(dim=0).clamp(min=1e-6))
        self.fitted.fill_(True)

    @torch.no_grad()
    def verify_isometry(self, patches: torch.Tensor) -> dict[str, float]:
        """How faithfully does this basis carry pixel-space distance?

        The reverse read is only worth anything if two views that are close in
        the world are close in embedding space, so this is the property that
        decides whether a basis is usable -- and it is a fixed number, knowable
        before training rather than after. Reported as the correlation between
        pairwise pixel distance and pairwise embedding distance, plus the
        fraction of pixel-space variance the subspace retains.
        """
        flat = patches.reshape(-1, self.patch * self.patch).to(torch.float32)
        embedded = (flat - flat.mean(dim=0, keepdim=True)) @ self.weight.T
        embedded = embedded * self.scale
        pixel_d = torch.cdist(flat, flat).reshape(-1)
        embed_d = torch.cdist(embedded, embedded).reshape(-1)
        keep = pixel_d > 0
        correlation = torch.corrcoef(
            torch.stack([pixel_d[keep], embed_d[keep]])
        )[0, 1]
        centred = flat - flat.mean(dim=0, keepdim=True)
        retained = ((centred @ self.weight.T) ** 2).sum() / (centred**2).sum()
        return {
            "distance_corr": float(correlation),
            "variance_retained": float(retained),
        }

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        lead = patches.shape[:-2]
        flat = patches.reshape(-1, self.patch * self.patch)
        if flat.shape[0] == 0:
            return patches.new_zeros(*lead, self.obs_dim)
        projected = (flat @ self.weight.T) * self.scale
        return self.norm(projected).reshape(*lead, self.obs_dim)


def info_nce(
    predicted: torch.Tensor,
    target: torch.Tensor,
    blocked: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contrastive loss and top-1 hits over a pool of candidate patches.

    The symbol world scored prediction with cross-entropy over a 45-way
    vocabulary. A patch has no vocabulary, and the two obvious replacements
    fail differently: regression to pixels is minimised by the *average* of the
    plausible continuations, so it produces blur and rewards hedging, while a
    reconstruction loss spends capacity on texture detail that has nothing to
    do with knowing where you are. Picking the true patch out of a pool of real
    ones asks exactly the question the symbol task asked -- *which* of these is
    about to appear -- and inherits its interpretation, chance included.

    Args:
        predicted / target: ``(N, D)``, both L2-normalised.
        blocked: ``(N, N)`` bool, candidates excluded from row i's denominator.
            Patches close in time along one walk overlap in pixels, so they are
            near-duplicates of the answer rather than negatives; counting them
            would train the encoder to separate observations that genuinely are
            the same thing seen twice.
        temperature: Softmax temperature on cosine similarity.

    Returns:
        ``(loss, hits)`` -- hits is a bool ``(N,)`` of top-1 correctness.
    """
    logits = predicted @ target.T / temperature
    logits = logits.masked_fill(blocked, -1e9)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = nn.functional.cross_entropy(logits, labels)
    return loss, (logits.argmax(dim=1) == labels)


def info_nce_grouped(
    predicted: torch.Tensor,
    target: torch.Tensor,
    blocked: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """InfoNCE run independently inside each group of walks.

    Exists so that **batch size stops changing the objective**. Scored against
    the whole batch, the candidate pool is ``batch * window``, so raising the
    batch from 64 to 4096 turns a 1280-way choice into an 81920-way one -- a
    harder task with a different loss scale, which would make a large-batch run
    incomparable to every run already measured. Worse, it would confound the
    thing a bigger batch is supposed to buy: a lower-variance gradient on the
    *same* problem.

    That is not hypothetical here. At one patch per step the pool was already
    hard enough that the loss sat at exactly ``ln(pool)`` for 500 iterations
    before anything moved, so pool size is known to be load-bearing.

    Groups all share one ``blocked`` mask because every group has the same
    ``(walks, timesteps)`` layout.

    Args:
        predicted / target: ``(G, N, D)``, L2-normalised, N = group * window.
        blocked: ``(N, N)`` bool, shared across groups.
    """
    logits = torch.bmm(predicted, target.transpose(1, 2)) / temperature
    logits = logits.masked_fill(blocked.unsqueeze(0), -1e9)
    n = logits.shape[1]
    labels = torch.arange(n, device=logits.device).expand(logits.shape[0], n)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, n), labels.reshape(-1)
    )
    return loss, (logits.argmax(dim=-1) == labels)


def blocked_mask(
    walk_id: torch.Tensor, time_id: torch.Tensor, radius: int
) -> torch.Tensor:
    """``(N, N)`` mask of same-walk, temporally-adjacent candidates.

    The diagonal is never blocked -- it is the answer.
    """
    same = walk_id[:, None] == walk_id[None, :]
    near = (time_id[:, None] - time_id[None, :]).abs() <= radius
    mask = same & near
    mask.fill_diagonal_(False)
    return mask
