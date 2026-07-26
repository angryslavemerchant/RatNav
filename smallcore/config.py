"""Hyperparameters.

Values are the starting points from the project brief (CLAUDE.md §8). They are
a single flat dataclass so a run can be described by one object and overridden
with ``dataclasses.replace``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- position stream --------------------------------------------------- #
    # Block-diagonal modules of the transition operators. Each module is
    # initialised as a rotation at its own frequency, so the modules settle at
    # geometrically-spaced spatial scales.
    # Equal widths, measured 2026-07-25 to match the brief's (30,30,24,18,18)
    # exactly on M1 (99.0% observation accuracy, 100% short-walk decoding,
    # 87.9% vs 81.3% on 200-step walks) while letting the position step run as
    # a single einsum instead of a loop over modules.
    module_dims: tuple[int, ...] = (24, 24, 24, 24, 24)
    module_freqs: tuple[float, ...] = (0.99, 0.3, 0.09, 0.03, 0.01)

    # --- observations ------------------------------------------------------ #
    n_observations: int = 45
    # Observations are compressed before entering memory as values: the memory
    # returns *contents*, and 45 one-hot dims is a wasteful way to carry them.
    obs_dim: int = 10
    # "symbol" -- a one-hot from a fixed vocabulary, projected linearly.
    # "patch"  -- a patch_size x patch_size window of an image, encoded by a
    # small convnet and scored contrastively (ROADMAP Rung 5). The recurrent
    # loop is identical either way; only what an observation *is* changes.
    observation_mode: str = "symbol"
    patch_size: int = 8
    # Softmax temperature for the contrastive objective, on cosine similarity.
    temperature: float = 0.07
    # Walks per contrastive group; 0 means "the whole batch". Set this whenever
    # the batch is scaled up, so the candidate pool -- and therefore the
    # difficulty of the task and the scale of the loss -- stays fixed and a
    # large-batch run remains comparable to the runs already measured.
    contrastive_group: int = 0

    # --- memory stream ----------------------------------------------------- #
    # Queries and keys are both position, through one tied projection: they
    # have to live in the same space for a dot product between them to mean
    # "am I near where I was?".
    key_dim: int = 64
    # Nonlinear grid->place bottleneck (ROADMAP Rung 0d). A LINEAR key
    # projection cannot turn periodic similarity into unimodal similarity --
    # key similarity is e(x)^T W^T W e(y), a reweighted version of the same
    # quadratic form -- so the retrieval objective's demand for a clean single
    # peak propagates back onto the recurrent state and forces it to be
    # place-like. Thresholding is what suppresses a periodic code's side lobes,
    # which is exactly how grid cells become place cells biologically.
    nonlinear_key: bool = False

    # --- the grid-cell recipe (M8) ----------------------------------------- #
    # Every published demonstration of grid cells emerging in a trained network
    # supplies three things this model has never had. M5 refuted six mechanisms
    # and every one of them REMOVED AN OBSTACLE to periodicity; not one of them
    # REWARDED it. These three do.
    #
    # 1. Nonnegative activations. Units here are tanh then LayerNorm, so a rate
    #    map is of a SIGNED quantity, where a grid cell is a nonnegative firing
    #    rate. CLAUDE.md flags this as the last untested structural suspect.
    #    LayerNorm cannot simply be kept: it centres, which reintroduces the
    #    negatives, so "relu" swaps in a nonnegative rescale instead.
    position_activation: str = "tanh"  # "tanh" or "relu"
    # 2. Place-cell-shaped targets. Sorscher et al. find it is the SHAPE of the
    #    target that yields hexagons -- raw (x, y) is not enough, because the
    #    minimal solution is a linear code in two dimensions with no pressure to
    #    repeat. A deliberate, labelled exception to "locations are analysis
    #    only", justified as a diagnostic. Zero disables it.
    w_place: float = 0.0
    n_place_cells: int = 256
    place_sigma: float = 0.5  # cells
    # 3. An activity cost. Sparse nonnegative codes are what make a periodic
    #    solution cheaper than a place-like one.
    l1_position_code: float = 0.0
    # Weight on the CONTRASTIVE objective itself. Set this to 0 with w_place > 0
    # and the model becomes a pure path integrator trained only to predict
    # place-cell targets -- which is what the published grid-cell models ARE.
    # If grids appear there and nowhere else, the architecture was always
    # capable and the MEMORY objective is what suppresses them, exactly as the
    # dot-product argument predicts: a periodic code has side lobes, so a query
    # at x also partly retrieves memories a full period away, and unimodal
    # similarity beats periodic similarity at retrieval.
    w_pred: float = 1.0

    # Continuous movement: the position stream takes a velocity vector instead
    # of an action index, and two learned generators replace the per-action
    # matrices (smallcore/continuous.py).
    continuous: bool = False
    # Step length for continuous walks (cells per step).
    speed: float = 1.0
    # Post-attention residual block, per the brief's one-layer/one-head start.
    model_dim: int = 64
    hidden_dim: int = 64

    # --- loss weights ------------------------------------------------------ #
    # L2 on the position code. Small number, large effect: this is the pressure
    # that pushes the code toward clean periodic structure.
    l2_position_code: float = 0.01
    # L2 on the weights (applied as Adam weight decay).
    weight_decay: float = 1e-5
    # Weight on predicting the observation from the position code alone. With
    # random walk starts the position code carries displacement rather than
    # absolute location, so this term is genuinely unsatisfiable and is kept
    # small -- it is a pressure on the code, not an objective to reach.
    w_pred_pos: float = 0.1
    # Squared error between the gated position and the pure path-integrated
    # one (M3). Keeps the gate from rewriting position wholesale on every
    # landmark: corrections should be nudges, not teleports.
    w_drift: float = 0.05

    # --- training ---------------------------------------------------------- #
    batch_size: int = 16
    lr: float = 9.4e-4
    lr_min: float = 8e-5
    lr_decay: float = 0.5
    lr_decay_every: int = 4000
    straight_bias: float = 2.0
    # Truncated-backprop window. The memory cache spans the whole walk; the
    # gradient does not.
    tbptt_window: int = 20

    @property
    def position_dim(self) -> int:
        return sum(self.module_dims)

    @property
    def readout_dim(self) -> int:
        """Width of the prediction head's output.

        A distribution over the vocabulary in the symbol world; a point in the
        observation embedding space, compared against encoded candidates, in the
        patch world.
        """
        return self.obs_dim if self.observation_mode == "patch" else self.n_observations

    def __post_init__(self) -> None:
        if len(self.module_dims) != len(self.module_freqs):
            raise ValueError("module_dims and module_freqs must align")
        if self.observation_mode not in ("symbol", "patch"):
            raise ValueError("observation_mode must be 'symbol' or 'patch'")
