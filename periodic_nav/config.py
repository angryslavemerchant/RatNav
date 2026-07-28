from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Periodic basis — cycles in cells for each module
    n_modules: int = 5
    module_cycles: tuple = (3.0, 4.2, 5.9, 8.2, 11.5)
    orient_per_module: int = 2
    learn_increments: bool = False

    # Memory / attention
    key_dim: int = 64
    obs_dim: int = 16

    # Drift gate
    gate_hidden: int = 64
    gate_bias_init: float = -2.0

    # Readout
    hidden_dim: int = 64

    # Environment
    n_observations: int = 45
    grid_width: int = 11
    grid_height: int = 11

    # Training
    batch_size: int = 16
    lr: float = 1e-3
    lr_min: float = 1e-4
    lr_decay_every: int = 4000
    lr_decay_factor: float = 0.5
    n_iterations: int = 10000
    tbptt_window: int = 20
    walk_length: int = 300
    straight_bias: float = 2.0

    # Loss weights
    w_pred_pos: float = 0.1
    w_drift: float = 0.05

    # Continuous / image world
    continuous: bool = False
    patch_size: int = 16
    patch_encoder: str = "dct"

    # Evaluation
    eval_every: int = 1000
    eval_walks: int = 32
    eval_lengths: tuple = (50, 100, 200, 300)
