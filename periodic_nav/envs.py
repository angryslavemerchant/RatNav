"""Environment constructors for the four difficulty levels.

All levels use SmallCore's Topology / Environment / walk-generation
infrastructure.  The levels differ only in how observations are
assigned to locations.

Level 1 — Unique: every cell gets a distinct observation.  No aliasing.
Level 2 — Tiled: a small observation pattern repeats across a larger grid.
Level 3 — Multi-tile: several tile types in a non-repeating arrangement.
Level 4 — Periodic tiles: the tile arrangement itself repeats.
"""

from __future__ import annotations

import numpy as np

from smallcore.graphs import (
    Environment,
    Topology,
    assign_observations,
    square_grid,
    NO_EDGE,
)


# ── topologies ──────────────────────────────────────────────────────


def torus_grid(width: int, height: int) -> Topology:
    """Square grid with wrapping edges (torus)."""
    n = width * height
    transitions = np.full((4, n), NO_EDGE, dtype=int)
    coords = np.zeros((n, 2))
    for r in range(height):
        for c in range(width):
            i = r * width + c
            coords[i] = [c, height - 1 - r]
            transitions[0, i] = ((r - 1) % height) * width + c      # north
            transitions[1, i] = r * width + (c + 1) % width          # east
            transitions[2, i] = ((r + 1) % height) * width + c      # south
            transitions[3, i] = r * width + (c - 1) % width          # west
    return Topology(
        name=f"torus_{width}x{height}",
        n_locations=n,
        n_actions=4,
        transitions=transitions,
        coords=coords,
        action_names=("north", "east", "south", "west"),
    )


# ── environment constructors ───────────────────────────────────────


def make_standard(
    width: int = 11,
    height: int = 11,
    n_obs: int = 45,
    rng: np.random.Generator | None = None,
) -> Environment:
    """Standard SmallCore environment for direct comparison."""
    if rng is None:
        rng = np.random.default_rng(42)
    return assign_observations(square_grid(width, height), n_obs, rng)


def make_level1(width: int = 11, height: int = 11) -> Environment:
    """Every location gets a unique observation."""
    topo = square_grid(width, height)
    n = width * height
    return Environment(topo, np.arange(n, dtype=int), n)


def make_level2(
    tile_w: int = 4,
    tile_h: int = 4,
    n_tile_obs: int = 8,
    n_repeats: int = 3,
    rng: np.random.Generator | None = None,
    use_torus: bool = False,
) -> Environment:
    """Large grid with periodically repeating observations.

    A ``tile_w × tile_h`` observation pattern is tiled ``n_repeats``
    times in each direction, producing a ``(tile_w*n_repeats)²`` world
    where every observation appears at ``n_repeats²`` locations.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    W, H = tile_w * n_repeats, tile_h * n_repeats
    if use_torus:
        topo = torus_grid(W, H)
    else:
        topo = square_grid(W, H)
    tile_obs = rng.integers(0, n_tile_obs, size=(tile_h, tile_w))
    obs = np.tile(tile_obs, (n_repeats, n_repeats)).flatten().astype(int)
    return Environment(topo, obs, n_tile_obs)


def make_level3(
    tile_w: int = 4,
    tile_h: int = 4,
    n_tile_obs: int = 8,
    n_tile_types: int = 4,
    layout_w: int = 3,
    layout_h: int = 3,
    rng: np.random.Generator | None = None,
) -> Environment:
    """Multiple tile types, non-repeating arrangement.

    Each tile type has its own observation pattern.  The arrangement of
    tiles across the grid is random (not periodic), so tile-boundary
    transitions disambiguate macro-position.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    W, H = tile_w * layout_w, tile_h * layout_h
    topo = square_grid(W, H)

    tile_patterns = [
        rng.integers(0, n_tile_obs, size=(tile_h, tile_w))
        for _ in range(n_tile_types)
    ]
    tile_assignment = rng.integers(0, n_tile_types, size=(layout_h, layout_w))

    obs = np.zeros((H, W), dtype=int)
    for tr in range(layout_h):
        for tc in range(layout_w):
            tt = tile_assignment[tr, tc]
            obs[
                tr * tile_h : (tr + 1) * tile_h,
                tc * tile_w : (tc + 1) * tile_w,
            ] = tile_patterns[tt]

    return Environment(topo, obs.flatten(), n_tile_obs)


def make_level4(
    tile_w: int = 4,
    tile_h: int = 4,
    n_tile_obs: int = 8,
    n_tile_types: int = 4,
    meta_w: int = 2,
    meta_h: int = 2,
    n_repeats: int = 2,
    rng: np.random.Generator | None = None,
) -> Environment:
    """Multiple tile types, periodically repeating arrangement.

    A ``meta_w × meta_h`` pattern of tile types is tiled ``n_repeats``
    times.  Both the within-tile observations *and* the tile layout
    are aliased — the model needs multi-scale tracking.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    meta_pattern = rng.integers(0, n_tile_types, size=(meta_h, meta_w))
    layout = np.tile(meta_pattern, (n_repeats, n_repeats))
    layout_h, layout_w = layout.shape

    tile_patterns = [
        rng.integers(0, n_tile_obs, size=(tile_h, tile_w))
        for _ in range(n_tile_types)
    ]

    W, H = tile_w * layout_w, tile_h * layout_h
    topo = square_grid(W, H)

    obs = np.zeros((H, W), dtype=int)
    for tr in range(layout_h):
        for tc in range(layout_w):
            tt = layout[tr, tc]
            obs[
                tr * tile_h : (tr + 1) * tile_h,
                tc * tile_w : (tc + 1) * tile_w,
            ] = tile_patterns[tt]

    return Environment(topo, obs.flatten(), n_tile_obs)
