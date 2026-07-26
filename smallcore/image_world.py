"""An image as the world: continuous position, a local patch as the observation.

M6 paired continuous movement with a piecewise-constant symbol field, which
asks for sub-cell precision while supplying no sub-cell information, and leaves
observation similarity carrying no spatial information at all -- so the reverse
read has nothing to work with. It reached 53% of ceiling where the discrete
world reached 98%. Continuous movement and continuous observation are one change
rather than two.

Here the observation is a ``patch x patch`` window **bilinearly sampled** at the
agent's real-valued position. Move half a cell and the patch shifts by half a
cell: the observation is now a smooth function of position, so a small
displacement is visible and observation similarity carries distance.

Controlled ambiguity
--------------------
The discrete world's ambiguity was exact by construction -- 45 symbols over 121
locations, ~2.7 locations per symbol -- and that mattered, because unique
observations turn prediction into table lookup and position never needs
representing. A random natural image gives no such guarantee, so the backdrop
here is built the same way the symbol world was: a lattice of cells, each
painted with one of ``n_motifs`` textures drawn from a fixed set.

A patch lying wholly inside one cell is ambiguous across every copy of that
motif. A patch straddling cells sees a *conjunction* and is far more
distinctive. With ``patch < cell`` both cases occur, in a ratio fixed by
``(1 - patch/cell)^2``, and `measure_ambiguity` reports what actually resulted
rather than what was intended.

Units
-----
Positions and velocities are in **cells**, not pixels, so the position stream's
module frequencies keep their meaning (cycle = ``2 / freq`` cells) across any
choice of ``cell``. Only `ImageEnvironment.observe` knows about pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def _smooth_noise(
    shape: tuple[int, int], sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Zero-mean unit-variance band-limited noise."""
    raw = rng.standard_normal(shape)
    out = gaussian_filter(raw, sigma=sigma, mode="wrap")
    return (out - out.mean()) / (out.std() + 1e-8)


@dataclass(frozen=True)
class ImageEnvironment:
    """A motif-tiled image, observed through a patch at a continuous position.

    Attributes:
        image: ``(height_px, width_px)`` backdrop, globally standardised.
        cell: Pixels per lattice cell.
        grid_w / grid_h: Arena extent in cells. Position is real-valued within.
        motifs: ``(n_motifs, cell, cell)`` texture set.
        cell_motifs: ``(grid_h * grid_w,)`` motif index per cell -- the direct
            analogue of ``Environment.observations`` in the discrete world.
        patch: Observation window in pixels.
    """

    image: np.ndarray
    cell: int
    grid_w: int
    grid_h: int
    motifs: np.ndarray | None
    cell_motifs: np.ndarray | None
    patch: int

    @property
    def n_cells(self) -> int:
        return self.grid_w * self.grid_h

    @property
    def n_motifs(self) -> int:
        """Zero for a photograph, which has no motif set behind it."""
        return 0 if self.motifs is None else len(self.motifs)

    @property
    def width_px(self) -> int:
        return self.grid_w * self.cell

    @property
    def height_px(self) -> int:
        return self.grid_h * self.cell

    @property
    def margin(self) -> float:
        """Closest a position may come to the border, in cells.

        Half a patch, plus a pixel of slack so bilinear sampling never has to
        clamp against the far edge.
        """
        return (self.patch / 2.0 + 1.0) / self.cell

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(lo_x, hi_x, lo_y, hi_y)`` in cells."""
        m = self.margin
        return m, self.grid_w - m, m, self.grid_h - m

    def observe(self, positions: np.ndarray) -> np.ndarray:
        """Patches at ``(N, 2)`` positions in cells -> ``(N, patch, patch)``.

        Bilinear, which is the whole point: the patch varies continuously with
        position rather than snapping to a cell.
        """
        px = np.asarray(positions, dtype=np.float64) * self.cell
        offsets = np.arange(self.patch) - (self.patch - 1) / 2.0
        xs = px[:, 0:1] + offsets  # (N, P) columns
        ys = px[:, 1:2] + offsets  # (N, P) rows

        h, w = self.image.shape
        x0 = np.floor(xs)
        y0 = np.floor(ys)
        wx = (xs - x0)[:, None, :]  # (N, 1, P)
        wy = (ys - y0)[:, :, None]  # (N, P, 1)
        x0i = np.clip(x0, 0, w - 1).astype(np.intp)[:, None, :]
        y0i = np.clip(y0, 0, h - 1).astype(np.intp)[:, :, None]
        x1i = np.clip(x0 + 1, 0, w - 1).astype(np.intp)[:, None, :]
        y1i = np.clip(y0 + 1, 0, h - 1).astype(np.intp)[:, :, None]

        top = self.image[y0i, x0i] * (1 - wx) + self.image[y0i, x1i] * wx
        bottom = self.image[y1i, x0i] * (1 - wx) + self.image[y1i, x1i] * wx
        return (top * (1 - wy) + bottom * wy).astype(np.float32)

    def cell_of(self, positions: np.ndarray) -> np.ndarray:
        """Lattice cell index per position. **Analysis only** -- never an input."""
        col = np.clip(np.floor(positions[:, 0]), 0, self.grid_w - 1).astype(int)
        row = np.clip(np.floor(positions[:, 1]), 0, self.grid_h - 1).astype(int)
        return row * self.grid_w + col


@dataclass(frozen=True)
class ImageWalk:
    """A trajectory across the image.

    At step ``t`` the agent is at ``positions[t]``, sees ``patches[t]`` and
    moves by ``velocities[t]``, arriving at ``positions[t + 1]``. The final
    velocity is zero, mirroring ``NO_ACTION``.

    ``positions`` is for analysis only, exactly as ``locations`` is in the
    discrete world. The model never sees it.
    """

    positions: np.ndarray  # (T, 2) cells
    patches: np.ndarray  # (T, P, P) float32
    velocities: np.ndarray  # (T, 2) cells

    def __len__(self) -> int:
        return len(self.positions)

    def validate(self, env: ImageEnvironment) -> None:
        """Raise if this walk is not a legal path through ``env``."""
        if not (len(self.positions) == len(self.patches) == len(self.velocities)):
            raise ValueError("walk arrays have inconsistent lengths")
        if np.any(self.velocities[-1] != 0):
            raise ValueError("final velocity must be zero")
        moved = self.positions[:-1] + self.velocities[:-1]
        if not np.allclose(moved, self.positions[1:], atol=1e-6):
            raise ValueError("velocities do not lead to the next position")
        lo_x, hi_x, lo_y, hi_y = env.bounds
        inside = (
            (self.positions[:, 0] >= lo_x - 1e-6)
            & (self.positions[:, 0] <= hi_x + 1e-6)
            & (self.positions[:, 1] >= lo_y - 1e-6)
            & (self.positions[:, 1] <= hi_y + 1e-6)
        )
        if not inside.all():
            raise ValueError("walk leaves the arena")
        if not np.allclose(env.observe(self.positions), self.patches, atol=1e-5):
            raise ValueError("patches do not match the environment")


def make_image_environment(
    grid_w: int,
    grid_h: int,
    cell: int,
    patch: int,
    n_motifs: int,
    rng: np.random.Generator,
    # How smooth each motif is, in pixels. This is the single most consequential
    # number in the world, and not for the reason it looks like. Two patches
    # compared elementwise correlate by the *texture's* autocorrelation at their
    # separation -- pixel (i, j) of one is a different place in the image from
    # pixel (i, j) of the other -- so it is `motif_sigma`, not the patch width,
    # that decides how far apart two views stay recognisably alike. At sigma 2 a
    # memory from a fifth of a cell away already correlates 0.33 with what is
    # actually there, which is to say memory is nearly useless; at sigma 4 it is
    # 0.46. Raising it costs ambiguity, since smoother motifs are less
    # distinctive, and `measure_ambiguity` is what prices that.
    motif_sigma: float = 4.0,
    blur: float = 0.0,
) -> ImageEnvironment:
    """Paint a fresh motif assignment onto a lattice and assemble the image.

    ``blur`` smooths the assembled image, which softens cell seams at the cost
    of making a cell's appearance depend slightly on its neighbours -- so the
    default is 0, keeping "same motif implies identical appearance" exact and
    ambiguity clean to measure.
    """
    # A patch wider than a cell always spans several motifs, so it is a
    # conjunction and aliases far less often -- ambiguity falls toward 1, and
    # `measure_ambiguity` is what says by how much. Allowed, because patch size
    # also sets how far two views stay similar, and that has to be traded
    # against ambiguity rather than fixed by fiat. Bounded so the trade stays
    # visible instead of silently disappearing.
    if patch > 4 * cell:
        raise ValueError("patch spans too many cells for ambiguity to survive")
    motifs = np.stack(
        [_smooth_noise((cell, cell), motif_sigma, rng) for _ in range(n_motifs)]
    )
    cell_motifs = rng.integers(0, n_motifs, size=grid_w * grid_h).astype(int)
    tiles = motifs[cell_motifs].reshape(grid_h, grid_w, cell, cell)
    image = tiles.transpose(0, 2, 1, 3).reshape(grid_h * cell, grid_w * cell)
    if blur > 0:
        image = gaussian_filter(image, sigma=blur, mode="reflect")
    image = (image - image.mean()) / (image.std() + 1e-8)
    return ImageEnvironment(
        image=image.astype(np.float32),
        cell=cell,
        grid_w=grid_w,
        grid_h=grid_h,
        motifs=motifs,
        cell_motifs=cell_motifs,
        patch=patch,
    )


def load_backdrop(path: str | Path) -> np.ndarray:
    """Read a photograph as a standardised greyscale float array.

    Photographs are the point of the exercise and also a hazard, so it is worth
    being clear about which properties change.

    What gets *better*: natural images carry structure at every scale, so two
    views stay recognisably alike over a far longer distance than the synthetic
    texture manages. That is not a stylistic preference -- it is the exact
    quantity that decides whether a memory from a fifth of a cell away is worth
    anything, and the synthetic world's short correlation length is what caps
    perfect-localisation retrieval near 5%.

    What gets *worse*: ambiguity stops being a number that was chosen and
    becomes an accident of the photograph. Sky is featureless, foliage is
    self-similar, and in such regions localisation is genuinely impossible, so a
    poor score cannot be attributed to the model. Always report
    `measure_ambiguity` alongside a result on a real image; it is the only thing
    standing between "the model failed" and "the picture had nothing in it".
    """
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as data:
            image = data[list(data.keys())[0]]
    else:
        import matplotlib.image as mpimg

        image = mpimg.imread(path)
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 3:  # luminance weights, not a flat mean
        image = image[..., :3] @ np.array([0.299, 0.587, 0.114])
    return ((image - image.mean()) / (image.std() + 1e-8)).astype(np.float32)


def split_backdrop(
    backdrop: np.ndarray, fraction: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Cut a photograph into disjoint left and right sources.

    Training arenas come from one side, held-out arenas from the other, which
    is the only way "unseen environment" means anything when every arena is a
    crop of the same picture. Restricting crops to overlapping *ranges* is not
    enough -- two crops from ranges that merely start apart can still share most
    of their pixels -- so the split is made on the array itself.
    """
    edge = int(backdrop.shape[1] * fraction)
    return backdrop[:, :edge], backdrop[:, edge:]


def crop_environments(
    backdrop: np.ndarray,
    n_environments: int,
    grid: int,
    cell: int,
    patch: int,
    rng: np.random.Generator,
    x_range: tuple[float, float] = (0.0, 1.0),
) -> list[ImageEnvironment]:
    """Cut ``n_environments`` arenas out of one photograph.

    Distinct crops of one image play the role that fresh motif assignments play
    in the synthetic world: same statistics, different layout. ``x_range``
    restricts crops to a horizontal band of the source, which is how training
    and held-out environments are kept genuinely disjoint -- overlapping crops
    would make "unseen" mean nothing.
    """
    height, width = backdrop.shape
    span = grid * cell
    if span > height or span > width:
        raise ValueError(
            f"backdrop {width}x{height} is smaller than a {span}x{span} arena"
        )
    lo = int(x_range[0] * (width - span))
    hi = max(lo + 1, int(x_range[1] * (width - span)))

    environments = []
    for _ in range(n_environments):
        x = int(rng.integers(lo, hi))
        y = int(rng.integers(0, height - span + 1))
        window = backdrop[y : y + span, x : x + span]
        window = (window - window.mean()) / (window.std() + 1e-8)
        environments.append(
            ImageEnvironment(
                image=window.astype(np.float32),
                cell=cell,
                grid_w=grid,
                grid_h=grid,
                motifs=None,
                cell_motifs=None,
                patch=patch,
            )
        )
    return environments


def generate_image_walks(
    env: ImageEnvironment,
    batch_size: int,
    length: int,
    rng: np.random.Generator,
    speed: float = 0.5,
    turn_sigma: float = 0.6,
) -> list[ImageWalk]:
    """Sample smooth walks that keep the whole patch inside the image.

    Heading follows a random walk, as in the continuous world. A step that
    would push the patch past the border is not truncated -- the heading is
    reflected and resampled until the step lands inside -- so a velocity always
    means "move by exactly this much". Clipping the step instead would break the
    rule the operator algebra depends on, in the same way silently converting a
    blocked grid move into a no-op would.
    """
    lo_x, hi_x, lo_y, hi_y = env.bounds
    positions = np.empty((batch_size, length, 2))
    velocities = np.zeros((batch_size, length, 2))

    current = np.stack(
        [rng.uniform(lo_x, hi_x, batch_size), rng.uniform(lo_y, hi_y, batch_size)],
        axis=1,
    )
    heading = rng.uniform(0, 2 * np.pi, batch_size)

    for step in range(length):
        positions[:, step] = current
        if step == length - 1:
            break

        proposal = current
        for _ in range(16):  # reflect and retry until the step lands inside
            proposal = current + speed * np.stack(
                [np.cos(heading), np.sin(heading)], axis=1
            )
            outside = (
                (proposal[:, 0] < lo_x)
                | (proposal[:, 0] > hi_x)
                | (proposal[:, 1] < lo_y)
                | (proposal[:, 1] > hi_y)
            )
            if not outside.any():
                break
            heading[outside] = rng.uniform(0, 2 * np.pi, int(outside.sum()))
        proposal = np.clip(proposal, [lo_x, lo_y], [hi_x, hi_y])

        velocities[:, step] = proposal - current
        current = proposal
        heading = heading + rng.normal(0, turn_sigma, batch_size)

    flat = env.observe(positions.reshape(-1, 2))
    patches = flat.reshape(batch_size, length, env.patch, env.patch)
    return [
        ImageWalk(
            positions=positions[i], patches=patches[i], velocities=velocities[i]
        )
        for i in range(batch_size)
    ]


def _correlate(patches: np.ndarray) -> np.ndarray:
    """Flatten to mean-removed unit vectors, so a dot product is a correlation."""
    flat = patches.reshape(len(patches), -1).astype(np.float64)
    flat -= flat.mean(axis=1, keepdims=True)
    return flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)


def measure_ambiguity(
    env: ImageEnvironment,
    rng: np.random.Generator,
    n_queries: int = 256,
    threshold: float = 0.9,
) -> dict:
    """How many places share an observation -- the "~2.7 locations per symbol"
    of the discrete world, measured rather than assumed.

    The comparison set is the query translated by **whole cells**, not a fresh
    random sample. That matters: two independently drawn positions almost never
    share a sub-cell offset, so random pairs report near-zero ambiguity even
    when the world is built to repeat. Translating by whole cells asks the
    question the construction actually answers -- *this* view, seen from
    somewhere else -- and recovers the aliasing the motif tiling puts there.

    ``cells_per_patch`` counts the query itself, so 1.0 means every patch is
    unique and prediction could in principle become lookup.

    ``random_pair_rate`` keeps the naive measure alongside it, because the two
    diverge sharply and the gap is informative: it says the ambiguity here is
    structured (same view, different place) rather than diffuse.
    """
    lo_x, hi_x, lo_y, hi_y = env.bounds
    queries = np.stack(
        [rng.uniform(lo_x, hi_x, n_queries), rng.uniform(lo_y, hi_y, n_queries)],
        axis=1,
    )
    shifts = np.stack(
        np.meshgrid(np.arange(env.grid_w), np.arange(env.grid_h), indexing="xy"),
        axis=-1,
    ).reshape(-1, 2).astype(np.float64)

    counts = []
    for query in queries:
        offset = query - np.floor(query)
        candidates = shifts + offset
        inside = (
            (candidates[:, 0] >= lo_x) & (candidates[:, 0] <= hi_x)
            & (candidates[:, 1] >= lo_y) & (candidates[:, 1] <= hi_y)
        )
        vectors = _correlate(env.observe(candidates[inside]))
        target = _correlate(env.observe(query[None]))[0]
        counts.append(int((vectors @ target > threshold).sum()))

    random_positions = np.stack(
        [rng.uniform(lo_x, hi_x, 512), rng.uniform(lo_y, hi_y, 512)], axis=1
    )
    random_vectors = _correlate(env.observe(random_positions))
    similarity = random_vectors @ random_vectors.T
    np.fill_diagonal(similarity, -1.0)

    return {
        "threshold": threshold,
        "cells_per_patch": float(np.mean(counts)),
        "max_cells_per_patch": int(np.max(counts)),
        "unique_fraction": float(np.mean(np.array(counts) <= 1)),
        "random_pair_rate": float((similarity > threshold).mean()),
    }
