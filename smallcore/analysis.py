"""Analysis harness.

Success cannot be read off a loss curve, so measurement tooling comes first.

    M1  linear probes -- can true location be decoded from the position code?
    M5  rate maps, periodicity scores, field scores

Everything here uses walk ``locations``, which the model itself never sees.
That is the point: locations are ground truth for *measurement*, never input.

A note on why rate maps need fixed-start walks
----------------------------------------------
From M2 onward walks begin at a random location, so the position code carries
displacement from an unknown origin. The map from location to code is then not
a function at all -- the same cell gets a different code in every walk, offset
by wherever that walk began -- and averaging activation per location across
walks would average unrelated things into mush.

Analysing on fixed-start walks makes displacement and location one-to-one, so a
rate map is well defined. The transition operators are start-agnostic, so this
measures the code the model actually learned rather than a special case of it.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage
from torch import nn


def fit_linear_probe(
    codes: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    iters: int = 400,
    lr: float = 0.05,
    weight_decay: float = 1e-4,
) -> nn.Linear:
    """Fit a multinomial logistic probe from ``codes`` ``(N, D)`` to ``labels``.

    Full-batch Adam; deliberately simple. The probe being *linear* is what
    makes the result meaningful -- it certifies the information is present in
    an easily-readable form, not merely recoverable by another network.
    """
    codes = codes.detach()
    probe = nn.Linear(codes.shape[1], n_classes).to(codes.device)
    optimiser = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(iters):
        optimiser.zero_grad()
        loss = nn.functional.cross_entropy(probe(codes), labels)
        loss.backward()
        optimiser.step()
    return probe


@torch.no_grad()
def probe_predictions(probe: nn.Linear, codes: torch.Tensor) -> torch.Tensor:
    """Predicted class per row of ``codes``."""
    return probe(codes.detach()).argmax(dim=-1)


@torch.no_grad()
def probe_accuracy(
    probe: nn.Linear, codes: torch.Tensor, labels: torch.Tensor
) -> float:
    """Fraction of ``labels`` the probe decodes correctly."""
    return (probe_predictions(probe, codes) == labels).float().mean().item()


# --------------------------------------------------------------------------- #
# M5: rate maps, periodicity, fields
# --------------------------------------------------------------------------- #


def rate_maps(
    activations: np.ndarray,
    locations: np.ndarray,
    width: int,
    height: int,
    smooth: float = 1.0,
) -> np.ndarray:
    """Mean activation per location, per unit, as ``(n_units, height, width)``.

    Args:
        activations: ``(N, n_units)`` activations pooled over all steps.
        locations: ``(N,)`` true location index for each row.
        smooth: Gaussian smoothing sigma in cells. Coverage is partial (a
            300-step walk visits ~72% of the grid), so unvisited cells are
            filled by smoothing the visit-weighted sums and dividing by the
            smoothed visit counts -- which interpolates holes rather than
            treating them as zero activation.
    """
    n_units = activations.shape[1]
    sums = np.zeros((n_units, height * width), dtype=np.float64)
    counts = np.zeros(height * width, dtype=np.float64)
    np.add.at(counts, locations, 1.0)
    for unit in range(n_units):
        np.add.at(sums[unit], locations, activations[:, unit])

    sums = sums.reshape(n_units, height, width)
    counts = counts.reshape(height, width)
    if smooth > 0:
        counts_s = ndimage.gaussian_filter(counts, smooth, mode="nearest")
        maps = np.stack(
            [ndimage.gaussian_filter(s, smooth, mode="nearest") for s in sums]
        )
        return maps / np.maximum(counts_s, 1e-9)
    return sums / np.maximum(counts, 1e-9)


def autocorrelogram(rate_map: np.ndarray) -> np.ndarray:
    """Normalised 2D autocorrelation of one rate map.

    Pearson correlation at each spatial lag, computed only over the overlapping
    region for that lag -- the standard construction, and the reason the result
    is ``(2h - 1, 2w - 1)`` rather than the map's own size.
    """
    height, width = rate_map.shape
    centred = rate_map - rate_map.mean()
    out = np.zeros((2 * height - 1, 2 * width - 1))
    for dy in range(-(height - 1), height):
        for dx in range(-(width - 1), width):
            ys = slice(max(0, dy), min(height, height + dy))
            xs = slice(max(0, dx), min(width, width + dx))
            a = centred[ys, xs]
            b = centred[
                slice(max(0, -dy), min(height, height - dy)),
                slice(max(0, -dx), min(width, width - dx)),
            ]
            if a.size < 4:
                continue
            a = a - a.mean()
            b = b - b.mean()
            denominator = np.sqrt((a * a).sum() * (b * b).sum())
            if denominator > 1e-12:
                out[dy + height - 1, dx + width - 1] = (a * b).sum() / denominator
    return out


def periodicity_score(rate_map: np.ndarray) -> float:
    """Hexagonal periodicity ("gridness") of one rate map.

    Correlate the autocorrelogram against rotations of itself and take

        min(corr at 60, 120) - max(corr at 30, 90, 150)

    A hexagonal lattice repeats every 60 degrees, so it correlates highly at 60
    and 120 and poorly at 30, 90 and 150; a square or striped pattern does not
    produce that contrast. **0.3-0.5 is the conventional threshold** for calling
    a unit periodic.

    The central peak is excluded (every map correlates perfectly with itself at
    zero lag, which would swamp the comparison) along with the outer rim, where
    few cells overlap and the correlation is noise. On an 11x11 arena the
    annulus is small, so treat scores as indicative rather than definitive --
    this measure was designed for arenas holding several grid periods.
    """
    correlogram = autocorrelogram(rate_map)
    size = correlogram.shape[0]
    centre = size // 2
    y, x = np.ogrid[:size, :size]
    radius = np.hypot(y - centre, x - centre)
    ring = (radius > size * 0.1) & (radius < size * 0.45)
    if ring.sum() < 8:
        return float("nan")

    base = correlogram[ring]
    correlations = {}
    for angle in (30, 60, 90, 120, 150):
        rotated = ndimage.rotate(correlogram, angle, reshape=False, order=1)
        other = rotated[ring]
        if base.std() < 1e-9 or other.std() < 1e-9:
            return float("nan")
        correlations[angle] = float(np.corrcoef(base, other)[0, 1])
    return min(correlations[60], correlations[120]) - max(
        correlations[30], correlations[90], correlations[150]
    )


def field_score(rate_map: np.ndarray, threshold: float = 0.5) -> float:
    """How concentrated a unit's activity is in ONE place.

    Threshold the map at ``threshold`` of its peak, label connected components,
    and return the firing mass in the largest component divided by the total.
    High means the unit fires in a single region -- a place field. Low means its
    activity is scattered, which is what a periodic unit looks like.
    """
    positive = rate_map - rate_map.min()
    peak = positive.max()
    if peak < 1e-9:
        return float("nan")
    mask = positive >= threshold * peak
    labels, n = ndimage.label(mask)
    if n == 0:
        return float("nan")
    masses = ndimage.sum(positive, labels, index=range(1, n + 1))
    return float(masses.max() / max(masses.sum(), 1e-9))


def spectral_structure(
    rate_map: np.ndarray, pad: int = 64, min_peak_ratio: float | None = None
) -> dict:
    """Classify a rate map by counting peaks in its 2D Fourier spectrum.

    The periodicity score in :func:`periodicity_score` asks one question --
    "is this hexagonal?" -- and a 1-D striped pattern answers "no" in a way
    that is indistinguishable from noise answering "no". Both land near zero.
    That blindness hid real structure: measured 2026-07-25, position units with
    obvious diagonal bands scored -0.01 to -0.02.

    Counting Fourier peaks separates the cases directly, because the number of
    peaks IS the symmetry:

        2 peaks  -> a band / stripe cell (one spatial frequency, one orientation)
        4 peaks  -> a square lattice
        6 peaks  -> a hexagonal lattice, i.e. a grid cell
        0 peaks  -> no dominant periodicity (a blob or a gradient)

    Bands matter in their own right: the standard account of how grid cells
    arise is interference between three bands at 60 degrees to each other, so a
    model producing bands has built the precursor rather than nothing.

    Returns wavelength in cells and orientation in degrees alongside the count.
    """
    centred = rate_map - rate_map.mean()
    if np.abs(centred).max() < 1e-9:
        return {"n_peaks": 0, "wavelength": float("nan"),
                "orientation": float("nan"), "peak_power": 0.0,
                "peak_ratio": 0.0, "radial_cv": 1.0}

    # Hann window suppresses the edge discontinuity that would otherwise smear
    # power across all frequencies; zero-padding buys frequency resolution.
    h, w = centred.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centred * window, s=(pad, pad))))

    centre = pad // 2
    y, x = np.ogrid[:pad, :pad]
    radius = np.hypot(y - centre, x - centre)
    # Exclude DC and the lowest frequencies, whose "period" exceeds the arena.
    valid = radius > 2.5
    if not valid.any():
        return {"n_peaks": 0, "wavelength": float("nan"),
                "orientation": float("nan"), "peak_power": 0.0,
                "peak_ratio": 0.0, "radial_cv": 1.0}

    masked = np.where(valid, spectrum, 0.0)
    peak = masked.max()
    if peak < 1e-9:
        return {"n_peaks": 0, "wavelength": float("nan"),
                "orientation": float("nan"), "peak_power": 0.0,
                "peak_ratio": 0.0, "radial_cv": 1.0}

    # Local maxima at >= 40% of the strongest peak. Fourier spectra of real
    # signals are symmetric, so genuine structure always appears in pairs.
    strong = (masked >= 0.4 * peak) & (
        masked >= ndimage.maximum_filter(masked, size=5) - 1e-12
    )
    labels, n = ndimage.label(strong)
    if n == 0:
        return {"n_peaks": 0, "wavelength": float("nan"),
                "orientation": float("nan"), "peak_power": 0.0,
                "peak_ratio": 0.0, "radial_cv": 1.0}
    centres = ndimage.center_of_mass(masked, labels, range(1, n + 1))

    best = max(centres, key=lambda c: masked[int(round(c[0])), int(round(c[1]))])
    dy, dx = best[0] - centre, best[1] - centre
    cycles_per_map = np.hypot(dy, dx) * (h / pad)  # rescale padded frequency
    wavelength = h / cycles_per_map if cycles_per_map > 1e-9 else float("nan")
    orientation = float(np.degrees(np.arctan2(dy, dx)) % 180.0)

    # How much the strongest peak stands out from the rest of the spectrum.
    # Counting peaks alone is NOT enough, and believing it produced a false
    # "7 of 30 units are hexagonal" on 2026-07-26: a smoothed noise map has a
    # lumpy spectrum too, and its lumps are local maxima above 40% of its own
    # maximum just as readily as a lattice's are. Peak COUNT says which
    # symmetry; peak RATIO says whether there is any signal to have a symmetry.
    # Dimensionless, so it survives the arbitrary scale of a rate map.
    background = float(np.median(masked[valid]))
    peak_ratio = float(peak / background) if background > 1e-12 else float("inf")

    # A real lattice puts every peak at the same spatial frequency, so the
    # spread of peak radii is near zero; noise scatters them.
    radii = [np.hypot(c[0] - centre, c[1] - centre) for c in centres]
    radial_cv = float(np.std(radii) / np.mean(radii)) if np.mean(radii) > 0 else 1.0

    if min_peak_ratio is not None and peak_ratio < min_peak_ratio:
        n = 0  # indistinguishable from noise; report no symmetry at all

    return {"n_peaks": int(n), "wavelength": float(wavelength),
            "orientation": orientation, "peak_power": float(peak),
            "peak_ratio": peak_ratio, "radial_cv": radial_cv}


def noise_peak_ratio(
    shape: tuple[int, int],
    smooth: float,
    rng: np.random.Generator,
    n_samples: int = 200,
    quantile: float = 0.99,
) -> float:
    """The peak ratio that structureless noise reaches, at a given quantile.

    Standing lesson 5 of the roadmap is to validate a metric on synthetic data
    before trusting it on real, and this is that validation made part of the
    measurement rather than a thing done once and forgotten. Rate maps are
    smoothed, and smoothing manufactures spectral lumps; how big those lumps
    get depends on the map size and the smoothing width, so the threshold
    cannot be a constant. Generate the matched null and read it off.

    Pass the result as ``min_peak_ratio`` to :func:`spectral_structure`.
    """
    ratios = []
    for _ in range(n_samples):
        noise = ndimage.gaussian_filter(
            rng.standard_normal(shape), smooth, mode="nearest"
        )
        ratios.append(spectral_structure(noise)["peak_ratio"])
    return float(np.quantile(ratios, quantile))


def rate_maps_coords(
    activations: np.ndarray,
    locations: np.ndarray,
    coords: np.ndarray,
    bin_size: float = 0.6,
    smooth: float = 1.2,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Rate maps binned in REAL space rather than by location index.

    :func:`rate_maps` reshapes the location index into a rectangle, which is
    correct only when the index grid *is* the geometry. It is not for a hex
    lattice: hex coordinates are offset (``col + 0.5 * (row % 2)``,
    ``row * sqrt(3) / 2``), so reshaping by index **shears the lattice**, and a
    hexagonal pattern drawn in a sheared frame cannot show six-fold symmetry in
    its autocorrelogram no matter what the model learned.

    Binning by ``topology.coords`` renders the map in the space the agent
    actually moves through. Returns ``(maps, extent)`` with maps shaped
    ``(n_units, ny, nx)``.
    """
    xy = coords[locations]
    x0, y0 = coords[:, 0].min(), coords[:, 1].min()
    x1, y1 = coords[:, 0].max(), coords[:, 1].max()
    nx = max(4, int(np.ceil((x1 - x0) / bin_size)) + 1)
    ny = max(4, int(np.ceil((y1 - y0) / bin_size)) + 1)

    ix = np.clip(((xy[:, 0] - x0) / bin_size).astype(int), 0, nx - 1)
    iy = np.clip(((xy[:, 1] - y0) / bin_size).astype(int), 0, ny - 1)
    flat = iy * nx + ix

    n_units = activations.shape[1]
    sums = np.zeros((n_units, ny * nx))
    counts = np.zeros(ny * nx)
    np.add.at(counts, flat, 1.0)
    for unit in range(n_units):
        np.add.at(sums[unit], flat, activations[:, unit])

    sums = sums.reshape(n_units, ny, nx)
    counts = counts.reshape(ny, nx)
    if smooth > 0:
        counts = ndimage.gaussian_filter(counts, smooth, mode="nearest")
        sums = np.stack(
            [ndimage.gaussian_filter(s, smooth, mode="nearest") for s in sums]
        )
    return sums / np.maximum(counts, 1e-9), (x0, x1, y0, y1)
