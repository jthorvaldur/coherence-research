"""Information Geometry for Delta.72.

Measures distance between probability distributions on a Riemannian manifold.
System state changes = geodesics on the statistical manifold. Coherence loss
= acceleration along the geodesic (the system is moving faster through
distribution space).

The Fisher Information Matrix defines the metric tensor on the statistical
manifold. The geodesic distance between two distributions is the minimum
information-theoretic "cost" of transforming one into the other.

For Delta.72: healthy system has slow movement on the manifold (low Fisher
divergence rate). Degrading system accelerates — the distribution is changing
rapidly.

Key metrics:
  - Fisher Information: local curvature of the log-likelihood
  - KL Divergence: asymmetric distance between distributions
  - Fisher-Rao distance: geodesic distance on statistical manifold
  - Divergence rate: speed of movement on the manifold

References:
  - Amari (1985) — Differential-Geometrical Methods in Statistics
  - Ay et al. (2017) — Information Geometry

Implementation: numpy, histogram-based probability estimation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def estimate_distribution(
    signal: NDArray,
    n_bins: int = 30,
    range_: tuple[float, float] | None = None,
) -> tuple[NDArray, NDArray]:
    """Estimate probability distribution from signal via histogram.

    Returns (probabilities, bin_centers).
    """
    if range_ is None:
        range_ = (signal.min() - 0.01, signal.max() + 0.01)
    counts, edges = np.histogram(signal, bins=n_bins, range=range_, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    dx = edges[1] - edges[0]
    probs = counts * dx
    probs = probs / (probs.sum() + 1e-15)  # normalize
    return probs, centers


def kl_divergence(p: NDArray, q: NDArray) -> float:
    """KL(P || Q) — Kullback-Leibler divergence."""
    mask = (p > 1e-15) & (q > 1e-15)
    if not mask.any():
        return 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def symmetric_kl(p: NDArray, q: NDArray) -> float:
    """Symmetric KL divergence: (KL(P||Q) + KL(Q||P)) / 2."""
    return (kl_divergence(p, q) + kl_divergence(q, p)) / 2


def fisher_information(
    signal: NDArray,
    n_bins: int = 30,
) -> float:
    """Estimate Fisher Information from signal.

    I(θ) = E[(d/dθ log p(x|θ))²]

    Approximated using the score function of the empirical distribution.
    High Fisher information = distribution is sharply peaked (informative).
    Low Fisher information = flat distribution (uncertain).
    """
    probs, centers = estimate_distribution(signal, n_bins)
    dx = centers[1] - centers[0] if len(centers) > 1 else 1.0

    # Score function: d/dx log p(x) ≈ (p(x+dx) - p(x-dx)) / (2dx * p(x))
    score_sq = np.zeros(len(probs))
    for i in range(1, len(probs) - 1):
        if probs[i] > 1e-15:
            dp = (probs[i + 1] - probs[i - 1]) / (2 * dx)
            score_sq[i] = (dp / probs[i]) ** 2

    return float(np.sum(score_sq * probs * dx))


def distribution_distance(
    signal1: NDArray,
    signal2: NDArray,
    n_bins: int = 30,
) -> dict:
    """Compute multiple distance measures between two signal distributions."""
    vmin = min(signal1.min(), signal2.min()) - 0.01
    vmax = max(signal1.max(), signal2.max()) + 0.01
    range_ = (vmin, vmax)

    p1, _ = estimate_distribution(signal1, n_bins, range_)
    p2, _ = estimate_distribution(signal2, n_bins, range_)

    # Hellinger distance
    hellinger = float(np.sqrt(np.sum((np.sqrt(p1) - np.sqrt(p2)) ** 2) / 2))

    # Jensen-Shannon divergence
    m = (p1 + p2) / 2
    js = (kl_divergence(p1, m) + kl_divergence(p2, m)) / 2

    return {
        "kl_forward": kl_divergence(p1, p2),
        "kl_reverse": kl_divergence(p2, p1),
        "symmetric_kl": symmetric_kl(p1, p2),
        "hellinger": hellinger,
        "jensen_shannon": js,
    }


def divergence_rate(
    signal: NDArray,
    window_size: int = 100,
    step: int = 20,
    n_bins: int = 20,
) -> list[dict]:
    """Compute rate of distributional change over time.

    Measures how fast the signal's distribution is changing — the "velocity"
    on the statistical manifold.
    """
    n = len(signal)
    results = []

    prev_dist = None
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        mid = start + window_size // 2
        window = signal[start:end]

        curr_dist, _ = estimate_distribution(window, n_bins)
        fi = fisher_information(window, n_bins)

        if prev_dist is not None:
            skl = symmetric_kl(prev_dist, curr_dist)
            hellinger = float(np.sqrt(np.sum((np.sqrt(prev_dist) - np.sqrt(curr_dist)) ** 2) / 2))
        else:
            skl = 0.0
            hellinger = 0.0

        results.append({
            "window_mid": mid,
            "lifecycle_pct": mid / n * 100,
            "fisher_info": fi,
            "divergence_rate": skl,
            "hellinger_rate": hellinger,
        })

        prev_dist = curr_dist

    return results


def info_geometry_summary(
    signal: NDArray,
    baseline: NDArray,
    **kwargs,
) -> dict:
    """Summary of information-geometric measures between signal and baseline."""
    dists = distribution_distance(signal, baseline, **kwargs)
    fi_signal = fisher_information(signal)
    fi_baseline = fisher_information(baseline)

    return {
        **dists,
        "fisher_info_signal": fi_signal,
        "fisher_info_baseline": fi_baseline,
        "fisher_ratio": fi_signal / fi_baseline if fi_baseline > 0 else 0,
    }
