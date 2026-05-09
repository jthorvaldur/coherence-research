"""Phase Space Reconstruction for Delta.72.

Embeds time series in higher-dimensional space via Takens' theorem to
reconstruct the system's attractor. Healthy system = clean, compact orbit.
Degrading system = smeared, expanding attractor. This formalizes the
M (Memory-of-Attractor) operator with geometric rigor.

Key metrics:
  - Attractor dimension (correlation dimension D2)
  - Attractor spread (mean distance from centroid)
  - Attractor compactness (ratio of hull volume to point count)
  - Trajectory divergence rate

References:
  - Takens (1981) — Detecting strange attractors in turbulence
  - Grassberger & Procaccia (1983) — Correlation dimension

Implementation: pure numpy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def time_delay_embedding(
    signal: NDArray,
    embedding_dim: int = 3,
    delay: int = 1,
) -> NDArray:
    """Embed 1D signal into m-dimensional phase space (Takens' theorem).

    Returns (N - (m-1)*tau, m) array of state vectors.
    """
    n = len(signal)
    n_vectors = n - (embedding_dim - 1) * delay
    if n_vectors <= 0:
        raise ValueError(f"Signal too short ({n}) for dim={embedding_dim}, delay={delay}")
    indices = np.arange(n_vectors)[:, None] + np.arange(embedding_dim) * delay
    return signal[indices]


def attractor_spread(
    signal: NDArray,
    embedding_dim: int = 3,
    delay: int = 1,
) -> float:
    """Mean distance of trajectory points from attractor centroid.

    Low spread = compact attractor (healthy, predictable).
    High spread = smeared attractor (degrading, unpredictable).
    """
    embedded = time_delay_embedding(signal, embedding_dim, delay)
    centroid = embedded.mean(axis=0)
    distances = np.sqrt(np.sum((embedded - centroid) ** 2, axis=1))
    return float(distances.mean())


def attractor_compactness(
    signal: NDArray,
    embedding_dim: int = 3,
    delay: int = 1,
) -> float:
    """Ratio of attractor variance to number of points (normalized spread).

    Compact attractor → low compactness value → healthy system.
    """
    embedded = time_delay_embedding(signal, embedding_dim, delay)
    variance = np.var(embedded, axis=0).sum()
    return float(variance / len(embedded))


def correlation_dimension(
    signal: NDArray,
    embedding_dim: int = 3,
    delay: int = 1,
    n_radii: int = 15,
    r_min_frac: float = 0.01,
    r_max_frac: float = 0.5,
) -> float:
    """Estimate correlation dimension D2 via Grassberger-Procaccia algorithm.

    D2 measures the fractal dimension of the attractor. Low D2 = simple dynamics
    (periodic orbit), high D2 = complex dynamics (chaos or noise).

    For Delta.72: if D2 increases over time, the system's dynamics are becoming
    more complex — a sign of degradation.
    """
    embedded = time_delay_embedding(signal, embedding_dim, delay)
    n = len(embedded)
    if n < 20:
        return 0.0

    # Compute pairwise distances (subsample for speed)
    max_pairs = 2000
    if n > max_pairs:
        idx = np.random.choice(n, max_pairs, replace=False)
        embedded = embedded[idx]
        n = max_pairs

    dists = []
    for i in range(n):
        d = np.sqrt(np.sum((embedded[i] - embedded[i + 1:]) ** 2, axis=1))
        dists.extend(d.tolist())
    dists = np.array(dists)
    dists = dists[dists > 0]

    if len(dists) < 10:
        return 0.0

    # Correlation integral C(r) for various radii
    r_min = np.percentile(dists, r_min_frac * 100)
    r_max = np.percentile(dists, r_max_frac * 100)
    if r_min <= 0 or r_max <= r_min:
        return 0.0

    radii = np.logspace(np.log10(r_min), np.log10(r_max), n_radii)
    n_pairs = len(dists)
    C = np.array([np.sum(dists < r) / n_pairs for r in radii])

    # Fit slope of log(C) vs log(r) in scaling region
    valid = C > 0
    if valid.sum() < 3:
        return 0.0

    log_r = np.log(radii[valid])
    log_C = np.log(C[valid])

    # Linear fit
    coeffs = np.polyfit(log_r, log_C, 1)
    return float(coeffs[0])  # slope = D2


def trajectory_divergence(
    signal: NDArray,
    embedding_dim: int = 3,
    delay: int = 1,
    n_neighbors: int = 5,
) -> float:
    """Mean rate of divergence between nearby trajectory points.

    High divergence = sensitive dependence on initial conditions → chaos.
    Related to Lyapunov exponents but computed geometrically.
    """
    embedded = time_delay_embedding(signal, embedding_dim, delay)
    n = len(embedded)
    if n < n_neighbors + 2:
        return 0.0

    divergences = []
    # Sample points and track their nearest neighbors
    sample_idx = np.linspace(0, n - 2, min(50, n - 1), dtype=int)

    for idx in sample_idx:
        point = embedded[idx]
        # Find nearest neighbors (excluding temporal neighbors)
        dists = np.sqrt(np.sum((embedded - point) ** 2, axis=1))
        dists[max(0, idx - 5):min(n, idx + 6)] = np.inf  # Theiler window
        nn_idx = np.argsort(dists)[:n_neighbors]
        nn_idx = nn_idx[nn_idx < n - 1]

        if len(nn_idx) == 0:
            continue

        # Divergence after 1 step
        for ni in nn_idx:
            if idx + 1 < n and ni + 1 < n:
                d0 = dists[ni]
                d1 = np.sqrt(np.sum((embedded[idx + 1] - embedded[ni + 1]) ** 2))
                if d0 > 1e-12:
                    divergences.append(d1 / d0)

    return float(np.mean(divergences)) if divergences else 0.0


def phase_space_summary(
    signal: NDArray,
    embedding_dim: int = 3,
    delay: int = 1,
) -> dict:
    """Compute all phase space metrics for a time series."""
    return {
        "spread": attractor_spread(signal, embedding_dim, delay),
        "compactness": attractor_compactness(signal, embedding_dim, delay),
        "correlation_dim": correlation_dimension(signal, embedding_dim, delay),
        "divergence": trajectory_divergence(signal, embedding_dim, delay),
        "n_embedded": len(signal) - (embedding_dim - 1) * delay,
    }


def rolling_phase_space(
    signal: NDArray,
    window_size: int = 100,
    step: int = 20,
    **kwargs,
) -> list[dict]:
    """Compute phase space metrics in rolling windows."""
    n = len(signal)
    results = []
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        mid = start + window_size // 2
        try:
            ps = phase_space_summary(signal[start:end], **kwargs)
            ps["window_mid"] = mid
            ps["lifecycle_pct"] = mid / n * 100
            results.append(ps)
        except Exception:
            pass
    return results
