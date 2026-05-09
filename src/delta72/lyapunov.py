"""Rolling Local Lyapunov Exponents for Delta.72.

The largest Lyapunov exponent (LLE) measures the average exponential rate
of divergence (or convergence) of nearby trajectories in phase space.
It is the gold standard for distinguishing chaos from stability:

  LLE > 0  =>  chaos  (nearby trajectories diverge exponentially)
  LLE ~ 0  =>  edge of chaos / periodic (neutral stability)
  LLE < 0  =>  stable attractor (nearby trajectories converge)

Delta.72 connection:
  If the largest Lyapunov exponent becomes positive, the system is chaotic --
  coherence loss (low Delta) in a chaotic regime is qualitatively different from
  coherence loss due to noise. A noisy-but-stable system (LLE < 0) can recover
  coherence with filtering; a chaotic system (LLE > 0) has fundamentally
  unpredictable dynamics where coherence loss reflects genuine structural
  instability. Rolling local Lyapunov exponents give a time-resolved "is this
  becoming chaotic?" metric, complementing RQA determinism and wavelet
  coherence.

Method: Rosenstein et al. (1993) — largest Lyapunov exponent from a time
  series. Advantages over Wolf (1985): works with small, noisy datasets;
  requires only nearest-neighbor search (no Jacobian estimation); robust
  to embedding parameter choices.

References:
  - Rosenstein, Collins, De Luca (1993) — "A practical method for calculating
    largest Lyapunov exponents from small data sets"
  - Takens (1981) — Delay embedding theorem
  - Kantz (1994) — Robust algorithm for Lyapunov estimation

Implementation: pure numpy, no external libraries needed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Phase-space embedding (shared with rqa.py but self-contained here)
# ---------------------------------------------------------------------------

def _time_delay_embedding(
    signal: NDArray,
    embedding_dim: int = 3,
    delay: int = 1,
) -> NDArray:
    """Embed 1D signal into m-dimensional phase space via Takens' theorem.

    Args:
        signal: 1D time series of length N.
        embedding_dim: Embedding dimension m (typically 2-5).
        delay: Time delay tau between coordinates.

    Returns:
        (N - (m-1)*tau, m) array of state vectors.
    """
    n = len(signal)
    n_vectors = n - (embedding_dim - 1) * delay
    if n_vectors <= 0:
        raise ValueError(
            f"Signal too short ({n}) for embedding_dim={embedding_dim}, delay={delay}"
        )

    indices = np.arange(n_vectors)[:, None] + np.arange(embedding_dim) * delay
    return signal[indices]


# ---------------------------------------------------------------------------
# Nearest-neighbor search
# ---------------------------------------------------------------------------

def _find_nearest_neighbors(
    embedded: NDArray,
    theiler_window: int = 10,
) -> NDArray:
    """Find the index of the nearest neighbor for each point in phase space.

    Excludes temporal neighbors within the Theiler window to avoid
    false-near-neighbors caused by temporal correlation rather than
    genuine phase-space proximity.

    Args:
        embedded: (N, m) array of embedded state vectors.
        theiler_window: Minimum temporal separation (indices) for valid
            neighbors. Prevents selecting neighbors that are simply
            adjacent in the time series.

    Returns:
        (N,) array of nearest-neighbor indices.
    """
    n = len(embedded)
    nn_indices = np.zeros(n, dtype=np.int64)

    for i in range(n):
        # Compute distances from point i to all other points
        diffs = embedded - embedded[i]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))

        # Exclude self and temporal neighbors
        lo = max(0, i - theiler_window)
        hi = min(n, i + theiler_window + 1)
        dists[lo:hi] = np.inf

        nn_indices[i] = np.argmin(dists)

    return nn_indices


# ---------------------------------------------------------------------------
# Rosenstein's method (1993)
# ---------------------------------------------------------------------------

def local_lyapunov(
    signal: NDArray,
    embedding_dim: int = 3,
    delay: int = 1,
    theiler_window: int = 10,
    max_iter: int | None = None,
    fit_range: tuple[int, int] | None = None,
) -> float:
    """Compute the largest Lyapunov exponent using Rosenstein's method (1993).

    Algorithm:
      1. Time-delay embed the signal into m-dimensional phase space.
      2. For each embedded point, find its nearest neighbor (excluding
         temporal neighbors within the Theiler window).
      3. Track how the distance between each nearest-neighbor pair
         evolves over time (divergence curves).
      4. Average log(divergence) across all pairs at each time step.
      5. Fit a line to the linear-growth region of <ln(divergence)> vs t.
         The slope is the largest Lyapunov exponent.

    Delta.72 interpretation:
      If the largest Lyapunov exponent becomes positive, the system is
      chaotic -- coherence loss (low Delta) in a chaotic regime is
      qualitatively different from coherence loss due to noise. A positive
      LLE means the system has sensitive dependence on initial conditions
      and nearby states diverge exponentially.

    Args:
        signal: 1D time series (float64 preferred).
        embedding_dim: Phase space dimension m.
        delay: Time delay tau for embedding.
        theiler_window: Minimum temporal separation for nearest neighbors.
        max_iter: Maximum number of divergence steps to track.
            Defaults to N_embedded // 4.
        fit_range: (start, end) index range for linear fit on the
            divergence curve. If None, uses (0, max_iter).

    Returns:
        Largest Lyapunov exponent (float). Positive = chaos, negative = stable.
    """
    signal = np.asarray(signal, dtype=np.float64)
    embedded = _time_delay_embedding(signal, embedding_dim, delay)
    n = len(embedded)

    if max_iter is None:
        max_iter = max(n // 4, 10)
    max_iter = min(max_iter, n - 1)

    # Find nearest neighbor for each point
    nn_indices = _find_nearest_neighbors(embedded, theiler_window)

    # Track divergence of nearest-neighbor pairs over time
    # divergence[k] = list of ln(dist) at k steps ahead
    divergence = [[] for _ in range(max_iter)]

    for i in range(n):
        j = nn_indices[i]

        # Initial distance
        d0 = np.sqrt(np.sum((embedded[i] - embedded[j]) ** 2))
        if d0 == 0:
            continue  # skip zero-distance pairs

        for k in range(1, max_iter):
            # Check that both i+k and j+k are valid indices
            if i + k >= n or j + k >= n:
                break

            dk = np.sqrt(np.sum((embedded[i + k] - embedded[j + k]) ** 2))
            if dk > 0:
                divergence[k].append(np.log(dk))

    # Average log-divergence at each time step
    mean_divergence = np.full(max_iter, np.nan)
    for k in range(max_iter):
        if divergence[k]:
            mean_divergence[k] = np.mean(divergence[k])

    # Remove NaN entries for fitting
    valid = ~np.isnan(mean_divergence)
    valid_indices = np.where(valid)[0]

    if len(valid_indices) < 2:
        return 0.0

    # Apply fit range
    if fit_range is not None:
        start, end = fit_range
        mask = (valid_indices >= start) & (valid_indices < end)
        valid_indices = valid_indices[mask]

    if len(valid_indices) < 2:
        return 0.0

    # Linear fit: slope of <ln(divergence)> vs time = LLE
    t_fit = valid_indices.astype(np.float64)
    y_fit = mean_divergence[valid_indices]

    # Least-squares linear fit (numpy polyfit degree 1)
    coeffs = np.polyfit(t_fit, y_fit, 1)
    slope = coeffs[0]

    return float(slope)


def divergence_curve(
    signal: NDArray,
    embedding_dim: int = 3,
    delay: int = 1,
    theiler_window: int = 10,
    max_iter: int | None = None,
) -> NDArray:
    """Return the full mean log-divergence curve (for visualization).

    This is the intermediate result from Rosenstein's method: the average
    <ln(d(t))> across all nearest-neighbor pairs. The slope of the
    linear-growth region gives the largest Lyapunov exponent.

    Useful for:
      - Visually confirming the linear region used for LLE estimation.
      - Diagnosing embedding parameter choices.
      - Plotting divergence dynamics in reports.

    Args:
        signal: 1D time series.
        embedding_dim: Phase space dimension.
        delay: Time delay for embedding.
        theiler_window: Minimum temporal separation for nearest neighbors.
        max_iter: Maximum divergence steps to track.

    Returns:
        (max_iter,) array of mean log-divergence values (may contain NaN
        at steps where no valid pairs existed).
    """
    signal = np.asarray(signal, dtype=np.float64)
    embedded = _time_delay_embedding(signal, embedding_dim, delay)
    n = len(embedded)

    if max_iter is None:
        max_iter = max(n // 4, 10)
    max_iter = min(max_iter, n - 1)

    nn_indices = _find_nearest_neighbors(embedded, theiler_window)

    divergence = [[] for _ in range(max_iter)]

    for i in range(n):
        j = nn_indices[i]
        d0 = np.sqrt(np.sum((embedded[i] - embedded[j]) ** 2))
        if d0 == 0:
            continue

        for k in range(1, max_iter):
            if i + k >= n or j + k >= n:
                break
            dk = np.sqrt(np.sum((embedded[i + k] - embedded[j + k]) ** 2))
            if dk > 0:
                divergence[k].append(np.log(dk))

    mean_div = np.full(max_iter, np.nan)
    for k in range(max_iter):
        if divergence[k]:
            mean_div[k] = np.mean(divergence[k])

    return mean_div


# ---------------------------------------------------------------------------
# Rolling window Lyapunov
# ---------------------------------------------------------------------------

def rolling_lyapunov(
    signal: NDArray,
    window_size: int = 100,
    step: int = 10,
    embedding_dim: int = 3,
    delay: int = 1,
    theiler_window: int = 10,
    max_iter: int | None = None,
    fit_range: tuple[int, int] | None = None,
) -> tuple[NDArray, NDArray]:
    """Compute rolling local Lyapunov exponent across a time series.

    Slides a window across the signal and computes the largest Lyapunov
    exponent for each window position. Gives a time-resolved measure of
    dynamical stability: "is this region of the signal becoming chaotic?"

    Delta.72 integration:
      Use alongside rolling Delta scores. If Delta drops AND LLE becomes
      positive, the system is entering chaotic degradation (qualitatively
      different from noise-driven coherence loss). If Delta drops but LLE
      remains negative, the loss is likely noise or drift-based and
      potentially recoverable.

    Args:
        signal: 1D time series.
        window_size: Number of points in each analysis window.
        step: Stride between successive windows.
        embedding_dim: Phase space dimension for embedding.
        delay: Time delay for embedding.
        theiler_window: Minimum temporal separation for nearest neighbors.
        max_iter: Maximum divergence steps per window.
        fit_range: Linear fit range for each window's LLE estimate.

    Returns:
        (lle_values, window_centers) where:
          - lle_values: (n_windows,) array of Lyapunov exponents.
          - window_centers: (n_windows,) array of center indices for each window.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    if window_size > n:
        raise ValueError(
            f"window_size ({window_size}) exceeds signal length ({n})"
        )

    starts = np.arange(0, n - window_size + 1, step)
    lle_values = np.zeros(len(starts))
    window_centers = np.zeros(len(starts))

    for idx, start in enumerate(starts):
        segment = signal[start : start + window_size]
        lle_values[idx] = local_lyapunov(
            segment,
            embedding_dim=embedding_dim,
            delay=delay,
            theiler_window=theiler_window,
            max_iter=max_iter,
            fit_range=fit_range,
        )
        window_centers[idx] = start + window_size / 2.0

    return lle_values, window_centers


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def lyapunov_summary(
    signal: NDArray,
    window_size: int = 100,
    step: int = 10,
    embedding_dim: int = 3,
    delay: int = 1,
    theiler_window: int = 10,
    max_iter: int | None = None,
    fit_range: tuple[int, int] | None = None,
) -> dict:
    """Convenience function: compute rolling Lyapunov and return summary stats.

    Returns dict with:
        - lle_global: Lyapunov exponent computed on the full signal.
        - lle_mean: Mean of rolling Lyapunov exponents.
        - lle_std: Standard deviation of rolling LLEs.
        - lle_max: Maximum rolling LLE (worst-case chaos).
        - lle_min: Minimum rolling LLE (most stable window).
        - lle_trend: Linear trend (slope) of rolling LLEs over time.
            Positive trend => system moving toward chaos.
        - frac_positive: Fraction of windows with positive LLE (chaotic fraction).
        - n_windows: Number of rolling windows computed.
        - regime: String classification ('stable', 'edge', or 'chaotic')
            based on the global LLE.
    """
    signal = np.asarray(signal, dtype=np.float64)

    kwargs = dict(
        embedding_dim=embedding_dim,
        delay=delay,
        theiler_window=theiler_window,
        max_iter=max_iter,
        fit_range=fit_range,
    )

    # Global LLE on full signal
    lle_global = local_lyapunov(signal, **kwargs)

    # Rolling LLE
    lle_values, window_centers = rolling_lyapunov(
        signal,
        window_size=window_size,
        step=step,
        **kwargs,
    )

    n_windows = len(lle_values)

    # Trend: linear fit of LLE values over window centers
    if n_windows >= 2:
        trend_coeffs = np.polyfit(window_centers, lle_values, 1)
        lle_trend = float(trend_coeffs[0])
    else:
        lle_trend = 0.0

    # Regime classification
    if lle_global > 0.01:
        regime = "chaotic"
    elif lle_global < -0.01:
        regime = "stable"
    else:
        regime = "edge"

    return {
        "lle_global": float(lle_global),
        "lle_mean": float(np.mean(lle_values)) if n_windows > 0 else 0.0,
        "lle_std": float(np.std(lle_values)) if n_windows > 0 else 0.0,
        "lle_max": float(np.max(lle_values)) if n_windows > 0 else 0.0,
        "lle_min": float(np.min(lle_values)) if n_windows > 0 else 0.0,
        "lle_trend": lle_trend,
        "frac_positive": float(np.mean(lle_values > 0)) if n_windows > 0 else 0.0,
        "n_windows": n_windows,
        "regime": regime,
    }
