"""Ergodic Theory for Delta.72.

Studies long-term average behavior of dynamical systems. A coherent system
is ergodic (time averages = ensemble averages). Loss of ergodicity =
coherence breakdown — the system no longer explores its state space uniformly.

For Delta.72: test whether the signal's time-average properties match its
ensemble properties. Breaking of ergodicity signals a phase transition
in the system's dynamics — exactly what Δ threshold behavior captures.

Key metrics:
  - Ergodicity breaking parameter: ratio of time-average variance to
    ensemble variance. EB = 1 → ergodic. EB >> 1 → non-ergodic.
  - Recurrence time statistics: distribution of return times to states
  - Time-average convergence rate: how fast time averages stabilize

References:
  - Birkhoff (1931) — Ergodic theorem
  - Thirumalai & Mountain (1993) — Ergodicity breaking parameter
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ergodicity_breaking(
    signal: NDArray,
    n_segments: int = 10,
) -> float:
    """Ergodicity Breaking parameter.

    EB = Var(time averages across segments) / Var(ensemble)

    EB ≈ 0 → ergodic (all segments have same statistics)
    EB >> 0 → non-ergodic (segments have different statistics)
    """
    n = len(signal)
    seg_len = n // n_segments
    if seg_len < 2:
        return 0.0

    # Time averages of each segment
    segment_means = []
    for i in range(n_segments):
        start = i * seg_len
        end = start + seg_len
        segment_means.append(signal[start:end].mean())

    var_time_avg = np.var(segment_means)
    var_ensemble = np.var(signal)

    if var_ensemble < 1e-15:
        return 0.0

    return float(var_time_avg / var_ensemble)


def recurrence_times(
    signal: NDArray,
    n_bins: int = 10,
) -> dict:
    """Distribution of recurrence times — how long until the signal returns
    to a previously visited state.

    Short mean recurrence → ergodic (system revisits states frequently).
    Long mean recurrence → non-ergodic or transient behavior.
    """
    # Discretize signal into bins
    vmin, vmax = signal.min(), signal.max()
    if vmax - vmin < 1e-12:
        return {"mean_recurrence": 0, "std_recurrence": 0, "max_recurrence": 0}

    bins = np.digitize(signal, np.linspace(vmin, vmax, n_bins + 1)[1:-1])

    # For each bin, compute return times
    all_returns = []
    last_visit = {}

    for t, b in enumerate(bins):
        if b in last_visit:
            all_returns.append(t - last_visit[b])
        last_visit[b] = t

    if not all_returns:
        return {"mean_recurrence": len(signal), "std_recurrence": 0, "max_recurrence": len(signal)}

    returns = np.array(all_returns)
    return {
        "mean_recurrence": float(returns.mean()),
        "std_recurrence": float(returns.std()),
        "max_recurrence": int(returns.max()),
        "median_recurrence": float(np.median(returns)),
    }


def time_average_convergence(
    signal: NDArray,
    n_points: int = 20,
) -> dict:
    """Measure how fast the cumulative time average converges.

    Fast convergence → ergodic (short memory, mixing).
    Slow convergence → non-ergodic (long memory, trending).
    """
    n = len(signal)
    checkpoints = np.linspace(n // 10, n, n_points, dtype=int)

    cum_means = []
    for cp in checkpoints:
        cum_means.append(float(signal[:cp].mean()))

    final_mean = signal.mean()
    deviations = np.abs(np.array(cum_means) - final_mean)

    # Convergence rate: fit exponential decay to deviations
    if deviations[0] > 0:
        # Simple estimate: ratio of early to late deviation
        early = deviations[:n_points // 3].mean()
        late = deviations[-n_points // 3:].mean()
        convergence_rate = float(np.log(early / late + 1e-12) / (n_points * 2 / 3))
    else:
        convergence_rate = float('inf')

    return {
        "convergence_rate": convergence_rate,
        "final_deviation": float(deviations[-1]),
        "initial_deviation": float(deviations[0]),
        "is_convergent": deviations[-1] < deviations[0] * 0.1,
    }


def ergodic_summary(
    signal: NDArray,
    n_segments: int = 10,
) -> dict:
    """Complete ergodic analysis of a time series."""
    eb = ergodicity_breaking(signal, n_segments)
    rt = recurrence_times(signal)
    tc = time_average_convergence(signal)

    return {
        "ergodicity_breaking": eb,
        "is_ergodic": eb < 0.1,
        **rt,
        **tc,
    }


def rolling_ergodic(
    signal: NDArray,
    window_size: int = 200,
    step: int = 50,
    **kwargs,
) -> list[dict]:
    """Compute ergodic metrics in rolling windows."""
    n = len(signal)
    results = []
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        mid = start + window_size // 2
        try:
            summary = ergodic_summary(signal[start:end], **kwargs)
            summary["window_mid"] = mid
            summary["lifecycle_pct"] = mid / n * 100
            results.append(summary)
        except Exception:
            pass
    return results
