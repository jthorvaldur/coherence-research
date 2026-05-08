"""Core Delta.72 coherence engine.

Δ = (P · A · R) / (D + N)

Where:
  P = Pattern Retention (correlation with expected baseline)
  A = Phase Alignment (temporal consistency / autocorrelation)
  R = Recovery (ability to return to baseline after deviation)
  D = Drift (mean deviation from baseline)
  N = Noise Amplification (variance / instability)

Extended operators:
  𝓜 = Memory-of-Attractor (does system remember how to return)
  𝓦 = Windowed Recovery (does it recover within time bounds)

Gated detection:
  ALERT if Δ < threshold AND 𝓜 < threshold AND 𝓦 < threshold
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Component calculations
# ---------------------------------------------------------------------------

def pattern_retention(signal: NDArray, baseline: NDArray) -> float:
    """P — Pearson correlation between signal and baseline."""
    if signal.std() == 0 or baseline.std() == 0:
        return 0.0
    return float(np.corrcoef(signal, baseline)[0, 1])


def phase_alignment(signal: NDArray, lag: int = 1) -> float:
    """A — Lag-1 autocorrelation as temporal consistency measure."""
    n = len(signal)
    if n < lag + 2:
        return 0.0
    x = signal[:-lag]
    y = signal[lag:]
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def recovery_score(signal: NDArray, baseline: NDArray, window: int | None = None) -> float:
    """R — Fraction of residual that decays toward zero over time.

    Computed as 1 - (trailing_residual / peak_residual).
    """
    residual = np.abs(signal - baseline)
    if residual.max() == 0:
        return 1.0
    if window is None:
        window = max(len(residual) // 4, 1)
    peak = residual.max()
    tail = residual[-window:].mean()
    return float(np.clip(1.0 - tail / peak, 0.0, 1.0))


def drift_score(signal: NDArray, baseline: NDArray) -> float:
    """D — Mean absolute deviation from baseline, normalized."""
    residual = signal - baseline
    return float(np.abs(residual).mean())


def noise_amplification(signal: NDArray, baseline: NDArray) -> float:
    """N — Variance of residuals (instability measure)."""
    residual = signal - baseline
    return float(np.var(residual))


# ---------------------------------------------------------------------------
# Composite Δ score
# ---------------------------------------------------------------------------

def coherence_score(
    signal: NDArray,
    baseline: NDArray,
    epsilon: float = 1e-8,
    lag: int = 1,
    recovery_window: int | None = None,
) -> dict:
    """Compute full Δ coherence score with all components.

    Returns dict with P, A, R, D, N, delta, and component breakdown.
    """
    P = pattern_retention(signal, baseline)
    A = phase_alignment(signal, lag=lag)
    R = recovery_score(signal, baseline, window=recovery_window)
    D = drift_score(signal, baseline)
    N = noise_amplification(signal, baseline)

    # Normalize numerator components to [0, 1] — P and A can be negative
    P_norm = max(P, 0.0)
    A_norm = max(A, 0.0)

    numerator = P_norm * A_norm * R
    denominator = D + N + epsilon

    delta = numerator / denominator

    return {
        "P": P,
        "A": A,
        "R": R,
        "D": D,
        "N": N,
        "P_norm": P_norm,
        "A_norm": A_norm,
        "numerator": numerator,
        "denominator": denominator,
        "delta": delta,
    }


# ---------------------------------------------------------------------------
# Extended operators
# ---------------------------------------------------------------------------

def memory_of_attractor(
    signal: NDArray,
    baseline: NDArray,
    n_shocks: int = 5,
    window: int | None = None,
) -> float:
    """𝓜 — Does the system remember how to return to its attractor?

    Measures average recovery across the top-N deviation episodes.
    High 𝓜 → system reliably returns to baseline after perturbation.
    """
    residual = np.abs(signal - baseline)
    if window is None:
        window = max(len(signal) // 20, 4)

    # Find top deviation peaks
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(residual, distance=window, height=residual.std())

    if len(peaks) == 0:
        return 1.0  # No deviations found → perfect memory

    # Sort by magnitude, take top n_shocks
    peak_heights = residual[peaks]
    top_idx = np.argsort(peak_heights)[-n_shocks:]
    top_peaks = peaks[top_idx]

    recoveries = []
    for pk in top_peaks:
        post_start = pk
        post_end = min(pk + window, len(residual))
        if post_end - post_start < 2:
            continue
        post_residual = residual[post_start:post_end]
        peak_val = residual[pk]
        if peak_val == 0:
            recoveries.append(1.0)
            continue
        tail_val = post_residual[-1]
        recoveries.append(float(np.clip(1.0 - tail_val / peak_val, 0.0, 1.0)))

    return float(np.mean(recoveries)) if recoveries else 1.0


def windowed_recovery(
    signal: NDArray,
    baseline: NDArray,
    max_recovery_steps: int | None = None,
    threshold_fraction: float = 0.5,
) -> float:
    """𝓦 — Does the system recover within time bounds?

    Measures fraction of deviation episodes that recover to < threshold_fraction
    of peak deviation within max_recovery_steps.
    """
    residual = np.abs(signal - baseline)
    if max_recovery_steps is None:
        max_recovery_steps = max(len(signal) // 10, 8)

    std = residual.std()
    if std == 0:
        return 1.0

    # Find deviation episodes (above 1 std)
    above = residual > std
    episodes = []
    in_episode = False
    start = 0
    for i, a in enumerate(above):
        if a and not in_episode:
            start = i
            in_episode = True
        elif not a and in_episode:
            episodes.append((start, i))
            in_episode = False
    if in_episode:
        episodes.append((start, len(above)))

    if not episodes:
        return 1.0

    recovered = 0
    for ep_start, ep_end in episodes:
        peak_val = residual[ep_start:ep_end].max()
        threshold = peak_val * threshold_fraction
        # Check recovery window after episode
        recovery_end = min(ep_end + max_recovery_steps, len(residual))
        if recovery_end <= ep_end:
            continue
        post = residual[ep_end:recovery_end]
        if len(post) > 0 and post.min() < threshold:
            recovered += 1

    return float(recovered / len(episodes))


# ---------------------------------------------------------------------------
# Gated alert check
# ---------------------------------------------------------------------------

def alert_check(
    signal: NDArray,
    baseline: NDArray,
    delta_threshold: float = 0.3,
    memory_threshold: float = 0.4,
    recovery_threshold: float = 0.4,
    **kwargs,
) -> dict:
    """Full gated detection: ALERT if Δ low AND 𝓜 low AND 𝓦 low.

    Returns dict with all scores, thresholds, and alert boolean.
    """
    scores = coherence_score(signal, baseline, **kwargs)
    M = memory_of_attractor(signal, baseline)
    W = windowed_recovery(signal, baseline)

    delta_low = scores["delta"] < delta_threshold
    memory_low = M < memory_threshold
    recovery_low = W < recovery_threshold
    alert = delta_low and memory_low and recovery_low

    return {
        **scores,
        "M": M,
        "W": W,
        "delta_threshold": delta_threshold,
        "memory_threshold": memory_threshold,
        "recovery_threshold": recovery_threshold,
        "delta_low": delta_low,
        "memory_low": memory_low,
        "recovery_low": recovery_low,
        "alert": alert,
    }


# ---------------------------------------------------------------------------
# Windowed / streaming coherence
# ---------------------------------------------------------------------------

class CoherenceEngine:
    """Streaming coherence scorer for time-series windows."""

    def __init__(
        self,
        window_size: int = 168,  # 1 week of hourly data
        delta_threshold: float = 0.3,
        memory_threshold: float = 0.4,
        recovery_threshold: float = 0.4,
    ):
        self.window_size = window_size
        self.delta_threshold = delta_threshold
        self.memory_threshold = memory_threshold
        self.recovery_threshold = recovery_threshold

    def score_window(self, signal: NDArray, baseline: NDArray) -> dict:
        """Score a single window."""
        return alert_check(
            signal,
            baseline,
            delta_threshold=self.delta_threshold,
            memory_threshold=self.memory_threshold,
            recovery_threshold=self.recovery_threshold,
        )

    def score_rolling(
        self,
        signal: NDArray,
        baseline: NDArray,
        step: int | None = None,
    ) -> list[dict]:
        """Score rolling windows across full time series."""
        if step is None:
            step = self.window_size // 4
        n = len(signal)
        results = []
        for start in range(0, n - self.window_size + 1, step):
            end = start + self.window_size
            result = self.score_window(signal[start:end], baseline[start:end])
            result["window_start"] = start
            result["window_end"] = end
            results.append(result)
        return results
