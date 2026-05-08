"""Synthetic signal generators for Delta.72 experiments.

Each generator returns (signal, baseline) arrays for use with the coherence engine.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def structured_with_noise(
    n: int = 1000,
    freq: float = 0.05,
    noise_level: float = 0.1,
    seed: int | None = None,
) -> tuple[NDArray, NDArray]:
    """Sinusoidal baseline + Gaussian noise.

    Used in Experiment 1 (Coherence vs Noise Threshold).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    baseline = np.sin(2 * np.pi * freq * t)
    signal = baseline + rng.normal(0, noise_level, n)
    return signal, baseline


def shock_recovery(
    n: int = 1000,
    freq: float = 0.05,
    shock_time: float = 0.4,
    shock_magnitude: float = 3.0,
    recovery_rate: float = 0.05,
    noise_level: float = 0.05,
    seed: int | None = None,
) -> tuple[NDArray, NDArray]:
    """Signal with injected shock and exponential recovery.

    Used in Experiment 2 (Recovery Dynamics After Shock).
    recovery_rate controls how fast the system returns: higher = faster.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    baseline = np.sin(2 * np.pi * freq * t)
    signal = baseline.copy()

    shock_idx = int(n * shock_time)
    for i in range(shock_idx, n):
        decay = np.exp(-recovery_rate * (i - shock_idx))
        signal[i] += shock_magnitude * decay

    signal += rng.normal(0, noise_level, n)
    return signal, baseline


def hidden_drift(
    n: int = 1000,
    freq: float = 0.05,
    drift_start: float = 0.3,
    drift_rate: float = 0.005,
    noise_level: float = 0.1,
    failure_threshold: float = 2.0,
    seed: int | None = None,
) -> tuple[NDArray, NDArray, int]:
    """Signal with slow drift + noise, simulating hidden degradation.

    Used in Experiment 3 (Hidden Drift Before Visible Failure).
    Returns (signal, baseline, failure_index).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    baseline = np.sin(2 * np.pi * freq * t)
    signal = baseline.copy()

    drift_idx = int(n * drift_start)
    drift = np.zeros(n)
    for i in range(drift_idx, n):
        drift[i] = drift_rate * (i - drift_idx)

    signal += drift + rng.normal(0, noise_level, n)

    # Find where drift exceeds failure threshold
    failure_index = n
    for i in range(drift_idx, n):
        if drift[i] > failure_threshold:
            failure_index = i
            break

    return signal, baseline, failure_index


def multi_coherence_shock(
    n: int = 1000,
    freq: float = 0.05,
    coherence_levels: tuple[float, ...] = (0.95, 0.7, 0.4, 0.1),
    shock_magnitude: float = 3.0,
    shock_time: float = 0.5,
    seed: int | None = None,
) -> list[tuple[NDArray, NDArray, float]]:
    """Multiple systems with varying coherence levels hit by same shock.

    Used in Experiment 4 (Shock Response vs Coherence).
    Returns list of (signal, baseline, coherence_param).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    baseline = np.sin(2 * np.pi * freq * t)
    results = []

    for coh in coherence_levels:
        noise = (1.0 - coh) * 1.5
        recovery_rate = coh * 0.1
        signal = baseline.copy()
        signal += rng.normal(0, noise, n)

        shock_idx = int(n * shock_time)
        for i in range(shock_idx, n):
            decay = np.exp(-recovery_rate * (i - shock_idx))
            signal[i] += shock_magnitude * decay

        results.append((signal, baseline.copy(), coh))

    return results


def cross_system_signals(
    n: int = 1000,
    noise_level: float = 0.3,
    seed: int | None = None,
) -> dict[str, tuple[NDArray, NDArray]]:
    """Different signal types for cross-system generalization.

    Used in Experiment 5.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    signals = {}

    # Sinusoidal
    baseline_sin = np.sin(2 * np.pi * 0.05 * t)
    signals["sinusoidal"] = (
        baseline_sin + rng.normal(0, noise_level, n),
        baseline_sin,
    )

    # Chaotic (logistic map)
    x = np.zeros(n)
    x[0] = 0.1
    r = 3.9  # chaotic regime
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    baseline_chaos = x.copy()
    signals["chaotic"] = (
        baseline_chaos + rng.normal(0, noise_level * 0.3, n),
        baseline_chaos,
    )

    # Piecewise
    baseline_pw = np.zeros(n)
    segment = n // 4
    baseline_pw[:segment] = np.linspace(0, 1, segment)
    baseline_pw[segment:2 * segment] = 1.0
    baseline_pw[2 * segment:3 * segment] = np.linspace(1, -1, segment)
    baseline_pw[3 * segment:] = np.linspace(-1, 0, n - 3 * segment)
    signals["piecewise"] = (
        baseline_pw + rng.normal(0, noise_level * 0.5, n),
        baseline_pw,
    )

    # Stochastic (random walk)
    steps = rng.choice([-1, 1], size=n) * 0.1
    baseline_rw = np.cumsum(steps)
    signals["stochastic"] = (
        baseline_rw + rng.normal(0, noise_level * 0.5, n),
        baseline_rw,
    )

    return signals


def monte_carlo_trial(
    n: int = 1000,
    seed: int | None = None,
) -> tuple[NDArray, NDArray, dict]:
    """Single randomized trial for Monte Carlo analysis.

    Used in Experiment 6. Randomizes drift timing, noise level, failure severity.
    Returns (signal, baseline, params).
    """
    rng = np.random.default_rng(seed)

    freq = rng.uniform(0.02, 0.1)
    drift_start = rng.uniform(0.2, 0.5)
    drift_rate = rng.uniform(0.001, 0.01)
    noise_level = rng.uniform(0.05, 0.5)
    failure_threshold = rng.uniform(1.0, 3.0)

    signal, baseline, failure_idx = hidden_drift(
        n=n,
        freq=freq,
        drift_start=drift_start,
        drift_rate=drift_rate,
        noise_level=noise_level,
        failure_threshold=failure_threshold,
        seed=seed,
    )

    params = {
        "freq": freq,
        "drift_start": drift_start,
        "drift_rate": drift_rate,
        "noise_level": noise_level,
        "failure_threshold": failure_threshold,
        "failure_index": failure_idx,
    }

    return signal, baseline, params
