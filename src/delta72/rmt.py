"""Random Matrix Theory / Renormalization Group for Delta.72.

Eigenvalue spectrum analysis of multi-sensor correlation matrices.
Inspired by Charles Martin's WeightWatcher (RMT for neural net weights).

In Delta.72 context: build a correlation matrix from multiple sensors at each
time step. As the system degrades, the eigenvalue spectrum departs from the
Marchenko-Pastur distribution (random matrix baseline). The largest eigenvalue
(λ_max) tracks system-wide coherence, while the bulk spectrum tracks noise.

Key concepts:
  - Marchenko-Pastur law: eigenvalue distribution of random correlation matrices
  - Tracy-Widom: distribution of λ_max for random matrices
  - Power-law α: tail exponent of eigenvalue distribution (heavy tail = coherent)
  - Participation ratio: how many eigenvalues contribute to the signal

If Δ threshold behavior follows scaling laws → predictable from universality
class rather than empirically tuned. This connects to renormalization group
theory — threshold becomes a phase transition.

References:
  - Marchenko & Pastur (1967) — Distribution of eigenvalues
  - Charles Martin — WeightWatcher (github.com/CalculatedContent/WeightWatcher)
  - Bouchaud & Potters (2003) — Theory of Financial Risk (RMT chapter)

Implementation: numpy + scipy.linalg.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def correlation_matrix(signals: NDArray) -> NDArray:
    """Compute correlation matrix from multi-sensor data.

    Args:
        signals: (T, N_sensors) array of time series data.

    Returns:
        (N, N) correlation matrix.
    """
    # Standardize each sensor
    means = signals.mean(axis=0)
    stds = signals.std(axis=0)
    stds[stds < 1e-12] = 1.0
    standardized = (signals - means) / stds
    return np.corrcoef(standardized, rowvar=False)


def eigenvalue_spectrum(signals: NDArray) -> NDArray:
    """Compute sorted eigenvalues of the sensor correlation matrix.

    Returns eigenvalues sorted in descending order.
    """
    C = correlation_matrix(signals)
    eigenvalues = np.linalg.eigvalsh(C)
    return np.sort(eigenvalues)[::-1]


def marchenko_pastur_bounds(
    n_samples: int,
    n_sensors: int,
) -> tuple[float, float]:
    """Theoretical Marchenko-Pastur bounds for random matrix eigenvalues.

    For a random (T × N) matrix with T samples and N sensors:
    λ_± = (1 ± √(N/T))²

    Eigenvalues outside these bounds indicate genuine signal (not noise).
    """
    q = n_sensors / n_samples
    sqrt_q = np.sqrt(q)
    lambda_minus = (1 - sqrt_q) ** 2
    lambda_plus = (1 + sqrt_q) ** 2
    return float(lambda_minus), float(lambda_plus)


def signal_eigenvalues(
    signals: NDArray,
) -> dict:
    """Identify signal vs noise eigenvalues using Marchenko-Pastur.

    Returns dict with signal eigenvalues (above MP bound), noise eigenvalues,
    and the fraction of variance explained by signal.
    """
    T, N = signals.shape
    eigs = eigenvalue_spectrum(signals)
    _, mp_upper = marchenko_pastur_bounds(T, N)

    signal_eigs = eigs[eigs > mp_upper]
    noise_eigs = eigs[eigs <= mp_upper]

    total_var = eigs.sum()
    signal_var = signal_eigs.sum() if len(signal_eigs) > 0 else 0

    return {
        "eigenvalues": eigs.tolist(),
        "signal_eigenvalues": signal_eigs.tolist(),
        "noise_eigenvalues": noise_eigs.tolist(),
        "n_signal": len(signal_eigs),
        "n_noise": len(noise_eigs),
        "mp_upper": mp_upper,
        "lambda_max": float(eigs[0]),
        "signal_fraction": float(signal_var / total_var) if total_var > 0 else 0,
        "participation_ratio": float(eigs.sum() ** 2 / (eigs ** 2).sum()) if (eigs ** 2).sum() > 0 else 0,
    }


def power_law_alpha(eigenvalues: NDArray, min_eig: float = 0.01) -> float:
    """Estimate power-law exponent α of eigenvalue distribution.

    Heavy tail (low α) indicates strong coherent structure.
    Light tail (high α) indicates noise-dominated spectrum.

    Uses simple log-log regression on the survival function.
    """
    eigs = eigenvalues[eigenvalues > min_eig]
    if len(eigs) < 5:
        return 0.0

    sorted_eigs = np.sort(eigs)[::-1]
    ranks = np.arange(1, len(sorted_eigs) + 1)
    survival = ranks / len(sorted_eigs)

    # Log-log fit
    log_eig = np.log(sorted_eigs)
    log_surv = np.log(survival)

    valid = np.isfinite(log_eig) & np.isfinite(log_surv)
    if valid.sum() < 3:
        return 0.0

    coeffs = np.polyfit(log_eig[valid], log_surv[valid], 1)
    return float(-coeffs[0])  # negative slope = α


def rmt_summary(
    signals: NDArray,
) -> dict:
    """Full RMT analysis of multi-sensor correlation matrix.

    Args:
        signals: (T, N_sensors) array.

    Returns:
        Dict with eigenvalue analysis, MP comparison, and power-law fit.
    """
    T, N = signals.shape
    sig_info = signal_eigenvalues(signals)
    eigs = np.array(sig_info["eigenvalues"])
    alpha = power_law_alpha(eigs)
    mp_lower, mp_upper = marchenko_pastur_bounds(T, N)

    return {
        **sig_info,
        "alpha": alpha,
        "mp_lower": mp_lower,
        "mp_upper": mp_upper,
        "n_samples": T,
        "n_sensors": N,
        "q_ratio": N / T,
    }


def rolling_rmt(
    signals: NDArray,
    window_size: int = 100,
    step: int = 20,
) -> list[dict]:
    """Compute RMT metrics in rolling windows over multi-sensor data.

    Args:
        signals: (T, N_sensors) array.
        window_size: Window length in samples.
        step: Step between windows.

    Returns:
        List of dicts with per-window RMT metrics.
    """
    T, N = signals.shape
    results = []

    for start in range(0, T - window_size + 1, step):
        end = start + window_size
        window = signals[start:end]
        mid = start + window_size // 2

        try:
            summary = rmt_summary(window)
            summary["window_mid"] = mid
            summary["lifecycle_pct"] = mid / T * 100
            results.append(summary)
        except Exception:
            pass

    return results
