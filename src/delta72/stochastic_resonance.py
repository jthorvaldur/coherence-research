"""Stochastic Resonance for Delta.72.

Noise can amplify weak signals in nonlinear systems. Relevant because
Δ shows threshold behavior — there may be an optimal noise level where
detection sensitivity is maximized. Connects to the 0.72 threshold concept.

Stochastic Resonance (SR): in a bistable or threshold system, adding an
intermediate amount of noise can ENHANCE signal detection. Too little noise
→ signal below threshold. Too much → signal drowned. Optimal noise exists.

For Delta.72: the noise component N is not purely destructive. At certain
levels, noise helps the system explore its attractor basin more completely,
which can actually improve coherence measurement. This analysis finds the
optimal noise level for coherence detection.

Key functions:
  - sr_sweep: sweep noise levels and measure detection quality
  - optimal_noise: find the noise level maximizing signal-to-noise ratio
  - sr_curve: the classic SR curve (SNR vs noise)

References:
  - Benzi, Sutera, Vulpiani (1981) — Stochastic resonance
  - Gammaitoni et al. (1998) — Stochastic resonance review
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from delta72.engine import coherence_score


def sr_sweep(
    signal: NDArray,
    baseline: NDArray,
    noise_levels: NDArray | None = None,
    n_trials: int = 20,
    n_levels: int = 30,
) -> dict:
    """Sweep noise levels and measure coherence detection quality.

    At each noise level, adds Gaussian noise to the signal and computes
    the coherence score. Averaging over trials reveals the SR curve.

    Returns dict with noise_levels, mean_deltas, std_deltas, and SNR.
    """
    if noise_levels is None:
        sig_std = signal.std()
        noise_levels = np.linspace(0.001 * sig_std, 3.0 * sig_std, n_levels)

    mean_deltas = []
    std_deltas = []

    for noise_amp in noise_levels:
        trial_deltas = []
        for _ in range(n_trials):
            noisy = signal + np.random.randn(len(signal)) * noise_amp
            scores = coherence_score(noisy, baseline)
            trial_deltas.append(scores["delta"])

        mean_deltas.append(float(np.mean(trial_deltas)))
        std_deltas.append(float(np.std(trial_deltas)))

    mean_deltas = np.array(mean_deltas)
    std_deltas = np.array(std_deltas)

    # SNR: mean / std (higher = better detection)
    snr = mean_deltas / (std_deltas + 1e-12)

    return {
        "noise_levels": noise_levels.tolist(),
        "mean_deltas": mean_deltas.tolist(),
        "std_deltas": std_deltas.tolist(),
        "snr": snr.tolist(),
    }


def optimal_noise(
    signal: NDArray,
    baseline: NDArray,
    **kwargs,
) -> dict:
    """Find the noise level maximizing signal-to-noise ratio.

    This is the stochastic resonance peak — the optimal noise for
    coherence detection.
    """
    result = sr_sweep(signal, baseline, **kwargs)

    snr = np.array(result["snr"])
    noise = np.array(result["noise_levels"])

    best_idx = int(np.argmax(snr))

    return {
        "optimal_noise": float(noise[best_idx]),
        "peak_snr": float(snr[best_idx]),
        "peak_delta": float(result["mean_deltas"][best_idx]),
        "zero_noise_delta": float(result["mean_deltas"][0]),
        "enhancement_ratio": float(result["mean_deltas"][best_idx] / (result["mean_deltas"][0] + 1e-12)),
        "best_idx": best_idx,
    }


def threshold_analysis(
    signal: NDArray,
    baseline: NDArray,
    thresholds: NDArray | None = None,
    noise_amp: float | None = None,
    n_trials: int = 50,
) -> dict:
    """Analyze detection performance across different Δ thresholds.

    Tests whether the canonical 0.72 threshold is optimal, or whether
    stochastic resonance effects shift the optimal threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 1.0, 20)

    if noise_amp is None:
        noise_amp = signal.std() * 0.1

    # Clean signal delta
    clean_scores = coherence_score(signal, baseline)
    clean_delta = clean_scores["delta"]

    # With noise: average over trials
    noisy_deltas = []
    for _ in range(n_trials):
        noisy = signal + np.random.randn(len(signal)) * noise_amp
        scores = coherence_score(noisy, baseline)
        noisy_deltas.append(scores["delta"])

    mean_noisy = float(np.mean(noisy_deltas))
    std_noisy = float(np.std(noisy_deltas))

    # For each threshold: is clean signal above? is noisy signal above?
    results = []
    for thr in thresholds:
        clean_above = clean_delta >= thr
        noisy_above = mean_noisy >= thr
        noisy_fraction_above = sum(1 for d in noisy_deltas if d >= thr) / len(noisy_deltas)

        results.append({
            "threshold": float(thr),
            "clean_above": bool(clean_above),
            "noisy_fraction_above": noisy_fraction_above,
        })

    return {
        "clean_delta": float(clean_delta),
        "noisy_delta_mean": mean_noisy,
        "noisy_delta_std": std_noisy,
        "noise_amp": noise_amp,
        "threshold_analysis": results,
    }
