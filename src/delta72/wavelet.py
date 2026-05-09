"""Wavelet Coherence for Delta.72.

Multi-scale decomposition of coherence — reveals whether coherence loss
is happening at fast timescales (minutes/hours), medium (daily), or slow
(weekly/seasonal). Turns the single Δ score into a scale-resolved spectrum.

Uses the Morlet wavelet for continuous wavelet transform (CWT), matching
frequency analysis conventions in geophysics and neuroscience.

Key functions:
  - cwt_morlet: Continuous wavelet transform with Morlet wavelet
  - wavelet_coherence: Cross-wavelet coherence between signal and baseline
  - scale_averaged_coherence: Average coherence within frequency bands

References:
  - Torrence & Compo (1998) — Wavelet analysis guide
  - Grinsted et al. (2004) — Cross-wavelet and wavelet coherence

Implementation: pure numpy, no external wavelet libraries needed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cwt_morlet(
    signal: NDArray,
    scales: NDArray | None = None,
    omega0: float = 6.0,
    n_scales: int = 32,
) -> tuple[NDArray, NDArray]:
    """Continuous Wavelet Transform using Morlet wavelet (time-domain).

    Args:
        signal: 1D time series of length N.
        scales: Array of scales. If None, auto-generated logarithmically.
        omega0: Central frequency of Morlet wavelet.
        n_scales: Number of scales to compute (if scales is None).

    Returns:
        (coefficients, scales) where coefficients is (n_scales, N) complex array.
    """
    n = len(signal)

    if scales is None:
        scales = np.logspace(np.log2(2), np.log2(max(n // 4, 4)), n_scales, base=2)

    coefficients = np.zeros((len(scales), n), dtype=complex)

    for i, scale in enumerate(scales):
        # Morlet wavelet in time domain
        wavelet_len = min(int(10 * scale), n)
        if wavelet_len % 2 == 0:
            wavelet_len += 1
        t = (np.arange(wavelet_len) - wavelet_len // 2) / scale
        wavelet = (np.pi ** -0.25) * np.exp(1j * omega0 * t) * np.exp(-t ** 2 / 2)
        wavelet /= np.sqrt(scale)

        # Convolve signal with conjugate wavelet
        conv = np.convolve(signal, np.conj(wavelet[::-1]), mode="same")
        coefficients[i] = conv[:n]

    return coefficients, scales


def wavelet_power(
    signal: NDArray,
    scales: NDArray | None = None,
    **kwargs,
) -> tuple[NDArray, NDArray]:
    """Compute wavelet power spectrum |W(s,t)|².

    Returns:
        (power, scales) where power is (n_scales, N) real array.
    """
    coeffs, scales = cwt_morlet(signal, scales, **kwargs)
    return np.abs(coeffs) ** 2, scales


def wavelet_coherence(
    signal: NDArray,
    baseline: NDArray,
    scales: NDArray | None = None,
    smoothing_window: int = 5,
    **kwargs,
) -> tuple[NDArray, NDArray]:
    """Wavelet coherence between signal and baseline.

    R²(s,t) = |S(W_xy)|² / (S(|W_x|²) · S(|W_y|²))

    where S is a smoothing operator (running mean in time).

    Returns:
        (coherence, scales) where coherence is (n_scales, N) real array in [0, 1].
    """
    n = min(len(signal), len(baseline))
    coeffs_s, scales = cwt_morlet(signal[:n], scales, **kwargs)
    coeffs_b, _ = cwt_morlet(baseline[:n], scales, **kwargs)

    # Cross-wavelet spectrum
    W_xy = coeffs_s * np.conj(coeffs_b)

    # Individual power spectra
    P_x = np.abs(coeffs_s) ** 2
    P_y = np.abs(coeffs_b) ** 2

    # Smooth in time dimension
    kernel = np.ones(smoothing_window) / smoothing_window

    def smooth_rows(arr: NDArray) -> NDArray:
        result = np.zeros_like(arr, dtype=np.complex128 if np.iscomplexobj(arr) else np.float64)
        for i in range(arr.shape[0]):
            if np.iscomplexobj(arr):
                result[i] = (
                    np.convolve(arr[i].real, kernel, mode="same")
                    + 1j * np.convolve(arr[i].imag, kernel, mode="same")
                )
            else:
                result[i] = np.convolve(arr[i], kernel, mode="same")
        return result

    S_xy = smooth_rows(W_xy)
    S_xx = smooth_rows(P_x)
    S_yy = smooth_rows(P_y)

    # Coherence: R² = |S(Wxy)|² / (S(|Wx|²) · S(|Wy|²))
    denominator = S_xx.real * S_yy.real
    denominator = np.maximum(denominator, 1e-15)

    coherence = np.abs(S_xy) ** 2 / denominator
    coherence = np.clip(coherence, 0, 1)

    return coherence, scales


def scale_averaged_coherence(
    signal: NDArray,
    baseline: NDArray,
    scale_bands: list[tuple[float, float]] | None = None,
    **kwargs,
) -> dict:
    """Average wavelet coherence within specified scale bands.

    Args:
        signal: Signal time series.
        baseline: Baseline time series.
        scale_bands: List of (min_scale, max_scale) tuples defining frequency bands.
            If None, auto-generates 3 bands (fast, medium, slow).

    Returns:
        Dict with per-band mean coherence and overall mean.
    """
    coh, scales = wavelet_coherence(signal, baseline, **kwargs)

    if scale_bands is None:
        s_min, s_max = scales.min(), scales.max()
        log_range = np.log2(s_max) - np.log2(s_min)
        s_third = log_range / 3
        scale_bands = [
            (s_min, 2 ** (np.log2(s_min) + s_third)),
            (2 ** (np.log2(s_min) + s_third), 2 ** (np.log2(s_min) + 2 * s_third)),
            (2 ** (np.log2(s_min) + 2 * s_third), s_max),
        ]

    band_names = ["fast", "medium", "slow"]
    result = {
        "overall_mean": float(coh.mean()),
        "overall_std": float(coh.std()),
        "bands": {},
    }

    for i, (s_lo, s_hi) in enumerate(scale_bands):
        mask = (scales >= s_lo) & (scales <= s_hi)
        if mask.any():
            band_coh = coh[mask].mean(axis=0)
            name = band_names[i] if i < len(band_names) else f"band_{i}"
            result["bands"][name] = {
                "scale_range": (float(s_lo), float(s_hi)),
                "mean_coherence": float(band_coh.mean()),
                "std_coherence": float(band_coh.std()),
            }

    return result


def wavelet_summary(
    signal: NDArray,
    baseline: NDArray,
    **kwargs,
) -> dict:
    """Compute wavelet coherence summary statistics."""
    return scale_averaged_coherence(signal, baseline, **kwargs)
