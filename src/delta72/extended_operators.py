"""Extended Coherence Operators from the Δ.72 Canon.

Implements the five extended unique operators from the canonical algorithm
(see jthorvaldur.github.io/r/d72/):

  𝓕 (Flower Return) — Return to organizing geometry after perturbation
  𝓜 (Memory-of-Attractor) — Already in engine.py, re-exported here
  𝓑 (Bounded Distortion) — Separation of deformation vs collapse
  𝓦 (Windowed Recovery) — Already in engine.py, re-exported here
  𝓛 (Loss-of-Flower) — Detection of irreversible pattern loss

Plus the canonical regime classification:
  Δ ≥ 0.72 → Coherent
  0.55 ≤ Δ < 0.72 → Distorted
  0.35 ≤ Δ < 0.55 → Fragmented
  Δ < 0.35 → Collapse

And the failure boundary condition:
  Collapse when D_accum + N_amp + Φ_error > P_retention + R_strength + A_capacity
  or: dD/dt > dR/dt
  or: D_{t+τ} ≥ D_t

References:
  - Δ.72 Coherence Framework — Canonical Algorithm
  - Hensgen, A. — Coherence Operator preprints (Zenodo)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

# Re-export core operators from engine.py
from delta72.engine import (
    memory_of_attractor,
    windowed_recovery,
    coherence_score,
)


# ---------------------------------------------------------------------------
# Regime classification (canonical thresholds)
# ---------------------------------------------------------------------------

def classify_regime(delta: float) -> str:
    """Classify system state per canonical Δ.72 regime boundaries.

    Δ ≥ 0.72 → Coherent
    0.55 ≤ Δ < 0.72 → Distorted
    0.35 ≤ Δ < 0.55 → Fragmented
    Δ < 0.35 → Collapse
    """
    if delta >= 0.72:
        return "coherent"
    elif delta >= 0.55:
        return "distorted"
    elif delta >= 0.35:
        return "fragmented"
    else:
        return "collapse"


def regime_color(regime: str) -> str:
    """CSS color for regime visualization."""
    return {
        "coherent": "#9ece6a",
        "distorted": "#e0af68",
        "fragmented": "#f7768e",
        "collapse": "#ff2222",
    }.get(regime, "#888888")


# ---------------------------------------------------------------------------
# 𝓕 — Flower Return Operator
# ---------------------------------------------------------------------------

def flower_return(
    signal: NDArray,
    baseline: NDArray,
    n_perturbations: int = 5,
    recovery_window: int | None = None,
) -> float:
    """𝓕 — Flower Return Operator.

    Measures return to organizing geometry after perturbation.
    Specifically: after the system deviates from its baseline pattern,
    does it return to a geometrically similar shape (not just similar
    magnitude)?

    Uses cross-correlation of the recovery segment with the baseline
    to measure geometric similarity, not just amplitude match.

    High 𝓕 → system returns to its original organizing pattern.
    Low 𝓕 → system recovers to a different geometry (structural change).
    """
    residual = np.abs(signal - baseline)
    if residual.max() == 0:
        return 1.0

    if recovery_window is None:
        recovery_window = max(len(signal) // 20, 4)

    # Find perturbation episodes
    peaks, _ = find_peaks(residual, distance=recovery_window, height=residual.std())
    if len(peaks) == 0:
        return 1.0

    # Sort by magnitude, take top n_perturbations
    peak_heights = residual[peaks]
    top_idx = np.argsort(peak_heights)[-n_perturbations:]
    top_peaks = peaks[top_idx]

    geometric_returns = []
    for pk in top_peaks:
        # Post-perturbation segment
        post_start = min(pk + recovery_window // 2, len(signal) - recovery_window)
        post_end = min(post_start + recovery_window, len(signal))
        if post_end - post_start < 4:
            continue

        post_segment = signal[post_start:post_end]
        base_segment = baseline[post_start:post_end]

        # Geometric similarity: normalized cross-correlation
        ps_norm = post_segment - post_segment.mean()
        bs_norm = base_segment - base_segment.mean()

        denom = np.sqrt(np.sum(ps_norm ** 2) * np.sum(bs_norm ** 2))
        if denom < 1e-12:
            geometric_returns.append(1.0)
        else:
            xcorr = float(np.sum(ps_norm * bs_norm) / denom)
            geometric_returns.append(max(xcorr, 0.0))

    return float(np.mean(geometric_returns)) if geometric_returns else 1.0


# ---------------------------------------------------------------------------
# 𝓑 — Bounded Distortion Operator
# ---------------------------------------------------------------------------

def bounded_distortion(
    signal: NDArray,
    baseline: NDArray,
    collapse_threshold: float = 3.0,
) -> float:
    """𝓑 — Bounded Distortion Operator.

    Separates deformation (bounded, recoverable) from collapse (unbounded,
    irreversible). Measures what fraction of the signal's deviation from
    baseline stays within bounded limits.

    High 𝓑 → deformation is bounded (system is deformed but not collapsing).
    Low 𝓑 → distortion exceeds bounds (approaching collapse).

    The collapse_threshold is in units of baseline standard deviation.
    """
    residual = np.abs(signal - baseline)
    baseline_std = baseline.std()

    if baseline_std < 1e-12:
        baseline_std = signal.std()
    if baseline_std < 1e-12:
        return 1.0

    # Normalized residual
    norm_residual = residual / baseline_std

    # Fraction of points within bounded distortion
    within_bounds = np.sum(norm_residual < collapse_threshold)

    return float(within_bounds / len(signal))


# ---------------------------------------------------------------------------
# 𝓛 — Loss-of-Flower Operator
# ---------------------------------------------------------------------------

def loss_of_flower(
    signal: NDArray,
    baseline: NDArray,
    window: int | None = None,
) -> float:
    """𝓛 — Loss-of-Flower Operator.

    Detects irreversible pattern loss — the point at which the system
    can no longer return to its organizing geometry regardless of recovery
    time. This is distinct from temporary coherence loss.

    Computed as: 1 - (max achievable cross-correlation in any window)

    Low 𝓛 → pattern is still recoverable (flower intact).
    High 𝓛 → pattern is irreversibly lost (flower destroyed).
    """
    n = len(signal)
    if window is None:
        window = max(n // 5, 8)

    max_corr = 0.0

    for start in range(0, n - window + 1, window // 2):
        end = start + window
        seg = signal[start:end]
        base = baseline[start:end]

        seg_norm = seg - seg.mean()
        base_norm = base - base.mean()

        denom = np.sqrt(np.sum(seg_norm ** 2) * np.sum(base_norm ** 2))
        if denom > 1e-12:
            corr = float(np.sum(seg_norm * base_norm) / denom)
            max_corr = max(max_corr, corr)

    return float(1.0 - max(max_corr, 0.0))


# ---------------------------------------------------------------------------
# Failure boundary condition
# ---------------------------------------------------------------------------

def failure_boundary(
    signal: NDArray,
    baseline: NDArray,
    tau: int = 10,
) -> dict:
    """Check the canonical failure boundary conditions.

    Collapse occurs when:
    1. D_accum + N_amp > P_retention + R_strength  (force balance)
    2. dD/dt > dR/dt  (drift accelerating faster than recovery)
    3. D_{t+τ} ≥ D_t  (drift is not decreasing)

    Returns dict with each condition's status and overall collapse assessment.
    """
    scores = coherence_score(signal, baseline)
    P, R, D, N = scores["P"], scores["R"], scores["D"], scores["N"]

    # Condition 1: force imbalance
    destructive = D + N
    restorative = max(P, 0) + R
    cond1 = destructive > restorative

    # Condition 2: drift rate vs recovery rate
    residual = np.abs(signal - baseline)
    if len(residual) > tau + 1:
        drift_early = residual[:len(residual) // 2].mean()
        drift_late = residual[len(residual) // 2:].mean()
        recovery_early = 1.0 - drift_early
        recovery_late = 1.0 - drift_late
        dD_dt = drift_late - drift_early
        dR_dt = recovery_late - recovery_early
        cond2 = dD_dt > dR_dt
    else:
        dD_dt, dR_dt = 0, 0
        cond2 = False

    # Condition 3: drift not decreasing
    if len(residual) > tau:
        D_t = residual[-tau - 1:-1].mean()
        D_t_tau = residual[-tau:].mean()
        cond3 = D_t_tau >= D_t
    else:
        cond3 = False

    return {
        "force_imbalance": cond1,
        "drift_accelerating": cond2,
        "drift_not_decreasing": cond3,
        "collapse": cond1 and (cond2 or cond3),
        "destructive_force": float(destructive),
        "restorative_force": float(restorative),
        "dD_dt": float(dD_dt),
        "dR_dt": float(dR_dt),
        "regime": classify_regime(scores["delta"]),
        "delta": scores["delta"],
    }


# ---------------------------------------------------------------------------
# Full extended analysis
# ---------------------------------------------------------------------------

def extended_analysis(
    signal: NDArray,
    baseline: NDArray,
    **kwargs,
) -> dict:
    """Run all extended operators on a signal/baseline pair.

    Returns dict with all 5 operator values, regime, and failure boundary.
    """
    scores = coherence_score(signal, baseline)
    F = flower_return(signal, baseline)
    M = memory_of_attractor(signal, baseline)
    B = bounded_distortion(signal, baseline)
    W = windowed_recovery(signal, baseline)
    L = loss_of_flower(signal, baseline)
    fb = failure_boundary(signal, baseline)

    return {
        **scores,
        "F": F,
        "M": M,
        "B": B,
        "W": W,
        "L": L,
        "regime": classify_regime(scores["delta"]),
        "failure_boundary": fb,
    }
