"""Bayesian Online Change Point Detection (BOCPD) for Delta.72.

Direct benchmark competitor — compares Delta.72's lead-time against BOCPD's
detection latency. If Δ detects structural degradation earlier than BOCPD
detects a change point, it validates the framework's early-warning capability.

BOCPD (Adams & MacKay, 2007) maintains a probability distribution over
the "run length" — how many observations since the last change point.
When the probability mass shifts to short run lengths, a change point
has been detected.

Key functions:
  - bocpd: Run Bayesian Online Change Point Detection
  - bocpd_alerts: Extract change point alerts from run-length posteriors
  - compare_delta_bocpd: Compare Delta.72 detection time vs BOCPD

References:
  - Adams & MacKay (2007) — Bayesian Online Changepoint Detection

Implementation: numpy with log-space computation for numerical stability.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _logsumexp(a: NDArray) -> float:
    """Numerically stable log-sum-exp."""
    a_max = a.max()
    if np.isinf(a_max):
        return float('-inf')
    return float(a_max + np.log(np.sum(np.exp(a - a_max))))


def bocpd(
    signal: NDArray,
    hazard_rate: float = 1 / 200,
    mu0: float | None = None,
    kappa0: float = 0.1,
    alpha0: float = 1.0,
    beta0: float | None = None,
) -> NDArray:
    """Bayesian Online Change Point Detection.

    Computes the run-length posterior probability at each time step using
    a Normal-Inverse-Gamma conjugate model with log-space arithmetic.

    Args:
        signal: 1D time series of length N.
        hazard_rate: Prior probability of a change point at each step.
        mu0: Prior mean. If None, estimated from first 10% of data.
        kappa0: Prior precision scaling (low = weak prior on mean).
        alpha0: Prior shape for inverse-gamma on variance.
        beta0: Prior rate for inverse-gamma. If None, estimated from data.

    Returns:
        (N,) array of change-point probabilities P(run_length=0 | data) at each step.
    """
    n = len(signal)

    # Auto-estimate priors if not provided
    warmup = max(10, n // 20)
    if mu0 is None:
        mu0 = float(np.mean(signal[:warmup]))
    if beta0 is None:
        beta0 = max(float(np.var(signal[:warmup])), 0.01)

    log_h = np.log(hazard_rate)
    log_1mh = np.log(1 - hazard_rate)

    # Run-length log-probabilities (unnormalized)
    # At time t, log_R[r] = log P(r_t = r, x_{1:t})
    log_R = np.array([0.0])  # log_R[0] = log(1) = 0

    # Sufficient statistics arrays — one per run length
    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])

    cp_probs = np.zeros(n)

    for t in range(n):
        x = signal[t]

        # Predictive log-probability under each run length
        # Student-t with parameters derived from Normal-Inverse-Gamma
        nu = 2 * alpha
        pred_var = beta * (kappa + 1) / (alpha * kappa)

        log_pi = np.zeros(len(mu))
        for i in range(len(mu)):
            sigma = np.sqrt(pred_var[i] * (nu[i] + 1) / nu[i])
            z = (x - mu[i]) / sigma
            log_pi[i] = (
                _lgamma_half(nu[i] + 1)
                - _lgamma_half(nu[i])
                - 0.5 * np.log(nu[i] * np.pi)
                - np.log(sigma)
                - (nu[i] + 1) / 2 * np.log(1 + z ** 2 / nu[i])
            )

        # Growth: log P(r_{t+1} = r+1, x_{1:t+1})
        log_growth = log_R + log_pi + log_1mh

        # Change point: log P(r_{t+1} = 0, x_{1:t+1})
        log_cp = _logsumexp(log_R + log_pi + log_h)

        # New run-length distribution
        new_log_R = np.empty(len(log_growth) + 1)
        new_log_R[0] = log_cp
        new_log_R[1:] = log_growth

        # Normalize to get posterior
        log_evidence = _logsumexp(new_log_R)
        new_log_R -= log_evidence

        # Store change-point probability
        cp_probs[t] = np.exp(new_log_R[0])

        log_R = new_log_R

        # Update sufficient statistics
        mu_new = (kappa * mu + x) / (kappa + 1)
        kappa_new = kappa + 1
        alpha_new = alpha + 0.5
        beta_new = beta + kappa * (x - mu) ** 2 / (2 * (kappa + 1))

        # Prepend prior for new run length = 0
        mu = np.concatenate([[mu0], mu_new])
        kappa = np.concatenate([[kappa0], kappa_new])
        alpha = np.concatenate([[alpha0], alpha_new])
        beta = np.concatenate([[beta0], beta_new])

    return cp_probs


def _lgamma_half(x: NDArray | float) -> NDArray | float:
    """Log-gamma, handling scalar and array."""
    from scipy.special import gammaln
    return gammaln(x / 2)


def bocpd_alerts(
    signal: NDArray,
    threshold: float = 0.3,
    min_spacing: int = 10,
    **kwargs,
) -> list[int]:
    """Extract change point alerts from BOCPD.

    Args:
        signal: 1D time series.
        threshold: Probability threshold for declaring a change point.
        min_spacing: Minimum spacing between consecutive alerts.
        **kwargs: Passed to bocpd().

    Returns:
        List of time indices where change points were detected.
    """
    cp_probs = bocpd(signal, **kwargs)

    alerts = []
    for t in range(len(cp_probs)):
        if cp_probs[t] > threshold:
            if not alerts or (t - alerts[-1]) >= min_spacing:
                alerts.append(t)

    return alerts


def compare_delta_bocpd(
    signal: NDArray,
    delta_alert_idx: int,
    bocpd_threshold: float = 0.3,
    **bocpd_kwargs,
) -> dict:
    """Compare Delta.72 detection time vs BOCPD detection time.

    Args:
        signal: Time series.
        delta_alert_idx: Index where Delta first alerted.
        bocpd_threshold: BOCPD alert threshold.
        **bocpd_kwargs: Passed to bocpd().

    Returns:
        Dict with detection times and lead-time comparison.
    """
    alerts = bocpd_alerts(signal, threshold=bocpd_threshold, **bocpd_kwargs)

    bocpd_first = alerts[0] if alerts else len(signal)

    return {
        "delta_alert_idx": delta_alert_idx,
        "bocpd_first_alert": bocpd_first,
        "bocpd_n_alerts": len(alerts),
        "bocpd_all_alerts": alerts[:10],
        "delta_leads_by": bocpd_first - delta_alert_idx,
        "delta_earlier": delta_alert_idx < bocpd_first,
    }


def bocpd_summary(
    signal: NDArray,
    threshold: float = 0.3,
    **kwargs,
) -> dict:
    """Compute BOCPD summary for a time series."""
    cp_probs = bocpd(signal, **kwargs)
    alerts = bocpd_alerts(signal, threshold=threshold, **kwargs)

    return {
        "n_change_points": len(alerts),
        "change_points": alerts[:20],
        "max_cp_prob": float(cp_probs.max()),
        "max_cp_idx": int(cp_probs.argmax()),
        "mean_cp_prob": float(cp_probs.mean()),
    }
