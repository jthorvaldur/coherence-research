"""Transfer Entropy for Delta.72.

Directed information flow between time series. Detects whether instability
propagates between systems (e.g., shared grid stress between buildings)
vs independent failure. Extends Δ from per-system to system-of-systems.

Transfer Entropy (Schreiber, 2000):
  TE(X→Y) = Σ p(y_{t+1}, y_t^k, x_t^l) * log[ p(y_{t+1}|y_t^k, x_t^l) / p(y_{t+1}|y_t^k) ]

High TE(X→Y) means X's past helps predict Y's future beyond Y's own past.
In Delta.72 context: if TE(Δ_building_A → Δ_building_B) is high, coherence
loss in A predicts coherence loss in B — shared instability.

Key functions:
  - transfer_entropy: Compute TE between two time series
  - net_transfer_entropy: TE(X→Y) - TE(Y→X) — net information flow direction
  - te_matrix: Pairwise TE for multiple time series

References:
  - Schreiber (2000) — Measuring Information Transfer

Implementation: pure numpy, histogram-based probability estimation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _discretize(signal: NDArray, n_bins: int = 8) -> NDArray:
    """Discretize continuous signal into bins for probability estimation."""
    vmin, vmax = signal.min(), signal.max()
    if vmax - vmin < 1e-12:
        return np.zeros(len(signal), dtype=int)
    bins = np.linspace(vmin, vmax, n_bins + 1)
    return np.clip(np.digitize(signal, bins[1:-1]), 0, n_bins - 1)


def transfer_entropy(
    source: NDArray,
    target: NDArray,
    k: int = 1,
    l: int = 1,
    delay: int = 1,
    n_bins: int = 8,
) -> float:
    """Compute Transfer Entropy from source to target.

    TE(source → target) measures how much source's past reduces
    uncertainty about target's future, beyond target's own past.

    Args:
        source: Source time series X.
        target: Target time series Y.
        k: History length for target (y_t^k).
        l: History length for source (x_t^l).
        delay: Prediction delay.
        n_bins: Number of bins for discretization.

    Returns:
        Transfer entropy in nats (natural log base).
    """
    n = min(len(source), len(target))
    max_lag = max(k, l) + delay
    if n <= max_lag + 1:
        return 0.0

    src_d = _discretize(source[:n], n_bins)
    tgt_d = _discretize(target[:n], n_bins)

    te = 0.0
    count = 0

    # Build joint histogram
    # States: (y_{t+1}, y_t^k, x_t^l)
    joint_yyx = {}  # (y_future, y_past_tuple, x_past_tuple) → count
    joint_yx = {}   # (y_past_tuple, x_past_tuple) → count
    joint_yy = {}   # (y_future, y_past_tuple) → count
    marg_y = {}     # (y_past_tuple,) → count

    for t in range(max_lag, n - delay):
        y_future = tgt_d[t + delay]
        y_past = tuple(tgt_d[t - k + 1:t + 1])
        x_past = tuple(src_d[t - l + 1:t + 1])

        key_yyx = (y_future, y_past, x_past)
        key_yx = (y_past, x_past)
        key_yy = (y_future, y_past)
        key_y = y_past

        joint_yyx[key_yyx] = joint_yyx.get(key_yyx, 0) + 1
        joint_yx[key_yx] = joint_yx.get(key_yx, 0) + 1
        joint_yy[key_yy] = joint_yy.get(key_yy, 0) + 1
        marg_y[key_y] = marg_y.get(key_y, 0) + 1
        count += 1

    if count == 0:
        return 0.0

    # Compute TE = Σ p(y',y,x) * log[ p(y'|y,x) / p(y'|y) ]
    for (y_f, y_p, x_p), n_yyx in joint_yyx.items():
        p_yyx = n_yyx / count
        p_yx = joint_yx.get((y_p, x_p), 1) / count
        p_yy = joint_yy.get((y_f, y_p), 1) / count
        p_y = marg_y.get(y_p, 1) / count

        # p(y'|y,x) = p(y',y,x) / p(y,x)
        # p(y'|y) = p(y',y) / p(y)
        cond_yx = p_yyx / p_yx if p_yx > 0 else 0
        cond_y = p_yy / p_y if p_y > 0 else 0

        if cond_yx > 0 and cond_y > 0:
            te += p_yyx * np.log(cond_yx / cond_y)

    return float(max(te, 0.0))


def net_transfer_entropy(
    x: NDArray,
    y: NDArray,
    **kwargs,
) -> dict:
    """Net transfer entropy: direction of information flow.

    Returns dict with TE(X→Y), TE(Y→X), net flow, and dominant direction.
    """
    te_xy = transfer_entropy(x, y, **kwargs)
    te_yx = transfer_entropy(y, x, **kwargs)

    return {
        "te_x_to_y": te_xy,
        "te_y_to_x": te_yx,
        "net": te_xy - te_yx,
        "dominant": "x→y" if te_xy > te_yx else "y→x" if te_yx > te_xy else "bidirectional",
    }


def te_matrix(
    signals: list[NDArray],
    labels: list[str] | None = None,
    **kwargs,
) -> dict:
    """Compute pairwise transfer entropy matrix.

    Args:
        signals: List of time series.
        labels: Optional names for each series.

    Returns:
        Dict with matrix (n x n), labels, and strongest links.
    """
    n = len(signals)
    if labels is None:
        labels = [f"s{i}" for i in range(n)]

    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = transfer_entropy(signals[i], signals[j], **kwargs)

    # Find strongest links
    links = []
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i, j] > 0:
                links.append({
                    "source": labels[i],
                    "target": labels[j],
                    "te": float(matrix[i, j]),
                })
    links.sort(key=lambda x: x["te"], reverse=True)

    return {
        "matrix": matrix.tolist(),
        "labels": labels,
        "strongest_links": links[:10],
        "mean_te": float(matrix[matrix > 0].mean()) if (matrix > 0).any() else 0,
    }
