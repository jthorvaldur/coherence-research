"""Granger Causality for Delta.72.

Statistical test for whether one time series helps predict another.
In Delta.72 context: does coherence loss in system A predict coherence
loss in system B? Discovers causal structure between systems.

Granger causality: X Granger-causes Y if past values of X improve
prediction of Y beyond Y's own past.

Implementation uses VAR (Vector Autoregression) residuals comparison.

References:
  - Granger (1969) — Investigating causal relations
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def granger_test(
    x: NDArray,
    y: NDArray,
    max_lag: int = 5,
    significance: float = 0.05,
) -> dict:
    """Test whether x Granger-causes y.

    Compares two models:
    - Restricted: y_t = Σ a_i * y_{t-i} + ε   (y's own past only)
    - Unrestricted: y_t = Σ a_i * y_{t-i} + Σ b_j * x_{t-j} + ε

    If unrestricted model has significantly lower RSS, x Granger-causes y.

    Returns dict with F-statistic, p-value, and significance.
    """
    n = min(len(x), len(y))
    if n <= 2 * max_lag + 2:
        return {"f_stat": 0, "p_value": 1, "significant": False, "lag": max_lag}

    x = x[:n].astype(np.float64)
    y = y[:n].astype(np.float64)

    best_result = {"f_stat": 0, "p_value": 1, "significant": False, "lag": 1}

    for lag in range(1, max_lag + 1):
        T = n - lag

        # Build design matrices
        Y = y[lag:]

        # Restricted model: y's own lags
        X_r = np.column_stack([y[lag - i - 1:n - i - 1] for i in range(lag)])
        X_r = np.column_stack([np.ones(T), X_r])

        # Unrestricted model: y's lags + x's lags
        X_u = np.column_stack([
            np.ones(T),
            *[y[lag - i - 1:n - i - 1] for i in range(lag)],
            *[x[lag - i - 1:n - i - 1] for i in range(lag)],
        ])

        # OLS residuals
        try:
            beta_r = np.linalg.lstsq(X_r, Y, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_u, Y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        rss_r = np.sum((Y - X_r @ beta_r) ** 2)
        rss_u = np.sum((Y - X_u @ beta_u) ** 2)

        # F-test
        df1 = lag  # additional parameters in unrestricted
        df2 = T - 2 * lag - 1
        if df2 <= 0 or rss_u <= 0:
            continue

        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)

        if p_value < best_result["p_value"]:
            best_result = {
                "f_stat": float(f_stat),
                "p_value": float(p_value),
                "significant": p_value < significance,
                "lag": lag,
            }

    return best_result


def granger_matrix(
    signals: list[NDArray],
    labels: list[str] | None = None,
    **kwargs,
) -> dict:
    """Pairwise Granger causality matrix.

    Returns dict with p-value matrix, significant links, and causal graph.
    """
    n = len(signals)
    if labels is None:
        labels = [f"s{i}" for i in range(n)]

    p_matrix = np.ones((n, n))
    f_matrix = np.zeros((n, n))
    links = []

    for i in range(n):
        for j in range(n):
            if i != j:
                result = granger_test(signals[i], signals[j], **kwargs)
                p_matrix[i, j] = result["p_value"]
                f_matrix[i, j] = result["f_stat"]
                if result["significant"]:
                    links.append({
                        "source": labels[i],
                        "target": labels[j],
                        "f_stat": result["f_stat"],
                        "p_value": result["p_value"],
                        "lag": result["lag"],
                    })

    links.sort(key=lambda x: x["p_value"])

    return {
        "p_matrix": p_matrix.tolist(),
        "f_matrix": f_matrix.tolist(),
        "labels": labels,
        "significant_links": links,
        "n_significant": len(links),
    }
