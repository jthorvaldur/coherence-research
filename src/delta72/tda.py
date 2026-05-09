"""Persistent Homology / Topological Data Analysis for Delta.72.

Tracks the "shape" of data as it evolves using persistent homology.
Persistence diagrams show stable vs dying topological features (connected
components, loops, voids). Applied to financial crash prediction,
cardiac arrhythmia detection, and material failure analysis.

In Delta.72 context: healthy system has stable topological features
(persistent cycles in phase space). Degrading system sees features
die rapidly — the "shape" of the attractor dissolves.

Key metrics:
  - Persistence entropy: Shannon entropy of persistence diagram lifetimes
  - Total persistence: sum of all feature lifetimes
  - Betti curves: count of topological features at each scale
  - Wasserstein distance: optimal transport between persistence diagrams

Implementation: Vietoris-Rips filtration on time-delay embedded data,
using a simplified persistent homology computation (pure numpy, H0 only
via union-find for connected components).

References:
  - Carlsson (2009) — Topology and Data
  - Edelsbrunner & Harer (2010) — Computational Topology
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class UnionFind:
    """Union-Find for tracking connected components in filtration."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.birth = [0.0] * n  # birth time of each component

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int, weight: float) -> float | None:
        """Union two components. Returns death time if a component dies."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return None  # already connected

        # Younger component dies (higher birth time)
        if self.birth[rx] > self.birth[ry]:
            rx, ry = ry, rx

        # ry dies at current weight
        death = weight

        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
            self.birth[ry] = min(self.birth[rx], self.birth[ry])
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1

        return death


def time_delay_embedding(signal: NDArray, dim: int = 3, delay: int = 1) -> NDArray:
    n = len(signal)
    n_vec = n - (dim - 1) * delay
    if n_vec <= 0:
        raise ValueError(f"Signal too short ({n}) for dim={dim}, delay={delay}")
    return signal[np.arange(n_vec)[:, None] + np.arange(dim) * delay]


def persistence_diagram_h0(
    signal: NDArray,
    embedding_dim: int = 3,
    delay: int = 1,
    max_points: int = 500,
) -> list[tuple[float, float]]:
    """Compute H0 persistence diagram (connected components) via Vietoris-Rips.

    Returns list of (birth, death) pairs for each topological feature.
    Birth = distance at which a component appears (0 for all points).
    Death = distance at which it merges with another component.
    """
    embedded = time_delay_embedding(signal, embedding_dim, delay)

    # Subsample if needed
    n = len(embedded)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=int)
        embedded = embedded[idx]
        n = max_points

    # Compute pairwise distances
    dists = np.sqrt(np.sum((embedded[:, None, :] - embedded[None, :, :]) ** 2, axis=2))

    # Build edge list sorted by weight
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dists[i, j], i, j))
    edges.sort()

    # Filtration via Union-Find
    uf = UnionFind(n)
    diagram = []

    for weight, i, j in edges:
        death = uf.union(i, j, weight)
        if death is not None:
            if death > 0:  # skip trivial (birth=death=0) features
                diagram.append((0.0, death))

    return diagram


def persistence_entropy(diagram: list[tuple[float, float]]) -> float:
    """Shannon entropy of persistence diagram lifetimes.

    High entropy = many features with similar lifetimes (complex topology).
    Low entropy = few dominant features (simple topology).
    """
    if not diagram:
        return 0.0

    lifetimes = np.array([d - b for b, d in diagram])
    lifetimes = lifetimes[lifetimes > 0]

    if len(lifetimes) == 0:
        return 0.0

    total = lifetimes.sum()
    if total <= 0:
        return 0.0

    probs = lifetimes / total
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def total_persistence(diagram: list[tuple[float, float]], p: int = 1) -> float:
    """Total p-persistence: sum of lifetime^p.

    High total persistence = stable, persistent topological features.
    Low total persistence = features die quickly (noisy/degraded system).
    """
    if not diagram:
        return 0.0
    lifetimes = np.array([d - b for b, d in diagram])
    return float(np.sum(lifetimes ** p))


def max_persistence(diagram: list[tuple[float, float]]) -> float:
    """Lifetime of the longest-lived feature."""
    if not diagram:
        return 0.0
    return float(max(d - b for b, d in diagram))


def n_persistent_features(
    diagram: list[tuple[float, float]],
    threshold_frac: float = 0.1,
) -> int:
    """Count features with lifetime > threshold_frac * max_lifetime."""
    if not diagram:
        return 0
    max_life = max(d - b for b, d in diagram)
    threshold = max_life * threshold_frac
    return sum(1 for b, d in diagram if (d - b) > threshold)


def tda_summary(
    signal: NDArray,
    embedding_dim: int = 3,
    delay: int = 1,
    max_points: int = 400,
) -> dict:
    """Compute TDA summary metrics for a time series."""
    diagram = persistence_diagram_h0(signal, embedding_dim, delay, max_points)

    return {
        "n_features": len(diagram),
        "persistence_entropy": persistence_entropy(diagram),
        "total_persistence": total_persistence(diagram),
        "max_persistence": max_persistence(diagram),
        "n_persistent": n_persistent_features(diagram),
    }


def rolling_tda(
    signal: NDArray,
    window_size: int = 100,
    step: int = 20,
    **kwargs,
) -> list[dict]:
    """Compute TDA metrics in rolling windows."""
    n = len(signal)
    results = []
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        mid = start + window_size // 2
        try:
            summary = tda_summary(signal[start:end], **kwargs)
            summary["window_mid"] = mid
            summary["lifecycle_pct"] = mid / n * 100
            results.append(summary)
        except Exception:
            pass
    return results
