"""Recurrence Quantification Analysis (RQA) for Delta.72.

Validates Pattern Retention (P) through nonlinear dynamics. While P uses
Pearson correlation (linear), RQA captures deterministic structure in the
system's phase-space trajectory — proving that coherence loss reflects
genuine dynamical degradation, not just linear decorrelation.

Key metrics:
  - Recurrence Rate (RR): fraction of recurrent points
  - Determinism (DET): fraction forming diagonal lines (predictable dynamics)
  - Laminarity (LAM): fraction forming vertical lines (trapped states)
  - Entropy (ENTR): Shannon entropy of diagonal line lengths (complexity)
  - Trapping Time (TT): mean vertical line length (time in laminar states)

References:
  - Eckmann, Kamphorst, Ruelle (1987) — Recurrence plots
  - Zbilut & Webber (1992) — Recurrence quantification analysis
  - Marwan et al. (2007) — Extended RQA measures

Implementation: pure numpy, no external RQA libraries needed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def time_delay_embedding(
    signal: NDArray,
    embedding_dim: int = 3,
    delay: int = 1,
) -> NDArray:
    """Embed 1D signal into m-dimensional phase space via Takens' theorem.

    Args:
        signal: 1D time series of length N.
        embedding_dim: Embedding dimension m (typically 2-5).
        delay: Time delay tau between coordinates.

    Returns:
        (N - (m-1)*tau, m) array of state vectors.
    """
    n = len(signal)
    n_vectors = n - (embedding_dim - 1) * delay
    if n_vectors <= 0:
        raise ValueError(f"Signal too short ({n}) for embedding_dim={embedding_dim}, delay={delay}")

    indices = np.arange(n_vectors)[:, None] + np.arange(embedding_dim) * delay
    return signal[indices]


def recurrence_matrix(
    signal: NDArray,
    embedding_dim: int = 3,
    delay: int = 1,
    threshold: float | None = None,
    threshold_pct: float = 10.0,
) -> NDArray:
    """Compute binary recurrence matrix from time series.

    Args:
        signal: 1D time series.
        embedding_dim: Phase space dimension.
        delay: Time delay for embedding.
        threshold: Fixed distance threshold (epsilon). If None, uses threshold_pct.
        threshold_pct: Percentage of max phase-space distance to use as threshold.

    Returns:
        (N', N') boolean recurrence matrix where N' = len(signal) - (m-1)*tau.
    """
    embedded = time_delay_embedding(signal, embedding_dim, delay)
    n = len(embedded)

    # Pairwise distances (L2 norm) — use broadcasting for memory efficiency
    # For signals < 2000 points, vectorized is fine
    if n <= 2000:
        diffs = embedded[:, None, :] - embedded[None, :, :]
        distances = np.sqrt(np.sum(diffs ** 2, axis=2))
    else:
        # Chunked computation for larger signals
        distances = np.zeros((n, n), dtype=np.float32)
        chunk = 500
        for i in range(0, n, chunk):
            i_end = min(i + chunk, n)
            diffs = embedded[i:i_end, None, :] - embedded[None, :, :]
            distances[i:i_end] = np.sqrt(np.sum(diffs ** 2, axis=2))

    if threshold is None:
        max_dist = distances.max()
        threshold = max_dist * (threshold_pct / 100.0)

    return distances <= threshold


def recurrence_rate(R: NDArray) -> float:
    """RR — Fraction of recurrent points in the recurrence matrix.

    RR = (1/N^2) * sum(R_ij)

    Higher RR → more recurrent states → system revisits similar configurations.
    """
    n = R.shape[0]
    return float(R.sum()) / (n * n)


def _diagonal_lines(R: NDArray, min_length: int = 2) -> list[int]:
    """Extract lengths of all diagonal lines in R (excluding main diagonal)."""
    n = R.shape[0]
    line_lengths = []

    # Diagonals above and below main
    for k in range(1, n):
        diag = np.diag(R, k)
        length = 0
        for val in diag:
            if val:
                length += 1
            else:
                if length >= min_length:
                    line_lengths.append(length)
                length = 0
        if length >= min_length:
            line_lengths.append(length)

    # Below main diagonal (symmetric for unthresholded, but compute anyway)
    for k in range(-1, -n, -1):
        diag = np.diag(R, k)
        length = 0
        for val in diag:
            if val:
                length += 1
            else:
                if length >= min_length:
                    line_lengths.append(length)
                length = 0
        if length >= min_length:
            line_lengths.append(length)

    return line_lengths


def _vertical_lines(R: NDArray, min_length: int = 2) -> list[int]:
    """Extract lengths of all vertical lines in R."""
    n = R.shape[0]
    line_lengths = []

    for col in range(n):
        length = 0
        for row in range(n):
            if R[row, col]:
                length += 1
            else:
                if length >= min_length:
                    line_lengths.append(length)
                length = 0
        if length >= min_length:
            line_lengths.append(length)

    return line_lengths


def determinism(R: NDArray, min_diag: int = 2) -> float:
    """DET — Fraction of recurrent points forming diagonal lines.

    High DET → deterministic dynamics (predictable trajectories).
    Low DET → stochastic dynamics (loss of predictable structure).

    This directly validates P (Pattern Retention): if DET drops,
    the system has lost its deterministic trajectory structure.
    """
    diag_lengths = _diagonal_lines(R, min_diag)
    if not diag_lengths:
        return 0.0

    total_diag_points = sum(diag_lengths)
    total_recurrent = int(R.sum()) - R.shape[0]  # exclude main diagonal
    if total_recurrent <= 0:
        return 0.0

    return float(total_diag_points) / total_recurrent


def laminarity(R: NDArray, min_vert: int = 2) -> float:
    """LAM — Fraction of recurrent points forming vertical lines.

    High LAM → system gets trapped in states (laminar phases).
    Increasing LAM can indicate impending transition — the system
    "sticks" in certain states before transitioning.
    """
    vert_lengths = _vertical_lines(R, min_vert)
    if not vert_lengths:
        return 0.0

    total_vert_points = sum(vert_lengths)
    total_recurrent = int(R.sum())
    if total_recurrent <= 0:
        return 0.0

    return float(total_vert_points) / total_recurrent


def trapping_time(R: NDArray, min_vert: int = 2) -> float:
    """TT — Mean length of vertical lines.

    Average time the system spends in laminar (trapped) states.
    Increasing TT → system increasingly gets stuck.
    """
    vert_lengths = _vertical_lines(R, min_vert)
    if not vert_lengths:
        return 0.0
    return float(np.mean(vert_lengths))


def entropy_diag(R: NDArray, min_diag: int = 2) -> float:
    """ENTR — Shannon entropy of diagonal line length distribution.

    Measures complexity of the deterministic structure.
    High entropy → complex dynamics with many different return patterns.
    Low entropy → simple, periodic dynamics.
    """
    diag_lengths = _diagonal_lines(R, min_diag)
    if not diag_lengths:
        return 0.0

    lengths = np.array(diag_lengths)
    # Frequency distribution
    unique, counts = np.unique(lengths, return_counts=True)
    probs = counts / counts.sum()

    return float(-np.sum(probs * np.log(probs)))


def longest_diagonal(R: NDArray, min_diag: int = 2) -> int:
    """Lmax — Length of the longest diagonal line.

    Inverse relates to largest positive Lyapunov exponent.
    Short Lmax → divergent trajectories → chaos.
    """
    diag_lengths = _diagonal_lines(R, min_diag)
    return max(diag_lengths) if diag_lengths else 0


def rqa_summary(
    signal: NDArray,
    embedding_dim: int = 3,
    delay: int = 1,
    threshold: float | None = None,
    threshold_pct: float = 10.0,
    min_diag: int = 2,
    min_vert: int = 2,
) -> dict:
    """Compute all RQA metrics for a time series.

    Returns dict with:
        RR, DET, LAM, ENTR, TT, Lmax, n_embedded
    """
    R = recurrence_matrix(signal, embedding_dim, delay, threshold, threshold_pct)

    return {
        "RR": recurrence_rate(R),
        "DET": determinism(R, min_diag),
        "LAM": laminarity(R, min_vert),
        "ENTR": entropy_diag(R, min_diag),
        "TT": trapping_time(R, min_vert),
        "Lmax": longest_diagonal(R, min_diag),
        "n_embedded": R.shape[0],
    }
