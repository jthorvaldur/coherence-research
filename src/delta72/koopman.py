"""Koopman Operator Theory for Delta.72.

Lifts nonlinear dynamics to infinite-dimensional linear space. Spectral
analysis of Koopman eigenfunctions reveals intrinsic system modes.
Coherence = stability of Koopman spectrum. Used in fluid dynamics, power grids.

The key insight: even nonlinear systems can be described by a linear operator
(the Koopman operator) acting on observables. The eigenvalues of this operator
encode the system's intrinsic timescales and frequencies.

For Delta.72: stable system has clustered Koopman eigenvalues (simple dynamics).
Degrading system sees eigenvalue scatter (complex, unpredictable dynamics).

Key functions:
  - dmd: Dynamic Mode Decomposition — finite-dimensional approximation
  - koopman_spectrum: Extract eigenvalues and modes
  - spectral_coherence: Measure stability of the Koopman spectrum

References:
  - Koopman (1931) — Hamiltonian systems and transformation in Hilbert space
  - Mezic (2005) — Spectral properties of dynamical systems
  - Kutz et al. (2016) — Dynamic Mode Decomposition

Implementation: numpy + scipy.linalg for SVD-based DMD.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def time_delay_matrix(
    signal: NDArray,
    n_delays: int = 10,
) -> NDArray:
    """Build Hankel matrix from time series for DMD.

    Returns (n_delays, N - n_delays) matrix where each column is a
    time-shifted copy of the signal.
    """
    n = len(signal)
    if n <= n_delays + 1:
        raise ValueError(f"Signal too short ({n}) for n_delays={n_delays}")
    return np.array([signal[i:n - n_delays + i] for i in range(n_delays)])


def dmd(
    signal: NDArray,
    n_delays: int = 10,
    rank: int | None = None,
    dt: float = 1.0,
) -> dict:
    """Dynamic Mode Decomposition — approximate Koopman operator.

    Computes the best-fit linear operator A such that X' ≈ A·X,
    where X and X' are time-shifted snapshot matrices.

    Args:
        signal: 1D time series.
        n_delays: Number of time delays for Hankel matrix.
        rank: Truncation rank for SVD (None = auto).
        dt: Time step for frequency computation.

    Returns:
        Dict with eigenvalues, frequencies, growth rates, and modes.
    """
    H = time_delay_matrix(signal, n_delays)

    X = H[:, :-1]  # snapshots at t
    Xp = H[:, 1:]  # snapshots at t+1

    # SVD of X
    U, S, Vh = np.linalg.svd(X, full_matrices=False)

    if rank is None:
        # Auto-select rank: keep singular values > 1% of max
        rank = max(1, int(np.sum(S > 0.01 * S[0])))
    rank = min(rank, len(S))

    U_r = U[:, :rank]
    S_r = S[:rank]
    V_r = Vh[:rank, :]

    # Build reduced DMD operator
    A_tilde = U_r.T @ Xp @ V_r.T @ np.diag(1.0 / S_r)

    # Eigendecomposition
    eigenvalues, W = np.linalg.eig(A_tilde)

    # DMD modes
    Phi = Xp @ V_r.T @ np.diag(1.0 / S_r) @ W

    # Continuous-time eigenvalues
    omega = np.log(eigenvalues + 1e-15) / dt

    return {
        "eigenvalues": eigenvalues,
        "frequencies": np.abs(omega.imag) / (2 * np.pi),
        "growth_rates": omega.real,
        "modes": Phi,
        "singular_values": S,
        "rank": rank,
    }


def koopman_spectrum(
    signal: NDArray,
    n_delays: int = 10,
    **kwargs,
) -> dict:
    """Extract Koopman spectrum summary.

    Returns eigenvalue magnitudes, dominant frequencies, and spectral entropy.
    """
    result = dmd(signal, n_delays, **kwargs)

    eig_mags = np.abs(result["eigenvalues"])
    freqs = result["frequencies"]
    growth = result["growth_rates"]

    # Spectral entropy: complexity of the Koopman spectrum
    eig_mags_norm = eig_mags / (eig_mags.sum() + 1e-12)
    spectral_entropy = float(-np.sum(eig_mags_norm * np.log(eig_mags_norm + 1e-12)))

    # Dominant mode: largest eigenvalue magnitude
    dom_idx = np.argmax(eig_mags)

    return {
        "n_modes": len(eig_mags),
        "spectral_entropy": spectral_entropy,
        "dominant_frequency": float(freqs[dom_idx]),
        "dominant_growth_rate": float(growth[dom_idx]),
        "max_eigenvalue_mag": float(eig_mags.max()),
        "mean_eigenvalue_mag": float(eig_mags.mean()),
        "n_growing_modes": int(np.sum(growth > 0.01)),
        "n_decaying_modes": int(np.sum(growth < -0.01)),
        "stability": "stable" if np.all(eig_mags <= 1.01) else "unstable",
    }


def rolling_koopman(
    signal: NDArray,
    window_size: int = 100,
    step: int = 20,
    **kwargs,
) -> list[dict]:
    """Compute Koopman spectrum in rolling windows."""
    n = len(signal)
    results = []
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        mid = start + window_size // 2
        try:
            spec = koopman_spectrum(signal[start:end], **kwargs)
            spec["window_mid"] = mid
            spec["lifecycle_pct"] = mid / n * 100
            results.append(spec)
        except Exception:
            pass
    return results
