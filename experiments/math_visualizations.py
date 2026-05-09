#!/usr/bin/env python3
"""Generate distinct visualization plots for every math module.

Each of the 14 math modules gets its own dedicated plot showing its
unique contribution to understanding system dynamics. All applied to
NASA C-MAPSS turbofan data (known degradation trajectories).

Outputs to results/math/ — one PNG per module.

Usage:
    uv run python experiments/math_visualizations.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from delta72.engine import coherence_score
from delta72.rqa import rqa_summary
from delta72.wavelet import wavelet_coherence, scale_averaged_coherence
from delta72.lyapunov import rolling_lyapunov, lyapunov_summary
from delta72.bocpd import bocpd
from delta72.transfer_entropy import transfer_entropy, net_transfer_entropy
from delta72.phase_space import time_delay_embedding, attractor_spread, correlation_dimension, rolling_phase_space
from delta72.rmt import rmt_summary, rolling_rmt, marchenko_pastur_bounds
from delta72.tda import tda_summary, rolling_tda
from delta72.granger import granger_test
from delta72.koopman import koopman_spectrum, rolling_koopman
from delta72.info_geometry import divergence_rate, fisher_information
from delta72.ergodic import ergodic_summary, rolling_ergodic
from delta72.stochastic_resonance import sr_sweep, optimal_noise
from delta72.extended_operators import (
    flower_return, bounded_distortion, loss_of_flower,
    classify_regime, failure_boundary, extended_analysis,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "nasa_turbofan"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "math"

COL_NAMES = ["unit_id", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
SENSORS = ["s2", "s3", "s4", "s7", "s11", "s12"]
BASELINE_FRAC = 0.20
WINDOW = 30


def setup_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
        "text.color": "#e0e0e0", "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0", "grid.color": "#2a2a4e",
        "grid.alpha": 0.5, "font.size": 11,
    })


def normalize(v):
    vmin, vmax = v.min(), v.max()
    return (v - vmin) / (vmax - vmin) if vmax - vmin > 1e-12 else np.zeros_like(v)


def load_data():
    df = pd.read_csv(DATA_DIR / "train_FD001.txt", sep=r"\s+", header=None, names=COL_NAMES)
    return df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)


def get_engine(df, uid):
    eng = df[df["unit_id"] == uid]
    raw = eng["s4"].values.astype(np.float64)  # LPT temp — clearest signal
    normed = normalize(raw)
    n = len(normed)
    baseline_end = max(int(n * BASELINE_FRAC), WINDOW + 1)
    baseline = normed[:baseline_end]
    return normed, baseline, n


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()
    t0 = time.time()

    print("=" * 60)
    print("  Math Module Visualizations — NASA C-MAPSS")
    print("=" * 60)

    df = load_data()
    lifetimes = df.groupby("unit_id")["cycle"].max()

    # Pick 1 representative engine (median lifetime)
    median_uid = int(lifetimes.sub(lifetimes.median()).abs().idxmin())
    # Pick a long-lived engine for richer dynamics
    long_uid = int(lifetimes.nlargest(5).index[2])
    print(f"  Primary engine: #{median_uid} ({lifetimes[median_uid]} cycles)")
    print(f"  Long engine: #{long_uid} ({lifetimes[long_uid]} cycles)")

    sig, base, n = get_engine(df, median_uid)
    sig_long, base_long, n_long = get_engine(df, long_uid)
    pct = np.linspace(0, 100, n)

    # ── 1. RQA: Recurrence Plot + DET/LAM over lifecycle ──
    print("  1. RQA recurrence plot...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Early recurrence matrix
    from delta72.rqa import recurrence_matrix
    early = sig[:60]
    R_early = recurrence_matrix(early, embedding_dim=3, delay=1, threshold_pct=15)
    ax1.imshow(R_early, cmap="binary", origin="lower", aspect="equal")
    ax1.set_title("Recurrence Plot — Early Life (Healthy)")
    ax1.set_xlabel("Time"); ax1.set_ylabel("Time")

    # Late recurrence matrix
    late = sig[-60:]
    R_late = recurrence_matrix(late, embedding_dim=3, delay=1, threshold_pct=15)
    ax2.imshow(R_late, cmap="binary", origin="lower", aspect="equal")
    ax2.set_title("Recurrence Plot — Late Life (Degraded)")
    ax2.set_xlabel("Time")

    # DET over lifecycle
    rqa_pts = []
    for start in range(0, n - WINDOW, WINDOW // 2):
        try:
            r = rqa_summary(sig[start:start + WINDOW], threshold_pct=15)
            rqa_pts.append((start / n * 100, r["DET"], r["LAM"]))
        except Exception:
            pass
    if rqa_pts:
        rp, rd, rl = zip(*rqa_pts)
        ax3.plot(rp, rd, "o-", color="#00d4ff", markersize=3, label="DET")
        ax3.plot(rp, rl, "s-", color="#50fa7b", markersize=3, label="LAM")
        ax3.set_xlabel("Lifecycle (%)"); ax3.set_ylabel("Value")
        ax3.set_title("RQA Metrics Over Lifecycle")
        ax3.legend(); ax3.grid(True)

    fig.suptitle(f"RQA — Recurrence Quantification Analysis (Engine #{median_uid})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "viz_rqa.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Wavelet: Scale-resolved coherence spectrogram ──
    print("  2. Wavelet coherence spectrogram...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

    base_tiled = np.tile(base, (n // len(base)) + 1)[:n]
    coh, scales = wavelet_coherence(sig, base_tiled, n_scales=20)
    im = ax1.imshow(coh, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                    extent=[0, 100, np.log2(scales[-1]), np.log2(scales[0])],
                    interpolation="nearest")
    ax1.set_ylabel("log₂(Scale)")
    ax1.set_title("Wavelet Coherence Spectrogram")
    plt.colorbar(im, ax=ax1, label="Coherence")

    # Scale-averaged bands
    sac = scale_averaged_coherence(sig, base_tiled, n_scales=20)
    for band, data in sac["bands"].items():
        color = {"fast": "#ff6b6b", "medium": "#ffb347", "slow": "#50fa7b"}[band]
        ax2.axhline(data["mean_coherence"], color=color, linestyle="--", alpha=0.7, label=f"{band}: {data['mean_coherence']:.3f}")
    ax2.set_ylabel("Mean Coherence")
    ax2.set_xlabel("Lifecycle (%)")
    ax2.set_title("Scale-Averaged Coherence Bands")
    ax2.legend(); ax2.grid(True); ax2.set_ylim(0, 1.05)

    fig.suptitle(f"Wavelet Coherence — Multi-Scale Decomposition (Engine #{median_uid})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "viz_wavelet.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 3. Lyapunov: Rolling LLE with chaos threshold ──
    print("  3. Lyapunov exponents...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    try:
        lle_vals, lle_centers = rolling_lyapunov(sig_long, window_size=min(80, n_long // 3), step=10)
        lle_pct = lle_centers / n_long * 100
        ax1.plot(lle_pct, lle_vals, color="#00d4ff", linewidth=1)
        ax1.axhline(0, color="#ff6b6b", linestyle="--", alpha=0.6, label="Chaos boundary")
        ax1.fill_between(lle_pct, 0, lle_vals, where=[v > 0 for v in lle_vals],
                         alpha=0.2, color="#ff6b6b", label="Chaotic")
        ax1.fill_between(lle_pct, 0, lle_vals, where=[v <= 0 for v in lle_vals],
                         alpha=0.2, color="#50fa7b", label="Stable")
        ax1.set_xlabel("Lifecycle (%)"); ax1.set_ylabel("Local Lyapunov Exponent")
        ax1.set_title(f"Rolling LLE (Engine #{long_uid})")
        ax1.legend(fontsize=8); ax1.grid(True)
    except Exception as e:
        ax1.text(0.5, 0.5, f"Error: {e}", transform=ax1.transAxes, ha="center")

    # Divergence curve for single window
    try:
        from delta72.lyapunov import divergence_curve
        div_early = divergence_curve(sig[:80])
        div_late = divergence_curve(sig[-80:])
        ax2.plot(div_early, color="#50fa7b", linewidth=2, label="Early life (healthy)")
        ax2.plot(div_late, color="#ff6b6b", linewidth=2, label="Late life (degraded)")
        ax2.set_xlabel("Time Steps"); ax2.set_ylabel("Mean Log Divergence")
        ax2.set_title("Nearest-Neighbor Divergence Curve")
        ax2.legend(); ax2.grid(True)
    except Exception:
        ax2.text(0.5, 0.5, "Divergence curve unavailable", transform=ax2.transAxes, ha="center")

    fig.suptitle("Lyapunov Exponents — Dynamical Stability", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "viz_lyapunov.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 4. BOCPD: Change-point probability over lifecycle ──
    print("  4. BOCPD change points...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)

    ax1.plot(pct, sig, color="#00d4ff", linewidth=0.5, alpha=0.7)
    ax1.set_ylabel("Signal (normalized)"); ax1.set_title(f"Sensor s4 — Engine #{median_uid}")
    ax1.grid(True)

    cp_probs = bocpd(sig, hazard_rate=1/50)
    ax2.fill_between(pct, 0, cp_probs, color="#ff6b6b", alpha=0.5)
    ax2.plot(pct, cp_probs, color="#ff6b6b", linewidth=0.5)
    ax2.set_xlabel("Lifecycle (%)"); ax2.set_ylabel("P(change point)")
    ax2.set_title("BOCPD — Change-Point Probability")
    ax2.grid(True)

    fig.suptitle("BOCPD — Bayesian Online Change Point Detection", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "viz_bocpd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 5. Transfer Entropy: Directed flow between sensors ──
    print("  5. Transfer entropy...")
    fig, ax = plt.subplots(figsize=(10, 8))

    eng_df = df[df["unit_id"] == median_uid]
    sensor_signals = [normalize(eng_df[s].values.astype(np.float64)) for s in SENSORS]
    te_mat = np.zeros((len(SENSORS), len(SENSORS)))
    for i in range(len(SENSORS)):
        for j in range(len(SENSORS)):
            if i != j:
                te_mat[i, j] = transfer_entropy(sensor_signals[i], sensor_signals[j], k=2, l=2, n_bins=6)

    im = ax.imshow(te_mat, cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(SENSORS))); ax.set_xticklabels(SENSORS, fontsize=9)
    ax.set_yticks(range(len(SENSORS))); ax.set_yticklabels(SENSORS, fontsize=9)
    ax.set_xlabel("Target"); ax.set_ylabel("Source")
    ax.set_title(f"Transfer Entropy Matrix — TE(source → target) (Engine #{median_uid})")
    for i in range(len(SENSORS)):
        for j in range(len(SENSORS)):
            if i != j:
                ax.text(j, i, f"{te_mat[i,j]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if te_mat[i,j] > te_mat.max()/2 else "black")
    plt.colorbar(im, label="Transfer Entropy (nats)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "viz_transfer_entropy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 6. Phase Space: Attractor reconstruction ──
    print("  6. Phase space reconstruction...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

    # 2D phase portrait: early vs late
    emb_early = time_delay_embedding(sig[:80], embedding_dim=2, delay=2)
    emb_late = time_delay_embedding(sig[-80:], embedding_dim=2, delay=2)
    ax1.plot(emb_early[:, 0], emb_early[:, 1], color="#50fa7b", linewidth=0.5, alpha=0.7)
    ax1.scatter(emb_early[:, 0], emb_early[:, 1], c="#50fa7b", s=5, alpha=0.5)
    ax1.set_title("Phase Portrait — Early (Healthy)"); ax1.set_xlabel("x(t)"); ax1.set_ylabel("x(t+τ)")
    ax1.grid(True)

    ax2.plot(emb_late[:, 0], emb_late[:, 1], color="#ff6b6b", linewidth=0.5, alpha=0.7)
    ax2.scatter(emb_late[:, 0], emb_late[:, 1], c="#ff6b6b", s=5, alpha=0.5)
    ax2.set_title("Phase Portrait — Late (Degraded)"); ax2.set_xlabel("x(t)"); ax2.set_ylabel("x(t+τ)")
    ax2.grid(True)

    # Attractor spread over lifecycle
    ps_results = rolling_phase_space(sig, window_size=min(60, n // 3), step=10)
    if ps_results:
        ps_pct = [r["lifecycle_pct"] for r in ps_results]
        ps_spread = [r["spread"] for r in ps_results]
        ps_compact = [r["compactness"] for r in ps_results]
        ax3.plot(ps_pct, ps_spread, "o-", color="#00d4ff", markersize=3, label="Spread")
        ax3.set_xlabel("Lifecycle (%)"); ax3.set_ylabel("Attractor Spread")
        ax3.set_title("Attractor Spread Over Lifecycle"); ax3.legend(); ax3.grid(True)

        ax4.plot(ps_pct, ps_compact, "s-", color="#bd93f9", markersize=3, label="Compactness")
        ax4.set_xlabel("Lifecycle (%)"); ax4.set_ylabel("Compactness")
        ax4.set_title("Attractor Compactness Over Lifecycle"); ax4.legend(); ax4.grid(True)

    fig.suptitle(f"Phase Space Reconstruction — Takens' Embedding (Engine #{median_uid})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "viz_phase_space.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 7. RMT: Eigenvalue spectrum + Marchenko-Pastur ──
    print("  7. Random Matrix Theory...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    eng_df = df[df["unit_id"] == median_uid]
    multi_early = np.column_stack([normalize(eng_df[s].values[:60].astype(np.float64)) for s in SENSORS])
    multi_late = np.column_stack([normalize(eng_df[s].values[-60:].astype(np.float64)) for s in SENSORS])

    rmt_early = rmt_summary(multi_early)
    rmt_late = rmt_summary(multi_late)

    eigs_e = sorted(rmt_early["eigenvalues"], reverse=True)
    eigs_l = sorted(rmt_late["eigenvalues"], reverse=True)
    mp_lo, mp_hi = marchenko_pastur_bounds(60, len(SENSORS))

    ax1.bar(range(len(eigs_e)), eigs_e, color="#50fa7b", alpha=0.7, label="Early")
    ax1.bar(range(len(eigs_l)), eigs_l, color="#ff6b6b", alpha=0.5, label="Late")
    ax1.axhline(mp_hi, color="white", linestyle="--", alpha=0.5, label=f"MP upper ({mp_hi:.2f})")
    ax1.set_xlabel("Eigenvalue Index"); ax1.set_ylabel("Eigenvalue")
    ax1.set_title("Eigenvalue Spectrum — Early vs Late")
    ax1.legend(fontsize=8); ax1.grid(True)

    # Rolling λ_max
    rmt_roll = rolling_rmt(np.column_stack([normalize(eng_df[s].values.astype(np.float64)) for s in SENSORS]),
                           window_size=40, step=10)
    if rmt_roll:
        rmt_pct = [r["lifecycle_pct"] for r in rmt_roll]
        rmt_lmax = [r["lambda_max"] for r in rmt_roll]
        rmt_pr = [r["participation_ratio"] for r in rmt_roll]
        ax2.plot(rmt_pct, rmt_lmax, "o-", color="#00d4ff", markersize=3, label="λ_max")
        ax2r = ax2.twinx()
        ax2r.plot(rmt_pct, rmt_pr, "s-", color="#bd93f9", markersize=3, alpha=0.7, label="Participation Ratio")
        ax2r.set_ylabel("Participation Ratio", color="#bd93f9")
        ax2.set_xlabel("Lifecycle (%)"); ax2.set_ylabel("λ_max", color="#00d4ff")
        ax2.set_title("Rolling RMT — λ_max and Participation Ratio")
        ax2.legend(loc="upper left", fontsize=8); ax2r.legend(loc="upper right", fontsize=8)
        ax2.grid(True)

    fig.suptitle(f"Random Matrix Theory — Eigenvalue Analysis (Engine #{median_uid})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "viz_rmt.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 8. TDA: Persistence diagram + rolling entropy ──
    print("  8. Persistent Homology / TDA...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    from delta72.tda import persistence_diagram_h0
    diag_early = persistence_diagram_h0(sig[:80], max_points=200)
    diag_late = persistence_diagram_h0(sig[-80:], max_points=200)

    if diag_early:
        births_e, deaths_e = zip(*diag_early)
        ax1.scatter(births_e, deaths_e, color="#50fa7b", s=10, alpha=0.5, label="Early")
    if diag_late:
        births_l, deaths_l = zip(*diag_late)
        ax1.scatter(births_l, deaths_l, color="#ff6b6b", s=10, alpha=0.5, label="Late")
    max_val = max([d for _, d in diag_early + diag_late]) if diag_early or diag_late else 1
    ax1.plot([0, max_val], [0, max_val], "w--", alpha=0.3)
    ax1.set_xlabel("Birth"); ax1.set_ylabel("Death")
    ax1.set_title("Persistence Diagrams — Early vs Late")
    ax1.legend(); ax1.grid(True)

    # Rolling persistence entropy
    tda_roll = rolling_tda(sig, window_size=min(60, n // 3), step=15, max_points=150)
    if tda_roll:
        tda_pct = [r["lifecycle_pct"] for r in tda_roll]
        tda_ent = [r["persistence_entropy"] for r in tda_roll]
        tda_tot = [r["total_persistence"] for r in tda_roll]
        ax2.plot(tda_pct, tda_ent, "o-", color="#00d4ff", markersize=3, label="Persistence Entropy")
        ax2r = ax2.twinx()
        ax2r.plot(tda_pct, tda_tot, "s-", color="#50fa7b", markersize=3, alpha=0.7, label="Total Persistence")
        ax2r.set_ylabel("Total Persistence", color="#50fa7b")
        ax2.set_xlabel("Lifecycle (%)"); ax2.set_ylabel("Entropy", color="#00d4ff")
        ax2.set_title("Rolling TDA Metrics")
        ax2.legend(loc="upper left", fontsize=8); ax2r.legend(loc="upper right", fontsize=8)
        ax2.grid(True)

    fig.suptitle(f"Persistent Homology — Topological Data Analysis (Engine #{median_uid})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "viz_tda.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 9. Granger: Causal network between sensors ──
    print("  9. Granger causality...")
    from delta72.granger import granger_matrix
    g_result = granger_matrix(sensor_signals, labels=SENSORS, max_lag=3)

    fig, ax = plt.subplots(figsize=(10, 8))
    p_mat = np.array(g_result["p_matrix"])
    sig_mat = (p_mat < 0.05).astype(float)
    im = ax.imshow(-np.log10(p_mat + 1e-15), cmap="YlOrRd", vmin=0, vmax=5, interpolation="nearest")
    ax.set_xticks(range(len(SENSORS))); ax.set_xticklabels(SENSORS)
    ax.set_yticks(range(len(SENSORS))); ax.set_yticklabels(SENSORS)
    ax.set_xlabel("Target (effect)"); ax.set_ylabel("Source (cause)")
    ax.set_title(f"Granger Causality — -log₁₀(p-value) (Engine #{median_uid})")
    for i in range(len(SENSORS)):
        for j in range(len(SENSORS)):
            if i != j:
                marker = "★" if p_mat[i, j] < 0.05 else ""
                ax.text(j, i, f"{marker}", ha="center", va="center", fontsize=12, color="white")
    plt.colorbar(im, label="-log₁₀(p-value)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "viz_granger.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 10. Koopman: DMD spectrum + rolling spectral entropy ──
    print("  10. Koopman operator...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    try:
        spec_early = koopman_spectrum(sig[:80], n_delays=8)
        spec_late = koopman_spectrum(sig[-80:], n_delays=8)
        from delta72.koopman import dmd
        dmd_early = dmd(sig[:80], n_delays=8)
        dmd_late = dmd(sig[-80:], n_delays=8)

        # Eigenvalue unit circle
        theta = np.linspace(0, 2 * np.pi, 100)
        ax1.plot(np.cos(theta), np.sin(theta), "w--", alpha=0.3, label="Unit circle")
        eigs_e = dmd_early["eigenvalues"]
        eigs_l = dmd_late["eigenvalues"]
        ax1.scatter(eigs_e.real, eigs_e.imag, c="#50fa7b", s=40, zorder=5, label="Early")
        ax1.scatter(eigs_l.real, eigs_l.imag, c="#ff6b6b", s=40, zorder=5, label="Late")
        ax1.set_xlabel("Re(λ)"); ax1.set_ylabel("Im(λ)")
        ax1.set_title("Koopman Eigenvalues on Unit Circle")
        ax1.set_aspect("equal"); ax1.legend(fontsize=8); ax1.grid(True)
    except Exception as e:
        ax1.text(0.5, 0.5, str(e), transform=ax1.transAxes, ha="center")

    # Rolling spectral entropy
    try:
        koop_roll = rolling_koopman(sig, window_size=min(60, n // 3), step=10, n_delays=6)
        if koop_roll:
            kp = [r["lifecycle_pct"] for r in koop_roll]
            ke = [r["spectral_entropy"] for r in koop_roll]
            ax2.plot(kp, ke, "o-", color="#bd93f9", markersize=3)
            ax2.set_xlabel("Lifecycle (%)"); ax2.set_ylabel("Spectral Entropy")
            ax2.set_title("Rolling Koopman Spectral Entropy")
            ax2.grid(True)
    except Exception:
        pass

    fig.suptitle(f"Koopman Operator Theory — Dynamic Mode Decomposition (Engine #{median_uid})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "viz_koopman.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 11. Information Geometry: Divergence rate on manifold ──
    print("  11. Information geometry...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    div_results = divergence_rate(sig, window_size=min(40, n // 4), step=10)
    if div_results:
        dp = [r["lifecycle_pct"] for r in div_results]
        dr = [r["divergence_rate"] for r in div_results]
        fi = [r["fisher_info"] for r in div_results]
        ax1.plot(dp, dr, color="#ff6b6b", linewidth=1)
        ax1.fill_between(dp, 0, dr, alpha=0.2, color="#ff6b6b")
        ax1.set_xlabel("Lifecycle (%)"); ax1.set_ylabel("Symmetric KL Divergence Rate")
        ax1.set_title("Speed on Statistical Manifold")
        ax1.grid(True)

        ax2.plot(dp, fi, color="#00d4ff", linewidth=1)
        ax2.set_xlabel("Lifecycle (%)"); ax2.set_ylabel("Fisher Information")
        ax2.set_title("Local Fisher Information (distribution sharpness)")
        ax2.grid(True)

    fig.suptitle(f"Information Geometry — Riemannian Divergence (Engine #{median_uid})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "viz_info_geometry.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 12. Ergodic Theory: Ergodicity breaking over lifecycle ──
    print("  12. Ergodic theory...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    erg_roll = rolling_ergodic(sig_long, window_size=min(100, n_long // 3), step=20)
    if erg_roll:
        ep = [r["lifecycle_pct"] for r in erg_roll]
        eb = [r["ergodicity_breaking"] for r in erg_roll]
        mr = [r["mean_recurrence"] for r in erg_roll]
        ax1.plot(ep, eb, "o-", color="#ffb347", markersize=3)
        ax1.axhline(0.1, color="#50fa7b", linestyle="--", alpha=0.5, label="Ergodic threshold")
        ax1.set_xlabel("Lifecycle (%)"); ax1.set_ylabel("Ergodicity Breaking Parameter")
        ax1.set_title("Ergodicity Breaking Over Lifecycle")
        ax1.legend(fontsize=8); ax1.grid(True)

        ax2.plot(ep, mr, "s-", color="#73daca", markersize=3)
        ax2.set_xlabel("Lifecycle (%)"); ax2.set_ylabel("Mean Recurrence Time")
        ax2.set_title("Recurrence Times (longer = less ergodic)")
        ax2.grid(True)

    fig.suptitle(f"Ergodic Theory — Time-Average vs Ensemble-Average (Engine #{long_uid})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "viz_ergodic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 13. Stochastic Resonance: SR curve ──
    print("  13. Stochastic resonance...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    base_tiled_short = np.tile(base, (WINDOW // len(base)) + 1)[:WINDOW]
    sr_result = sr_sweep(sig[:WINDOW], base_tiled_short, n_trials=30, n_levels=25)
    noise = sr_result["noise_levels"]
    snr = sr_result["snr"]
    mean_d = sr_result["mean_deltas"]

    ax1.plot(noise, snr, "o-", color="#bd93f9", markersize=4)
    best = np.argmax(snr)
    ax1.axvline(noise[best], color="#50fa7b", linestyle="--", alpha=0.6,
                label=f"Optimal noise = {noise[best]:.4f}")
    ax1.set_xlabel("Noise Amplitude"); ax1.set_ylabel("SNR (mean/std)")
    ax1.set_title("Stochastic Resonance — SNR vs Noise")
    ax1.legend(fontsize=8); ax1.grid(True)

    ax2.plot(noise, mean_d, "o-", color="#00d4ff", markersize=4)
    ax2.fill_between(noise,
                     [m - s for m, s in zip(mean_d, sr_result["std_deltas"])],
                     [m + s for m, s in zip(mean_d, sr_result["std_deltas"])],
                     alpha=0.2, color="#00d4ff")
    ax2.set_xlabel("Noise Amplitude"); ax2.set_ylabel("Mean Delta")
    ax2.set_title("Delta vs Noise Level (±1σ)")
    ax2.grid(True)

    fig.suptitle("Stochastic Resonance — Optimal Noise for Coherence Detection", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "viz_stochastic_resonance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 14. Extended Operators: Regime map + canonical operators ──
    print("  14. Extended canonical operators...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Rolling extended analysis
    ext_pts = []
    for start in range(0, n - WINDOW, WINDOW // 2):
        end = start + WINDOW
        base_w = np.tile(base, (WINDOW // len(base)) + 1)[:WINDOW]
        try:
            ea = extended_analysis(sig[start:end], base_w)
            ea["pct"] = start / n * 100
            ext_pts.append(ea)
        except Exception:
            pass

    if ext_pts:
        ep = [e["pct"] for e in ext_pts]

        # Regime map
        regimes = [e["regime"] for e in ext_pts]
        regime_colors = {"coherent": "#50fa7b", "distorted": "#ffb347", "fragmented": "#ff6b6b", "collapse": "#ff2222"}
        for i in range(len(ep) - 1):
            ax1.axvspan(ep[i], ep[i + 1], color=regime_colors.get(regimes[i], "#888"), alpha=0.3)
        ax1.plot(ep, [e["delta"] for e in ext_pts], color="white", linewidth=1)
        ax1.axhline(0.72, color="#50fa7b", linestyle=":", alpha=0.5, label="0.72 Coherent")
        ax1.axhline(0.55, color="#ffb347", linestyle=":", alpha=0.5, label="0.55 Distorted")
        ax1.axhline(0.35, color="#ff6b6b", linestyle=":", alpha=0.5, label="0.35 Fragmented")
        ax1.set_xlabel("Lifecycle (%)"); ax1.set_ylabel("Δ")
        ax1.set_title("Canonical Regime Classification")
        ax1.legend(fontsize=7, ncol=2); ax1.grid(True)

        # F, B, L operators
        ax2.plot(ep, [e["F"] for e in ext_pts], "o-", color="#50fa7b", markersize=3, label="F (Flower Return)")
        ax2.plot(ep, [e["B"] for e in ext_pts], "s-", color="#00d4ff", markersize=3, label="B (Bounded Dist.)")
        ax2.plot(ep, [e["L"] for e in ext_pts], "^-", color="#ff6b6b", markersize=3, label="L (Loss-of-Flower)")
        ax2.set_xlabel("Lifecycle (%)"); ax2.set_ylabel("Operator Value")
        ax2.set_title("Extended Unique Operators")
        ax2.legend(fontsize=8); ax2.grid(True)

        # Failure boundary forces
        ax3.plot(ep, [e["failure_boundary"]["destructive_force"] for e in ext_pts],
                 color="#ff6b6b", linewidth=1.5, label="Destructive (D+N)")
        ax3.plot(ep, [e["failure_boundary"]["restorative_force"] for e in ext_pts],
                 color="#50fa7b", linewidth=1.5, label="Restorative (P+R)")
        ax3.set_xlabel("Lifecycle (%)"); ax3.set_ylabel("Force")
        ax3.set_title("Failure Boundary — Force Balance")
        ax3.legend(); ax3.grid(True)

        # M and W (already in engine)
        ax4.plot(ep, [e["M"] for e in ext_pts], "o-", color="#00d4ff", markersize=3, label="M (Memory)")
        ax4.plot(ep, [e["W"] for e in ext_pts], "s-", color="#bd93f9", markersize=3, label="W (Windowed Recovery)")
        ax4.set_xlabel("Lifecycle (%)"); ax4.set_ylabel("Operator Value")
        ax4.set_title("Core Extended Operators (M, W)")
        ax4.legend(); ax4.grid(True)

    fig.suptitle(f"Extended Canonical Operators — Δ.72 Framework (Engine #{median_uid})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "viz_extended_operators.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    elapsed = time.time() - t0
    print(f"\n  All 14 visualizations generated in {elapsed:.1f}s")
    print(f"  Output: {OUTPUT_DIR}/viz_*.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
