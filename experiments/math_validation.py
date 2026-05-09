#!/usr/bin/env python3
"""Mathematical Extensions Validation — Apply all math modules to NASA data.

Runs RQA, Wavelet Coherence, Lyapunov Exponents, and BOCPD on NASA C-MAPSS
turbofan data to validate each extension against known degradation trajectories.

Outputs:
  - results/math/rqa_vs_delta.png      — RQA DET/LAM over engine lifecycle
  - results/math/wavelet_scales.png    — Wavelet coherence by scale band
  - results/math/lyapunov_rolling.png  — Rolling Lyapunov exponent over lifecycle
  - results/math/bocpd_detection.png   — BOCPD change points vs Delta alerts
  - results/math/math_summary.png      — Combined summary
  - results/math/math_results.json     — Full results

Usage:
    uv run python experiments/math_validation.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from delta72.engine import coherence_score
from delta72.rqa import rqa_summary
from delta72.wavelet import wavelet_coherence, scale_averaged_coherence
from delta72.lyapunov import rolling_lyapunov, lyapunov_summary
from delta72.bocpd import bocpd, bocpd_alerts


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "nasa_turbofan"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "math"

COL_NAMES = ["unit_id", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
SENSOR = "s4"  # LPT outlet temp — clearest degradation signal
N_ENGINES = 8  # sample for speed
BASELINE_FRAC = 0.20
WINDOW = 30


def setup_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "text.color": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "grid.color": "#2a2a4e",
        "grid.alpha": 0.5,
        "font.size": 11,
    })


def normalize(v):
    vmin, vmax = v.min(), v.max()
    return (v - vmin) / (vmax - vmin) if vmax - vmin > 1e-12 else np.zeros_like(v)


def load_data():
    path = DATA_DIR / "train_FD001.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COL_NAMES)
    return df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    print("=" * 60)
    print("  Math Extensions Validation — NASA C-MAPSS")
    print("=" * 60)
    t0 = time.time()

    df = load_data()
    lifetimes = df.groupby("unit_id")["cycle"].max()
    sampled = lifetimes.sort_values().iloc[
        np.linspace(0, len(lifetimes) - 1, N_ENGINES, dtype=int)
    ].index.tolist()

    print(f"  Engines: {sampled}")

    # Per-engine results
    all_rqa = []
    all_wavelet = []
    all_lyapunov = []
    all_bocpd = []
    all_delta = []

    for uid in sampled:
        eng = df[df["unit_id"] == uid]
        raw = eng[SENSOR].values.astype(np.float64)
        normed = normalize(raw)
        n = len(normed)
        max_cycle = int(eng["cycle"].max())
        baseline_end = max(int(n * BASELINE_FRAC), WINDOW + 1)
        baseline = normed[:baseline_end]

        print(f"  Engine {uid}: {n} cycles")

        # --- Delta rolling ---
        deltas = []
        for start in range(0, n - WINDOW + 1, 10):
            end = start + WINDOW
            base_w = np.tile(baseline, (WINDOW // len(baseline)) + 1)[:WINDOW]
            scores = coherence_score(normed[start:end], base_w)
            pct = (start + WINDOW // 2) / n * 100
            deltas.append({"pct": pct, "delta": scores["delta"], "P": scores["P"]})
        all_delta.append({"uid": uid, "points": deltas})

        # --- RQA rolling ---
        rqa_pts = []
        for start in range(0, n - WINDOW + 1, WINDOW // 2):
            end = start + WINDOW
            try:
                r = rqa_summary(normed[start:end], embedding_dim=3, delay=1, threshold_pct=15)
                pct = (start + WINDOW // 2) / n * 100
                rqa_pts.append({"pct": pct, **r})
            except Exception:
                pass
        all_rqa.append({"uid": uid, "points": rqa_pts})

        # --- Wavelet coherence ---
        wav_pts = []
        for start in range(0, n - WINDOW * 2 + 1, WINDOW):
            end = start + WINDOW * 2
            sig_seg = normed[start:end]
            base_seg = np.tile(baseline, (len(sig_seg) // len(baseline)) + 1)[:len(sig_seg)]
            try:
                s = scale_averaged_coherence(sig_seg, base_seg, n_scales=12)
                pct = (start + WINDOW) / n * 100
                wav_pts.append({"pct": pct, **{k: v["mean_coherence"] for k, v in s["bands"].items()}, "overall": s["overall_mean"]})
            except Exception:
                pass
        all_wavelet.append({"uid": uid, "points": wav_pts})

        # --- Lyapunov rolling ---
        try:
            lsummary = lyapunov_summary(normed, window_size=min(80, n // 3), step=max(10, n // 20))
            lle_vals, lle_centers = rolling_lyapunov(normed, window_size=min(80, n // 3), step=max(10, n // 20))
            lyap_pts = [{"pct": c / n * 100, "lle": float(v)} for c, v in zip(lle_centers, lle_vals)]
            all_lyapunov.append({"uid": uid, "points": lyap_pts, "summary": lsummary})
        except Exception as e:
            all_lyapunov.append({"uid": uid, "points": [], "summary": {"regime": "error", "lle_global": 0}})
            print(f"    Lyapunov error: {e}")

        # --- BOCPD ---
        try:
            cp_probs = bocpd(normed, hazard_rate=1 / 50)
            alerts = bocpd_alerts(normed, threshold=0.1, hazard_rate=1 / 50)
            bocpd_pts = [{"idx": int(a), "pct": a / n * 100} for a in alerts[:20]]
            all_bocpd.append({"uid": uid, "points": bocpd_pts, "n_cps": len(alerts),
                              "max_prob": float(cp_probs.max()), "max_idx": int(cp_probs.argmax())})
        except Exception as e:
            all_bocpd.append({"uid": uid, "points": [], "n_cps": 0, "max_prob": 0, "max_idx": 0})
            print(f"    BOCPD error: {e}")

    # ---- Plotting ----
    print("\n  Generating plots...")

    # Plot 1: RQA DET + LAM over lifecycle
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, N_ENGINES))
    for i, eng in enumerate(all_rqa):
        pcts = [p["pct"] for p in eng["points"]]
        dets = [p["DET"] for p in eng["points"]]
        lams = [p["LAM"] for p in eng["points"]]
        ax1.plot(pcts, dets, color=colors[i], alpha=0.7, linewidth=1, label=f"#{eng['uid']}")
        ax2.plot(pcts, lams, color=colors[i], alpha=0.7, linewidth=1)
    ax1.set_ylabel("DET (Determinism)")
    ax1.set_title("RQA: Determinism Over Engine Lifecycle")
    ax1.legend(fontsize=7, ncol=4)
    ax1.grid(True)
    ax2.set_xlabel("Lifecycle (%)")
    ax2.set_ylabel("LAM (Laminarity)")
    ax2.set_title("RQA: Laminarity Over Engine Lifecycle")
    ax2.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rqa_vs_delta.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: rqa_vs_delta.png")

    # Plot 2: Wavelet scales over lifecycle
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    band_names = ["fast", "medium", "slow"]
    for bi, band in enumerate(band_names):
        for i, eng in enumerate(all_wavelet):
            pcts = [p["pct"] for p in eng["points"] if band in p]
            vals = [p[band] for p in eng["points"] if band in p]
            if pcts:
                axes[bi].plot(pcts, vals, color=colors[i], alpha=0.7, linewidth=1,
                              label=f"#{eng['uid']}" if bi == 0 else None)
        axes[bi].set_ylabel(f"{band.capitalize()} Coherence")
        axes[bi].set_title(f"Wavelet Coherence — {band.capitalize()} Scale")
        axes[bi].grid(True)
        axes[bi].set_ylim(0, 1.05)
    axes[0].legend(fontsize=7, ncol=4)
    axes[2].set_xlabel("Lifecycle (%)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "wavelet_scales.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: wavelet_scales.png")

    # Plot 3: Rolling Lyapunov
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, eng in enumerate(all_lyapunov):
        pcts = [p["pct"] for p in eng["points"]]
        lles = [p["lle"] for p in eng["points"]]
        if pcts:
            ax.plot(pcts, lles, color=colors[i], alpha=0.7, linewidth=1, label=f"#{eng['uid']}")
    ax.axhline(0, color="#ff6b6b", linestyle="--", alpha=0.5, label="Chaos threshold")
    ax.set_xlabel("Lifecycle (%)")
    ax.set_ylabel("Local Lyapunov Exponent")
    ax.set_title("Rolling Lyapunov Exponent Over Engine Lifecycle")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lyapunov_rolling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: lyapunov_rolling.png")

    # Plot 4: BOCPD change points vs Delta
    fig, axes = plt.subplots(min(4, N_ENGINES), 1, figsize=(14, 3 * min(4, N_ENGINES)), sharex=True)
    if min(4, N_ENGINES) == 1:
        axes = [axes]
    for i in range(min(4, N_ENGINES)):
        ax = axes[i]
        eng_d = all_delta[i]
        eng_b = all_bocpd[i]
        pcts = [p["pct"] for p in eng_d["points"]]
        deltas = [p["delta"] for p in eng_d["points"]]
        # Normalize deltas
        d_arr = np.array(deltas)
        if d_arr[:3].mean() > 0:
            d_norm = d_arr / d_arr[:3].mean()
        else:
            d_norm = d_arr
        ax.plot(pcts, np.clip(d_norm, 0, 2), color="#00d4ff", linewidth=1, label="Delta")
        ax.axhline(0.3, color="#ff6b6b", linestyle="--", alpha=0.4)
        # BOCPD change points
        for cp in eng_b["points"]:
            ax.axvline(cp["pct"], color="#50fa7b", linestyle=":", alpha=0.6)
        ax.set_ylabel(f"#{eng_d['uid']}")
        ax.set_ylim(0, 2)
        ax.grid(True)
        if i == 0:
            ax.set_title("BOCPD Change Points (green) vs Delta Coherence (blue)")
            ax.legend(fontsize=7)
    axes[-1].set_xlabel("Lifecycle (%)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "bocpd_detection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: bocpd_detection.png")

    # Plot 5: Combined summary — lifecycle binned averages
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    n_bins = 10
    bin_edges = np.linspace(0, 100, n_bins + 1)

    # Delta over lifecycle
    all_d_pcts = []
    all_d_vals = []
    for eng in all_delta:
        for p in eng["points"]:
            d = p["delta"]
            early = [pp["delta"] for pp in eng["points"][:3]]
            em = np.mean(early) if early else 1
            all_d_pcts.append(p["pct"])
            all_d_vals.append(d / em if em > 0 else d)
    _plot_binned(ax1, all_d_pcts, all_d_vals, bin_edges, "#00d4ff", "Delta (normalized)")
    ax1.set_title("Delta Over Lifecycle")
    ax1.set_ylabel("Mean Normalized Delta")

    # RQA DET over lifecycle
    all_r_pcts = [p["pct"] for eng in all_rqa for p in eng["points"]]
    all_r_dets = [p["DET"] for eng in all_rqa for p in eng["points"]]
    _plot_binned(ax2, all_r_pcts, all_r_dets, bin_edges, "#50fa7b", "RQA DET")
    ax2.set_title("RQA Determinism Over Lifecycle")
    ax2.set_ylabel("Mean DET")

    # Wavelet overall over lifecycle
    all_w_pcts = [p["pct"] for eng in all_wavelet for p in eng["points"] if "overall" in p]
    all_w_vals = [p["overall"] for eng in all_wavelet for p in eng["points"] if "overall" in p]
    _plot_binned(ax3, all_w_pcts, all_w_vals, bin_edges, "#bd93f9", "Wavelet Coherence")
    ax3.set_title("Wavelet Coherence Over Lifecycle")
    ax3.set_ylabel("Mean Coherence")

    # Lyapunov over lifecycle
    all_l_pcts = [p["pct"] for eng in all_lyapunov for p in eng["points"]]
    all_l_vals = [p["lle"] for eng in all_lyapunov for p in eng["points"]]
    _plot_binned(ax4, all_l_pcts, all_l_vals, bin_edges, "#ff6b6b", "Lyapunov (LLE)")
    ax4.axhline(0, color="white", linestyle="--", alpha=0.3)
    ax4.set_title("Lyapunov Exponent Over Lifecycle")
    ax4.set_ylabel("Mean LLE")

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel("Lifecycle (%)")
        ax.grid(True)

    fig.suptitle("Math Extensions Summary — All Metrics Over Engine Lifecycle", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "math_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: math_summary.png")

    elapsed = time.time() - t0

    # Save JSON
    result = {
        "n_engines": N_ENGINES,
        "engines": sampled,
        "sensor": SENSOR,
        "rqa": {
            "description": "Recurrence Quantification Analysis — nonlinear dynamics validation of P",
            "n_data_points": sum(len(e["points"]) for e in all_rqa),
        },
        "wavelet": {
            "description": "Wavelet Coherence — multi-scale decomposition of coherence",
            "n_data_points": sum(len(e["points"]) for e in all_wavelet),
        },
        "lyapunov": {
            "description": "Rolling local Lyapunov exponent — dynamical stability measure",
            "regimes": {e["uid"]: e["summary"].get("regime", "?") for e in all_lyapunov},
        },
        "bocpd": {
            "description": "Bayesian Online Change Point Detection — benchmark competitor",
            "change_points_per_engine": {e["uid"]: e["n_cps"] for e in all_bocpd},
        },
        "elapsed_s": round(elapsed, 1),
    }
    with open(OUTPUT_DIR / "math_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n  Results saved. Elapsed: {elapsed:.1f}s")
    print("=" * 60)


def _plot_binned(ax, pcts, vals, bin_edges, color, label):
    pcts = np.array(pcts)
    vals = np.array(vals)
    centers = []
    means = []
    stds = []
    for i in range(len(bin_edges) - 1):
        mask = (pcts >= bin_edges[i]) & (pcts < bin_edges[i + 1])
        if mask.sum() > 0:
            centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            means.append(vals[mask].mean())
            stds.append(vals[mask].std())
    if centers:
        ax.fill_between(centers, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                         alpha=0.15, color=color)
        ax.plot(centers, means, "o-", color=color, linewidth=2, label=label)
        ax.legend(fontsize=8)


if __name__ == "__main__":
    main()
