#!/usr/bin/env python3
"""Experiment 12 — Credit Card Fraud Detection.

Applies Delta.72 to 284,807 credit card transactions. 30 features (Time,
V1-V28 PCA components, Amount) with binary fraud labels (0.17% fraud rate).

Tests whether coherence breakdown in transaction patterns precedes fraud.
Treats the time-ordered transaction stream as a time series, building
baselines from early legitimate transactions.

Outputs:
  - results/credit_card/exp12_overview.png
  - results/credit_card/exp12_precision_recall.png
  - results/credit_card/exp12_feature_importance.png
  - results/credit_card/exp12_results.json

Usage:
    uv run python experiments/exp_credit_card.py
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


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "credit_card"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "credit_card"

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
WINDOW_SIZE = 500    # transactions per window
STEP_SIZE = 100
BASELINE_N = 5000    # first 5000 legit transactions for baseline
DELTA_THRESHOLD = 0.3
VARIANCE_ZSCORE = 2.5


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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    print("=" * 60)
    print("  Experiment 12 — Credit Card Fraud Detection")
    print("=" * 60)
    t0 = time.time()

    df = pd.read_csv(DATA_DIR / "creditcard.csv")
    df["Class"] = df["Class"].astype(int)
    n = len(df)
    n_fraud = int(df["Class"].sum())
    print(f"  Loaded: {n:,} transactions, {n_fraud} fraud ({n_fraud/n:.3%})")

    # Use top 6 features by variance (most informative PCA components)
    top_features = df[FEATURE_COLS].var().nlargest(6).index.tolist()
    print(f"  Top features by variance: {top_features}")

    # Build baseline from first BASELINE_N legitimate transactions
    legit = df[df["Class"] == 0].head(BASELINE_N)
    baselines = {}
    for feat in top_features:
        baselines[feat] = normalize(legit[feat].values.astype(np.float64))

    # Rolling coherence across all transactions
    labels = df["Class"].values
    feature_deltas = {}

    for feat in top_features:
        raw = df[feat].values.astype(np.float64)
        normed = normalize(raw)
        base = baselines[feat]

        deltas = []
        for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
            end = start + WINDOW_SIZE
            sig_win = normed[start:end]
            base_win = np.tile(base, (WINDOW_SIZE // len(base)) + 1)[:WINDOW_SIZE]
            scores = coherence_score(sig_win, base_win)
            mid = start + WINDOW_SIZE // 2
            fraud_in_window = int(labels[start:end].sum())
            deltas.append({
                "mid": mid,
                "delta": scores["delta"],
                "fraud_count": fraud_in_window,
                "has_fraud": fraud_in_window > 0,
            })
        feature_deltas[feat] = deltas

    # System-level aggregation
    n_windows = len(feature_deltas[top_features[0]])
    system_deltas = []
    for i in range(n_windows):
        d_vals = [feature_deltas[f][i]["delta"] for f in top_features]
        mid = feature_deltas[top_features[0]][i]["mid"]
        has_fraud = feature_deltas[top_features[0]][i]["has_fraud"]
        fraud_count = feature_deltas[top_features[0]][i]["fraud_count"]
        system_deltas.append({
            "mid": mid,
            "delta": float(np.mean(d_vals)),
            "has_fraud": has_fraud,
            "fraud_count": fraud_count,
        })

    # Normalize
    sys_arr = np.array([d["delta"] for d in system_deltas])
    if len(sys_arr) > 5 and sys_arr[:5].mean() > 0:
        d_norm = sys_arr / sys_arr[:5].mean()
        d_norm = np.clip(d_norm, 0, 2.0)
    else:
        d_norm = sys_arr

    for i, dn in enumerate(d_norm):
        system_deltas[i]["delta_norm"] = float(dn)

    # Detection metrics
    coh_pred = np.array([d["delta_norm"] < DELTA_THRESHOLD for d in system_deltas])
    fraud_truth = np.array([d["has_fraud"] for d in system_deltas])

    tp = int(np.sum(coh_pred & fraud_truth))
    fp = int(np.sum(coh_pred & ~fraud_truth))
    fn = int(np.sum(~coh_pred & fraud_truth))
    tn = int(np.sum(~coh_pred & ~fraud_truth))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Variance baseline
    var_pred = np.zeros(n_windows, dtype=bool)
    for feat in top_features[:3]:
        raw = df[feat].values.astype(np.float64)
        normed = normalize(raw)
        base_var = np.var(baselines[feat])
        for i in range(n_windows):
            mid = system_deltas[i]["mid"]
            start = max(0, mid - WINDOW_SIZE // 2)
            end = min(n, mid + WINDOW_SIZE // 2)
            win_var = np.var(normed[start:end])
            if base_var > 1e-12 and win_var / base_var > VARIANCE_ZSCORE:
                var_pred[i] = True

    var_tp = int(np.sum(var_pred & fraud_truth))
    var_fp = int(np.sum(var_pred & ~fraud_truth))
    var_fn = int(np.sum(~var_pred & fraud_truth))
    var_prec = var_tp / (var_tp + var_fp) if (var_tp + var_fp) > 0 else 0
    var_rec = var_tp / (var_tp + var_fn) if (var_tp + var_fn) > 0 else 0
    var_f1 = 2 * var_prec * var_rec / (var_prec + var_rec) if (var_prec + var_rec) > 0 else 0

    # Mean delta for fraud vs legit windows
    fraud_windows = [d["delta_norm"] for d in system_deltas if d["has_fraud"]]
    legit_windows = [d["delta_norm"] for d in system_deltas if not d["has_fraud"]]
    mean_delta_fraud = float(np.mean(fraud_windows)) if fraud_windows else 0
    mean_delta_legit = float(np.mean(legit_windows)) if legit_windows else 0

    print(f"\n  Results:")
    print(f"    Coherence — P={precision:.3f} R={recall:.3f} F1={f1:.3f}")
    print(f"    Variance  — P={var_prec:.3f} R={var_rec:.3f} F1={var_f1:.3f}")
    print(f"    Mean Delta (fraud): {mean_delta_fraud:.3f}")
    print(f"    Mean Delta (legit): {mean_delta_legit:.3f}")

    # --- Plots ---
    print("\n  Generating plots...")

    # Plot 1: Overview — delta over transaction stream
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    mids = [d["mid"] for d in system_deltas]
    d_norms = [d["delta_norm"] for d in system_deltas]
    fraud_mids = [d["mid"] for d in system_deltas if d["has_fraud"]]
    fraud_d = [d["delta_norm"] for d in system_deltas if d["has_fraud"]]

    ax1.plot(mids, d_norms, color="#00d4ff", linewidth=0.5, alpha=0.7)
    ax1.scatter(fraud_mids, fraud_d, color="#ff6b6b", s=8, alpha=0.5, label="Fraud windows", zorder=5)
    ax1.axhline(DELTA_THRESHOLD, color="#ff6b6b", linestyle="--", alpha=0.4)
    ax1.set_ylabel("Delta (normalized)")
    ax1.set_title(f"Experiment 12: Credit Card Fraud — {n:,} transactions, {n_fraud} fraud")
    ax1.legend(fontsize=8)
    ax1.grid(True)

    # Fraud count per window
    fraud_counts = [d["fraud_count"] for d in system_deltas]
    ax2.bar(mids, fraud_counts, width=STEP_SIZE, color="#ff6b6b", alpha=0.5)
    ax2.set_xlabel("Transaction Index")
    ax2.set_ylabel("Fraud Count per Window")
    ax2.set_title("Fraud Distribution Across Transaction Stream")
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp12_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: exp12_overview.png")

    # Plot 2: Precision/Recall comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    methods = ["Delta", "Variance"]
    precs = [precision, var_prec]
    recs = [recall, var_rec]
    f1s = [f1, var_f1]

    x = np.arange(2)
    w = 0.25
    ax1.bar(x - w, precs, w, label="Precision", color="#00d4ff", alpha=0.8)
    ax1.bar(x, recs, w, label="Recall", color="#50fa7b", alpha=0.8)
    ax1.bar(x + w, f1s, w, label="F1", color="#bd93f9", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.set_title("Detection Metrics: Delta vs Variance")
    ax1.legend()
    ax1.grid(True, axis="y")

    # Delta distribution: fraud vs legit
    ax2.hist(legit_windows, bins=30, alpha=0.6, color="#50fa7b", label="Legit", density=True)
    if fraud_windows:
        ax2.hist(fraud_windows, bins=30, alpha=0.6, color="#ff6b6b", label="Fraud", density=True)
    ax2.set_xlabel("Delta (normalized)")
    ax2.set_ylabel("Density")
    ax2.set_title("Delta Distribution: Fraud vs Legitimate")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp12_precision_recall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: exp12_precision_recall.png")

    # Plot 3: Per-feature delta heatmap
    n_feats = len(top_features)
    n_win = len(feature_deltas[top_features[0]])
    matrix = np.zeros((n_feats, n_win))
    for fi, feat in enumerate(top_features):
        for wi, d in enumerate(feature_deltas[feat]):
            matrix[fi, wi] = d["delta"]
    for fi in range(n_feats):
        row = matrix[fi]
        em = row[:5].mean() if len(row) > 5 else row.mean()
        if em > 0:
            matrix[fi] = row / em

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=2, interpolation="nearest")
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(top_features, fontsize=8)
    ax.set_xlabel("Window Index")
    ax.set_title("Per-Feature Normalized Delta")
    plt.colorbar(im, ax=ax, pad=0.02, label="Normalized Delta")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp12_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: exp12_feature_importance.png")

    elapsed = time.time() - t0

    stats = {
        "n_transactions": n,
        "n_fraud": n_fraud,
        "fraud_rate": round(n_fraud / n, 5),
        "n_windows": n_windows,
        "coherence_precision": round(precision, 3),
        "coherence_recall": round(recall, 3),
        "coherence_f1": round(f1, 3),
        "variance_precision": round(var_prec, 3),
        "variance_recall": round(var_rec, 3),
        "variance_f1": round(var_f1, 3),
        "mean_delta_fraud": round(mean_delta_fraud, 3),
        "mean_delta_legit": round(mean_delta_legit, 3),
        "top_features": top_features,
    }

    result = {
        "experiment": 12,
        "name": "Credit Card Fraud Detection",
        "dataset": f"Credit Card Fraud — {n:,} transactions, 30 features, {n_fraud} fraud",
        "config": {
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "baseline_n": BASELINE_N,
            "delta_threshold": DELTA_THRESHOLD,
            "variance_zscore": VARIANCE_ZSCORE,
            "n_features_used": len(top_features),
        },
        "stats": stats,
        "elapsed_s": round(elapsed, 1),
    }

    with open(OUTPUT_DIR / "exp12_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Saved to {OUTPUT_DIR / 'exp12_results.json'}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
