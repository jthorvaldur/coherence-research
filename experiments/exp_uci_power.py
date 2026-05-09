#!/usr/bin/env python3
"""Experiment 10 — UCI Household Electric Power Consumption.

Applies Delta.72 to 4 years of 1-minute resolution household electricity data.
7 features: global active/reactive power, voltage, intensity, 3 sub-meters.
~2M readings from a single household in Sceaux, France (Dec 2006 - Nov 2010).

Unlike Exp 7 (hourly office buildings with hour-of-week baselines), this dataset
tests Delta.72 at fine granularity on residential data with no labeled anomalies.
We build daily baselines (minute-of-day profiles) and detect structural
coherence loss — periods where the household's consumption pattern breaks down.

Outputs:
  - results/uci_power/exp10_overview.png           — full timeline + Delta
  - results/uci_power/exp10_monthly_heatmap.png    — coherence by month
  - results/uci_power/exp10_daily_profile.png      — baseline profile vs actual
  - results/uci_power/exp10_alert_distribution.png — when alerts occur
  - results/uci_power/exp10_multi_feature.png      — per-feature Delta decomposition
  - results/uci_power/exp10_results.json           — full results

Usage:
    uv run python experiments/exp_uci_power.py
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

from delta72.engine import coherence_score, memory_of_attractor, windowed_recovery


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "uci_power"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "uci_power"

FEATURES = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

# Resample to hourly for tractability (2M rows → ~35K)
RESAMPLE_FREQ = "1h"
WINDOW_SIZE = 168       # 1 week of hourly data
STEP_SIZE = 24          # 1 day step
BASELINE_WEEKS = 8      # 8 weeks to build baseline profile
DELTA_THRESHOLD = 0.3
MEMORY_THRESHOLD = 0.4
RECOVERY_THRESHOLD = 0.4
VARIANCE_ZSCORE = 2.5


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
        "figure.figsize": (14, 6),
        "font.size": 11,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load and preprocess UCI household power data."""
    path = DATA_DIR / "household_power_consumption.txt"
    df = pd.read_csv(
        path,
        sep=";",
        na_values="?",
        low_memory=False,
    )
    # Combine Date + Time into datetime index
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.set_index("datetime").drop(columns=["Date", "Time"])

    # Convert to numeric
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where all features are NaN
    df = df.dropna(how="all", subset=FEATURES)

    # Resample to hourly (mean)
    df = df[FEATURES].resample(RESAMPLE_FREQ).mean()

    # Forward-fill small gaps, drop remaining NaN
    df = df.ffill(limit=3).dropna()

    return df


def normalize_feature(values: np.ndarray) -> np.ndarray:
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    if vmax - vmin < 1e-12:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def build_hourly_baseline(df: pd.DataFrame, feature: str, n_weeks: int = 8) -> np.ndarray:
    """Build hour-of-week baseline from first n_weeks of data."""
    baseline_end = n_weeks * 168  # hours in n_weeks
    baseline_data = df[feature].iloc[:baseline_end]

    # Hour-of-week profile (168 values)
    profile = np.zeros(168)
    counts = np.zeros(168)
    for i, (idx, val) in enumerate(baseline_data.items()):
        how = idx.weekday() * 24 + idx.hour
        profile[how] += val
        counts[how] += 1

    counts[counts == 0] = 1
    return profile / counts


def analyze_power(df: pd.DataFrame) -> dict:
    """Run coherence analysis on household power data."""
    n_hours = len(df)

    # Per-feature rolling coherence
    feature_deltas = {}
    feature_baselines = {}

    for feat in FEATURES:
        raw = df[feat].values.astype(np.float64)
        normed = normalize_feature(raw)

        # Build baseline profile
        profile = build_hourly_baseline(df, feat, BASELINE_WEEKS)
        profile_normed = normalize_feature(profile)

        # Expand profile to full length
        baseline_full = np.zeros(n_hours)
        for i, idx in enumerate(df.index):
            how = idx.weekday() * 24 + idx.hour
            baseline_full[i] = profile_normed[how]

        feature_baselines[feat] = baseline_full

        deltas = []
        for start in range(0, n_hours - WINDOW_SIZE + 1, STEP_SIZE):
            end = start + WINDOW_SIZE
            sig_window = normed[start:end]
            base_window = baseline_full[start:end]

            scores = coherence_score(sig_window, base_window)
            mid = start + WINDOW_SIZE // 2
            deltas.append((mid, scores["delta"]))

        feature_deltas[feat] = deltas

    # System-level aggregation
    n_windows = len(feature_deltas[FEATURES[0]])
    system_mids = [feature_deltas[FEATURES[0]][i][0] for i in range(n_windows)]
    system_deltas = []
    system_Ms = []
    system_Ws = []

    for i in range(n_windows):
        d_vals = [feature_deltas[f][i][1] for f in FEATURES]
        system_deltas.append(float(np.mean(d_vals)))

    sys_arr = np.array(system_deltas)

    # Normalize
    if len(sys_arr) > 5 and sys_arr[:5].mean() > 0:
        delta_normalized = sys_arr / sys_arr[:5].mean()
        delta_normalized = np.clip(delta_normalized, 0, 2.0)
    else:
        delta_normalized = sys_arr

    # Compute M and W for system
    primary = df[FEATURES[0]].values.astype(np.float64)
    primary_normed = normalize_feature(primary)
    primary_baseline = feature_baselines[FEATURES[0]]

    for i in range(n_windows):
        mid = system_mids[i]
        start = max(0, mid - WINDOW_SIZE // 2)
        end = min(n_hours, mid + WINDOW_SIZE // 2)
        sig_win = primary_normed[start:end]
        base_win = primary_baseline[start:end]
        if len(sig_win) >= 8:
            M = memory_of_attractor(sig_win, base_win)
            W = windowed_recovery(sig_win, base_win)
        else:
            M, W = 1.0, 1.0
        system_Ms.append(M)
        system_Ws.append(W)

    # Alerts
    n_alerts = 0
    n_var_alerts = 0
    alert_indices = []

    for i in range(len(delta_normalized)):
        if delta_normalized[i] < DELTA_THRESHOLD:
            n_alerts += 1
            alert_indices.append(system_mids[i])

    # Variance alerts
    for feat in FEATURES[:3]:
        raw = df[feat].values.astype(np.float64)
        normed = normalize_feature(raw)
        baseline = feature_baselines[feat]
        baseline_var = np.var(normed[:WINDOW_SIZE * 3] - baseline[:WINDOW_SIZE * 3])

        for start in range(0, n_hours - WINDOW_SIZE + 1, STEP_SIZE):
            end = start + WINDOW_SIZE
            residual = normed[start:end] - baseline[start:end]
            win_var = np.var(residual)
            if baseline_var > 1e-12 and win_var / baseline_var > VARIANCE_ZSCORE:
                n_var_alerts += 1
                break  # one per feature

    # Monthly coherence
    monthly = {}
    for i in range(n_windows):
        mid = system_mids[i]
        if mid < len(df):
            month_key = df.index[mid].strftime("%Y-%m")
            monthly.setdefault(month_key, []).append(delta_normalized[i])

    monthly_means = {k: float(np.mean(v)) for k, v in sorted(monthly.items())}

    # Alert hours distribution
    alert_hours = []
    for aidx in alert_indices:
        if aidx < len(df):
            alert_hours.append(df.index[aidx].hour)

    return {
        "n_hours": n_hours,
        "n_windows": n_windows,
        "n_alerts": n_alerts,
        "n_var_alerts": n_var_alerts,
        "coherence_only": max(0, n_alerts - n_var_alerts),
        "mean_delta": float(delta_normalized.mean()),
        "system_mids": system_mids,
        "system_deltas": system_deltas,
        "delta_normalized": delta_normalized.tolist(),
        "system_Ms": system_Ms,
        "system_Ws": system_Ws,
        "monthly_means": monthly_means,
        "alert_hours": alert_hours,
        "alert_indices": alert_indices[:100],
        "feature_deltas": {f: [(int(m), float(d)) for m, d in v] for f, v in feature_deltas.items()},
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_overview(result: dict, df: pd.DataFrame, output_dir: Path):
    """Plot 1: Full timeline with Delta overlay."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Raw power
    ax1.plot(df.index, df["Global_active_power"].values, color="#00d4ff", alpha=0.3, linewidth=0.3)
    ax1.set_ylabel("Active Power (kW)")
    ax1.set_title("Experiment 10: UCI Household Power — 4 Years, 1-Minute Resolution")
    ax1.grid(True)

    # Delta
    mids = result["system_mids"]
    mid_dates = [df.index[m] if m < len(df) else df.index[-1] for m in mids]
    ax2.plot(mid_dates, result["delta_normalized"], color="#00d4ff", linewidth=0.8)
    ax2.axhline(DELTA_THRESHOLD, color="#ff6b6b", linestyle="--", alpha=0.6, label=f"Threshold ({DELTA_THRESHOLD})")
    ax2.fill_between(mid_dates, 0, result["delta_normalized"],
                     where=[d < DELTA_THRESHOLD for d in result["delta_normalized"]],
                     alpha=0.2, color="#ff6b6b")
    ax2.set_ylabel("Delta (normalized)")
    ax2.set_title(f"System Coherence — {result['n_alerts']} alerts detected")
    ax2.legend(fontsize=8)
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / "exp10_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: exp10_overview.png")


def plot_monthly_heatmap(result: dict, output_dir: Path):
    """Plot 2: Monthly coherence heatmap."""
    months = list(result["monthly_means"].keys())
    values = list(result["monthly_means"].values())

    fig, ax = plt.subplots(figsize=(16, 4))
    matrix = np.array(values).reshape(1, -1)
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=2.0, interpolation="nearest")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=7)
    ax.set_yticks([])
    ax.set_title("Experiment 10: Monthly Mean Coherence")
    plt.colorbar(im, ax=ax, pad=0.02, label="Normalized Delta")

    fig.tight_layout()
    fig.savefig(output_dir / "exp10_monthly_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: exp10_monthly_heatmap.png")


def plot_daily_profile(df: pd.DataFrame, output_dir: Path):
    """Plot 3: Baseline daily profile vs actual data."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Compute hourly profile across entire dataset
    hourly = df["Global_active_power"].groupby(df.index.hour).agg(["mean", "std"])
    hours = hourly.index

    ax.fill_between(hours, hourly["mean"] - hourly["std"], hourly["mean"] + hourly["std"],
                    alpha=0.2, color="#00d4ff")
    ax.plot(hours, hourly["mean"], color="#00d4ff", linewidth=2, label="Mean (all data)")

    # Early period
    early = df["Global_active_power"].iloc[:BASELINE_WEEKS * 168]
    early_hourly = early.groupby(early.index.hour).mean()
    ax.plot(early_hourly.index, early_hourly.values, color="#50fa7b", linewidth=2,
            linestyle="--", label=f"Baseline ({BASELINE_WEEKS} weeks)")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Active Power (kW)")
    ax.set_title("Experiment 10: Daily Consumption Profile — Baseline vs Overall")
    ax.legend()
    ax.grid(True)
    ax.set_xticks(range(0, 24, 2))

    fig.tight_layout()
    fig.savefig(output_dir / "exp10_daily_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: exp10_daily_profile.png")


def plot_alert_distribution(result: dict, output_dir: Path):
    """Plot 4: When do alerts occur (hour of day + day of week)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    hours = result["alert_hours"]
    if hours:
        ax1.hist(hours, bins=24, range=(0, 24), color="#ff6b6b", alpha=0.7, edgecolor="#16213e")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Alert Count")
    ax1.set_title("Coherence Alerts by Hour")
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(True, axis="y")

    # Monthly alert counts
    monthly = result["monthly_means"]
    months = list(monthly.keys())
    low_months = [m for m, v in monthly.items() if v < DELTA_THRESHOLD]
    high_months = [m for m, v in monthly.items() if v >= DELTA_THRESHOLD]

    colors = ["#ff6b6b" if monthly[m] < DELTA_THRESHOLD else "#50fa7b" for m in months]
    ax2.bar(range(len(months)), [monthly[m] for m in months], color=colors, alpha=0.7)
    ax2.set_xticks(range(len(months)))
    ax2.set_xticklabels(months, rotation=45, ha="right", fontsize=6)
    ax2.axhline(DELTA_THRESHOLD, color="white", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Mean Coherence")
    ax2.set_title("Monthly Mean Coherence (red = below threshold)")
    ax2.grid(True, axis="y")

    fig.suptitle("Experiment 10: Alert Patterns", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "exp10_alert_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: exp10_alert_distribution.png")


def plot_multi_feature(result: dict, output_dir: Path):
    """Plot 5: Per-feature Delta decomposition."""
    n_features = len(FEATURES)
    n_windows = len(result["feature_deltas"][FEATURES[0]])

    matrix = np.zeros((n_features, n_windows))
    for fi, feat in enumerate(FEATURES):
        for wi, (_, d) in enumerate(result["feature_deltas"][feat]):
            matrix[fi, wi] = d

    # Normalize per feature
    for fi in range(n_features):
        row = matrix[fi]
        early_mean = row[:5].mean() if len(row) > 5 else row.mean()
        if early_mean > 0:
            matrix[fi] = row / early_mean

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=2.0, interpolation="nearest")
    ax.set_yticks(range(n_features))
    ax.set_yticklabels([f[:20] for f in FEATURES], fontsize=8)
    ax.set_xlabel("Window Index")
    ax.set_title("Experiment 10: Per-Feature Normalized Delta")
    plt.colorbar(im, ax=ax, pad=0.02, label="Normalized Delta")

    fig.tight_layout()
    fig.savefig(output_dir / "exp10_multi_feature.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: exp10_multi_feature.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    print("=" * 60)
    print("  Experiment 10 — UCI Household Power Consumption")
    print("=" * 60)
    t0 = time.time()

    print("\n  Loading data...")
    df = load_data()
    print(f"  Loaded: {len(df)} hourly readings")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Features: {len(FEATURES)}")

    print("\n  Analyzing...")
    result = analyze_power(df)

    print(f"  Windows: {result['n_windows']}")
    print(f"  Coherence alerts: {result['n_alerts']}")
    print(f"  Variance alerts: {result['n_var_alerts']}")
    print(f"  Coherence-only: {result['coherence_only']}")
    print(f"  Mean Delta: {result['mean_delta']:.4f}")

    print("\n  Generating plots...")
    plot_overview(result, df, OUTPUT_DIR)
    plot_monthly_heatmap(result, OUTPUT_DIR)
    plot_daily_profile(df, OUTPUT_DIR)
    plot_alert_distribution(result, OUTPUT_DIR)
    plot_multi_feature(result, OUTPUT_DIR)

    elapsed = time.time() - t0

    # Save results
    exp_result = {
        "experiment": 10,
        "name": "UCI Household Electric Power Consumption",
        "dataset": f"4 years, 1-minute resolution, resampled to hourly ({len(df)} readings)",
        "features_used": FEATURES,
        "config": {
            "resample_freq": RESAMPLE_FREQ,
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "baseline_weeks": BASELINE_WEEKS,
            "delta_threshold": DELTA_THRESHOLD,
            "memory_threshold": MEMORY_THRESHOLD,
            "recovery_threshold": RECOVERY_THRESHOLD,
            "variance_zscore": VARIANCE_ZSCORE,
        },
        "stats": {
            "n_hours": result["n_hours"],
            "n_windows": result["n_windows"],
            "n_alerts": result["n_alerts"],
            "n_var_alerts": result["n_var_alerts"],
            "coherence_only": result["coherence_only"],
            "mean_delta": result["mean_delta"],
            "monthly_means": result["monthly_means"],
        },
        "elapsed_s": round(elapsed, 1),
    }

    results_path = OUTPUT_DIR / "exp10_results.json"
    with open(results_path, "w") as f:
        json.dump(exp_result, f, indent=2)

    print(f"\n  Results saved to {results_path}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
