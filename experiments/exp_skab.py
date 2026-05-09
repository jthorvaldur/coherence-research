#!/usr/bin/env python3
"""Experiment 9 — SKAB Industrial Valve Fault Detection.

Applies the Delta.72 coherence framework to the Skoltech Anomaly Benchmark
(SKAB) — a testbed with 8 sensors monitoring a water circulation system
under various valve fault conditions.

Each experiment file starts with normal operation (anomaly=0) then transitions
to a fault condition (anomaly=1). The anomaly-free baseline file provides
~9400 points of healthy reference data.

For each experiment:
  - Builds baseline from anomaly-free reference data
  - Normalizes each sensor to [0, 1]
  - Computes rolling Delta coherence across 8 sensors
  - Aggregates into system-level Delta score
  - Detects when Delta first drops below threshold
  - Compares lead-time vs variance-based detection
  - Evaluates precision/recall/F1 against labeled anomalies

Outputs:
  - results/skab/exp9_example_experiment.png  — single experiment with coherence overlay
  - results/skab/exp9_f1_comparison.png       — F1 by fault category, Delta vs variance
  - results/skab/exp9_detection_timeline.png  — detection time vs anomaly onset
  - results/skab/exp9_sensor_heatmap.png      — per-sensor Delta heatmap
  - results/skab/exp9_summary.png             — aggregate metrics bar chart
  - results/skab/exp9_results.json            — full results

Usage:
    uv run python experiments/exp_skab.py
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

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from delta72.engine import (
    coherence_score,
    memory_of_attractor,
    windowed_recovery,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "skab" / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "skab"

SENSOR_COLS = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]

WINDOW_SIZE = 60       # rolling window size in samples (~1 min at 1 Hz)
STEP_SIZE = 10         # step between windows
DELTA_THRESHOLD = 0.3
MEMORY_THRESHOLD = 0.4
RECOVERY_THRESHOLD = 0.4
VARIANCE_ZSCORE = 2.5  # variance alert threshold


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

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
        "figure.figsize": (12, 6),
        "font.size": 11,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_baseline() -> pd.DataFrame:
    """Load anomaly-free reference data."""
    path = DATA_DIR / "anomaly-free" / "anomaly-free.csv"
    df = pd.read_csv(path, sep=";", parse_dates=["datetime"])
    return df


def load_experiment(path: Path) -> pd.DataFrame:
    """Load a single experiment CSV."""
    df = pd.read_csv(path, sep=";", parse_dates=["datetime"])
    return df


def get_experiment_files() -> list[tuple[str, Path]]:
    """Enumerate all experiment files grouped by category."""
    experiments = []
    for category in ["valve1", "valve2", "other"]:
        cat_dir = DATA_DIR / category
        if cat_dir.exists():
            for f in sorted(cat_dir.glob("*.csv")):
                experiments.append((category, f))
    return experiments


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_sensor(values: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-12:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


# ---------------------------------------------------------------------------
# Per-experiment analysis
# ---------------------------------------------------------------------------

def analyze_experiment(
    exp_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    category: str,
    filename: str,
) -> dict:
    """Run coherence analysis on a single SKAB experiment."""
    n_points = len(exp_df)
    labels = exp_df["anomaly"].values.astype(int) if "anomaly" in exp_df.columns else np.zeros(n_points)
    anomaly_onset = int(np.argmax(labels)) if labels.any() else n_points

    # Per-sensor rolling coherence
    sensor_deltas = {}
    for scol in SENSOR_COLS:
        raw_exp = exp_df[scol].values.astype(np.float64)
        raw_base = baseline_df[scol].values.astype(np.float64)

        # Normalize experiment data
        normed_exp = normalize_sensor(raw_exp)
        normed_base = normalize_sensor(raw_base)

        # Build baseline window: tile from reference data
        deltas_for_sensor = []
        for start in range(0, n_points - WINDOW_SIZE + 1, STEP_SIZE):
            end = start + WINDOW_SIZE
            sig_window = normed_exp[start:end]

            # Baseline window: sample from anomaly-free reference
            base_start = start % (len(normed_base) - WINDOW_SIZE)
            base_window = normed_base[base_start:base_start + WINDOW_SIZE]
            if len(base_window) < WINDOW_SIZE:
                base_window = normed_base[:WINDOW_SIZE]

            scores = coherence_score(sig_window, base_window)
            mid = start + WINDOW_SIZE // 2
            deltas_for_sensor.append((mid, scores["delta"]))

        sensor_deltas[scol] = deltas_for_sensor

    # Aggregate system-level Delta
    n_windows = len(sensor_deltas[SENSOR_COLS[0]])
    system_mids = [sensor_deltas[SENSOR_COLS[0]][i][0] for i in range(n_windows)]
    system_deltas = []
    for i in range(n_windows):
        d_vals = [sensor_deltas[s][i][1] for s in SENSOR_COLS]
        system_deltas.append(float(np.mean(d_vals)))

    sys_arr = np.array(system_deltas)
    sys_mids = np.array(system_mids)

    # Normalize deltas relative to early windows
    if len(sys_arr) > 3 and sys_arr[:3].mean() > 0:
        delta_normalized = sys_arr / sys_arr[:3].mean()
        delta_normalized = np.clip(delta_normalized, 0, 2.0)
    else:
        delta_normalized = sys_arr

    # Detection: first sustained drop below threshold
    delta_alert_idx = n_points  # default: no detection
    for i in range(len(delta_normalized) - 1):
        if delta_normalized[i] < DELTA_THRESHOLD and delta_normalized[i + 1] < DELTA_THRESHOLD:
            delta_alert_idx = int(sys_mids[i])
            break

    # Variance-based detection
    var_alert_idx = n_points
    for scol in SENSOR_COLS:
        raw = exp_df[scol].values.astype(np.float64)
        normed = normalize_sensor(raw)
        # Baseline variance from anomaly-free reference
        base_normed = normalize_sensor(baseline_df[scol].values.astype(np.float64))
        baseline_var = np.var(base_normed[:WINDOW_SIZE * 3])

        for start in range(0, n_points - WINDOW_SIZE + 1, STEP_SIZE):
            end = start + WINDOW_SIZE
            win_var = np.var(normed[start:end])
            if baseline_var > 1e-12 and win_var / baseline_var > VARIANCE_ZSCORE:
                mid = start + WINDOW_SIZE // 2
                if mid < var_alert_idx:
                    var_alert_idx = mid
                break

    # Compute M and W on full experiment
    system_Ms = []
    system_Ws = []
    for i in range(n_windows):
        mid = sys_mids[i]
        start = max(0, mid - WINDOW_SIZE // 2)
        end = min(n_points, mid + WINDOW_SIZE // 2)
        # Use first sensor as representative for M/W
        raw = exp_df[SENSOR_COLS[0]].values.astype(np.float64)
        normed = normalize_sensor(raw)
        base_normed = normalize_sensor(baseline_df[SENSOR_COLS[0]].values.astype(np.float64))
        sig_win = normed[start:end]
        base_win = base_normed[:len(sig_win)]
        if len(sig_win) >= 8:
            M = memory_of_attractor(sig_win, base_win)
            W = windowed_recovery(sig_win, base_win)
        else:
            M, W = 1.0, 1.0
        system_Ms.append(M)
        system_Ws.append(W)

    # Precision/Recall/F1 for coherence detection
    # Label each point as alert if it falls in a window where Delta < threshold
    coh_pred = np.zeros(n_points, dtype=int)
    for i in range(n_windows):
        if delta_normalized[i] < DELTA_THRESHOLD:
            start = max(0, int(sys_mids[i]) - WINDOW_SIZE // 2)
            end = min(n_points, int(sys_mids[i]) + WINDOW_SIZE // 2)
            coh_pred[start:end] = 1

    var_pred = np.zeros(n_points, dtype=int)
    # Mark variance alerts
    for scol in SENSOR_COLS:
        raw = exp_df[scol].values.astype(np.float64)
        normed = normalize_sensor(raw)
        base_normed = normalize_sensor(baseline_df[scol].values.astype(np.float64))
        baseline_var = np.var(base_normed[:WINDOW_SIZE * 3])
        for start in range(0, n_points - WINDOW_SIZE + 1, STEP_SIZE):
            end = start + WINDOW_SIZE
            win_var = np.var(normed[start:end])
            if baseline_var > 1e-12 and win_var / baseline_var > VARIANCE_ZSCORE:
                var_pred[start:end] = 1

    # Compute precision/recall/F1
    def prf(pred, truth):
        tp = int(np.sum((pred == 1) & (truth == 1)))
        fp = int(np.sum((pred == 1) & (truth == 0)))
        fn = int(np.sum((pred == 0) & (truth == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    coh_p, coh_r, coh_f1 = prf(coh_pred, labels)
    var_p, var_r, var_f1 = prf(var_pred, labels)

    # Lead time: how many samples before anomaly onset
    delta_lead = max(0, anomaly_onset - delta_alert_idx) if delta_alert_idx < anomaly_onset else 0
    var_lead = max(0, anomaly_onset - var_alert_idx) if var_alert_idx < anomaly_onset else 0

    return {
        "category": category,
        "filename": filename,
        "n_points": n_points,
        "anomaly_onset": anomaly_onset,
        "n_anomaly_points": int(labels.sum()),
        "delta_alert_idx": delta_alert_idx,
        "var_alert_idx": var_alert_idx,
        "delta_lead": delta_lead,
        "var_lead": var_lead,
        "delta_detected": delta_alert_idx < n_points,
        "var_detected": var_alert_idx < n_points,
        "coherence_precision": coh_p,
        "coherence_recall": coh_r,
        "coherence_f1": coh_f1,
        "variance_precision": var_p,
        "variance_recall": var_r,
        "variance_f1": var_f1,
        "system_mids": [int(m) for m in sys_mids],
        "system_deltas": system_deltas,
        "system_Ms": system_Ms,
        "system_Ws": system_Ws,
        "delta_normalized": delta_normalized.tolist(),
        "sensor_deltas": {s: [(int(c), float(d)) for c, d in v] for s, v in sensor_deltas.items()},
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_example_experiment(result: dict, exp_df: pd.DataFrame, baseline_df: pd.DataFrame, output_dir: Path):
    """Plot 1: Single experiment with sensor traces + coherence overlay."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    colors = ["#00d4ff", "#ff6b6b", "#50fa7b", "#bd93f9", "#ffb347", "#73daca", "#f78c6c", "#c3e88d"]

    n = result["n_points"]
    x = np.arange(n)

    # Top: Sensor traces
    ax1 = axes[0]
    for i, scol in enumerate(SENSOR_COLS):
        raw = exp_df[scol].values.astype(np.float64)
        normed = normalize_sensor(raw)
        ax1.plot(x, normed, color=colors[i], alpha=0.6, linewidth=0.7, label=scol)

    # Shade anomaly region
    if result["anomaly_onset"] < n:
        ax1.axvspan(result["anomaly_onset"], n, alpha=0.15, color="#ff6b6b", label="Anomaly region")
        ax1.axvline(result["anomaly_onset"], color="#ff6b6b", linestyle="--", alpha=0.6)

    ax1.set_ylabel("Normalized Sensor Value")
    ax1.set_title(f"Experiment 9: {result['category']}/{result['filename']} — {n} samples")
    ax1.legend(fontsize=7, loc="upper left", ncol=4)
    ax1.grid(True)

    # Middle: System Delta, M, W
    ax2 = axes[1]
    mids = result["system_mids"]
    ax2.plot(mids, result["delta_normalized"], color="#00d4ff", linewidth=1.5, label="Delta (normalized)")
    ax2.plot(mids, result["system_Ms"], color="#50fa7b", linewidth=1.0, alpha=0.7, label="M (attractor)")
    ax2.plot(mids, result["system_Ws"], color="#bd93f9", linewidth=1.0, alpha=0.7, label="W (recovery)")
    ax2.axhline(DELTA_THRESHOLD, color="#ff6b6b", linestyle="--", alpha=0.6, label=f"Threshold ({DELTA_THRESHOLD})")

    if result["delta_detected"]:
        ax2.axvline(result["delta_alert_idx"], color="#00d4ff", linestyle="--", alpha=0.8,
                    label=f"Delta alert @ {result['delta_alert_idx']}")
    if result["var_detected"]:
        ax2.axvline(result["var_alert_idx"], color="#ff6b6b", linestyle="--", alpha=0.8,
                    label=f"Var alert @ {result['var_alert_idx']}")
    if result["anomaly_onset"] < n:
        ax2.axvline(result["anomaly_onset"], color="#ffb347", linestyle=":", alpha=0.8,
                    label=f"Anomaly onset @ {result['anomaly_onset']}")

    ax2.set_ylabel("Score")
    ax2.set_title(f"System Coherence — F1: {result['coherence_f1']:.3f} (coh) vs {result['variance_f1']:.3f} (var)")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True)

    # Bottom: Per-sensor Delta heatmap
    ax3 = axes[2]
    n_sensors = len(SENSOR_COLS)
    n_windows = len(result["sensor_deltas"][SENSOR_COLS[0]])
    matrix = np.zeros((n_sensors, n_windows))
    for si, scol in enumerate(SENSOR_COLS):
        for wi, (_, d) in enumerate(result["sensor_deltas"][scol]):
            matrix[si, wi] = d

    # Normalize per sensor
    for si in range(n_sensors):
        row = matrix[si]
        if row[:3].mean() > 0:
            matrix[si] = row / row[:3].mean()

    im = ax3.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=2.0, interpolation="nearest")
    ax3.set_yticks(range(n_sensors))
    ax3.set_yticklabels([s[:12] for s in SENSOR_COLS], fontsize=7)
    ax3.set_xlabel("Window Index")
    ax3.set_title("Per-Sensor Delta Decomposition")
    plt.colorbar(im, ax=ax3, pad=0.02, label="Normalized Delta")

    fig.tight_layout()
    fig.savefig(output_dir / "exp9_example_experiment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: exp9_example_experiment.png")


def plot_f1_comparison(results_by_cat: dict, output_dir: Path):
    """Plot 2: F1 by category, Delta vs variance."""
    categories = list(results_by_cat.keys())
    n_cats = len(categories)

    coh_f1s = [np.mean([r["coherence_f1"] for r in results_by_cat[c]]) for c in categories]
    var_f1s = [np.mean([r["variance_f1"] for r in results_by_cat[c]]) for c in categories]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_cats)
    w = 0.35
    ax.bar(x - w/2, coh_f1s, w, label="Delta Coherence", color="#00d4ff", alpha=0.8)
    ax.bar(x + w/2, var_f1s, w, label="Variance", color="#ff6b6b", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("F1 Score (macro avg)")
    ax.set_title("Experiment 9: SKAB — F1 by Fault Category")
    ax.legend()
    ax.grid(True, axis="y")

    for i, (cf, vf) in enumerate(zip(coh_f1s, var_f1s)):
        ax.text(i - w/2, cf + 0.01, f"{cf:.3f}", ha="center", fontsize=8, color="#00d4ff")
        ax.text(i + w/2, vf + 0.01, f"{vf:.3f}", ha="center", fontsize=8, color="#ff6b6b")

    fig.tight_layout()
    fig.savefig(output_dir / "exp9_f1_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: exp9_f1_comparison.png")


def plot_detection_timeline(all_results: list[dict], output_dir: Path):
    """Plot 3: Detection time vs anomaly onset for each experiment."""
    fig, ax = plt.subplots(figsize=(14, 6))

    detected_results = [r for r in all_results if r["anomaly_onset"] < r["n_points"]]
    detected_results.sort(key=lambda r: r["anomaly_onset"])

    for i, r in enumerate(detected_results):
        name = f"{r['category']}/{r['filename']}"
        # Anomaly onset
        ax.plot(r["anomaly_onset"], i, "s", color="#ffb347", markersize=6)
        # Delta detection
        if r["delta_detected"]:
            ax.plot(r["delta_alert_idx"], i, "o", color="#00d4ff", markersize=5)
            ax.plot([r["delta_alert_idx"], r["anomaly_onset"]], [i, i],
                    color="#00d4ff", alpha=0.3, linewidth=1)
        # Variance detection
        if r["var_detected"]:
            ax.plot(r["var_alert_idx"], i, "^", color="#ff6b6b", markersize=4, alpha=0.7)

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Experiment")
    ax.set_title("Experiment 9: Detection Timeline — Delta (circle) vs Anomaly Onset (square)")

    # Legend
    ax.plot([], [], "o", color="#00d4ff", label="Delta alert")
    ax.plot([], [], "s", color="#ffb347", label="Anomaly onset")
    ax.plot([], [], "^", color="#ff6b6b", label="Variance alert")
    ax.legend(loc="upper right")
    ax.grid(True, axis="x")

    fig.tight_layout()
    fig.savefig(output_dir / "exp9_detection_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: exp9_detection_timeline.png")


def plot_sensor_heatmap(all_results: list[dict], output_dir: Path):
    """Plot 4: Per-sensor mean Delta across all experiments."""
    n_exp = len(all_results)
    n_sensors = len(SENSOR_COLS)

    matrix = np.zeros((n_exp, n_sensors))
    labels = []
    for i, r in enumerate(all_results):
        labels.append(f"{r['category']}/{r['filename']}")
        for j, scol in enumerate(SENSOR_COLS):
            sensor_d = [d for _, d in r["sensor_deltas"][scol]]
            matrix[i, j] = np.mean(sensor_d) if sensor_d else 0

    fig, ax = plt.subplots(figsize=(14, max(8, n_exp * 0.3)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", interpolation="nearest")
    ax.set_xticks(range(n_sensors))
    ax.set_xticklabels([s[:12] for s in SENSOR_COLS], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_exp))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Experiment 9: SKAB — Per-Sensor Mean Delta by Experiment")
    plt.colorbar(im, ax=ax, pad=0.02, label="Mean Delta")

    fig.tight_layout()
    fig.savefig(output_dir / "exp9_sensor_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: exp9_sensor_heatmap.png")


def plot_summary(stats: dict, output_dir: Path):
    """Plot 5: Summary metrics bar chart."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Detection rates
    methods = ["Delta", "Variance"]
    rates = [stats["delta_detection_rate"], stats["var_detection_rate"]]
    bars = ax1.bar(methods, rates, color=["#00d4ff", "#ff6b6b"], alpha=0.8)
    ax1.set_ylabel("Detection Rate")
    ax1.set_title("Detection Rate")
    ax1.set_ylim(0, 1.15)
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{rate:.1%}", ha="center", fontsize=12, fontweight="bold")
    ax1.grid(True, axis="y")

    # F1 scores
    f1s = [stats["coherence_f1"], stats["variance_f1"]]
    bars = ax2.bar(methods, f1s, color=["#00d4ff", "#ff6b6b"], alpha=0.8)
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Overall F1 Score")
    ax2.set_ylim(0, max(f1s) * 1.3 + 0.05)
    for bar, f1 in zip(bars, f1s):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{f1:.3f}", ha="center", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y")

    # Mean lead times
    leads = [stats["delta_mean_lead"], stats["var_mean_lead"]]
    bars = ax3.bar(methods, leads, color=["#00d4ff", "#ff6b6b"], alpha=0.8)
    ax3.set_ylabel("Lead Time (samples)")
    ax3.set_title("Mean Lead Time Before Anomaly")
    for bar, lead in zip(bars, leads):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{lead:.0f}", ha="center", fontsize=12, fontweight="bold")
    ax3.grid(True, axis="y")

    # Per-experiment F1 scatter
    ax4.set_xlabel("Delta F1")
    ax4.set_ylabel("Variance F1")
    ax4.set_title("Per-Experiment F1: Delta vs Variance")
    for r in stats.get("per_experiment", []):
        color = "#00d4ff" if r["coherence_f1"] >= r["variance_f1"] else "#ff6b6b"
        ax4.scatter(r["coherence_f1"], r["variance_f1"], c=color, s=25, alpha=0.7)
    ax4.plot([0, 1], [0, 1], color="white", linestyle="--", alpha=0.3, label="Equal")
    ax4.legend()
    ax4.grid(True)

    fig.suptitle(f"Experiment 9: SKAB Industrial Fault Detection ({stats['n_experiments']} experiments)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "exp9_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: exp9_summary.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    print("=" * 60)
    print("  Experiment 9 — SKAB Industrial Valve Fault Detection")
    print("=" * 60)
    t_total = time.time()

    # Load baseline
    print("\n  Loading anomaly-free baseline...")
    baseline_df = load_baseline()
    print(f"  Baseline: {len(baseline_df)} samples, {len(SENSOR_COLS)} sensors")

    # Enumerate experiments
    experiments = get_experiment_files()
    print(f"  Found {len(experiments)} experiment files")

    # Analyze each experiment
    all_results = []
    results_by_cat = {}
    example_result = None

    for i, (category, filepath) in enumerate(experiments):
        exp_df = load_experiment(filepath)
        result = analyze_experiment(exp_df, baseline_df, category, filepath.stem)

        all_results.append(result)
        results_by_cat.setdefault(category, []).append(result)

        # Pick best example: highest coherence F1 with detected anomaly
        if result["delta_detected"] and result["anomaly_onset"] < result["n_points"]:
            if example_result is None or result["coherence_f1"] > example_result["coherence_f1"]:
                example_result = result
                example_df = exp_df

        if (i + 1) % 10 == 0 or i == len(experiments) - 1:
            print(f"    [{i+1}/{len(experiments)}] {category}/{filepath.stem} "
                  f"F1={result['coherence_f1']:.3f} lead={result['delta_lead']}")

    print(f"\n  Analyzed {len(all_results)} experiments")

    # Generate plots
    print("\n  Generating plots...")

    if example_result:
        plot_example_experiment(example_result, example_df, baseline_df, OUTPUT_DIR)

    plot_f1_comparison(results_by_cat, OUTPUT_DIR)
    plot_detection_timeline(all_results, OUTPUT_DIR)
    plot_sensor_heatmap(all_results, OUTPUT_DIR)

    # Aggregate stats
    total_elapsed = time.time() - t_total
    detected_coh = [r for r in all_results if r["delta_detected"]]
    detected_var = [r for r in all_results if r["var_detected"]]
    coh_wins = sum(1 for r in all_results if r["coherence_f1"] > r["variance_f1"])

    stats = {
        "n_experiments": len(all_results),
        "delta_detection_rate": len(detected_coh) / len(all_results) if all_results else 0,
        "var_detection_rate": len(detected_var) / len(all_results) if all_results else 0,
        "coherence_f1": float(np.mean([r["coherence_f1"] for r in all_results])),
        "variance_f1": float(np.mean([r["variance_f1"] for r in all_results])),
        "coherence_precision": float(np.mean([r["coherence_precision"] for r in all_results])),
        "coherence_recall": float(np.mean([r["coherence_recall"] for r in all_results])),
        "variance_precision": float(np.mean([r["variance_precision"] for r in all_results])),
        "variance_recall": float(np.mean([r["variance_recall"] for r in all_results])),
        "delta_mean_lead": float(np.mean([r["delta_lead"] for r in detected_coh])) if detected_coh else 0,
        "var_mean_lead": float(np.mean([r["var_lead"] for r in detected_var])) if detected_var else 0,
        "coh_wins": coh_wins,
        "per_experiment": [
            {
                "category": r["category"],
                "filename": r["filename"],
                "n_points": r["n_points"],
                "anomaly_onset": r["anomaly_onset"],
                "coherence_f1": r["coherence_f1"],
                "variance_f1": r["variance_f1"],
                "coherence_precision": r["coherence_precision"],
                "coherence_recall": r["coherence_recall"],
                "delta_lead": r["delta_lead"],
                "var_lead": r["var_lead"],
                "delta_detected": r["delta_detected"],
                "var_detected": r["var_detected"],
            }
            for r in all_results
        ],
    }

    plot_summary(stats, OUTPUT_DIR)

    # Print summary
    print("\n" + "=" * 60)
    print("  Experiment 9 Results Summary")
    print("=" * 60)
    print(f"  Experiments analyzed: {stats['n_experiments']}")
    print(f"  Delta detection rate: {stats['delta_detection_rate']:.1%}")
    print(f"  Variance detection rate: {stats['var_detection_rate']:.1%}")
    print(f"  Delta F1 (macro): {stats['coherence_f1']:.3f}")
    print(f"  Variance F1 (macro): {stats['variance_f1']:.3f}")
    print(f"  Delta mean lead: {stats['delta_mean_lead']:.0f} samples")
    print(f"  Variance mean lead: {stats['var_mean_lead']:.0f} samples")
    print(f"  Delta wins on F1: {coh_wins}/{len(all_results)} experiments")
    print(f"  Total elapsed: {total_elapsed:.1f}s")
    print("=" * 60)

    # Category breakdown
    print("\n  By category:")
    for cat, cat_results in results_by_cat.items():
        cat_f1 = np.mean([r["coherence_f1"] for r in cat_results])
        var_f1 = np.mean([r["variance_f1"] for r in cat_results])
        print(f"    {cat}: coh_F1={cat_f1:.3f}, var_F1={var_f1:.3f}, n={len(cat_results)}")

    # Save results
    exp_result = {
        "experiment": 9,
        "name": "SKAB Industrial Valve Fault Detection",
        "dataset": f"SKAB (Skoltech Anomaly Benchmark) — {len(all_results)} experiments, 8 sensors",
        "sensors_used": SENSOR_COLS,
        "config": {
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "delta_threshold": DELTA_THRESHOLD,
            "memory_threshold": MEMORY_THRESHOLD,
            "recovery_threshold": RECOVERY_THRESHOLD,
            "variance_zscore": VARIANCE_ZSCORE,
        },
        "stats": stats,
        "categories": {
            cat: {
                "n_files": len(cat_results),
                "coherence_f1": float(np.mean([r["coherence_f1"] for r in cat_results])),
                "variance_f1": float(np.mean([r["variance_f1"] for r in cat_results])),
                "coherence_precision": float(np.mean([r["coherence_precision"] for r in cat_results])),
                "coherence_recall": float(np.mean([r["coherence_recall"] for r in cat_results])),
                "variance_precision": float(np.mean([r["variance_precision"] for r in cat_results])),
                "variance_recall": float(np.mean([r["variance_recall"] for r in cat_results])),
            }
            for cat, cat_results in results_by_cat.items()
        },
        "elapsed_s": total_elapsed,
    }

    results_path = OUTPUT_DIR / "exp9_results.json"
    with open(results_path, "w") as f:
        json.dump(exp_result, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
