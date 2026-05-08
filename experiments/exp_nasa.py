#!/usr/bin/env python3
"""Experiment 8 — NASA C-MAPSS Turbofan Engine Degradation.

Applies the Delta.72 coherence framework to the NASA Commercial Modular
Aero-Propulsion System Simulation (C-MAPSS) dataset — FD001 subset.

Each engine runs until failure. The training set provides full run-to-failure
trajectories for ~100 engines with 21 sensor channels + 3 operational settings.

For each engine:
  - Uses the first 20% of cycles as the "healthy baseline"
  - Computes rolling Delta coherence on the 6 most degradation-sensitive sensors
  - Aggregates multi-sensor coherence into a single system-level Delta score
  - Computes M (memory-of-attractor) and W (windowed recovery) operators
  - Determines when Delta first drops below alert threshold
  - Compares lead-time against variance-based detection
  - Evaluates detection vs actual Remaining Useful Life (RUL)

Outputs:
  - results/nasa/exp8_example_engine.png    — single engine degradation + coherence
  - results/nasa/exp8_lead_time_dist.png    — lead-time distribution across engines
  - results/nasa/exp8_delta_vs_rul.png      — scatter: Delta at detection vs actual RUL
  - results/nasa/exp8_detection_comparison.png — bar chart: Delta vs variance detection
  - results/nasa/exp8_multi_engine_heatmap.png — heatmap of coherence across engines
  - results/nasa/exp8_results.json          — full results

Usage:
    uv run python experiments/exp_nasa.py
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

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "nasa_turbofan"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "nasa"

# Sensors that show clearest degradation trends in FD001
# (sensors 2, 3, 4, 7, 11, 12 — 1-indexed in literature, 0-indexed here as s2..s12)
SENSOR_COLS = ["s2", "s3", "s4", "s7", "s11", "s12"]
SENSOR_NAMES = {
    "s2": "Total temp (LPC outlet)",
    "s3": "Total temp (HPC outlet)",
    "s4": "Total temp (LPT outlet)",
    "s7": "Total pressure (HPC outlet)",
    "s11": "Static pressure (HPC outlet)",
    "s12": "Fuel flow ratio (Ps30/P30)",
}

BASELINE_FRACTION = 0.20   # first 20% of cycles = healthy baseline
WINDOW_SIZE = 30           # rolling window size in cycles
STEP_SIZE = 5              # step between windows
DELTA_THRESHOLD = 0.3      # coherence alert threshold
MEMORY_THRESHOLD = 0.4
RECOVERY_THRESHOLD = 0.4
VARIANCE_ZSCORE = 2.0      # variance alert threshold (x times baseline variance)


# ---------------------------------------------------------------------------
# Plot style (same dark theme as run_all.py / exp7)
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

COL_NAMES = (
    ["unit_id", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)


def load_train_data() -> pd.DataFrame:
    """Load train_FD001.txt — space-delimited, no header."""
    path = DATA_DIR / "train_FD001.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COL_NAMES)
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    return df


def load_test_rul() -> pd.DataFrame:
    """Load RUL_FD001.txt — one RUL value per test engine."""
    path = DATA_DIR / "RUL_FD001.txt"
    rul = pd.read_csv(path, sep=r"\s+", header=None, names=["rul"])
    rul["unit_id"] = rul.index + 1
    return rul


def compute_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add RUL column: for training data, RUL = max_cycle - current_cycle."""
    max_cycles = df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    df = df.merge(max_cycles, on="unit_id")
    df["rul"] = df["max_cycle"] - df["cycle"]
    return df


# ---------------------------------------------------------------------------
# Per-sensor normalization
# ---------------------------------------------------------------------------

def normalize_sensor(values: np.ndarray) -> np.ndarray:
    """Min-max normalize a sensor to [0, 1]."""
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-12:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


# ---------------------------------------------------------------------------
# Per-engine coherence analysis
# ---------------------------------------------------------------------------

def analyze_engine(
    engine_df: pd.DataFrame,
    unit_id: int,
) -> dict:
    """Run full coherence analysis on a single engine's lifecycle.

    Returns a dict with all scores, detection times, and metadata.
    """
    cycles = engine_df["cycle"].values
    n_cycles = len(cycles)
    max_cycle = int(cycles.max())

    # Baseline window: first 20% of cycles
    baseline_end = max(int(n_cycles * BASELINE_FRACTION), WINDOW_SIZE + 1)

    # Per-sensor rolling coherence
    sensor_deltas = {}  # sensor -> list of (cycle_mid, delta)
    sensor_Ms = {}
    sensor_Ws = {}

    for scol in SENSOR_COLS:
        raw = engine_df[scol].values.astype(np.float64)
        normed = normalize_sensor(raw)

        # Baseline: repeat early-life pattern to match signal length
        baseline_segment = normed[:baseline_end]

        deltas_for_sensor = []
        Ms_for_sensor = []
        Ws_for_sensor = []

        for start in range(0, n_cycles - WINDOW_SIZE + 1, STEP_SIZE):
            end = start + WINDOW_SIZE
            sig_window = normed[start:end]

            # Build baseline window: tile early-life data to match window size
            base_window = np.tile(
                baseline_segment,
                (WINDOW_SIZE // len(baseline_segment)) + 1,
            )[:WINDOW_SIZE]

            scores = coherence_score(sig_window, base_window)
            mid_cycle = int(cycles[start + WINDOW_SIZE // 2])

            deltas_for_sensor.append((mid_cycle, scores["delta"]))

            # M and W only if we have enough data
            if len(sig_window) >= 8:
                M = memory_of_attractor(sig_window, base_window)
                W = windowed_recovery(sig_window, base_window)
            else:
                M, W = 1.0, 1.0

            Ms_for_sensor.append((mid_cycle, M))
            Ws_for_sensor.append((mid_cycle, W))

        sensor_deltas[scol] = deltas_for_sensor
        sensor_Ms[scol] = Ms_for_sensor
        sensor_Ws[scol] = Ws_for_sensor

    # Aggregate: system-level Delta = mean across sensors at each time point
    # All sensors produce the same cycle midpoints
    n_windows = len(sensor_deltas[SENSOR_COLS[0]])
    system_cycles = [sensor_deltas[SENSOR_COLS[0]][i][0] for i in range(n_windows)]
    system_deltas = []
    system_Ms = []
    system_Ws = []

    for i in range(n_windows):
        d_vals = [sensor_deltas[s][i][1] for s in SENSOR_COLS]
        m_vals = [sensor_Ms[s][i][1] for s in SENSOR_COLS]
        w_vals = [sensor_Ws[s][i][1] for s in SENSOR_COLS]
        system_deltas.append(float(np.mean(d_vals)))
        system_Ms.append(float(np.mean(m_vals)))
        system_Ws.append(float(np.mean(w_vals)))

    system_deltas_arr = np.array(system_deltas)
    system_cycles_arr = np.array(system_cycles)

    # Detection: first cycle where system Delta drops below threshold
    # Normalize deltas to [0, 1] range for threshold comparison
    if system_deltas_arr.max() > 0:
        delta_normalized = system_deltas_arr / system_deltas_arr[:max(3, len(system_deltas_arr) // 5)].mean()
        delta_normalized = np.clip(delta_normalized, 0, 2.0)
    else:
        delta_normalized = system_deltas_arr

    # Find first sustained drop: Delta falls below threshold for 2+ consecutive windows
    delta_alert_cycle = max_cycle  # default: no detection
    for i in range(len(delta_normalized) - 1):
        if delta_normalized[i] < DELTA_THRESHOLD and delta_normalized[i + 1] < DELTA_THRESHOLD:
            delta_alert_cycle = int(system_cycles_arr[i])
            break

    # Variance-based detection
    # Rolling variance of raw sensor data compared to baseline variance
    var_alert_cycle = max_cycle
    for scol in SENSOR_COLS:
        raw = engine_df[scol].values.astype(np.float64)
        normed = normalize_sensor(raw)
        baseline_var = np.var(normed[:baseline_end])

        for start in range(baseline_end, n_cycles - WINDOW_SIZE + 1, STEP_SIZE):
            end = start + WINDOW_SIZE
            win_var = np.var(normed[start:end])
            if baseline_var > 1e-12 and win_var / baseline_var > VARIANCE_ZSCORE:
                mid_cycle = int(cycles[start + WINDOW_SIZE // 2])
                if mid_cycle < var_alert_cycle:
                    var_alert_cycle = mid_cycle
                break

    # RUL at detection
    delta_rul_at_detect = max_cycle - delta_alert_cycle
    var_rul_at_detect = max_cycle - var_alert_cycle

    # Lead time: cycles before failure
    delta_lead = max_cycle - delta_alert_cycle
    var_lead = max_cycle - var_alert_cycle

    return {
        "unit_id": unit_id,
        "max_cycle": max_cycle,
        "n_cycles": n_cycles,
        "system_cycles": system_cycles,
        "system_deltas": system_deltas,
        "system_Ms": system_Ms,
        "system_Ws": system_Ws,
        "delta_normalized": delta_normalized.tolist(),
        "sensor_deltas": {s: [(c, d) for c, d in v] for s, v in sensor_deltas.items()},
        "delta_alert_cycle": delta_alert_cycle,
        "var_alert_cycle": var_alert_cycle,
        "delta_lead": delta_lead,
        "var_lead": var_lead,
        "delta_rul_at_detect": delta_rul_at_detect,
        "var_rul_at_detect": var_rul_at_detect,
        "delta_detected": delta_alert_cycle < max_cycle,
        "var_detected": var_alert_cycle < max_cycle,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_example_engine(
    engine_result: dict,
    engine_df: pd.DataFrame,
    output_dir: Path,
):
    """Plot 1: Single engine degradation curve with coherence overlay."""
    uid = engine_result["unit_id"]
    cycles = engine_df["cycle"].values
    max_cycle = engine_result["max_cycle"]
    sys_cycles = engine_result["system_cycles"]
    sys_deltas = engine_result["delta_normalized"]
    sys_Ms = engine_result["system_Ms"]
    sys_Ws = engine_result["system_Ws"]

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Top: Raw sensor traces (normalized)
    ax1 = axes[0]
    colors_sensor = ["#00d4ff", "#ff6b6b", "#50fa7b", "#bd93f9", "#ffb347", "#73daca"]
    for i, scol in enumerate(SENSOR_COLS):
        raw = engine_df[scol].values.astype(np.float64)
        normed = normalize_sensor(raw)
        ax1.plot(cycles, normed, color=colors_sensor[i], alpha=0.7, linewidth=0.8,
                 label=f"{scol}: {SENSOR_NAMES[scol]}")

    baseline_end_cycle = int(cycles[max(int(len(cycles) * BASELINE_FRACTION), 1)])
    ax1.axvline(baseline_end_cycle, color="white", linestyle="--", alpha=0.4,
                label=f"Baseline end ({baseline_end_cycle})")
    ax1.set_ylabel("Normalized Sensor Value")
    ax1.set_title(f"Experiment 8: Engine {uid} — Sensor Degradation ({max_cycle} cycles to failure)")
    ax1.legend(fontsize=7, loc="upper left", ncol=2)
    ax1.grid(True)

    # Middle: System-level Delta, M, W
    ax2 = axes[1]
    ax2.plot(sys_cycles, sys_deltas, color="#00d4ff", linewidth=1.5, label="Delta (normalized)")
    ax2.plot(sys_cycles, sys_Ms, color="#50fa7b", linewidth=1.0, alpha=0.7, label="M (attractor)")
    ax2.plot(sys_cycles, sys_Ws, color="#bd93f9", linewidth=1.0, alpha=0.7, label="W (recovery)")
    ax2.axhline(DELTA_THRESHOLD, color="#ff6b6b", linestyle="--", alpha=0.6,
                label=f"Alert threshold ({DELTA_THRESHOLD})")

    if engine_result["delta_detected"]:
        ax2.axvline(engine_result["delta_alert_cycle"], color="#00d4ff", linestyle="--",
                    alpha=0.8, label=f"Delta alert @ cycle {engine_result['delta_alert_cycle']}")
    if engine_result["var_detected"]:
        ax2.axvline(engine_result["var_alert_cycle"], color="#ff6b6b", linestyle="--",
                    alpha=0.8, label=f"Variance alert @ cycle {engine_result['var_alert_cycle']}")

    ax2.set_ylabel("Score")
    ax2.set_title(f"Engine {uid} — System Coherence (lead: {engine_result['delta_lead']} cycles)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True)

    # Bottom: Per-sensor Delta breakdown
    ax3 = axes[2]
    for i, scol in enumerate(SENSOR_COLS):
        scycles = [c for c, _ in engine_result["sensor_deltas"][scol]]
        sdeltas = [d for _, d in engine_result["sensor_deltas"][scol]]
        # Normalize per-sensor
        sd_arr = np.array(sdeltas)
        if sd_arr[:max(3, len(sd_arr) // 5)].mean() > 0:
            sd_norm = sd_arr / sd_arr[:max(3, len(sd_arr) // 5)].mean()
        else:
            sd_norm = sd_arr
        ax3.plot(scycles, sd_norm, color=colors_sensor[i], alpha=0.7, linewidth=0.8,
                 label=scol)

    ax3.axhline(DELTA_THRESHOLD, color="#ff6b6b", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Cycle")
    ax3.set_ylabel("Per-Sensor Delta (normalized)")
    ax3.set_title(f"Engine {uid} — Per-Sensor Coherence Decomposition")
    ax3.legend(fontsize=8, loc="upper right", ncol=3)
    ax3.grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / "exp8_example_engine.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: exp8_example_engine.png")


def plot_lead_time_distribution(
    all_results: list[dict],
    output_dir: Path,
):
    """Plot 2: Lead-time distribution across all engines."""
    delta_leads = [r["delta_lead"] for r in all_results if r["delta_detected"]]
    var_leads = [r["var_lead"] for r in all_results if r["var_detected"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    if delta_leads:
        ax1.hist(delta_leads, bins=25, color="#00d4ff", alpha=0.7,
                 label=f"Delta (n={len(delta_leads)})", edgecolor="#16213e")
    if var_leads:
        ax1.hist(var_leads, bins=25, color="#ff6b6b", alpha=0.5,
                 label=f"Variance (n={len(var_leads)})", edgecolor="#16213e")

    ax1.set_xlabel("Lead Time (cycles before failure)")
    ax1.set_ylabel("Count (engines)")
    ax1.set_title("Lead-Time Distribution")
    ax1.legend()
    ax1.grid(True)

    # Box plot
    box_data = []
    box_labels = []
    if delta_leads:
        box_data.append(delta_leads)
        box_labels.append("Delta")
    if var_leads:
        box_data.append(var_leads)
        box_labels.append("Variance")

    if box_data:
        bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                         medianprops=dict(color="white", linewidth=2))
        colors_box = ["#00d4ff", "#ff6b6b"]
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

    ax2.set_ylabel("Lead Time (cycles)")
    ax2.set_title("Lead-Time Comparison")
    ax2.grid(True, axis="y")

    fig.suptitle("Experiment 8: NASA C-MAPSS — Lead-Time Analysis", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "exp8_lead_time_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: exp8_lead_time_dist.png")


def plot_delta_vs_rul(
    all_results: list[dict],
    output_dir: Path,
):
    """Plot 3: Delta detection cycle vs actual RUL — scatter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Delta lead vs engine lifetime
    lifetimes = [r["max_cycle"] for r in all_results]
    delta_leads = [r["delta_lead"] for r in all_results]
    detected = [r["delta_detected"] for r in all_results]

    colors = ["#00d4ff" if d else "#ff6b6b" for d in detected]
    ax1.scatter(lifetimes, delta_leads, c=colors, s=20, alpha=0.7)
    ax1.set_xlabel("Engine Lifetime (total cycles)")
    ax1.set_ylabel("Delta Lead Time (cycles before failure)")
    ax1.set_title("Lead Time vs Engine Lifetime")
    ax1.grid(True)

    # Add annotation
    mean_lead = np.mean([l for l, d in zip(delta_leads, detected) if d])
    ax1.axhline(mean_lead, color="#50fa7b", linestyle="--", alpha=0.6,
                label=f"Mean lead = {mean_lead:.0f} cycles")
    ax1.legend()

    # Right: % of life remaining at detection
    pct_remaining = []
    for r in all_results:
        if r["delta_detected"] and r["max_cycle"] > 0:
            pct_remaining.append(r["delta_lead"] / r["max_cycle"] * 100)

    if pct_remaining:
        ax2.hist(pct_remaining, bins=20, color="#50fa7b", alpha=0.7, edgecolor="#16213e")
        mean_pct = np.mean(pct_remaining)
        ax2.axvline(mean_pct, color="white", linestyle="--", alpha=0.7,
                    label=f"Mean = {mean_pct:.1f}%")
    ax2.set_xlabel("% of Engine Life Remaining at Detection")
    ax2.set_ylabel("Count")
    ax2.set_title("How Early Does Delta Alert?")
    ax2.legend()
    ax2.grid(True)

    fig.suptitle("Experiment 8: Delta Alert vs Remaining Useful Life", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "exp8_delta_vs_rul.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: exp8_delta_vs_rul.png")


def plot_detection_comparison(
    all_results: list[dict],
    output_dir: Path,
):
    """Plot 4: Detection rate + mean lead time comparison bar chart."""
    n_total = len(all_results)
    delta_detected = [r for r in all_results if r["delta_detected"]]
    var_detected = [r for r in all_results if r["var_detected"]]

    delta_rate = len(delta_detected) / n_total
    var_rate = len(var_detected) / n_total

    delta_mean_lead = np.mean([r["delta_lead"] for r in delta_detected]) if delta_detected else 0
    var_mean_lead = np.mean([r["var_lead"] for r in var_detected]) if var_detected else 0

    delta_median_lead = np.median([r["delta_lead"] for r in delta_detected]) if delta_detected else 0
    var_median_lead = np.median([r["var_lead"] for r in var_detected]) if var_detected else 0

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Detection rate
    methods = ["Delta Coherence", "Variance"]
    rates = [delta_rate, var_rate]
    bars = ax1.bar(methods, rates, color=["#00d4ff", "#ff6b6b"], alpha=0.8)
    ax1.set_ylabel("Detection Rate")
    ax1.set_title("Detection Rate (pre-failure alerts)")
    ax1.set_ylim(0, 1.15)
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{rate:.1%}", ha="center", fontsize=12, fontweight="bold")
    ax1.grid(True, axis="y")

    # Mean/Median lead time
    x = np.arange(2)
    means = [delta_mean_lead, var_mean_lead]
    medians = [delta_median_lead, var_median_lead]
    ax2.bar(x - 0.15, means, 0.3, label="Mean", color="#00d4ff", alpha=0.8)
    ax2.bar(x + 0.15, medians, 0.3, label="Median", color="#50fa7b", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.set_ylabel("Lead Time (cycles)")
    ax2.set_title("Mean/Median Lead Time Before Failure")
    ax2.legend()
    ax2.grid(True, axis="y")

    # Per-engine lead comparison scatter
    for r in all_results:
        color = "#00d4ff" if r["delta_lead"] > r["var_lead"] else "#ff6b6b"
        ax3.scatter(r["delta_lead"], r["var_lead"], c=color, s=15, alpha=0.6)

    max_lead = max(max(r["delta_lead"] for r in all_results),
                   max(r["var_lead"] for r in all_results))
    ax3.plot([0, max_lead], [0, max_lead], color="white", linestyle="--", alpha=0.3,
             label="Equal line")
    ax3.set_xlabel("Delta Lead Time (cycles)")
    ax3.set_ylabel("Variance Lead Time (cycles)")
    ax3.set_title("Per-Engine: Delta vs Variance Lead Time")
    ax3.legend()
    ax3.grid(True)

    # Advantage histogram
    advantages = [r["delta_lead"] - r["var_lead"] for r in all_results]
    ax4.hist(advantages, bins=25, color="#bd93f9", alpha=0.7, edgecolor="#16213e")
    mean_adv = np.mean(advantages)
    ax4.axvline(mean_adv, color="white", linestyle="--", alpha=0.7,
                label=f"Mean advantage = {mean_adv:.0f} cycles")
    ax4.axvline(0, color="#ff6b6b", linestyle=":", alpha=0.5)
    ax4.set_xlabel("Delta Advantage (cycles earlier than variance)")
    ax4.set_ylabel("Count")
    ax4.set_title("Delta Lead-Time Advantage per Engine")
    ax4.legend()
    ax4.grid(True)

    fig.suptitle(f"Experiment 8: Detection Comparison ({n_total} engines)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "exp8_detection_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: exp8_detection_comparison.png")


def plot_multi_engine_heatmap(
    all_results: list[dict],
    output_dir: Path,
):
    """Plot 5: Heatmap of coherence across engines over normalized lifecycle."""
    # Normalize each engine's lifecycle to [0, 1] with 50 bins
    n_bins = 50
    n_engines = len(all_results)

    # Sort by lifetime for visual clarity
    sorted_results = sorted(all_results, key=lambda r: r["max_cycle"])

    matrix = np.full((n_engines, n_bins), np.nan)

    for i, r in enumerate(sorted_results):
        sys_cycles = np.array(r["system_cycles"])
        sys_deltas = np.array(r["delta_normalized"])
        max_c = r["max_cycle"]

        if max_c == 0 or len(sys_cycles) == 0:
            continue

        # Normalize cycles to [0, 1]
        norm_cycles = sys_cycles / max_c

        for j in range(n_bins):
            bin_start = j / n_bins
            bin_end = (j + 1) / n_bins
            mask = (norm_cycles >= bin_start) & (norm_cycles < bin_end)
            if mask.any():
                matrix[i, j] = np.mean(sys_deltas[mask])

    # Custom colormap
    cmap = LinearSegmentedColormap.from_list(
        "coherence",
        ["#ff6b6b", "#ffb347", "#50fa7b", "#00d4ff"],
    )

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=0, vmax=1.5,
        interpolation="nearest",
        origin="lower",
    )

    ax.set_xlabel("Normalized Lifecycle (0% = new, 100% = failure)")
    ax.set_ylabel("Engine (sorted by lifetime)")
    ax.set_title("Experiment 8: Multi-Engine Coherence Heatmap — NASA C-MAPSS FD001")

    # X ticks as percentages
    n_xticks = 11
    xtick_pos = np.linspace(0, n_bins - 1, n_xticks)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([f"{int(x / n_bins * 100)}%" for x in xtick_pos])

    # Y ticks: sample engine IDs
    n_yticks = min(10, n_engines)
    ytick_pos = np.linspace(0, n_engines - 1, n_yticks, dtype=int)
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels([f"#{sorted_results[p]['unit_id']}" for p in ytick_pos])

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Normalized Delta (higher = more coherent)")
    cbar.ax.yaxis.label.set_color("#e0e0e0")
    cbar.ax.tick_params(colors="#e0e0e0")

    fig.tight_layout()
    fig.savefig(output_dir / "exp8_multi_engine_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: exp8_multi_engine_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    print("=" * 60)
    print("  Experiment 8 — NASA C-MAPSS Turbofan Degradation")
    print("=" * 60)
    t_total = time.time()

    # Load data
    print("\n  Loading train_FD001.txt...")
    df = load_train_data()
    df = compute_train_rul(df)
    n_engines = df["unit_id"].nunique()
    print(f"  Loaded: {len(df)} rows, {n_engines} engines")
    print(f"  Cycle range: {df['cycle'].min()} to {df.groupby('unit_id')['cycle'].max().max()}")

    # Analyze each engine
    print(f"\n  Analyzing {n_engines} engines...")
    all_results = []

    # Pick a good example engine: one with medium lifetime and clear degradation
    lifetimes = df.groupby("unit_id")["cycle"].max()
    median_lifetime = lifetimes.median()
    # Find engine closest to median
    example_uid = int(lifetimes.sub(median_lifetime).abs().idxmin())

    for uid in sorted(df["unit_id"].unique()):
        engine_df = df[df["unit_id"] == uid].copy()

        # Skip engines with very few cycles (can't compute rolling windows)
        if len(engine_df) < WINDOW_SIZE + 10:
            print(f"    [Engine {uid}] Skipped — only {len(engine_df)} cycles")
            continue

        result = analyze_engine(engine_df, uid)
        all_results.append(result)

        if uid % 20 == 0 or uid == n_engines:
            print(f"    [Engine {uid}/{n_engines}] "
                  f"life={result['max_cycle']} "
                  f"delta_lead={result['delta_lead']} "
                  f"var_lead={result['var_lead']} "
                  f"detected={'Y' if result['delta_detected'] else 'N'}")

    print(f"\n  Analyzed {len(all_results)} engines (skipped {n_engines - len(all_results)})")

    # Generate plots
    print("\n  Generating plots...")

    # Plot 1: Example engine
    example_result = next((r for r in all_results if r["unit_id"] == example_uid), all_results[0])
    example_df = df[df["unit_id"] == example_result["unit_id"]].copy()
    plot_example_engine(example_result, example_df, OUTPUT_DIR)

    # Plot 2: Lead-time distribution
    plot_lead_time_distribution(all_results, OUTPUT_DIR)

    # Plot 3: Delta vs RUL
    plot_delta_vs_rul(all_results, OUTPUT_DIR)

    # Plot 4: Detection comparison
    plot_detection_comparison(all_results, OUTPUT_DIR)

    # Plot 5: Multi-engine heatmap
    plot_multi_engine_heatmap(all_results, OUTPUT_DIR)

    # Aggregate statistics
    total_elapsed = time.time() - t_total
    delta_detected = [r for r in all_results if r["delta_detected"]]
    var_detected = [r for r in all_results if r["var_detected"]]

    stats = {
        "n_engines": len(all_results),
        "delta_detection_rate": len(delta_detected) / len(all_results),
        "var_detection_rate": len(var_detected) / len(all_results),
        "delta_mean_lead": float(np.mean([r["delta_lead"] for r in delta_detected])) if delta_detected else 0,
        "delta_median_lead": float(np.median([r["delta_lead"] for r in delta_detected])) if delta_detected else 0,
        "var_mean_lead": float(np.mean([r["var_lead"] for r in var_detected])) if var_detected else 0,
        "var_median_lead": float(np.median([r["var_lead"] for r in var_detected])) if var_detected else 0,
        "mean_advantage": float(np.mean([r["delta_lead"] - r["var_lead"] for r in all_results])),
        "mean_lifetime": float(np.mean([r["max_cycle"] for r in all_results])),
        "mean_pct_remaining": float(np.mean([
            r["delta_lead"] / r["max_cycle"] * 100
            for r in all_results if r["delta_detected"] and r["max_cycle"] > 0
        ])) if delta_detected else 0,
        "example_engine": example_result["unit_id"],
    }

    # Summary
    print("\n" + "=" * 60)
    print("  Experiment 8 Results Summary")
    print("=" * 60)
    print(f"  Engines analyzed: {stats['n_engines']}")
    print(f"  Delta detection rate: {stats['delta_detection_rate']:.1%}")
    print(f"  Variance detection rate: {stats['var_detection_rate']:.1%}")
    print(f"  Delta mean lead time: {stats['delta_mean_lead']:.1f} cycles")
    print(f"  Delta median lead time: {stats['delta_median_lead']:.1f} cycles")
    print(f"  Variance mean lead time: {stats['var_mean_lead']:.1f} cycles")
    print(f"  Mean advantage: {stats['mean_advantage']:.1f} cycles")
    print(f"  Mean % life remaining at detection: {stats['mean_pct_remaining']:.1f}%")
    print(f"  Total elapsed: {total_elapsed:.1f}s")
    print("=" * 60)

    # Save results
    exp_result = {
        "experiment": 8,
        "name": "NASA C-MAPSS Turbofan Engine Degradation",
        "dataset": "FD001 (100 engines, 21 sensors, single operating condition)",
        "sensors_used": SENSOR_COLS,
        "config": {
            "baseline_fraction": BASELINE_FRACTION,
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "delta_threshold": DELTA_THRESHOLD,
            "memory_threshold": MEMORY_THRESHOLD,
            "recovery_threshold": RECOVERY_THRESHOLD,
            "variance_zscore": VARIANCE_ZSCORE,
        },
        "stats": stats,
        "per_engine": [
            {
                "unit_id": r["unit_id"],
                "max_cycle": r["max_cycle"],
                "delta_alert_cycle": r["delta_alert_cycle"],
                "var_alert_cycle": r["var_alert_cycle"],
                "delta_lead": r["delta_lead"],
                "var_lead": r["var_lead"],
                "delta_detected": r["delta_detected"],
                "var_detected": r["var_detected"],
            }
            for r in all_results
        ],
        "elapsed_s": total_elapsed,
    }

    results_path = OUTPUT_DIR / "exp8_results.json"
    with open(results_path, "w") as f:
        json.dump(exp_result, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    # Also update main experiment_results.json
    main_results_path = OUTPUT_DIR.parent / "experiment_results.json"
    if main_results_path.exists():
        with open(main_results_path) as f:
            main_results = json.load(f)
    else:
        main_results = {}
    main_results["exp8"] = exp_result
    with open(main_results_path, "w") as f:
        json.dump(main_results, f, indent=2, default=str)
    print(f"  Updated {main_results_path}")

    print(f"  Plots saved to {OUTPUT_DIR}/exp8_*.png")


if __name__ == "__main__":
    main()
