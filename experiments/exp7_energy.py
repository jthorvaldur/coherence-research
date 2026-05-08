#!/usr/bin/env python3
"""Experiment 7 — Real Data Validation on Energy Systems.

Applies the Delta.72 coherence framework to real hourly electricity load data
from four office buildings (Hog_office_Betsy, Hog_office_Nia, Lamb_office_Vasiliki,
Rat_office_Avis).

For each building:
  - Builds an hour-of-week baseline from the full dataset
  - Computes rolling Delta coherence scores using CoherenceEngine
  - Computes M (memory-of-attractor) and W (windowed recovery) operators
  - Runs the gated alert check
  - Compares alert timing vs simple variance-based alerts
  - Generates per-building plots (load vs baseline, coherence timeline, alert comparison)
  - Generates a cross-building instability heatmap

Usage:
    uv run python experiments/exp7_energy.py
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
    CoherenceEngine,
    coherence_score,
    memory_of_attractor,
    windowed_recovery,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "extracted"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"

BUILDINGS = [
    "Hog_office_Betsy",
    "Hog_office_Nia",
    "Lamb_office_Vasiliki",
    "Rat_office_Avis",
]

WINDOW_SIZE = 168      # 1 week of hourly data
STEP_SIZE = 24         # slide by 1 day
DELTA_THRESHOLD = 0.3
MEMORY_THRESHOLD = 0.4
RECOVERY_THRESHOLD = 0.4
VARIANCE_ZSCORE = 2.0  # for simple variance-based alert comparison


# ---------------------------------------------------------------------------
# Plot style (same dark theme as run_all.py)
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

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load electricity and metadata CSVs."""
    print("  Loading electricity data...")
    elec_path = DATA_DIR / "electricity_phase1_sample.csv"
    # Only load timestamp + target building columns
    cols_to_load = ["timestamp"] + BUILDINGS
    elec = pd.read_csv(elec_path, usecols=cols_to_load, parse_dates=["timestamp"])
    elec = elec.sort_values("timestamp").reset_index(drop=True)

    print("  Loading metadata...")
    meta = pd.read_csv(DATA_DIR / "metadata_phase1.csv")

    return elec, meta


# ---------------------------------------------------------------------------
# Baseline construction
# ---------------------------------------------------------------------------

def build_hour_of_week_baseline(series: pd.Series, timestamps: pd.Series) -> np.ndarray:
    """Build an hour-of-week baseline (168 bins: 0=Mon 00:00, 167=Sun 23:00).

    Returns a baseline array the same length as the input series.
    """
    hour_of_week = timestamps.dt.dayofweek * 24 + timestamps.dt.hour
    df = pd.DataFrame({"value": series.values, "how": hour_of_week.values})
    medians = df.groupby("how")["value"].median()

    baseline = medians.reindex(hour_of_week.values).values
    return baseline.astype(np.float64)


# ---------------------------------------------------------------------------
# Variance-based alert (naive comparator)
# ---------------------------------------------------------------------------

def variance_alerts(signal: np.ndarray, baseline: np.ndarray,
                    window: int, step: int, z_threshold: float = 2.0) -> list[dict]:
    """Simple rolling variance alert: flag if window variance exceeds z_threshold * global variance."""
    residuals = signal - baseline
    global_var = np.var(residuals[np.isfinite(residuals)])

    alerts = []
    n = len(signal)
    for start in range(0, n - window + 1, step):
        end = start + window
        seg_resid = residuals[start:end]
        valid = seg_resid[np.isfinite(seg_resid)]
        if len(valid) < window // 2:
            alerts.append({"window_start": start, "window_end": end, "alert": False, "var_ratio": 0.0})
            continue
        win_var = np.var(valid)
        var_ratio = win_var / (global_var + 1e-12)
        alerts.append({
            "window_start": start,
            "window_end": end,
            "alert": var_ratio > z_threshold,
            "var_ratio": float(var_ratio),
        })
    return alerts


# ---------------------------------------------------------------------------
# Per-building analysis
# ---------------------------------------------------------------------------

def analyze_building(
    building: str,
    elec: pd.DataFrame,
    meta: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Full analysis for one building. Returns summary dict."""
    print(f"\n  [{building}] Starting analysis...")
    t0 = time.time()

    ts = elec["timestamp"]
    raw = elec[building].values.astype(np.float64)

    # Handle NaN: forward-fill then back-fill small gaps
    nan_count = int(np.isnan(raw).sum())
    series = pd.Series(raw).ffill().bfill()
    signal = series.values

    # Build baseline
    baseline = build_hour_of_week_baseline(series, ts)

    n = len(signal)
    print(f"    Data points: {n}, NaN filled: {nan_count}")

    # --- Rolling coherence via CoherenceEngine ---
    engine = CoherenceEngine(
        window_size=WINDOW_SIZE,
        delta_threshold=DELTA_THRESHOLD,
        memory_threshold=MEMORY_THRESHOLD,
        recovery_threshold=RECOVERY_THRESHOLD,
    )
    print(f"    Computing rolling coherence (window={WINDOW_SIZE}, step={STEP_SIZE})...")
    rolling = engine.score_rolling(signal, baseline, step=STEP_SIZE)

    # Extract time series from rolling results
    window_starts = [r["window_start"] for r in rolling]
    window_midpoints = [r["window_start"] + WINDOW_SIZE // 2 for r in rolling]
    deltas = [r["delta"] for r in rolling]
    Ms = [r["M"] for r in rolling]
    Ws = [r["W"] for r in rolling]
    alerts_coherence = [r["alert"] for r in rolling]

    # Map window midpoints to dates
    mid_dates = [ts.iloc[min(m, len(ts) - 1)] for m in window_midpoints]

    # --- Variance-based alerts ---
    var_alerts = variance_alerts(signal, baseline, WINDOW_SIZE, STEP_SIZE, z_threshold=VARIANCE_ZSCORE)
    alerts_variance = [r["alert"] for r in var_alerts]
    var_ratios = [r["var_ratio"] for r in var_alerts]

    # --- Summary statistics ---
    n_coherence_alerts = sum(alerts_coherence)
    n_variance_alerts = sum(alerts_variance)

    # Coherence-only alerts (flagged by coherence but NOT by variance)
    coherence_only = sum(1 for c, v in zip(alerts_coherence, alerts_variance) if c and not v)
    # Variance-only alerts
    variance_only = sum(1 for c, v in zip(alerts_coherence, alerts_variance) if not c and v)
    # Both alert
    both_alert = sum(1 for c, v in zip(alerts_coherence, alerts_variance) if c and v)

    # Global metrics over entire signal
    global_scores = coherence_score(signal, baseline)
    global_M = memory_of_attractor(signal, baseline)
    global_W = windowed_recovery(signal, baseline)

    # Building metadata
    meta_row = meta[meta["building_id"] == building]
    sqft = float(meta_row["sqft"].values[0]) if len(meta_row) > 0 and pd.notna(meta_row["sqft"].values[0]) else None
    usage = str(meta_row["primaryspaceusage"].values[0]) if len(meta_row) > 0 else "Unknown"

    result = {
        "building": building,
        "usage": usage,
        "sqft": sqft,
        "n_datapoints": n,
        "nan_filled": nan_count,
        "global_delta": global_scores["delta"],
        "global_P": global_scores["P"],
        "global_A": global_scores["A"],
        "global_R": global_scores["R"],
        "global_D": global_scores["D"],
        "global_N": global_scores["N"],
        "global_M": global_M,
        "global_W": global_W,
        "n_windows": len(rolling),
        "n_coherence_alerts": n_coherence_alerts,
        "n_variance_alerts": n_variance_alerts,
        "coherence_only_alerts": coherence_only,
        "variance_only_alerts": variance_only,
        "both_alerts": both_alert,
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "min_delta": float(np.min(deltas)),
        "mean_M": float(np.mean(Ms)),
        "mean_W": float(np.mean(Ws)),
        "elapsed_s": 0.0,  # will be updated
    }

    # -----------------------------------------------------------------------
    # Plot 1: Load vs Baseline
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    ax1 = axes[0]
    ax1.plot(ts, signal, color="#00d4ff", alpha=0.4, linewidth=0.3, label="Actual Load")
    ax1.plot(ts, baseline, color="#ff6b6b", linewidth=0.8, alpha=0.8, label="Hour-of-Week Baseline")
    ax1.set_ylabel("Load (kWh)")
    ax1.set_title(f"Experiment 7: {building} — Load vs Baseline")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True)

    # Plot 2: Coherence timeline
    ax2 = axes[1]
    ax2.plot(mid_dates, deltas, color="#00d4ff", linewidth=1.0, label="Delta (coherence)", alpha=0.9)
    ax2.plot(mid_dates, Ms, color="#50fa7b", linewidth=0.8, label="M (attractor)", alpha=0.7)
    ax2.plot(mid_dates, Ws, color="#bd93f9", linewidth=0.8, label="W (recovery)", alpha=0.7)
    ax2.axhline(DELTA_THRESHOLD, color="#ff6b6b", linestyle="--", alpha=0.5, label=f"Alert threshold ({DELTA_THRESHOLD})")

    # Shade coherence alert windows
    for i, (alert, date) in enumerate(zip(alerts_coherence, mid_dates)):
        if alert:
            ax2.axvspan(
                mid_dates[max(i - 1, 0)], mid_dates[min(i + 1, len(mid_dates) - 1)],
                color="#ff6b6b", alpha=0.1,
            )

    ax2.set_ylabel("Score")
    ax2.set_title(f"{building} — Coherence Timeline")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True)

    # Plot 3: Alert comparison
    ax3 = axes[2]
    coh_alert_dates = [d for d, a in zip(mid_dates, alerts_coherence) if a]
    var_alert_dates = [d for d, a in zip(mid_dates, alerts_variance) if a]

    if coh_alert_dates:
        ax3.scatter(coh_alert_dates, [1.0] * len(coh_alert_dates),
                    color="#00d4ff", s=8, alpha=0.7, label=f"Coherence alerts ({n_coherence_alerts})")
    if var_alert_dates:
        ax3.scatter(var_alert_dates, [0.5] * len(var_alert_dates),
                    color="#ff6b6b", s=8, alpha=0.7, label=f"Variance alerts ({n_variance_alerts})")

    ax3.set_yticks([0.5, 1.0])
    ax3.set_yticklabels(["Variance", "Coherence"])
    ax3.set_ylim(0, 1.5)
    ax3.set_xlabel("Date")
    ax3.set_title(f"{building} — Alert Comparison (Coherence-only: {coherence_only}, Variance-only: {variance_only})")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, axis="x")

    fig.tight_layout()
    plot_path = output_dir / f"exp7_{building.lower()}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {plot_path.name}")

    result["elapsed_s"] = time.time() - t0

    # Return rolling data for heatmap
    return result, {
        "dates": mid_dates,
        "deltas": deltas,
        "Ms": Ms,
        "Ws": Ws,
        "alerts_coherence": alerts_coherence,
        "alerts_variance": alerts_variance,
    }


# ---------------------------------------------------------------------------
# Cross-building heatmap
# ---------------------------------------------------------------------------

def generate_heatmap(
    all_rolling: dict[str, dict],
    output_dir: Path,
):
    """Generate cross-building instability heatmap (1 - delta over time)."""
    print("\n  Generating cross-building heatmap...")

    # Align all buildings to the shortest common timeline
    min_len = min(len(v["deltas"]) for v in all_rolling.values())
    buildings = list(all_rolling.keys())

    # Build matrix: rows = buildings, cols = time windows
    # Instability = 1 - min(delta, 1) so values in [0, 1] where 1 = most unstable
    matrix = np.zeros((len(buildings), min_len))
    for i, bld in enumerate(buildings):
        deltas = np.array(all_rolling[bld]["deltas"][:min_len])
        # Clip deltas to [0, reasonable max] then normalize
        clipped = np.clip(deltas, 0, np.percentile(deltas, 95) + 0.01)
        max_val = clipped.max() if clipped.max() > 0 else 1.0
        matrix[i, :] = 1.0 - (clipped / max_val)

    # Use dates from first building for x-axis
    dates = all_rolling[buildings[0]]["dates"][:min_len]

    # Custom colormap: dark blue (stable) -> yellow -> red (unstable)
    cmap = LinearSegmentedColormap.from_list(
        "instability",
        ["#16213e", "#00d4ff", "#ffb347", "#ff6b6b"],
    )

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # Y-axis: building names
    ax.set_yticks(range(len(buildings)))
    ax.set_yticklabels([b.replace("_", " ") for b in buildings], fontsize=10)

    # X-axis: sampled dates
    n_ticks = 12
    tick_positions = np.linspace(0, min_len - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([dates[i].strftime("%Y-%m") for i in tick_positions], rotation=45, fontsize=9)

    ax.set_xlabel("Date")
    ax.set_title("Experiment 7: Cross-Building Instability Heatmap (darker = more stable)")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Instability (1 - normalized Delta)")
    cbar.ax.yaxis.label.set_color("#e0e0e0")
    cbar.ax.tick_params(colors="#e0e0e0")

    fig.tight_layout()
    heatmap_path = output_dir / "exp7_heatmap.png"
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {heatmap_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    print("=" * 60)
    print("  Experiment 7 — Real Data Validation on Energy Systems")
    print("=" * 60)
    t_total = time.time()

    elec, meta = load_data()

    all_results = {}
    all_rolling = {}

    for building in BUILDINGS:
        result, rolling_data = analyze_building(building, elec, meta, OUTPUT_DIR)
        all_results[building] = result
        all_rolling[building] = rolling_data

    # Cross-building heatmap
    generate_heatmap(all_rolling, OUTPUT_DIR)

    total_elapsed = time.time() - t_total

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  Experiment 7 Results Summary")
    print("=" * 60)
    for bld, r in all_results.items():
        print(f"\n  {bld}:")
        print(f"    Global Delta: {r['global_delta']:.4f}  |  M: {r['global_M']:.3f}  |  W: {r['global_W']:.3f}")
        print(f"    Mean rolling Delta: {r['mean_delta']:.4f}  |  Median: {r['median_delta']:.4f}")
        print(f"    Coherence alerts: {r['n_coherence_alerts']}  |  Variance alerts: {r['n_variance_alerts']}")
        print(f"    Coherence-only: {r['coherence_only_alerts']}  |  Variance-only: {r['variance_only_alerts']}")

    print(f"\n  Total elapsed: {total_elapsed:.1f}s")
    print("=" * 60)

    # --- Save results JSON ---
    exp7_result = {
        "experiment": 7,
        "name": "Real Data Validation on Energy Systems",
        "buildings": all_results,
        "config": {
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "delta_threshold": DELTA_THRESHOLD,
            "memory_threshold": MEMORY_THRESHOLD,
            "recovery_threshold": RECOVERY_THRESHOLD,
            "variance_zscore": VARIANCE_ZSCORE,
        },
        "elapsed_s": total_elapsed,
    }

    results_path = OUTPUT_DIR / "exp7_results.json"
    with open(results_path, "w") as f:
        json.dump(exp7_result, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    # Also update the main experiment_results.json
    main_results_path = OUTPUT_DIR / "experiment_results.json"
    if main_results_path.exists():
        with open(main_results_path) as f:
            main_results = json.load(f)
    else:
        main_results = {}

    main_results["exp7"] = exp7_result
    with open(main_results_path, "w") as f:
        json.dump(main_results, f, indent=2, default=str)
    print(f"  Updated {main_results_path}")

    print(f"  Plots saved to {OUTPUT_DIR}/exp7_*.png")


if __name__ == "__main__":
    main()
