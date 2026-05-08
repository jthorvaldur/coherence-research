#!/usr/bin/env python3
"""Experiment 8 — NAB (Numenta Anomaly Benchmark) Validation.

Applies the Delta.72 coherence framework to the full NAB benchmark suite (58 time
series across 7 categories). Compares coherence-based anomaly detection against
NAB's labeled anomaly windows and a simple variance-based detector.

For each time series:
  - Builds a rolling-mean baseline (168-point rolling mean)
  - Runs CoherenceEngine rolling scorer
  - Compares coherence alert windows against NAB labeled anomaly windows
  - Computes precision, recall, F1

Aggregates results by NAB category and generates comparison plots.

Usage:
    uv run python experiments/exp_nab.py
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

from delta72.engine import CoherenceEngine, coherence_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "nab"
LABELS_DIR = Path(__file__).resolve().parent.parent / "data" / "nab_labels"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "nab"

# CoherenceEngine parameters
WINDOW_SIZE = 168         # rolling window for coherence scoring
STEP_SIZE = 24            # step between windows
BASELINE_WINDOW = 168     # rolling mean window for baseline
DELTA_THRESHOLD = 0.3     # coherence alert threshold
MEMORY_THRESHOLD = 0.4
RECOVERY_THRESHOLD = 0.4

# Variance detector parameters
VAR_WINDOW = 168
VAR_ZSCORE = 2.5          # flag if window variance > z * global variance

# Categories to benchmark (skip artificialNoAnomaly — no labels)
CATEGORIES = [
    "artificialWithAnomaly",
    "realAWSCloudwatch",
    "realAdExchange",
    "realKnownCause",
    "realTraffic",
    "realTweets",
]


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

def load_nab_labels() -> dict:
    """Load NAB combined anomaly windows."""
    labels_path = LABELS_DIR / "combined_windows.json"
    with open(labels_path) as f:
        return json.load(f)


def load_nab_csv(category: str, filename: str) -> pd.DataFrame:
    """Load a single NAB CSV file."""
    path = DATA_DIR / category / filename
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Baseline construction
# ---------------------------------------------------------------------------

def build_rolling_mean_baseline(values: np.ndarray, window: int = 168) -> np.ndarray:
    """Build a rolling-mean baseline. Pads edges with the nearest available mean."""
    series = pd.Series(values)
    rolling = series.rolling(window=window, center=True, min_periods=1).mean()
    return rolling.values.astype(np.float64)


# ---------------------------------------------------------------------------
# Anomaly window matching
# ---------------------------------------------------------------------------

def timestamps_to_index_set(
    timestamps: pd.Series,
    windows: list[list[str]],
) -> set[int]:
    """Convert NAB anomaly windows (timestamp pairs) to a set of integer indices."""
    anomaly_indices = set()
    for window in windows:
        start = pd.Timestamp(window[0])
        end = pd.Timestamp(window[1])
        mask = (timestamps >= start) & (timestamps <= end)
        anomaly_indices.update(timestamps.index[mask].tolist())
    return anomaly_indices


def alert_windows_to_index_set(
    alerts: list[dict],
) -> set[int]:
    """Convert CoherenceEngine rolling results to set of indices covered by alert windows."""
    alert_indices = set()
    for a in alerts:
        if a.get("alert", False):
            alert_indices.update(range(a["window_start"], a["window_end"]))
    return alert_indices


def compute_pr_f1(
    predicted: set[int],
    actual: set[int],
    total_length: int,
) -> dict:
    """Compute precision, recall, F1 for point-based anomaly detection."""
    if not predicted and not actual:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(actual)}
    if not actual:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": len(predicted), "fn": 0}

    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ---------------------------------------------------------------------------
# Variance-based detector (comparison baseline)
# ---------------------------------------------------------------------------

def variance_detector(
    signal: np.ndarray,
    baseline: np.ndarray,
    window: int = 168,
    step: int = 24,
    z_threshold: float = 2.5,
) -> list[dict]:
    """Simple rolling variance anomaly detector."""
    residuals = signal - baseline
    global_var = np.var(residuals[np.isfinite(residuals)])

    results = []
    n = len(signal)
    for start in range(0, n - window + 1, step):
        end = start + window
        seg_resid = residuals[start:end]
        valid = seg_resid[np.isfinite(seg_resid)]
        if len(valid) < window // 2:
            results.append({"window_start": start, "window_end": end, "alert": False})
            continue
        win_var = np.var(valid)
        var_ratio = win_var / (global_var + 1e-12)
        results.append({
            "window_start": start,
            "window_end": end,
            "alert": var_ratio > z_threshold,
        })
    return results


# ---------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------

def analyze_file(
    category: str,
    filename: str,
    anomaly_windows: list[list[str]],
) -> dict:
    """Analyze a single NAB file. Returns result dict."""
    df = load_nab_csv(category, filename)
    values = df["value"].values.astype(np.float64)
    timestamps = df["timestamp"]
    n = len(values)

    # Handle NaN
    series = pd.Series(values).ffill().bfill()
    signal = series.values

    # Build baseline
    baseline = build_rolling_mean_baseline(signal, window=BASELINE_WINDOW)

    # Ground truth anomaly indices
    gt_indices = timestamps_to_index_set(timestamps, anomaly_windows)

    # --- Coherence detector ---
    engine = CoherenceEngine(
        window_size=min(WINDOW_SIZE, n // 2),  # adapt for short series
        delta_threshold=DELTA_THRESHOLD,
        memory_threshold=MEMORY_THRESHOLD,
        recovery_threshold=RECOVERY_THRESHOLD,
    )
    effective_window = min(WINDOW_SIZE, n // 2)
    effective_step = max(effective_window // 7, 1)

    if n > effective_window:
        rolling = engine.score_rolling(signal, baseline, step=effective_step)
        coh_alert_indices = alert_windows_to_index_set(rolling)
        deltas = [r["delta"] for r in rolling]
        window_mids = [r["window_start"] + effective_window // 2 for r in rolling]
        n_coh_alerts = sum(1 for r in rolling if r["alert"])
    else:
        rolling = []
        coh_alert_indices = set()
        deltas = []
        window_mids = []
        n_coh_alerts = 0

    coh_metrics = compute_pr_f1(coh_alert_indices, gt_indices, n)

    # --- Variance detector ---
    if n > effective_window:
        var_results = variance_detector(
            signal, baseline,
            window=effective_window, step=effective_step,
            z_threshold=VAR_ZSCORE,
        )
        var_alert_indices = alert_windows_to_index_set(var_results)
        n_var_alerts = sum(1 for r in var_results if r["alert"])
    else:
        var_alert_indices = set()
        n_var_alerts = 0

    var_metrics = compute_pr_f1(var_alert_indices, gt_indices, n)

    return {
        "category": category,
        "filename": filename,
        "n_points": n,
        "n_anomaly_points": len(gt_indices),
        "n_anomaly_windows": len(anomaly_windows),
        "coherence": {
            **coh_metrics,
            "n_alerts": n_coh_alerts,
            "n_alert_points": len(coh_alert_indices),
        },
        "variance": {
            **var_metrics,
            "n_alerts": n_var_alerts,
            "n_alert_points": len(var_alert_indices),
        },
        "deltas": deltas,
        "window_mids": window_mids,
        "signal": signal.tolist(),
        "baseline": baseline.tolist(),
        "gt_indices": sorted(gt_indices),
        "coh_alert_indices": sorted(coh_alert_indices),
        "var_alert_indices": sorted(var_alert_indices),
        "timestamps": [str(t) for t in timestamps],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_category_f1_comparison(
    category_results: dict[str, dict],
    output_dir: Path,
):
    """Bar chart comparing coherence vs variance F1 by category."""
    categories = list(category_results.keys())
    coh_f1s = [category_results[c]["coherence_f1"] for c in categories]
    var_f1s = [category_results[c]["variance_f1"] for c in categories]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width / 2, coh_f1s, width, color="#00d4ff", alpha=0.85,
                   label="Delta Coherence", edgecolor="#00d4ff", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, var_f1s, width, color="#ff6b6b", alpha=0.85,
                   label="Variance Detector", edgecolor="#ff6b6b", linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars1, coh_f1s):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9, color="#00d4ff")
    for bar, val in zip(bars2, var_f1s):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9, color="#ff6b6b")

    short_names = [c.replace("artificial", "artif.").replace("real", "r.") for c in categories]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Experiment 8: NAB Benchmark — F1 by Category (Coherence vs Variance)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "nab_f1_by_category.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: nab_f1_by_category.png")


def plot_precision_recall_comparison(
    category_results: dict[str, dict],
    output_dir: Path,
):
    """Side-by-side precision and recall comparison by category."""
    categories = list(category_results.keys())
    coh_prec = [category_results[c]["coherence_precision"] for c in categories]
    coh_rec = [category_results[c]["coherence_recall"] for c in categories]
    var_prec = [category_results[c]["variance_precision"] for c in categories]
    var_rec = [category_results[c]["variance_recall"] for c in categories]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(categories))
    width = 0.35
    short_names = [c.replace("artificial", "artif.").replace("real", "r.") for c in categories]

    # Precision
    ax1.bar(x - width / 2, coh_prec, width, color="#00d4ff", alpha=0.85, label="Coherence")
    ax1.bar(x + width / 2, var_prec, width, color="#ff6b6b", alpha=0.85, label="Variance")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Precision")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Precision by Category")
    ax1.legend()
    ax1.grid(True, axis="y")

    # Recall
    ax2.bar(x - width / 2, coh_rec, width, color="#00d4ff", alpha=0.85, label="Coherence")
    ax2.bar(x + width / 2, var_rec, width, color="#ff6b6b", alpha=0.85, label="Variance")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Recall")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Recall by Category")
    ax2.legend()
    ax2.grid(True, axis="y")

    fig.suptitle("Experiment 8: Precision & Recall Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "nab_precision_recall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: nab_precision_recall.png")


def plot_example_timeseries(
    file_results: list[dict],
    output_dir: Path,
    n_examples: int = 4,
):
    """Plot example time series with alerts vs ground truth labels.

    Picks examples from different categories with the most interesting results.
    """
    # Select files that have anomaly labels and coherence alerts
    candidates = [
        r for r in file_results
        if r["n_anomaly_windows"] > 0 and r["coherence"]["n_alerts"] > 0
    ]
    # Sort by F1 descending — show the best coherence detections
    candidates.sort(key=lambda r: r["coherence"]["f1"], reverse=True)

    # Pick from different categories
    seen_cats = set()
    selected = []
    for r in candidates:
        if r["category"] not in seen_cats and len(selected) < n_examples:
            selected.append(r)
            seen_cats.add(r["category"])
    # Fill remaining if needed
    for r in candidates:
        if len(selected) >= n_examples:
            break
        if r not in selected:
            selected.append(r)

    if not selected:
        print("  No suitable examples found for time series plot.")
        return

    fig, axes = plt.subplots(len(selected), 1, figsize=(16, 4 * len(selected)))
    if len(selected) == 1:
        axes = [axes]

    for ax, r in zip(axes, selected):
        signal = np.array(r["signal"])
        baseline = np.array(r["baseline"])
        n = len(signal)

        # Plot signal and baseline
        ax.plot(signal, color="#00d4ff", alpha=0.5, linewidth=0.5, label="Signal")
        ax.plot(baseline, color="white", alpha=0.4, linewidth=0.8, label="Baseline")

        # Shade ground truth anomaly regions
        gt = r["gt_indices"]
        if gt:
            _shade_regions(ax, gt, n, color="#ff6b6b", alpha=0.15, label="NAB Label")

        # Shade coherence alert regions
        coh = r["coh_alert_indices"]
        if coh:
            _shade_regions(ax, coh, n, color="#00d4ff", alpha=0.12, label="Coherence Alert")

        short_name = r["filename"].replace(".csv", "")
        f1_coh = r["coherence"]["f1"]
        f1_var = r["variance"]["f1"]
        ax.set_title(
            f"{r['category']}/{short_name} — "
            f"Coherence F1={f1_coh:.2f}, Variance F1={f1_var:.2f}",
            fontsize=10,
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True)

    fig.suptitle("Experiment 8: Example Detections on NAB Data", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "nab_example_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: nab_example_timeseries.png")


def _shade_regions(ax, indices: list[int], n: int, color: str, alpha: float, label: str):
    """Shade contiguous index regions on an axis."""
    if not indices:
        return
    # Group contiguous indices
    regions = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx > prev + 1:
            regions.append((start, prev))
            start = idx
        prev = idx
    regions.append((start, prev))

    for i, (s, e) in enumerate(regions):
        ax.axvspan(s, e, color=color, alpha=alpha, label=label if i == 0 else None)


def plot_overall_summary(
    category_results: dict[str, dict],
    overall: dict,
    output_dir: Path,
):
    """Overall summary plot: metric cards + aggregated bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overall F1/precision/recall comparison
    metrics = ["Precision", "Recall", "F1"]
    coh_vals = [overall["coherence_precision"], overall["coherence_recall"], overall["coherence_f1"]]
    var_vals = [overall["variance_precision"], overall["variance_recall"], overall["variance_f1"]]

    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax1.bar(x - width / 2, coh_vals, width, color="#00d4ff", alpha=0.85, label="Coherence")
    bars2 = ax1.bar(x + width / 2, var_vals, width, color="#ff6b6b", alpha=0.85, label="Variance")

    for bar, val in zip(bars1, coh_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, color="#00d4ff")
    for bar, val in zip(bars2, var_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, color="#ff6b6b")

    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_ylim(0, 1.15)
    ax1.set_title("Overall NAB Benchmark Performance")
    ax1.legend()
    ax1.grid(True, axis="y")

    # Right: per-category F1 stacked as a horizontal bar
    categories = list(category_results.keys())
    coh_f1s = [category_results[c]["coherence_f1"] for c in categories]
    var_f1s = [category_results[c]["variance_f1"] for c in categories]

    y = np.arange(len(categories))
    short_names = [c.replace("artificial", "artif.").replace("real", "r.") for c in categories]

    ax2.barh(y + 0.15, coh_f1s, 0.3, color="#00d4ff", alpha=0.85, label="Coherence")
    ax2.barh(y - 0.15, var_f1s, 0.3, color="#ff6b6b", alpha=0.85, label="Variance")
    ax2.set_yticks(y)
    ax2.set_yticklabels(short_names, fontsize=9)
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("F1 Score")
    ax2.set_title("F1 by Category")
    ax2.legend(loc="lower right")
    ax2.grid(True, axis="x")

    fig.suptitle("Experiment 8: NAB Benchmark Summary", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "nab_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: nab_summary.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    print("=" * 60)
    print("  Experiment 8 — NAB Benchmark Validation")
    print("=" * 60)
    t_total = time.time()

    # Load labels
    all_labels = load_nab_labels()
    print(f"  Loaded {len(all_labels)} label entries")

    # Process each file
    all_file_results = []
    category_file_results: dict[str, list[dict]] = {c: [] for c in CATEGORIES}

    for category in CATEGORIES:
        cat_dir = DATA_DIR / category
        if not cat_dir.exists():
            print(f"  WARNING: Category directory not found: {category}")
            continue

        csv_files = sorted(cat_dir.glob("*.csv"))
        print(f"\n  [{category}] Processing {len(csv_files)} files...")

        for csv_file in csv_files:
            filename = csv_file.name
            label_key = f"{category}/{filename}"
            anomaly_windows = all_labels.get(label_key, [])

            try:
                result = analyze_file(category, filename, anomaly_windows)
                all_file_results.append(result)
                category_file_results[category].append(result)
                status = (
                    f"F1(coh)={result['coherence']['f1']:.2f} "
                    f"F1(var)={result['variance']['f1']:.2f}"
                )
                print(f"    {filename}: {status}")
            except Exception as e:
                print(f"    {filename}: ERROR — {e}")

    # --- Aggregate by category ---
    category_summary = {}
    for cat, results in category_file_results.items():
        if not results:
            continue

        # Macro-averaged metrics across files in category
        coh_f1s = [r["coherence"]["f1"] for r in results]
        coh_precs = [r["coherence"]["precision"] for r in results]
        coh_recs = [r["coherence"]["recall"] for r in results]
        var_f1s = [r["variance"]["f1"] for r in results]
        var_precs = [r["variance"]["precision"] for r in results]
        var_recs = [r["variance"]["recall"] for r in results]

        category_summary[cat] = {
            "n_files": len(results),
            "coherence_f1": float(np.mean(coh_f1s)),
            "coherence_precision": float(np.mean(coh_precs)),
            "coherence_recall": float(np.mean(coh_recs)),
            "variance_f1": float(np.mean(var_f1s)),
            "variance_precision": float(np.mean(var_precs)),
            "variance_recall": float(np.mean(var_recs)),
            "per_file": [
                {
                    "filename": r["filename"],
                    "n_points": r["n_points"],
                    "n_anomaly_windows": r["n_anomaly_windows"],
                    "coherence_f1": r["coherence"]["f1"],
                    "coherence_precision": r["coherence"]["precision"],
                    "coherence_recall": r["coherence"]["recall"],
                    "variance_f1": r["variance"]["f1"],
                    "variance_precision": r["variance"]["precision"],
                    "variance_recall": r["variance"]["recall"],
                }
                for r in results
            ],
        }

    # --- Overall metrics ---
    all_coh_f1 = [r["coherence"]["f1"] for r in all_file_results]
    all_coh_prec = [r["coherence"]["precision"] for r in all_file_results]
    all_coh_rec = [r["coherence"]["recall"] for r in all_file_results]
    all_var_f1 = [r["variance"]["f1"] for r in all_file_results]
    all_var_prec = [r["variance"]["precision"] for r in all_file_results]
    all_var_rec = [r["variance"]["recall"] for r in all_file_results]

    overall = {
        "n_files": len(all_file_results),
        "coherence_f1": float(np.mean(all_coh_f1)),
        "coherence_precision": float(np.mean(all_coh_prec)),
        "coherence_recall": float(np.mean(all_coh_rec)),
        "variance_f1": float(np.mean(all_var_f1)),
        "variance_precision": float(np.mean(all_var_prec)),
        "variance_recall": float(np.mean(all_var_rec)),
    }

    # --- Plots ---
    print("\n  Generating plots...")
    plot_category_f1_comparison(category_summary, OUTPUT_DIR)
    plot_precision_recall_comparison(category_summary, OUTPUT_DIR)
    plot_example_timeseries(all_file_results, OUTPUT_DIR)
    plot_overall_summary(category_summary, overall, OUTPUT_DIR)

    # --- Print summary ---
    total_elapsed = time.time() - t_total

    print("\n" + "=" * 60)
    print("  Experiment 8: NAB Benchmark Results")
    print("=" * 60)

    print(f"\n  Overall ({overall['n_files']} files):")
    print(f"    Coherence —  P={overall['coherence_precision']:.3f}  "
          f"R={overall['coherence_recall']:.3f}  F1={overall['coherence_f1']:.3f}")
    print(f"    Variance  —  P={overall['variance_precision']:.3f}  "
          f"R={overall['variance_recall']:.3f}  F1={overall['variance_f1']:.3f}")

    for cat, summary in category_summary.items():
        print(f"\n  {cat} ({summary['n_files']} files):")
        print(f"    Coherence F1={summary['coherence_f1']:.3f}  "
              f"(P={summary['coherence_precision']:.3f}, R={summary['coherence_recall']:.3f})")
        print(f"    Variance  F1={summary['variance_f1']:.3f}  "
              f"(P={summary['variance_precision']:.3f}, R={summary['variance_recall']:.3f})")

    print(f"\n  Total elapsed: {total_elapsed:.1f}s")
    print("=" * 60)

    # --- Save JSON ---
    # Strip large arrays from results before saving
    save_results = {
        "experiment": 8,
        "name": "NAB Benchmark Validation",
        "overall": overall,
        "categories": category_summary,
        "config": {
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "baseline_window": BASELINE_WINDOW,
            "delta_threshold": DELTA_THRESHOLD,
            "memory_threshold": MEMORY_THRESHOLD,
            "recovery_threshold": RECOVERY_THRESHOLD,
            "var_window": VAR_WINDOW,
            "var_zscore": VAR_ZSCORE,
        },
        "elapsed_s": total_elapsed,
    }

    results_path = OUTPUT_DIR / "nab_results.json"
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    # Update main experiment_results.json
    main_results_path = OUTPUT_DIR.parent / "experiment_results.json"
    if main_results_path.exists():
        with open(main_results_path) as f:
            main_results = json.load(f)
    else:
        main_results = {}
    main_results["exp8"] = save_results
    with open(main_results_path, "w") as f:
        json.dump(main_results, f, indent=2, default=str)
    print(f"  Updated {main_results_path}")

    print(f"  Plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
