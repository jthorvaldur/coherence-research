#!/usr/bin/env python3
"""Experiment 11 — PhysioNet MIT-BIH Arrhythmia Detection.

Applies Delta.72 to ECG heart rhythm data. The MIT-BIH Arrhythmia Database
contains 48 half-hour recordings from 47 subjects, with cardiologist annotations
marking each heartbeat (Normal, PVC, LBBB, RBBB, etc.).

Delta.72 tests whether coherence breakdown precedes arrhythmic episodes.
Normal rhythm = high coherence with baseline sinus pattern.
Arrhythmia = structural departure from normal rhythm.

Outputs:
  - results/ecg/exp11_example_record.png   — single ECG + Delta overlay
  - results/ecg/exp11_detection_summary.png — detection rates across records
  - results/ecg/exp11_arrhythmia_types.png  — coherence by arrhythmia type
  - results/ecg/exp11_lead_time.png         — lead-time before arrhythmia onset
  - results/ecg/exp11_results.json          — full results

Usage:
    uv run python experiments/exp_ecg.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from delta72.engine import coherence_score

# Try importing wfdb
try:
    import wfdb
except ImportError:
    print("Install wfdb: uv add wfdb")
    sys.exit(1)


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "ecg"

# MIT-BIH records (subset — most clinically interesting)
RECORDS = ["100", "101", "103", "105", "106", "108", "109",
           "111", "112", "113", "114", "115", "116", "117",
           "118", "119", "121", "122", "123", "124",
           "200", "201", "202", "203", "205", "207", "208",
           "209", "210", "212", "213", "214", "215", "217",
           "219", "220", "221", "222", "223", "228", "230",
           "231", "232", "233", "234"]

# Normal beat symbols
NORMAL_BEATS = {"N", "L", "R", "e", "j"}
# Abnormal beat symbols (PVC, LBBB, RBBB, etc.)
ABNORMAL_BEATS = {"V", "A", "F", "!", "a", "J", "S", "E", "/", "f", "Q"}

WINDOW_SIZE = 1080  # 3 seconds at 360 Hz
STEP_SIZE = 360     # 1 second step
DELTA_THRESHOLD = 0.3
BASELINE_SECONDS = 30  # first 30s for baseline


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


def analyze_record(record_id: str) -> dict | None:
    """Analyze a single MIT-BIH record using rolling-baseline coherence.

    Uses both channels independently: each channel's signal is compared to its
    own rolling-mean baseline (like NAB experiment), then per-channel deltas are
    averaged to a system-level score. This avoids the inter-lead morphology
    mismatch that makes raw cross-channel correlation uninformative.
    """
    try:
        record = wfdb.rdrecord(record_id, pn_dir="mitdb")
        ann = wfdb.rdann(record_id, "atr", pn_dir="mitdb")
    except Exception as e:
        print(f"    Skip {record_id}: {e}")
        return None

    n_channels = min(record.p_signal.shape[1], 2)
    fs = record.fs
    n = len(record.p_signal[:, 0])

    # Build rolling-mean baseline per channel (like NAB approach)
    rolling_window = int(BASELINE_SECONDS * fs)  # 30s rolling mean
    channels = []
    baselines = []
    for c in range(n_channels):
        ch = record.p_signal[:, c].astype(np.float64)
        normed = normalize(ch)
        # Centered rolling mean as baseline
        kernel = np.ones(rolling_window) / rolling_window
        baseline = np.convolve(normed, kernel, mode='same')
        channels.append(normed)
        baselines.append(baseline)

    # Map annotations to per-sample labels
    beat_labels = np.zeros(n, dtype=int)  # 0 = normal, 1 = abnormal
    for sample, symbol in zip(ann.sample, ann.symbol):
        if sample < n and symbol in ABNORMAL_BEATS:
            start = max(0, sample - fs // 2)
            end = min(n, sample + fs // 2)
            beat_labels[start:end] = 1

    n_abnormal = int(beat_labels.sum())
    abnormal_frac = n_abnormal / n

    # Count arrhythmia types
    arrhythmia_counts = {}
    for symbol in ann.symbol:
        if symbol in ABNORMAL_BEATS:
            arrhythmia_counts[symbol] = arrhythmia_counts.get(symbol, 0) + 1

    # Rolling coherence: per-channel delta, then average
    deltas = []
    for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        ch_deltas = []
        ch_ps = []
        for c in range(n_channels):
            scores = coherence_score(channels[c][start:end], baselines[c][start:end])
            ch_deltas.append(scores["delta"])
            ch_ps.append(scores["P"])
        mid = start + WINDOW_SIZE // 2
        is_abnormal = beat_labels[start:end].any()

        deltas.append({
            "mid": mid,
            "delta": float(np.mean(ch_deltas)),
            "P": float(np.mean(ch_ps)),
            "is_abnormal": bool(is_abnormal),
            "time_s": mid / fs,
        })

    # Normalize against baseline period mean delta
    d_arr = np.array([d["delta"] for d in deltas])
    baseline_windows = int(BASELINE_SECONDS / (STEP_SIZE / fs))
    baseline_mean = d_arr[:baseline_windows].mean() if baseline_windows > 0 and d_arr[:baseline_windows].mean() > 0 else d_arr[d_arr > 0].mean() if (d_arr > 0).any() else 1.0
    d_norm = d_arr / baseline_mean
    for i, dn in enumerate(d_norm):
        deltas[i]["delta_norm"] = float(np.clip(dn, 0, 2))

    # Detection metrics
    n_windows = len(deltas)
    n_alert_windows = sum(1 for d in deltas if d["delta_norm"] < DELTA_THRESHOLD)
    n_abnormal_windows = sum(1 for d in deltas if d["is_abnormal"])
    n_true_pos = sum(1 for d in deltas if d["delta_norm"] < DELTA_THRESHOLD and d["is_abnormal"])
    n_false_pos = sum(1 for d in deltas if d["delta_norm"] < DELTA_THRESHOLD and not d["is_abnormal"])
    n_false_neg = sum(1 for d in deltas if d["delta_norm"] >= DELTA_THRESHOLD and d["is_abnormal"])

    precision = n_true_pos / (n_true_pos + n_false_pos) if (n_true_pos + n_false_pos) > 0 else 0
    recall = n_true_pos / (n_true_pos + n_false_neg) if (n_true_pos + n_false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Lead time: for first arrhythmia episode, how many seconds before does Delta drop?
    first_abnormal_idx = next((i for i, d in enumerate(deltas) if d["is_abnormal"]), None)
    first_alert_idx = next((i for i, d in enumerate(deltas) if d["delta_norm"] < DELTA_THRESHOLD), None)
    lead_time_s = 0
    if first_abnormal_idx is not None and first_alert_idx is not None:
        if first_alert_idx < first_abnormal_idx:
            lead_time_s = (deltas[first_abnormal_idx]["time_s"] - deltas[first_alert_idx]["time_s"])

    return {
        "record_id": record_id,
        "n_samples": n,
        "fs": fs,
        "duration_s": n / fs,
        "n_beats": len(ann.sample),
        "n_abnormal_beats": sum(1 for s in ann.symbol if s in ABNORMAL_BEATS),
        "abnormal_frac": abnormal_frac,
        "arrhythmia_types": arrhythmia_counts,
        "n_windows": n_windows,
        "n_alert_windows": n_alert_windows,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "lead_time_s": lead_time_s,
        "mean_delta_normal": float(np.mean([d["delta_norm"] for d in deltas if not d["is_abnormal"]])) if any(not d["is_abnormal"] for d in deltas) else 0,
        "mean_delta_abnormal": float(np.mean([d["delta_norm"] for d in deltas if d["is_abnormal"]])) if any(d["is_abnormal"] for d in deltas) else 0,
        "deltas_all": deltas,  # full for plotting
        "deltas": deltas[:200],  # truncate for JSON size
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    print("=" * 60)
    print("  Experiment 11 — PhysioNet MIT-BIH Arrhythmia")
    print("=" * 60)
    t0 = time.time()

    all_results = []
    example_result = None

    for i, rec_id in enumerate(RECORDS):
        result = analyze_record(rec_id)
        if result is None:
            continue
        all_results.append(result)

        # Pick example: record with good mix of normal/abnormal and real discrimination
        if result["n_abnormal_beats"] > 20:
            frac = result["abnormal_frac"]
            # Prefer records with 5-50% abnormal and decent F1 (real separation)
            if 0.05 < frac < 0.50 and result["f1"] > 0.15:
                score = result["f1"] * (1 - abs(frac - 0.25))  # favor ~25% abnormal
                best_score = example_result.get("_sel_score", 0) if example_result else 0
                if score > best_score:
                    result["_sel_score"] = score
                    example_result = result

        if (i + 1) % 10 == 0 or i == len(RECORDS) - 1:
            print(f"    [{i+1}/{len(RECORDS)}] Record {rec_id}: "
                  f"beats={result['n_beats']}, abnormal={result['n_abnormal_beats']}, "
                  f"F1={result['f1']:.3f}")

    print(f"\n  Analyzed {len(all_results)} records")

    # Aggregate stats
    records_with_arrhythmia = [r for r in all_results if r["n_abnormal_beats"] > 0]
    records_normal_only = [r for r in all_results if r["n_abnormal_beats"] == 0]

    mean_f1 = np.mean([r["f1"] for r in records_with_arrhythmia]) if records_with_arrhythmia else 0
    mean_precision = np.mean([r["precision"] for r in records_with_arrhythmia]) if records_with_arrhythmia else 0
    mean_recall = np.mean([r["recall"] for r in records_with_arrhythmia]) if records_with_arrhythmia else 0
    mean_lead = np.mean([r["lead_time_s"] for r in records_with_arrhythmia if r["lead_time_s"] > 0])

    # Mean delta for normal vs abnormal windows
    all_normal_deltas = [r["mean_delta_normal"] for r in all_results if r["mean_delta_normal"] > 0]
    all_abnormal_deltas = [r["mean_delta_abnormal"] for r in records_with_arrhythmia if r["mean_delta_abnormal"] > 0]

    print(f"\n  Records with arrhythmia: {len(records_with_arrhythmia)}")
    print(f"  Mean F1: {mean_f1:.3f}")
    print(f"  Mean precision: {mean_precision:.3f}")
    print(f"  Mean recall: {mean_recall:.3f}")
    print(f"  Mean delta (normal windows): {np.mean(all_normal_deltas):.3f}" if all_normal_deltas else "")
    print(f"  Mean delta (abnormal windows): {np.mean(all_abnormal_deltas):.3f}" if all_abnormal_deltas else "")

    # --- Plots ---
    print("\n  Generating plots...")

    # Plot 1: Example record — full recording with raw ECG + delta overlay
    if example_result:
        # Re-load raw ECG for the example record
        try:
            rec = wfdb.rdrecord(example_result["record_id"], pn_dir="mitdb")
            raw_ch1 = rec.p_signal[:, 0]
            raw_fs = rec.fs
        except Exception:
            raw_ch1 = None
            raw_fs = example_result["fs"]

        all_deltas = example_result.get("deltas_all", example_result["deltas"])
        times = [d["time_s"] for d in all_deltas]
        d_norms = [d["delta_norm"] for d in all_deltas]
        abnormal = [d["is_abnormal"] for d in all_deltas]

        n_panels = 3 if raw_ch1 is not None else 2
        fig, axes = plt.subplots(n_panels, 1, figsize=(16, 3.5 * n_panels))
        ax_idx = 0

        # Panel: raw ECG trace
        if raw_ch1 is not None:
            ax = axes[ax_idx]; ax_idx += 1
            ecg_times = np.arange(len(raw_ch1)) / raw_fs
            ax.plot(ecg_times, raw_ch1, color="#73daca", linewidth=0.3, alpha=0.8)
            # Shade abnormal regions
            beat_labels = np.zeros(len(raw_ch1), dtype=bool)
            ann = wfdb.rdann(example_result["record_id"], "atr", pn_dir="mitdb")
            for sample, symbol in zip(ann.sample, ann.symbol):
                if sample < len(raw_ch1) and symbol in ABNORMAL_BEATS:
                    s = max(0, sample - raw_fs // 2)
                    e = min(len(raw_ch1), sample + raw_fs // 2)
                    beat_labels[s:e] = True
            ax.fill_between(ecg_times, raw_ch1.min(), raw_ch1.max(),
                           where=beat_labels, alpha=0.15, color="#ff6b6b")
            ax.set_ylabel("ECG (mV)")
            ax.set_title(f"Record {example_result['record_id']} — Raw ECG (MLII) with annotated arrhythmia regions")
            ax.grid(True)

        # Panel: delta overlay
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(times, d_norms, color="#00d4ff", linewidth=0.8)
        ax.axhline(DELTA_THRESHOLD, color="#ff6b6b", linestyle="--", alpha=0.6, label=f"Threshold ({DELTA_THRESHOLD})")
        ax.fill_between(times, 0, max(d_norms) if d_norms else 1,
                        where=abnormal, alpha=0.15, color="#ff6b6b", label="Abnormal region")
        ax.set_ylabel("Delta (normalized)")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{example_result['n_abnormal_beats']} abnormal beats, F1={example_result['f1']:.3f}, P={example_result['precision']:.3f}, R={example_result['recall']:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True)

        # Panel: delta distribution
        ax = axes[ax_idx]
        normal_d = [d["delta_norm"] for d in all_deltas if not d["is_abnormal"]]
        abnormal_d = [d["delta_norm"] for d in all_deltas if d["is_abnormal"]]
        if normal_d:
            ax.hist(normal_d, bins=30, alpha=0.6, color="#50fa7b", label=f"Normal (n={len(normal_d)})", density=True)
        if abnormal_d:
            ax.hist(abnormal_d, bins=30, alpha=0.6, color="#ff6b6b", label=f"Abnormal (n={len(abnormal_d)})", density=True)
        ax.set_xlabel("Delta (normalized)")
        ax.set_ylabel("Density")
        ax.set_title("Delta Distribution: Normal vs Abnormal Windows")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "exp11_example_record.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: exp11_example_record.png (record {example_result['record_id']})")

    # Plot 2: Detection summary across records
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    rec_ids = [r["record_id"] for r in records_with_arrhythmia]
    f1s = [r["f1"] for r in records_with_arrhythmia]
    ax1.barh(range(len(rec_ids)), f1s, color="#00d4ff", alpha=0.7)
    ax1.set_yticks(range(len(rec_ids)))
    ax1.set_yticklabels(rec_ids, fontsize=7)
    ax1.set_xlabel("F1 Score")
    ax1.set_title("F1 by Record (arrhythmia records only)")
    ax1.grid(True, axis="x")

    # Normal vs abnormal delta comparison
    if all_normal_deltas and all_abnormal_deltas:
        ax2.boxplot([all_normal_deltas, all_abnormal_deltas], tick_labels=["Normal", "Abnormal"],
                    patch_artist=True,
                    boxprops=dict(facecolor="#00d4ff", alpha=0.5),
                    medianprops=dict(color="white", linewidth=2))
    ax2.set_ylabel("Mean Delta (normalized)")
    ax2.set_title("Delta: Normal vs Abnormal Windows")
    ax2.grid(True, axis="y")

    fig.suptitle("Experiment 11: MIT-BIH Arrhythmia Detection", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp11_detection_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: exp11_detection_summary.png")

    # Plot 3: Arrhythmia types
    all_types = {}
    for r in all_results:
        for t, c in r.get("arrhythmia_types", {}).items():
            all_types[t] = all_types.get(t, 0) + c

    if all_types:
        fig, ax = plt.subplots(figsize=(10, 5))
        types_sorted = sorted(all_types.items(), key=lambda x: x[1], reverse=True)
        names = [t[0] for t in types_sorted[:10]]
        counts = [t[1] for t in types_sorted[:10]]
        ax.bar(names, counts, color="#bd93f9", alpha=0.7)
        ax.set_ylabel("Count")
        ax.set_title("Arrhythmia Types Across All Records")
        ax.grid(True, axis="y")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "exp11_arrhythmia_types.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: exp11_arrhythmia_types.png")

    # Plot 4: Lead time
    lead_times = [r["lead_time_s"] for r in records_with_arrhythmia if r["lead_time_s"] > 0]
    if lead_times:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(lead_times, bins=20, color="#50fa7b", alpha=0.7, edgecolor="#16213e")
        ax.axvline(np.mean(lead_times), color="white", linestyle="--", label=f"Mean = {np.mean(lead_times):.1f}s")
        ax.set_xlabel("Lead Time (seconds before first arrhythmia)")
        ax.set_ylabel("Count")
        ax.set_title("Delta Lead Time Before Arrhythmia Onset")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "exp11_lead_time.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: exp11_lead_time.png")

    elapsed = time.time() - t0

    # Save results
    stats = {
        "n_records": len(all_results),
        "n_with_arrhythmia": len(records_with_arrhythmia),
        "n_normal_only": len(records_normal_only),
        "mean_f1": round(mean_f1, 3),
        "mean_precision": round(mean_precision, 3),
        "mean_recall": round(mean_recall, 3),
        "mean_lead_s": round(float(mean_lead), 1) if lead_times else 0,
        "mean_delta_normal": round(float(np.mean(all_normal_deltas)), 3) if all_normal_deltas else 0,
        "mean_delta_abnormal": round(float(np.mean(all_abnormal_deltas)), 3) if all_abnormal_deltas else 0,
        "arrhythmia_types": all_types,
        "per_record": [
            {
                "record_id": r["record_id"],
                "n_beats": r["n_beats"],
                "n_abnormal": r["n_abnormal_beats"],
                "f1": round(r["f1"], 3),
                "precision": round(r["precision"], 3),
                "recall": round(r["recall"], 3),
                "lead_time_s": round(r["lead_time_s"], 1),
            }
            for r in all_results
        ],
    }

    exp_result = {
        "experiment": 11,
        "name": "PhysioNet MIT-BIH Arrhythmia Detection",
        "dataset": f"MIT-BIH Arrhythmia Database — {len(all_results)} records, 360 Hz, 2 channels",
        "config": {
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "delta_threshold": DELTA_THRESHOLD,
            "baseline_seconds": BASELINE_SECONDS,
        },
        "stats": stats,
        "elapsed_s": round(elapsed, 1),
    }

    with open(OUTPUT_DIR / "exp11_results.json", "w") as f:
        json.dump(exp_result, f, indent=2)

    print(f"\n  Saved to {OUTPUT_DIR / 'exp11_results.json'}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
