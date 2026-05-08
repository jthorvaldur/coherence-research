#!/usr/bin/env python3
"""Delta.72 Coherence Framework — Full Experiment Suite (Experiments 1–6).

GPU-parallelizable via PyTorch/NumPy vectorization.
Outputs results to ../results/ as JSON + PNG plots.

Usage:
    python experiments/run_all.py [--n-monte-carlo 500] [--output-dir results]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from delta72.engine import coherence_score, alert_check, memory_of_attractor, windowed_recovery
from delta72.signals import (
    structured_with_noise,
    shock_recovery,
    hidden_drift,
    multi_coherence_shock,
    cross_system_signals,
    monte_carlo_trial,
)


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
# Experiment 1 — Coherence vs Noise Threshold
# ---------------------------------------------------------------------------

def experiment_1(output_dir: Path) -> dict:
    """Determine if coherence collapses at a predictable threshold as noise increases."""
    print("  [Exp 1] Coherence vs Noise Threshold...")
    t0 = time.time()

    noise_levels = np.linspace(0.01, 3.0, 60)
    deltas = []
    components = {"P": [], "A": [], "R": [], "D": [], "N_val": []}

    for nl in noise_levels:
        signal, baseline = structured_with_noise(n=2000, noise_level=nl, seed=42)
        scores = coherence_score(signal, baseline)
        deltas.append(scores["delta"])
        components["P"].append(scores["P"])
        components["A"].append(scores["A"])
        components["R"].append(scores["R"])
        components["D"].append(scores["D"])
        components["N_val"].append(scores["N"])

    deltas = np.array(deltas)

    # Find threshold — point where Δ drops below 50% of initial value
    initial_delta = deltas[0]
    half_delta = initial_delta * 0.5
    threshold_idx = int(np.argmax(deltas < half_delta)) if np.any(deltas < half_delta) else len(deltas) - 1
    threshold_noise = float(noise_levels[threshold_idx])

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(noise_levels, deltas, color="#00d4ff", linewidth=2, label="Δ coherence")
    ax1.axvline(threshold_noise, color="#ff6b6b", linestyle="--", alpha=0.8,
                label=f"Threshold ≈ {threshold_noise:.2f}")
    ax1.set_xlabel("Noise Level (σ)")
    ax1.set_ylabel("Δ Score")
    ax1.set_title("Experiment 1: Coherence vs Noise")
    ax1.legend()
    ax1.grid(True)

    for key, vals in components.items():
        ax2.plot(noise_levels, vals, linewidth=1.5, label=key, alpha=0.8)
    ax2.set_xlabel("Noise Level (σ)")
    ax2.set_ylabel("Component Value")
    ax2.set_title("Component Decomposition")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / "exp1_coherence_vs_noise.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    result = {
        "experiment": 1,
        "name": "Coherence vs Noise Threshold",
        "threshold_noise": threshold_noise,
        "threshold_delta": float(deltas[threshold_idx]),
        "noise_levels": noise_levels.tolist(),
        "deltas": deltas.tolist(),
        "elapsed_s": time.time() - t0,
    }
    return result


# ---------------------------------------------------------------------------
# Experiment 2 — Recovery Dynamics After Shock
# ---------------------------------------------------------------------------

def experiment_2(output_dir: Path) -> dict:
    """Test whether coherence captures recovery differences after perturbation."""
    print("  [Exp 2] Recovery Dynamics After Shock...")
    t0 = time.time()

    recovery_rates = [0.005, 0.02, 0.05, 0.1, 0.2]
    labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    results_data = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(recovery_rates)))

    for i, (rate, label) in enumerate(zip(recovery_rates, labels)):
        signal, baseline = shock_recovery(
            n=2000, shock_magnitude=4.0, recovery_rate=rate, seed=42
        )
        scores = coherence_score(signal, baseline)
        M = memory_of_attractor(signal, baseline)
        W = windowed_recovery(signal, baseline)

        results_data.append({
            "recovery_rate": rate,
            "label": label,
            "delta": scores["delta"],
            "M": M,
            "W": W,
            "P": scores["P"],
            "R": scores["R"],
        })

        ax1.plot(signal[:800], color=colors[i], alpha=0.7, label=f"{label} (rate={rate})")

    ax1.plot(baseline[:800], color="white", linestyle="--", alpha=0.4, label="Baseline")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Signal")
    ax1.set_title("Experiment 2: Shock Response by Recovery Rate")
    ax1.legend(fontsize=8)
    ax1.grid(True)

    rates = [r["recovery_rate"] for r in results_data]
    ax2.plot(rates, [r["delta"] for r in results_data], "o-", color="#00d4ff", label="Δ", linewidth=2)
    ax2.plot(rates, [r["M"] for r in results_data], "s-", color="#ff6b6b", label="M (attractor)", linewidth=2)
    ax2.plot(rates, [r["W"] for r in results_data], "^-", color="#50fa7b", label="W (recovery)", linewidth=2)
    ax2.set_xlabel("Recovery Rate")
    ax2.set_ylabel("Score")
    ax2.set_title("Coherence Metrics vs Recovery Rate")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / "exp2_recovery_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "experiment": 2,
        "name": "Recovery Dynamics After Shock",
        "results": results_data,
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Experiment 3 — Hidden Drift Before Visible Failure
# ---------------------------------------------------------------------------

def experiment_3(output_dir: Path) -> dict:
    """Detect drift before obvious system breakdown. Compare Δ vs variance vs z-score."""
    print("  [Exp 3] Hidden Drift Before Visible Failure...")
    t0 = time.time()

    signal, baseline, failure_idx = hidden_drift(
        n=3000, drift_rate=0.003, noise_level=0.15, failure_threshold=2.0, seed=42
    )

    # Rolling metrics
    window = 200
    step = 20
    delta_series = []
    variance_series = []
    zscore_series = []
    time_points = []

    for start in range(0, len(signal) - window, step):
        end = start + window
        seg_sig = signal[start:end]
        seg_base = baseline[start:end]

        scores = coherence_score(seg_sig, seg_base)
        delta_series.append(scores["delta"])

        residual = seg_sig - seg_base
        variance_series.append(float(np.var(residual)))
        zscore_series.append(float(np.abs(residual).mean() / (residual.std() + 1e-8)))
        time_points.append(start + window // 2)

    delta_series = np.array(delta_series)
    variance_series = np.array(variance_series)
    zscore_series = np.array(zscore_series)
    time_points = np.array(time_points)

    # Detection times (first crossing of alert threshold)
    delta_alert_idx = np.where(delta_series < 0.5)[0]
    var_alert_idx = np.where(variance_series > variance_series[:10].mean() * 3)[0]
    z_alert_idx = np.where(zscore_series > 2.0)[0]

    delta_detect = int(time_points[delta_alert_idx[0]]) if len(delta_alert_idx) > 0 else len(signal)
    var_detect = int(time_points[var_alert_idx[0]]) if len(var_alert_idx) > 0 else len(signal)
    z_detect = int(time_points[z_alert_idx[0]]) if len(z_alert_idx) > 0 else len(signal)

    lead_time_delta = failure_idx - delta_detect
    lead_time_var = failure_idx - var_detect
    lead_time_z = failure_idx - z_detect

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax1.plot(signal, color="#00d4ff", alpha=0.6, linewidth=0.5)
    ax1.plot(baseline, color="white", linestyle="--", alpha=0.4)
    ax1.axvline(failure_idx, color="#ff6b6b", linestyle="--", label=f"Failure @ {failure_idx}")
    ax1.set_ylabel("Signal")
    ax1.set_title("Experiment 3: Hidden Drift Detection")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time_points, delta_series, color="#00d4ff", linewidth=1.5, label="Δ coherence")
    ax2.axhline(0.5, color="#00d4ff", linestyle=":", alpha=0.5)
    if delta_detect < len(signal):
        ax2.axvline(delta_detect, color="#00d4ff", linestyle="--", alpha=0.8,
                    label=f"Δ detects @ {delta_detect}")
    ax2.axvline(failure_idx, color="#ff6b6b", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Δ Score")
    ax2.legend()
    ax2.grid(True)

    ax3.plot(time_points, variance_series / variance_series.max(), color="#ff6b6b",
             linewidth=1.5, label="Variance (norm)")
    ax3.plot(time_points, zscore_series / zscore_series.max(), color="#50fa7b",
             linewidth=1.5, label="Z-score (norm)")
    if var_detect < len(signal):
        ax3.axvline(var_detect, color="#ff6b6b", linestyle="--", alpha=0.8,
                    label=f"Var detects @ {var_detect}")
    if z_detect < len(signal):
        ax3.axvline(z_detect, color="#50fa7b", linestyle="--", alpha=0.8,
                    label=f"Z detects @ {z_detect}")
    ax3.axvline(failure_idx, color="white", linestyle="--", alpha=0.3)
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Normalized Score")
    ax3.legend()
    ax3.grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / "exp3_hidden_drift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "experiment": 3,
        "name": "Hidden Drift Before Visible Failure",
        "failure_index": failure_idx,
        "detection_times": {
            "delta": delta_detect,
            "variance": var_detect,
            "zscore": z_detect,
        },
        "lead_times": {
            "delta": lead_time_delta,
            "variance": lead_time_var,
            "zscore": lead_time_z,
        },
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Experiment 4 — Shock Response vs Coherence
# ---------------------------------------------------------------------------

def experiment_4(output_dir: Path) -> dict:
    """Quantify deviation magnitude vs coherence level."""
    print("  [Exp 4] Shock Response vs Coherence Level...")
    t0 = time.time()

    systems = multi_coherence_shock(n=2000, shock_magnitude=4.0, seed=42)
    results_data = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(systems)))

    for i, (signal, baseline, coh_param) in enumerate(systems):
        residual = np.abs(signal - baseline)
        peak_dev = float(residual.max())
        scores = coherence_score(signal, baseline)

        # Time to return to 1-std after shock
        shock_idx = 1000
        post_residual = residual[shock_idx:]
        std_baseline = residual[:shock_idx].std()
        ttr = len(post_residual)
        for j, r in enumerate(post_residual):
            if r < std_baseline * 1.5:
                ttr = j
                break

        results_data.append({
            "coherence_param": coh_param,
            "peak_deviation": peak_dev,
            "time_to_return": ttr,
            "delta": scores["delta"],
            "post_shock_stability": float(post_residual[-200:].std()),
        })

        ax1.plot(signal[900:1500], color=colors[i], alpha=0.8,
                label=f"Coh={coh_param:.2f}")

    ax1.axvline(100, color="white", linestyle="--", alpha=0.3, label="Shock")
    ax1.set_xlabel("Time Step (offset)")
    ax1.set_ylabel("Signal")
    ax1.set_title("Experiment 4: Shock Response by Coherence")
    ax1.legend(fontsize=9)
    ax1.grid(True)

    coh_params = [r["coherence_param"] for r in results_data]
    ax2.bar(range(len(results_data)),
            [r["peak_deviation"] for r in results_data],
            color=[colors[i] for i in range(len(results_data))],
            alpha=0.8)
    ax2.set_xticks(range(len(results_data)))
    ax2.set_xticklabels([f"{c:.2f}" for c in coh_params])
    ax2.set_xlabel("Coherence Parameter")
    ax2.set_ylabel("Peak Deviation")
    ax2.set_title("Peak Deviation vs Coherence Level")
    ax2.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "exp4_shock_response.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "experiment": 4,
        "name": "Shock Response vs Coherence",
        "results": results_data,
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Experiment 5 — Cross-System Generalization
# ---------------------------------------------------------------------------

def experiment_5(output_dir: Path) -> dict:
    """Test if Δ generalizes across different signal types."""
    print("  [Exp 5] Cross-System Generalization...")
    t0 = time.time()

    noise_sweep = np.linspace(0.01, 2.0, 30)
    system_results = {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = {"sinusoidal": "#00d4ff", "chaotic": "#ff6b6b", "piecewise": "#50fa7b", "stochastic": "#bd93f9"}

    for idx, (name, color) in enumerate(colors.items()):
        deltas_at_noise = []
        for nl in noise_sweep:
            signals = cross_system_signals(n=2000, noise_level=nl, seed=42)
            sig, base = signals[name]
            scores = coherence_score(sig, base)
            deltas_at_noise.append(scores["delta"])

        system_results[name] = {
            "deltas": deltas_at_noise,
            "mean_delta": float(np.mean(deltas_at_noise)),
        }

        axes[idx].plot(noise_sweep, deltas_at_noise, color=color, linewidth=2)
        axes[idx].set_xlabel("Noise Level")
        axes[idx].set_ylabel("Δ Score")
        axes[idx].set_title(f"{name.capitalize()}")
        axes[idx].grid(True)

    fig.suptitle("Experiment 5: Cross-System Δ Generalization", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "exp5_cross_system.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Consistency metric: coefficient of variation across systems at each noise level
    all_deltas = np.array([system_results[name]["deltas"] for name in colors])
    cv_per_noise = np.std(all_deltas, axis=0) / (np.mean(all_deltas, axis=0) + 1e-8)
    mean_cv = float(np.mean(cv_per_noise))

    return {
        "experiment": 5,
        "name": "Cross-System Generalization",
        "systems": {k: v["mean_delta"] for k, v in system_results.items()},
        "cross_system_cv": mean_cv,
        "consistent": mean_cv < 1.0,
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Experiment 6 — Monte Carlo Lead-Time Analysis
# ---------------------------------------------------------------------------

def experiment_6(output_dir: Path, n_trials: int = 500) -> dict:
    """Evaluate statistical robustness across many randomized trials."""
    print(f"  [Exp 6] Monte Carlo Lead-Time Analysis (N={n_trials})...")
    t0 = time.time()

    detection_results = []
    window = 200
    step = 20

    for trial in range(n_trials):
        signal, baseline, params = monte_carlo_trial(n=2000, seed=trial)
        failure_idx = params["failure_index"]

        # Rolling Δ detection
        delta_detect = len(signal)
        var_detect = len(signal)

        for start in range(0, len(signal) - window, step):
            end = start + window
            seg = signal[start:end]
            base = baseline[start:end]
            midpoint = start + window // 2

            if delta_detect == len(signal):
                scores = coherence_score(seg, base)
                if scores["delta"] < 0.5:
                    delta_detect = midpoint

            if var_detect == len(signal):
                residual = seg - base
                if np.var(residual) > 0.5:
                    var_detect = midpoint

        detection_results.append({
            "trial": trial,
            "failure_idx": failure_idx,
            "delta_detect": delta_detect,
            "var_detect": var_detect,
            "delta_lead": failure_idx - delta_detect,
            "var_lead": failure_idx - var_detect,
            "delta_detected": delta_detect < failure_idx,
            "var_detected": var_detect < failure_idx,
            "noise_level": params["noise_level"],
            "drift_rate": params["drift_rate"],
        })

    # Aggregate statistics
    delta_detected = [r for r in detection_results if r["delta_detected"]]
    var_detected = [r for r in detection_results if r["var_detected"]]

    delta_leads = [r["delta_lead"] for r in delta_detected]
    var_leads = [r["var_lead"] for r in var_detected]

    stats = {
        "delta_detection_rate": len(delta_detected) / n_trials,
        "var_detection_rate": len(var_detected) / n_trials,
        "delta_mean_lead": float(np.mean(delta_leads)) if delta_leads else 0,
        "var_mean_lead": float(np.mean(var_leads)) if var_leads else 0,
        "delta_median_lead": float(np.median(delta_leads)) if delta_leads else 0,
        "var_median_lead": float(np.median(var_leads)) if var_leads else 0,
    }

    # Plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Lead time distributions
    if delta_leads:
        ax1.hist(delta_leads, bins=40, color="#00d4ff", alpha=0.7, label="Δ Lead Time")
    if var_leads:
        ax1.hist(var_leads, bins=40, color="#ff6b6b", alpha=0.5, label="Variance Lead Time")
    ax1.set_xlabel("Lead Time (steps before failure)")
    ax1.set_ylabel("Count")
    ax1.set_title("Lead Time Distribution")
    ax1.legend()
    ax1.grid(True)

    # Detection rate comparison
    methods = ["Δ Coherence", "Variance"]
    rates = [stats["delta_detection_rate"], stats["var_detection_rate"]]
    bars = ax2.bar(methods, rates, color=["#00d4ff", "#ff6b6b"], alpha=0.8)
    ax2.set_ylabel("Detection Rate")
    ax2.set_title("Detection Rate Comparison")
    ax2.set_ylim(0, 1.1)
    for bar, rate in zip(bars, rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.1%}", ha="center", fontsize=12)
    ax2.grid(True, axis="y")

    # Lead time vs noise level
    noise_levels = [r["noise_level"] for r in detection_results]
    delta_leads_all = [r["delta_lead"] for r in detection_results]
    ax3.scatter(noise_levels, delta_leads_all, alpha=0.3, s=10, color="#00d4ff")
    ax3.set_xlabel("Noise Level")
    ax3.set_ylabel("Δ Lead Time")
    ax3.set_title("Lead Time vs Noise Level")
    ax3.grid(True)

    # Mean lead time bar
    means = [stats["delta_mean_lead"], stats["var_mean_lead"]]
    medians = [stats["delta_median_lead"], stats["var_median_lead"]]
    x = np.arange(2)
    ax4.bar(x - 0.15, means, 0.3, label="Mean", color="#00d4ff", alpha=0.8)
    ax4.bar(x + 0.15, medians, 0.3, label="Median", color="#50fa7b", alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(["Δ Coherence", "Variance"])
    ax4.set_ylabel("Lead Time (steps)")
    ax4.set_title("Mean/Median Lead Time")
    ax4.legend()
    ax4.grid(True, axis="y")

    fig.suptitle(f"Experiment 6: Monte Carlo ({n_trials} trials)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "exp6_monte_carlo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "experiment": 6,
        "name": "Monte Carlo Lead-Time Analysis",
        "n_trials": n_trials,
        "stats": stats,
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Delta.72 Experiment Suite")
    parser.add_argument("--n-monte-carlo", type=int, default=500, help="Monte Carlo trials (Exp 6)")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_plot_style()

    print("=" * 60)
    print("  Delta.72 Coherence Framework — Experiment Suite v1.0")
    print("=" * 60)
    t_total = time.time()

    all_results = {}

    all_results["exp1"] = experiment_1(output_dir)
    all_results["exp2"] = experiment_2(output_dir)
    all_results["exp3"] = experiment_3(output_dir)
    all_results["exp4"] = experiment_4(output_dir)
    all_results["exp5"] = experiment_5(output_dir)
    all_results["exp6"] = experiment_6(output_dir, n_trials=args.n_monte_carlo)

    total_elapsed = time.time() - t_total

    # Summary
    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)
    print(f"  Exp 1: Noise threshold ≈ {all_results['exp1']['threshold_noise']:.3f}")
    print(f"  Exp 3: Δ lead time = {all_results['exp3']['lead_times']['delta']} steps "
          f"(variance = {all_results['exp3']['lead_times']['variance']})")
    print(f"  Exp 5: Cross-system CV = {all_results['exp5']['cross_system_cv']:.3f} "
          f"({'consistent' if all_results['exp5']['consistent'] else 'inconsistent'})")
    print(f"  Exp 6: Δ detection rate = {all_results['exp6']['stats']['delta_detection_rate']:.1%}, "
          f"mean lead = {all_results['exp6']['stats']['delta_mean_lead']:.0f} steps")
    print(f"\n  Total elapsed: {total_elapsed:.1f}s")
    print("=" * 60)

    # Save JSON
    results_file = output_dir / "experiment_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_file}")
    print(f"  Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
