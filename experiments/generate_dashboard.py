#!/usr/bin/env python3
"""Generate self-contained HTML dashboard from experiment results.

Usage:
    python experiments/generate_dashboard.py [--results-dir results] [--output results/dashboard.html]
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def img_to_base64(path: Path) -> str:
    """Encode image file to base64 data URI."""
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"


def score_color(value: float, low: float = 0.3, high: float = 0.7) -> str:
    """Map score to CSS color: red < low, orange < high, green >= high."""
    if value < low:
        return "#ff6b6b"
    elif value < high:
        return "#ffb347"
    return "#50fa7b"


def lead_color(delta_lead: int, other_lead: int) -> str:
    """Green if delta leads, red if behind."""
    if delta_lead > other_lead:
        return "#50fa7b"
    elif delta_lead < other_lead:
        return "#ff6b6b"
    return "#e0e0e0"


def build_metric_card(label: str, value: str, color: str = "#00d4ff") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {color};">{value}</div>
        <div class="metric-label">{label}</div>
    </div>"""


def generate_dashboard(results_dir: Path, output_path: Path):
    results_file = results_dir / "experiment_results.json"
    if not results_file.exists():
        print(f"No results found at {results_file}. Run experiments first.")
        return

    with open(results_file) as f:
        results = json.load(f)

    # Encode plot images
    plots = {}
    for i in range(1, 7):
        plot_files = list(results_dir.glob(f"exp{i}_*.png"))
        if plot_files:
            plots[f"exp{i}"] = img_to_base64(plot_files[0])

    # Build summary metrics
    exp1 = results.get("exp1", {})
    exp3 = results.get("exp3", {})
    exp5 = results.get("exp5", {})
    exp6 = results.get("exp6", {})

    threshold = exp1.get("threshold_noise", 0)
    delta_lead = exp3.get("lead_times", {}).get("delta", 0)
    var_lead = exp3.get("lead_times", {}).get("variance", 0)
    cross_cv = exp5.get("cross_system_cv", 0)
    mc_detect = exp6.get("stats", {}).get("delta_detection_rate", 0)
    mc_mean_lead = exp6.get("stats", {}).get("delta_mean_lead", 0)

    # --- Experiment 2 table ---
    exp2_rows = ""
    for r in results.get("exp2", {}).get("results", []):
        d_color = score_color(r["delta"])
        m_color = score_color(r["M"])
        w_color = score_color(r["W"])
        exp2_rows += f"""
        <tr>
            <td>{r['label']}</td>
            <td>{r['recovery_rate']}</td>
            <td style="color: {d_color}; font-weight: bold;">{r['delta']:.4f}</td>
            <td style="color: {m_color}; font-weight: bold;">{r['M']:.3f}</td>
            <td style="color: {w_color}; font-weight: bold;">{r['W']:.3f}</td>
        </tr>"""

    # --- Experiment 4 table ---
    exp4_rows = ""
    for r in results.get("exp4", {}).get("results", []):
        exp4_rows += f"""
        <tr>
            <td>{r['coherence_param']:.2f}</td>
            <td style="font-weight: bold;">{r['peak_deviation']:.2f}</td>
            <td>{r['time_to_return']}</td>
            <td style="color: {score_color(r['delta'])}; font-weight: bold;">{r['delta']:.4f}</td>
            <td>{r['post_shock_stability']:.4f}</td>
        </tr>"""

    # --- Experiment 5 table ---
    exp5_rows = ""
    for name, mean_d in exp5.get("systems", {}).items():
        exp5_rows += f"""
        <tr>
            <td>{name.capitalize()}</td>
            <td style="color: {score_color(mean_d)}; font-weight: bold;">{mean_d:.4f}</td>
        </tr>"""

    # --- Experiment 6 comparison ---
    mc_stats = exp6.get("stats", {})

    # Total elapsed
    total_elapsed = sum(results.get(f"exp{i}", {}).get("elapsed_s", 0) for i in range(1, 7))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Δ.72 Coherence Framework — Experiment Dashboard</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
        min-height: 100vh;
        padding: 2rem;
    }}
    .header {{
        text-align: center;
        padding: 2rem 0 3rem;
        border-bottom: 1px solid #2a2a4e;
        margin-bottom: 2rem;
    }}
    .header h1 {{
        font-size: 2.2rem;
        background: linear-gradient(90deg, #00d4ff, #50fa7b, #bd93f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }}
    .header .subtitle {{
        color: #8892b0;
        font-size: 0.95rem;
    }}
    .equation {{
        text-align: center;
        padding: 1.5rem;
        background: rgba(0,212,255,0.05);
        border: 1px solid rgba(0,212,255,0.2);
        border-radius: 12px;
        margin: 1.5rem auto;
        max-width: 600px;
        font-size: 1.3rem;
        letter-spacing: 2px;
    }}
    .metrics-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }}
    .metric-card {{
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }}
    .metric-value {{
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.3rem;
    }}
    .metric-label {{
        font-size: 0.75rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .experiment {{
        margin: 2.5rem 0;
        padding: 2rem;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
    }}
    .experiment h2 {{
        font-size: 1.2rem;
        color: #00d4ff;
        margin-bottom: 0.3rem;
    }}
    .experiment .exp-desc {{
        color: #8892b0;
        font-size: 0.85rem;
        margin-bottom: 1.2rem;
    }}
    .plot-img {{
        width: 100%;
        max-width: 900px;
        border-radius: 8px;
        margin: 1rem auto;
        display: block;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.85rem;
    }}
    th {{
        background: rgba(0,212,255,0.1);
        color: #00d4ff;
        padding: 0.6rem;
        text-align: left;
        border-bottom: 1px solid rgba(0,212,255,0.2);
    }}
    td {{
        padding: 0.5rem 0.6rem;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }}
    tr:hover {{ background: rgba(255,255,255,0.02); }}
    .comparison-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin: 1rem 0;
    }}
    .comparison-card {{
        background: rgba(255,255,255,0.02);
        border-radius: 8px;
        padding: 1.2rem;
        border: 1px solid rgba(255,255,255,0.06);
    }}
    .comparison-card h3 {{
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
    }}
    .footer {{
        text-align: center;
        padding: 2rem 0;
        color: #555;
        font-size: 0.8rem;
        border-top: 1px solid #2a2a4e;
        margin-top: 2rem;
    }}
    .alert-badge {{
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }}
    .alert-green {{ background: rgba(80,250,123,0.15); color: #50fa7b; }}
    .alert-red {{ background: rgba(255,107,107,0.15); color: #ff6b6b; }}
</style>
</head>
<body>

<div class="header">
    <h1>Δ.72 Coherence Framework</h1>
    <div class="subtitle">GPU Experiment Suite v1.0 — Instability Detection via Coherence Metrics</div>
</div>

<div class="equation">
    Δ = (P · A · R) / (D + N)
</div>

<div class="metrics-grid">
    {build_metric_card("Noise Threshold", f"{threshold:.2f}σ", "#00d4ff")}
    {build_metric_card("Δ Lead Time (Exp 3)", f"{delta_lead} steps", lead_color(delta_lead, var_lead))}
    {build_metric_card("Var Lead Time (Exp 3)", f"{var_lead} steps", "#8892b0")}
    {build_metric_card("Cross-System CV", f"{cross_cv:.3f}", score_color(1-cross_cv))}
    {build_metric_card("MC Detection Rate", f"{mc_detect:.0%}", score_color(mc_detect))}
    {build_metric_card("MC Mean Lead", f"{mc_mean_lead:.0f} steps", "#bd93f9")}
    {build_metric_card("Total Runtime", f"{total_elapsed:.1f}s", "#8892b0")}
</div>

<!-- Experiment 1 -->
<div class="experiment">
    <h2>Experiment 1 — Coherence vs Noise Threshold</h2>
    <div class="exp-desc">Does coherence collapse at a predictable threshold as noise increases?</div>
    {"<img class='plot-img' src='" + plots.get('exp1', '') + "'/>" if plots.get('exp1') else ""}
    <p>Coherence drops sharply at <strong>σ ≈ {threshold:.3f}</strong>, confirming threshold behavior.
    Below this noise level, the Δ metric maintains structural sensitivity. Above it, signal degrades into noise.</p>
</div>

<!-- Experiment 2 -->
<div class="experiment">
    <h2>Experiment 2 — Recovery Dynamics After Shock</h2>
    <div class="exp-desc">Does coherence capture recovery differences after perturbation?</div>
    {"<img class='plot-img' src='" + plots.get('exp2', '') + "'/>" if plots.get('exp2') else ""}
    <table>
        <thead><tr><th>Recovery</th><th>Rate</th><th>Δ</th><th>𝓜</th><th>𝓦</th></tr></thead>
        <tbody>{exp2_rows}</tbody>
    </table>
</div>

<!-- Experiment 3 -->
<div class="experiment">
    <h2>Experiment 3 — Hidden Drift Before Visible Failure</h2>
    <div class="exp-desc">Can Δ detect drift significantly earlier than variance or z-score?</div>
    {"<img class='plot-img' src='" + plots.get('exp3', '') + "'/>" if plots.get('exp3') else ""}
    <div class="comparison-grid">
        <div class="comparison-card">
            <h3 style="color: #00d4ff;">Δ Coherence</h3>
            <p>Detected at step <strong>{exp3.get('detection_times', {}).get('delta', '?')}</strong></p>
            <p>Lead time: <strong style="color: {lead_color(delta_lead, var_lead)};">{delta_lead} steps</strong></p>
        </div>
        <div class="comparison-card">
            <h3 style="color: #ff6b6b;">Variance</h3>
            <p>Detected at step <strong>{exp3.get('detection_times', {}).get('variance', '?')}</strong></p>
            <p>Lead time: <strong>{var_lead} steps</strong></p>
        </div>
    </div>
</div>

<!-- Experiment 4 -->
<div class="experiment">
    <h2>Experiment 4 — Shock Response vs Coherence</h2>
    <div class="exp-desc">Lower coherence → larger deviation + slower recovery?</div>
    {"<img class='plot-img' src='" + plots.get('exp4', '') + "'/>" if plots.get('exp4') else ""}
    <table>
        <thead><tr><th>Coherence Param</th><th>Peak Deviation</th><th>Time to Return</th><th>Δ</th><th>Post-Shock σ</th></tr></thead>
        <tbody>{exp4_rows}</tbody>
    </table>
</div>

<!-- Experiment 5 -->
<div class="experiment">
    <h2>Experiment 5 — Cross-System Generalization</h2>
    <div class="exp-desc">Does Δ remain consistent across signal types?</div>
    {"<img class='plot-img' src='" + plots.get('exp5', '') + "'/>" if plots.get('exp5') else ""}
    <table>
        <thead><tr><th>Signal Type</th><th>Mean Δ</th></tr></thead>
        <tbody>{exp5_rows}</tbody>
    </table>
    <p>Cross-system coefficient of variation: <strong style="color: {score_color(1-cross_cv)};">{cross_cv:.3f}</strong>
    <span class="alert-badge {'alert-green' if cross_cv < 1.0 else 'alert-red'}">
        {'CONSISTENT' if cross_cv < 1.0 else 'INCONSISTENT'}
    </span></p>
</div>

<!-- Experiment 6 -->
<div class="experiment">
    <h2>Experiment 6 — Monte Carlo Lead-Time Analysis</h2>
    <div class="exp-desc">Statistical robustness across {exp6.get('n_trials', '?')} randomized trials.</div>
    {"<img class='plot-img' src='" + plots.get('exp6', '') + "'/>" if plots.get('exp6') else ""}
    <div class="comparison-grid">
        <div class="comparison-card">
            <h3 style="color: #00d4ff;">Δ Coherence</h3>
            <p>Detection rate: <strong>{mc_stats.get('delta_detection_rate', 0):.1%}</strong></p>
            <p>Mean lead: <strong>{mc_stats.get('delta_mean_lead', 0):.0f}</strong> steps</p>
            <p>Median lead: <strong>{mc_stats.get('delta_median_lead', 0):.0f}</strong> steps</p>
        </div>
        <div class="comparison-card">
            <h3 style="color: #ff6b6b;">Variance</h3>
            <p>Detection rate: <strong>{mc_stats.get('var_detection_rate', 0):.1%}</strong></p>
            <p>Mean lead: <strong>{mc_stats.get('var_mean_lead', 0):.0f}</strong> steps</p>
            <p>Median lead: <strong>{mc_stats.get('var_median_lead', 0):.0f}</strong> steps</p>
        </div>
    </div>
</div>

<div class="footer">
    Δ.72 Coherence Framework — allison-research — Generated from experiment suite v1.0
</div>

</body>
</html>"""

    output_path.write_text(html)
    print(f"Dashboard written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Delta.72 experiment dashboard")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/dashboard.html")
    args = parser.parse_args()

    generate_dashboard(Path(args.results_dir), Path(args.output))
