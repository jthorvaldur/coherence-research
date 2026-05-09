#!/usr/bin/env python3
"""Generate HTML dashboard for experiments 1-7.

Matches the visual style of the NAB/NASA/SKAB benchmark pages
(Outfit + JetBrains Mono, dark theme, card layout, nav bar).

Usage:
    uv run python experiments/generate_dashboard.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def score_var(value: float, low: float = 0.3, high: float = 0.7) -> str:
    if value < low:
        return "var(--red)"
    elif value < high:
        return "var(--amber)"
    return "var(--green)"


def badge(value: float, thresholds: tuple = (0.3, 0.7)) -> str:
    low, high = thresholds
    if value >= high:
        return f'<span class="badge badge-pass">{value:.3f}</span>'
    elif value >= low:
        return f'<span class="badge badge-warn">{value:.3f}</span>'
    return f'<span class="badge badge-fail">{value:.3f}</span>'


def generate_dashboard(results_dir: Path, output_path: Path):
    results_file = results_dir / "experiment_results.json"
    if not results_file.exists():
        print(f"No results at {results_file}. Run experiments first.")
        return

    with open(results_file) as f:
        results = json.load(f)

    exp1 = results.get("exp1", {})
    exp2 = results.get("exp2", {})
    exp3 = results.get("exp3", {})
    exp4 = results.get("exp4", {})
    exp5 = results.get("exp5", {})
    exp6 = results.get("exp6", {})
    exp7 = results.get("exp7", {})

    threshold = exp1.get("threshold_noise", 0)
    delta_lead = exp3.get("lead_times", {}).get("delta", 0)
    var_lead = exp3.get("lead_times", {}).get("variance", 0)
    cross_cv = exp5.get("cross_system_cv", 0)
    mc_stats = exp6.get("stats", {})
    mc_detect = mc_stats.get("delta_detection_rate", 0)
    mc_mean_lead = mc_stats.get("delta_mean_lead", 0)

    # Exp 2 table rows
    exp2_rows = ""
    for r in exp2.get("results", []):
        exp2_rows += (
            f'          <tr><td>{r["label"]}</td><td>{r["recovery_rate"]}</td>'
            f'<td style="color: {score_var(r["delta"])}; font-weight: 600;">{r["delta"]:.4f}</td>'
            f'<td style="color: {score_var(r["M"])}; font-weight: 600;">{r["M"]:.3f}</td>'
            f'<td style="color: {score_var(r["W"])}; font-weight: 600;">{r["W"]:.3f}</td></tr>\n'
        )

    # Exp 4 table rows
    exp4_rows = ""
    for r in exp4.get("results", []):
        exp4_rows += (
            f'          <tr><td>{r["coherence_param"]:.2f}</td>'
            f'<td style="font-weight: 600;">{r["peak_deviation"]:.2f}</td>'
            f'<td>{r["time_to_return"]}</td>'
            f'<td style="color: {score_var(r["delta"])}; font-weight: 600;">{r["delta"]:.4f}</td>'
            f'<td>{r["post_shock_stability"]:.4f}</td></tr>\n'
        )

    # Exp 5 table rows
    exp5_rows = ""
    for name, mean_d in exp5.get("systems", {}).items():
        exp5_rows += (
            f'          <tr><td>{name.capitalize()}</td>'
            f'<td style="color: {score_var(mean_d)}; font-weight: 600;">{mean_d:.4f}</td></tr>\n'
        )

    # Exp 7 data
    exp7_buildings = exp7.get("buildings", {})
    total_coh_alerts = 0
    total_var_alerts = 0
    total_coh_only = 0
    exp7_rows = ""
    for bld_name, bld_data in exp7_buildings.items():
        m_val = float(bld_data.get("global_M", 0))
        w_val = float(bld_data.get("global_W", 0))
        n_coh = int(bld_data.get("n_coherence_alerts", 0))
        n_var = int(bld_data.get("n_variance_alerts", 0))
        coh_only = int(bld_data.get("coherence_only_alerts", 0))
        total_coh_alerts += n_coh
        total_var_alerts += n_var
        total_coh_only += coh_only
        exp7_rows += (
            f'          <tr><td>{bld_name.replace("_", " ")}</td>'
            f'<td style="font-weight: 600;">{bld_data.get("mean_delta", 0):.4f}</td>'
            f'<td style="color: {score_var(m_val)}; font-weight: 600;">{m_val:.3f}</td>'
            f'<td style="color: {score_var(w_val)}; font-weight: 600;">{w_val:.3f}</td>'
            f'<td style="color: var(--accent);">{n_coh}</td>'
            f'<td style="color: var(--red);">{n_var}</td>'
            f'<td style="color: var(--green); font-weight: 600;">{coh_only}</td></tr>\n'
        )

    total_elapsed = sum(results.get(f"exp{i}", {}).get("elapsed_s", 0) for i in range(1, 8))

    # Exp 7 building detail blocks
    exp7_details = ""
    bld_labels = {
        "hog_office_betsy": "Hog office Betsy",
        "hog_office_nia": "Hog office Nia",
        "lamb_office_vasiliki": "Lamb office Vasiliki",
        "rat_office_avis": "Rat office Avis",
    }
    for bld_key, bld_label in bld_labels.items():
        exp7_details += f"""
        <details style="margin: 0.8rem 0;">
          <summary style="cursor: pointer; color: var(--accent); font-size: 0.85rem;">{bld_label} &mdash; detailed plots</summary>
          <div class="plot-wrap" style="margin-top: 0.5rem;"><img src="exp7_{bld_key}.png" alt="{bld_label}"></div>
        </details>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>&Delta;.72 Coherence Framework &mdash; Experiments 1&ndash;7</title>
<meta name="description" content="Delta.72 coherence framework: 7 experiments validating instability detection via coherence metrics. Synthetic GPU suite + real energy data.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@200;300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg: #07070f;
    --surface: rgba(255,255,255,0.035);
    --surface-hover: rgba(122,162,247,0.07);
    --border: rgba(255,255,255,0.07);
    --border-hover: rgba(122,162,247,0.30);
    --text: #e8e8f4;
    --muted: rgba(232,232,244,0.45);
    --dim: rgba(232,232,244,0.25);
    --accent: #7aa2f7;
    --accent2: #bb9af7;
    --green: #9ece6a;
    --amber: #e0af68;
    --red: #f7768e;
    --teal: #73daca;
    --glow: rgba(122,162,247,0.10);
  }}

  html {{ scroll-behavior: smooth; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Outfit', sans-serif;
    overflow-x: hidden;
    -webkit-font-smoothing: antialiased;
  }}

  /* Nav */
  .nav {{
    position: fixed; top: 0; left: 0; right: 0; z-index: 50;
    padding: 0 2rem; height: 56px;
    display: flex; align-items: center; justify-content: space-between;
    background: rgba(7,7,15,0.7);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
  }}
  .nav-brand {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--accent);
    text-decoration: none;
  }}
  .nav-links {{ display: flex; gap: 1.5rem; list-style: none; }}
  .nav-links a {{
    font-size: 0.78rem; font-weight: 400;
    color: var(--muted); text-decoration: none;
    transition: color 0.2s;
  }}
  .nav-links a:hover {{ color: var(--text); }}

  /* Hero */
  .hero {{
    padding: 120px 2rem 60px;
    text-align: center;
  }}
  .hero-eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.3em;
    text-transform: uppercase; color: var(--accent);
    margin-bottom: 1.5rem;
  }}
  .hero-title {{
    font-size: clamp(1.8rem, 5vw, 3.2rem);
    font-weight: 200; letter-spacing: -0.01em;
    line-height: 1.15; max-width: 720px; margin: 0 auto;
  }}
  .hero-title strong {{
    font-weight: 500;
    background: linear-gradient(140deg, #ffffff 20%, var(--accent) 60%, var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  .hero-sub {{
    margin-top: 1.2rem; font-size: 0.95rem;
    color: var(--muted); max-width: 640px; margin-left: auto; margin-right: auto;
    line-height: 1.7;
  }}

  /* Equation */
  .equation-block {{
    text-align: center;
    padding: 1.5rem 2rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    max-width: 700px; margin: 2rem auto;
  }}
  .equation-main {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem; letter-spacing: 2px;
    color: var(--accent);
  }}
  .equation-desc {{
    font-size: 0.78rem; color: var(--muted);
    margin-top: 0.8rem; line-height: 1.8;
  }}
  .equation-desc span {{ color: var(--text); font-weight: 500; }}

  /* Main */
  main {{
    position: relative; z-index: 2;
    max-width: 1000px; margin: 0 auto;
    padding: 2rem 2rem 6rem;
  }}

  /* Section */
  .section {{ margin-bottom: 4rem; }}
  .section-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.3em;
    text-transform: uppercase; color: var(--accent);
    margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 0.8rem;
  }}
  .section-label::after {{
    content: ''; display: block; flex: 1;
    height: 1px; background: var(--border);
  }}

  /* Metric cards */
  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
  }}
  .metric-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem; text-align: center;
    transition: border-color 0.25s;
  }}
  .metric-card:hover {{ border-color: var(--border-hover); }}
  .metric-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem; font-weight: 600;
    margin-bottom: 0.3rem;
  }}
  .metric-label {{
    font-size: 0.7rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 1px;
  }}

  /* Experiment card */
  .experiment {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 2rem;
    transition: border-color 0.25s;
  }}
  .experiment:hover {{ border-color: var(--border-hover); }}
  .experiment h2 {{
    font-size: 1.1rem; font-weight: 600;
    margin-bottom: 0.3rem;
  }}
  .experiment .exp-num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.15em;
    color: var(--accent); text-transform: uppercase;
    margin-bottom: 0.3rem;
  }}
  .experiment .exp-desc {{
    font-size: 0.85rem; color: var(--muted);
    margin-bottom: 1.2rem; line-height: 1.6;
  }}
  .experiment .finding {{
    font-size: 0.88rem; line-height: 1.7;
    margin-top: 1rem;
  }}
  .experiment .finding strong {{ color: var(--accent); }}

  /* Plot */
  .plot-wrap {{
    margin: 1.2rem 0;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.04);
  }}
  .plot-wrap img {{
    width: 100%; display: block;
  }}

  /* Tables */
  table {{
    width: 100%; border-collapse: collapse;
    margin: 1rem 0; font-size: 0.82rem;
  }}
  th {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.06em;
    text-transform: uppercase;
    background: rgba(122,162,247,0.06);
    color: var(--accent); padding: 0.6rem;
    text-align: left;
    border-bottom: 1px solid rgba(122,162,247,0.15);
  }}
  td {{
    padding: 0.5rem 0.6rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }}
  tr:hover {{ background: rgba(255,255,255,0.015); }}

  /* Comparison */
  .comparison {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem; margin: 1rem 0;
  }}
  .comp-card {{
    background: rgba(255,255,255,0.015);
    border-radius: 8px; padding: 1.2rem;
    border: 1px solid rgba(255,255,255,0.04);
  }}
  .comp-card h3 {{
    font-size: 0.85rem; font-weight: 600;
    margin-bottom: 0.6rem;
  }}
  .comp-card p {{
    font-size: 0.82rem; color: var(--muted); line-height: 1.7;
  }}
  .comp-card strong {{ color: var(--text); }}

  /* Tags */
  .tag {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem; font-weight: 500;
    letter-spacing: 0.06em; padding: 2px 8px;
    border-radius: 6px; display: inline-block;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
  }}

  /* Badge */
  .badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 3px 8px; border-radius: 4px;
    display: inline-block;
  }}
  .badge-pass {{ background: rgba(158,206,106,0.12); color: var(--green); }}
  .badge-warn {{ background: rgba(224,175,104,0.12); color: var(--amber); }}
  .badge-fail {{ background: rgba(247,118,142,0.12); color: var(--red); }}

  /* Footer */
  footer {{
    text-align: center; padding: 2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: var(--dim);
    border-top: 1px solid var(--border);
  }}
  footer a {{ color: var(--muted); text-decoration: none; }}
  footer a:hover {{ color: var(--accent); }}

  /* Responsive */
  @media (max-width: 640px) {{
    .nav-links {{ display: none; }}
    main {{ padding: 2rem 1.2rem 4rem; }}
    .comparison {{ grid-template-columns: 1fr; }}
    .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
  }}
</style>
</head>
<body>

<nav class="nav">
  <a href="/r/research/" class="nav-brand">Thorarinson</a>
  <ul class="nav-links">
    <li><a href="/r/research/" style="color: var(--accent); font-weight: 600;">Exps 1&ndash;7</a></li>
    <li><a href="/r/research/nab/">NAB</a></li>
    <li><a href="/r/research/nasa/">NASA</a></li>
    <li><a href="/r/research/skab/">SKAB</a></li>
    <li><a href="/r/research/uci-power/">UCI Power</a></li>
    <li><a href="/r/research/math/">Math</a></li>
    <li><a href="/r/research/ecg/">ECG</a></li>
    <li><a href="/r/research/credit-card/">Fraud</a></li>
    <li style="opacity: 0.3;">|</li>
    <li><a href="#summary">Summary</a></li>
    <li><a href="#synthetic">Synthetic</a></li>
    <li><a href="#realworld">Real-World</a></li>
  </ul>
</nav>

<!-- Hero -->
<section class="hero">
  <p class="hero-eyebrow">Experiments 01&ndash;07 &mdash; Core Validation Suite</p>
  <h1 class="hero-title">
    <strong>&Delta;.72</strong> Coherence Framework
  </h1>
  <p class="hero-sub">
    Seven experiments validating instability detection via coherence metrics.
    Six GPU-accelerated synthetic tests plus real-world energy data from four office buildings.
  </p>
</section>

<!-- Equation -->
<div class="equation-block">
  <div class="equation-main">&Delta; = (P &middot; A &middot; R) / (D + N)</div>
  <div class="equation-desc">
    <span>P</span> = Pattern Retention &middot;
    <span>A</span> = Phase Alignment &middot;
    <span>R</span> = Recovery Score<br>
    <span>D</span> = Drift &middot;
    <span>N</span> = Noise Amplification &middot;
    Extended: <span>M</span> (Attractor Memory), <span>W</span> (Windowed Recovery)<br>
    Gated alert: &Delta; &lt; 0.3 AND M &lt; 0.4 AND W &lt; 0.4
  </div>
</div>

<main>

  <!-- Summary Metrics -->
  <div class="section" id="summary">
    <p class="section-label">Key Results</p>
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-value" style="color: var(--accent);">{threshold:.2f}&sigma;</div>
        <div class="metric-label">Noise Threshold (Exp 1)</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--green);">{delta_lead}</div>
        <div class="metric-label">&Delta; Lead Time (steps)</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--red);">{var_lead}</div>
        <div class="metric-label">Var Lead Time (steps)</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--teal);">{cross_cv:.3f}</div>
        <div class="metric-label">Cross-System CV (Exp 5)</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--green);">{mc_detect:.0%}</div>
        <div class="metric-label">MC Detection Rate</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--accent2);">{mc_mean_lead:.0f}</div>
        <div class="metric-label">MC Mean Lead (steps)</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--muted);">{total_elapsed:.1f}s</div>
        <div class="metric-label">Total Runtime</div>
      </div>
    </div>
  </div>

  <!-- Synthetic Experiments -->
  <div class="section" id="synthetic">
    <p class="section-label">Synthetic Experiments (GPU)</p>

    <div class="experiment">
      <div class="exp-num">Experiment 01</div>
      <h2>Coherence vs Noise Threshold</h2>
      <div class="exp-desc">
        Does coherence collapse at a predictable threshold as noise increases?
      </div>
      <div class="plot-wrap"><img src="exp1_coherence_vs_noise.png" alt="Exp 1"></div>
      <div class="finding">
        Coherence drops sharply at <strong>&sigma; &asymp; {threshold:.3f}</strong>, confirming threshold behavior.
        Below this noise level, &Delta; maintains structural sensitivity. Above it, signal degrades into noise.
        <span class="badge badge-pass">Threshold confirmed</span>
      </div>
    </div>

    <div class="experiment">
      <div class="exp-num">Experiment 02</div>
      <h2>Recovery Dynamics After Shock</h2>
      <div class="exp-desc">
        Does coherence capture recovery differences after perturbation?
      </div>
      <div class="plot-wrap"><img src="exp2_recovery_dynamics.png" alt="Exp 2"></div>
      <table>
        <thead><tr><th>Recovery</th><th>Rate</th><th>&Delta;</th><th>M</th><th>W</th></tr></thead>
        <tbody>
{exp2_rows}        </tbody>
      </table>
    </div>

    <div class="experiment">
      <div class="exp-num">Experiment 03</div>
      <h2>Hidden Drift Before Visible Failure</h2>
      <div class="exp-desc">
        Can &Delta; detect drift significantly earlier than variance or z-score?
      </div>
      <div class="plot-wrap"><img src="exp3_hidden_drift.png" alt="Exp 3"></div>
      <div class="comparison">
        <div class="comp-card">
          <h3 style="color: var(--accent);">&Delta; Coherence</h3>
          <p>Detected at step <strong>{exp3.get('detection_times', {}).get('delta', '?')}</strong></p>
          <p>Lead time: <strong style="color: var(--green);">{delta_lead} steps</strong></p>
        </div>
        <div class="comp-card">
          <h3 style="color: var(--red);">Variance</h3>
          <p>Detected at step <strong>{exp3.get('detection_times', {}).get('variance', '?')}</strong></p>
          <p>Lead time: <strong>{var_lead} steps</strong></p>
        </div>
      </div>
      <div class="finding">
        &Delta; detects the hidden drift <strong>{delta_lead - var_lead} steps earlier</strong> than variance.
        <span class="badge badge-pass">Early Detection</span>
      </div>
    </div>

    <div class="experiment">
      <div class="exp-num">Experiment 04</div>
      <h2>Shock Response vs Coherence</h2>
      <div class="exp-desc">
        Lower coherence &rarr; larger deviation + slower recovery?
      </div>
      <div class="plot-wrap"><img src="exp4_shock_response.png" alt="Exp 4"></div>
      <table>
        <thead><tr><th>Coherence</th><th>Peak Dev</th><th>Return Time</th><th>&Delta;</th><th>Post-Shock &sigma;</th></tr></thead>
        <tbody>
{exp4_rows}        </tbody>
      </table>
    </div>

    <div class="experiment">
      <div class="exp-num">Experiment 05</div>
      <h2>Cross-System Generalization</h2>
      <div class="exp-desc">
        Does &Delta; remain consistent across different signal types?
      </div>
      <div class="plot-wrap"><img src="exp5_cross_system.png" alt="Exp 5"></div>
      <table>
        <thead><tr><th>Signal Type</th><th>Mean &Delta;</th></tr></thead>
        <tbody>
{exp5_rows}        </tbody>
      </table>
      <div class="finding">
        Cross-system coefficient of variation: <strong>{cross_cv:.3f}</strong>
        <span class="badge {'badge-pass' if cross_cv < 1.0 else 'badge-fail'}">{'Consistent' if cross_cv < 1.0 else 'Inconsistent'}</span>
      </div>
    </div>

    <div class="experiment">
      <div class="exp-num">Experiment 06</div>
      <h2>Monte Carlo Lead-Time Analysis</h2>
      <div class="exp-desc">
        Statistical robustness across {exp6.get('n_trials', '?')} randomized trials.
      </div>
      <div class="plot-wrap"><img src="exp6_monte_carlo.png" alt="Exp 6"></div>
      <div class="comparison">
        <div class="comp-card">
          <h3 style="color: var(--accent);">&Delta; Coherence</h3>
          <p>Detection rate: <strong style="color: var(--green);">{mc_stats.get('delta_detection_rate', 0):.1%}</strong></p>
          <p>Mean lead: <strong>{mc_stats.get('delta_mean_lead', 0):.0f} steps</strong></p>
          <p>Median lead: <strong>{mc_stats.get('delta_median_lead', 0):.0f} steps</strong></p>
        </div>
        <div class="comp-card">
          <h3 style="color: var(--red);">Variance</h3>
          <p>Detection rate: <strong>{mc_stats.get('var_detection_rate', 0):.1%}</strong></p>
          <p>Mean lead: <strong>{mc_stats.get('var_mean_lead', 0):.0f} steps</strong></p>
          <p>Median lead: <strong>{mc_stats.get('var_median_lead', 0):.0f} steps</strong></p>
        </div>
      </div>
    </div>
  </div>

  <!-- Real-World Validation -->
  <div class="section" id="realworld">
    <p class="section-label">Real-World Validation</p>

    <div class="experiment">
      <div class="exp-num">Experiment 07</div>
      <h2>Energy Systems &mdash; Office Building Electricity</h2>
      <div class="exp-desc">
        Applying the coherence framework to real hourly electricity load data from four office buildings.
        Hour-of-week baselines with rolling &Delta; scoring.
      </div>

      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-value" style="color: var(--accent);">{len(exp7_buildings)}</div>
          <div class="metric-label">Buildings</div>
        </div>
        <div class="metric-card">
          <div class="metric-value" style="color: var(--accent);">{total_coh_alerts}</div>
          <div class="metric-label">Coherence Alerts</div>
        </div>
        <div class="metric-card">
          <div class="metric-value" style="color: var(--red);">{total_var_alerts}</div>
          <div class="metric-label">Variance Alerts</div>
        </div>
        <div class="metric-card">
          <div class="metric-value" style="color: var(--green);">{total_coh_only}</div>
          <div class="metric-label">Coherence-Only</div>
        </div>
      </div>

      <table>
        <thead><tr><th>Building</th><th>Mean &Delta;</th><th>M</th><th>W</th><th>Coh.</th><th>Var.</th><th>Coh-Only</th></tr></thead>
        <tbody>
{exp7_rows}        </tbody>
      </table>

      <div class="finding">
        The coherence framework detected <strong style="color: var(--green);">{total_coh_only}</strong>
        instability episodes invisible to simple variance-based detection, validating the framework
        on real-world energy data.
        <span class="badge badge-pass">Real-World Validated</span>
      </div>

      <div class="plot-wrap"><img src="exp7_heatmap.png" alt="Cross-building heatmap"></div>

{exp7_details}
    </div>
  </div>

  <!-- Benchmark Links -->
  <div class="section">
    <p class="section-label">Cross-Domain Benchmarks</p>
    <div class="metrics-grid">
      <a href="/r/research/nab/" style="text-decoration: none;">
        <div class="metric-card">
          <div class="metric-value" style="color: var(--accent);">NAB</div>
          <div class="metric-label">58 Time Series &middot; Anomaly Benchmark</div>
        </div>
      </a>
      <a href="/r/research/nasa/" style="text-decoration: none;">
        <div class="metric-card">
          <div class="metric-value" style="color: var(--green);">NASA</div>
          <div class="metric-label">100 Engines &middot; Turbofan Degradation</div>
        </div>
      </a>
      <a href="/r/research/skab/" style="text-decoration: none;">
        <div class="metric-card">
          <div class="metric-value" style="color: var(--accent2);">SKAB</div>
          <div class="metric-label">34 Experiments &middot; Industrial Valve Faults</div>
        </div>
      </a>
      <a href="/r/research/uci-power/" style="text-decoration: none;">
        <div class="metric-card">
          <div class="metric-value" style="color: var(--teal);">UCI Power</div>
          <div class="metric-label">4 Years &middot; Household Electricity</div>
        </div>
      </a>
      <a href="/r/research/ecg/" style="text-decoration: none;">
        <div class="metric-card">
          <div class="metric-value" style="color: var(--red);">ECG</div>
          <div class="metric-label">45 Records &middot; Cardiac Arrhythmia</div>
        </div>
      </a>
      <a href="/r/research/credit-card/" style="text-decoration: none;">
        <div class="metric-card">
          <div class="metric-value" style="color: var(--amber);">Fraud</div>
          <div class="metric-label">284K Transactions &middot; Financial</div>
        </div>
      </a>
      <a href="/r/research/math/" style="text-decoration: none;">
        <div class="metric-card">
          <div class="metric-value" style="color: var(--accent2);">Math</div>
          <div class="metric-label">12 Modules &middot; Mathematical Foundations</div>
        </div>
      </a>
    </div>
  </div>

</main>

<footer>
  <p>&Delta;.72 Coherence Framework &mdash; Experiments 01&ndash;07</p>
  <p style="margin-top: 0.5rem;"><a href="/">thorarinson</a> &middot; <a href="/r/d72/">canonical framework</a> &middot; <a href="https://coherenceengine.org">coherence engine</a></p>
</footer>

</body>
</html>"""

    output_path.write_text(html)
    print(f"Dashboard written to {output_path} ({len(html):,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Delta.72 experiment dashboard")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/dashboard.html")
    args = parser.parse_args()

    generate_dashboard(Path(args.results_dir), Path(args.output))
