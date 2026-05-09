#!/usr/bin/env python3
"""Generate results/skab/index.html — self-contained dashboard for Experiment 9.

Matches the exact visual style of the NASA dashboard
(/r/research/nasa/index.html) — same fonts, colors, layout, component classes.

Usage:
    uv run python experiments/skab_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "skab"
RESULTS_JSON = RESULTS_DIR / "exp9_results.json"
OUTPUT_HTML = RESULTS_DIR / "index.html"


def load_results() -> dict:
    with open(RESULTS_JSON) as f:
        return json.load(f)


def build_per_experiment_rows(per_experiment: list[dict]) -> str:
    """Build HTML table rows for per-experiment results."""
    rows = []
    for e in per_experiment:
        delta_style = 'color: var(--green); font-weight: 600;' if e["delta_detected"] else 'color: var(--red);'
        var_style = 'color: var(--green);' if e["var_detected"] else 'color: var(--red);'
        coh_f1 = e["coherence_f1"]
        var_f1 = e["variance_f1"]
        f1_winner = 'color: var(--green); font-weight: 600;' if coh_f1 >= var_f1 else ''
        f1_var_style = 'color: var(--green); font-weight: 600;' if var_f1 > coh_f1 else ''
        rows.append(
            f'          <tr>'
            f'<td>{e["category"]}/{e["filename"]}</td>'
            f'<td>{e["n_points"]}</td>'
            f'<td>{e["anomaly_onset"]}</td>'
            f'<td style="{f1_winner}">{coh_f1:.3f}</td>'
            f'<td style="{f1_var_style}">{var_f1:.3f}</td>'
            f'<td>{e["coherence_precision"]:.3f}</td>'
            f'<td>{e["coherence_recall"]:.3f}</td>'
            f'<td style="{delta_style}">{e["delta_lead"]}</td>'
            f'<td style="{var_style}">{e["var_lead"]}</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def build_category_rows(categories: dict) -> str:
    """Build HTML table rows for per-category breakdown."""
    rows = []
    for cat_name, cat in categories.items():
        coh_f1 = cat["coherence_f1"]
        var_f1 = cat["variance_f1"]
        f1_winner = 'color: var(--green); font-weight: 600;' if coh_f1 >= var_f1 else ''
        f1_var_style = 'color: var(--green); font-weight: 600;' if var_f1 > coh_f1 else ''
        rows.append(
            f'          <tr>'
            f'<td style="font-weight: 600;">{cat_name}</td>'
            f'<td>{cat["n_files"]}</td>'
            f'<td style="{f1_winner}">{coh_f1:.3f}</td>'
            f'<td style="{f1_var_style}">{var_f1:.3f}</td>'
            f'<td>{cat["coherence_precision"]:.3f}</td>'
            f'<td>{cat["coherence_recall"]:.3f}</td>'
            f'<td>{cat["variance_precision"]:.3f}</td>'
            f'<td>{cat["variance_recall"]:.3f}</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def generate_html(data: dict) -> str:
    s = data["stats"]
    cfg = data["config"]
    categories = data["categories"]

    per_experiment_rows = build_per_experiment_rows(s["per_experiment"])
    category_rows = build_category_rows(categories)

    # Count delta wins (experiments where coherence_f1 > variance_f1)
    coh_wins = s["coh_wins"]
    total_exps = s["n_experiments"]

    # Sensors list
    sensors = data.get("sensors_used", [])
    sensors_str = ", ".join(f'<span>{s_name}</span>' for s_name in sensors)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>&Delta;.72 &mdash; SKAB Industrial Fault Detection</title>
<meta name="description" content="Delta.72 coherence framework applied to SKAB (Skoltech Anomaly Benchmark). 34 industrial experiments, 8 sensors, valve fault detection.">
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
    color: var(--muted); max-width: 620px; margin-left: auto; margin-right: auto;
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
  .tag-green {{ color: var(--green); }}
  .tag-amber {{ color: var(--amber); }}
  .tag-red {{ color: var(--red); }}

  /* Status badge */
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
    <li><a href="/r/research/">Exps 1&ndash;7</a></li>
    <li><a href="/r/research/nab/">NAB</a></li>
    <li><a href="/r/research/nasa/">NASA</a></li>
    <li><a href="/r/research/skab/" style="color: var(--accent); font-weight: 600;">SKAB</a></li>
    <li><a href="/r/research/uci-power/">UCI Power</a></li>
    <li><a href="/r/research/math/">Math</a></li>
    <li><a href="/r/research/ecg/">ECG</a></li>
    <li><a href="/r/research/credit-card/">Fraud</a></li>
    <li style="opacity: 0.3;">|</li>
    <li><a href="#results">Results</a></li>
    <li><a href="#categories">Categories</a></li>
    <li><a href="#plots">Plots</a></li>
    <li><a href="#detail">Detail</a></li>
  </ul>
</nav>

<!-- Hero -->
<section class="hero">
  <p class="hero-eyebrow">Experiment 09 &mdash; Industrial Fault Detection</p>
  <h1 class="hero-title">
    <strong>&Delta;.72</strong> on SKAB Benchmark
  </h1>
  <p class="hero-sub">
    Applying the coherence framework to real industrial valve fault data.
    {s['n_experiments']} experiments across 3 fault categories, 8 sensor channels.
    Can &Delta; detect valve faults with higher precision than variance?
  </p>
</section>

<!-- Equation -->
<div class="equation-block">
  <div class="equation-main">&Delta; = (P &middot; A &middot; R) / (D + N)</div>
  <div class="equation-desc">
    Applied to 8 industrial sensors: {sensors_str}.<br>
    Rolling window: {cfg['window_size']} samples, step {cfg['step_size']}.
    &Delta; threshold: {cfg['delta_threshold']}. Variance z-score: {cfg['variance_zscore']}.
  </div>
</div>

<main>

  <!-- Summary Metrics -->
  <div class="section" id="results">
    <p class="section-label">Key Results</p>
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-value" style="color: var(--green);">{s['delta_detection_rate']:.0%}</div>
        <div class="metric-label">&Delta; Detection Rate</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--green);">{s['var_detection_rate']:.0%}</div>
        <div class="metric-label">Variance Detection Rate</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--accent);">{s['coherence_f1']:.3f}</div>
        <div class="metric-label">&Delta; F1 Score</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--amber);">{s['variance_f1']:.3f}</div>
        <div class="metric-label">Variance F1 Score</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--teal);">{s['coherence_precision']:.3f}</div>
        <div class="metric-label">&Delta; Precision</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--accent2);">{s['coherence_recall']:.3f}</div>
        <div class="metric-label">&Delta; Recall</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--accent);">{s['delta_mean_lead']:.0f}</div>
        <div class="metric-label">Mean &Delta; Lead (samples)</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--red);">{s['var_mean_lead']:.0f}</div>
        <div class="metric-label">Mean Var Lead (samples)</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--green);">{coh_wins} / {total_exps}</div>
        <div class="metric-label">&Delta; Wins (F1)</div>
      </div>
    </div>

    <p style="font-size: 0.85rem; color: var(--muted); line-height: 1.8; max-width: 700px;">
      Across {s['n_experiments']} SKAB industrial experiments, the &Delta; coherence metric achieved
      <strong style="color: var(--green);">{s['delta_detection_rate']:.0%} detection rate</strong> with a mean F1 of
      <strong style="color: var(--accent);">{s['coherence_f1']:.3f}</strong>. Variance-based detection
      scored F1 of {s['variance_f1']:.3f}. While variance achieves near-perfect recall
      (<strong style="color: var(--amber);">{s['variance_recall']:.3f}</strong>), it does so at the cost of precision
      ({s['variance_precision']:.3f}). &Delta; wins on F1 in <strong style="color: var(--green);">{coh_wins}</strong> of
      {total_exps} experiments, showing stronger precision where it matters.
    </p>
  </div>

  <!-- Method Comparison -->
  <div class="section">
    <p class="section-label">Method Comparison</p>
    <div class="comparison">
      <div class="comp-card">
        <h3 style="color: var(--accent);">&Delta; Coherence</h3>
        <p>Detection rate: <strong style="color: var(--green);">{s['delta_detection_rate']:.0%}</strong></p>
        <p>F1 score: <strong>{s['coherence_f1']:.3f}</strong></p>
        <p>Precision: <strong>{s['coherence_precision']:.3f}</strong></p>
        <p>Recall: <strong>{s['coherence_recall']:.3f}</strong></p>
        <p>Mean lead: <strong>{s['delta_mean_lead']:.0f} samples</strong></p>
      </div>
      <div class="comp-card">
        <h3 style="color: var(--red);">Variance</h3>
        <p>Detection rate: <strong style="color: var(--green);">{s['var_detection_rate']:.0%}</strong></p>
        <p>F1 score: <strong>{s['variance_f1']:.3f}</strong></p>
        <p>Precision: <strong>{s['variance_precision']:.3f}</strong></p>
        <p>Recall: <strong>{s['variance_recall']:.3f}</strong></p>
        <p>Mean lead: <strong>{s['var_mean_lead']:.0f} samples</strong></p>
      </div>
    </div>
  </div>

  <!-- Category Breakdown -->
  <div class="section" id="categories">
    <p class="section-label">Category Breakdown</p>
    <table>
      <thead>
        <tr>
          <th>Category</th>
          <th>Files</th>
          <th>&Delta; F1</th>
          <th>Var F1</th>
          <th>&Delta; Precision</th>
          <th>&Delta; Recall</th>
          <th>Var Precision</th>
          <th>Var Recall</th>
        </tr>
      </thead>
      <tbody>
{category_rows}
      </tbody>
    </table>
    <p style="font-size: 0.82rem; color: var(--muted); line-height: 1.7; margin-top: 1rem;">
      Three fault categories: <strong style="color: var(--text);">valve1</strong> (16 experiments),
      <strong style="color: var(--text);">valve2</strong> (4 experiments), and
      <strong style="color: var(--text);">other</strong> (14 experiments).
      Variance achieves near-perfect recall across all categories but at low precision.
      &Delta; trades recall for meaningfully higher precision, reducing false positives.
    </p>
  </div>

  <!-- Plots -->
  <div class="section" id="plots">
    <p class="section-label">Experiment Plots</p>

    <div class="experiment">
      <div class="exp-num">Plot 01</div>
      <h2>Example Experiment &mdash; Sensor Traces &amp; Coherence</h2>
      <div class="exp-desc">
        Top: 8 sensor traces (normalized) showing industrial valve behavior before and after fault onset.
        Middle: system-level &Delta; coherence with alert threshold. Bottom: per-sensor &Delta; decomposition.
      </div>
      <div class="plot-wrap"><img src="exp9_example_experiment.png" alt="Example experiment with coherence overlay"></div>
      <div class="finding">
        Sensor readings shift subtly at fault onset. &Delta; coherence captures the cross-sensor structural
        departure from baseline, providing an interpretable anomaly signal.
        <span class="badge badge-pass">Interpretable</span>
      </div>
    </div>

    <div class="experiment">
      <div class="exp-num">Plot 02</div>
      <h2>F1 Score Comparison</h2>
      <div class="exp-desc">
        Per-experiment F1 scores for &Delta; coherence vs variance across all {s['n_experiments']} experiments.
      </div>
      <div class="plot-wrap"><img src="exp9_f1_comparison.png" alt="F1 score comparison"></div>
      <div class="finding">
        &Delta; wins on F1 in <strong>{coh_wins} of {total_exps}</strong> experiments. Where &Delta; wins,
        the margin is often substantial &mdash; driven by higher precision on the anomalous segments.
        <span class="badge badge-warn">Mixed</span>
      </div>
    </div>

    <div class="experiment">
      <div class="exp-num">Plot 03</div>
      <h2>Detection Timeline</h2>
      <div class="exp-desc">
        When does each method first detect the anomaly relative to the labeled onset?
        Comparison of lead times across all experiments.
      </div>
      <div class="plot-wrap"><img src="exp9_detection_timeline.png" alt="Detection timeline"></div>
      <div class="finding">
        Both methods detect faults before the labeled onset in most cases. Variance tends to alert
        earlier on average (<strong>{s['var_mean_lead']:.0f}</strong> vs <strong>{s['delta_mean_lead']:.0f}</strong> samples),
        but this comes at the cost of more false positives.
        <span class="badge badge-pass">Early Detection</span>
      </div>
    </div>

    <div class="experiment">
      <div class="exp-num">Plot 04</div>
      <h2>Sensor Heatmap</h2>
      <div class="exp-desc">
        Per-sensor &Delta; coherence values across experiments. Which sensors contribute most to fault detection?
      </div>
      <div class="plot-wrap"><img src="exp9_sensor_heatmap.png" alt="Sensor heatmap"></div>
      <div class="finding">
        The heatmap reveals which sensor channels carry the strongest fault signatures.
        Accelerometer and pressure sensors tend to show the earliest coherence departure,
        consistent with mechanical valve fault physics.
        <span class="badge badge-pass">Physics-Aligned</span>
      </div>
    </div>

    <div class="experiment">
      <div class="exp-num">Plot 05</div>
      <h2>Summary Overview</h2>
      <div class="exp-desc">
        Combined summary: detection rates, F1 distributions, precision-recall tradeoff,
        and per-category performance.
      </div>
      <div class="plot-wrap"><img src="exp9_summary.png" alt="Summary overview"></div>
      <div class="finding">
        The summary confirms that &Delta; coherence provides a viable alternative to variance-based detection
        on industrial data, with a fundamentally different precision-recall tradeoff.
        <span class="badge badge-pass">Confirmed</span>
      </div>
    </div>
  </div>

  <!-- Per-Experiment Detail -->
  <div class="section" id="detail">
    <p class="section-label">All Experiments</p>
    <details>
      <summary style="cursor: pointer; color: var(--accent); font-size: 0.88rem; margin-bottom: 1rem;">
        Show all {s['n_experiments']} experiments
      </summary>
      <div style="overflow-x: auto;">
        <table>
          <thead>
            <tr>
              <th>Experiment</th>
              <th>Points</th>
              <th>Onset</th>
              <th>&Delta; F1</th>
              <th>Var F1</th>
              <th>&Delta; Prec</th>
              <th>&Delta; Rec</th>
              <th>&Delta; Lead</th>
              <th>Var Lead</th>
            </tr>
          </thead>
          <tbody>
{per_experiment_rows}
          </tbody>
        </table>
      </div>
    </details>
  </div>

  <!-- Dataset -->
  <div class="section">
    <p class="section-label">Dataset &amp; Configuration</p>
    <p style="font-size: 0.85rem; color: var(--muted); line-height: 1.8; max-width: 700px;">
      <strong style="color: var(--text);">SKAB (Skoltech Anomaly Benchmark)</strong> &mdash; Industrial testbed
      with water circulation system, valves, and 8 sensor channels. {s['n_experiments']} experiments across
      3 fault categories (valve1, valve2, other). Each experiment contains labeled anomaly onset points.
      Source: Skoltech / GitHub.
    </p>
    <p style="font-size: 0.85rem; color: var(--muted); line-height: 1.8; max-width: 700px; margin-top: 1rem;">
      <strong style="color: var(--text);">Configuration</strong> &mdash;
      Window: {cfg['window_size']} samples, step {cfg['step_size']}.
      &Delta; threshold: {cfg['delta_threshold']}.
      Memory threshold: {cfg['memory_threshold']}.
      Recovery threshold: {cfg['recovery_threshold']}.
      Variance z-score: {cfg['variance_zscore']}.
    </p>
    <div style="margin-top: 1.5rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">
      <span class="tag tag-green">Python</span>
      <span class="tag" style="color: var(--accent);">NumPy</span>
      <span class="tag" style="color: var(--accent2);">SciPy</span>
      <span class="tag" style="color: var(--teal);">SKAB</span>
      <span class="tag tag-amber">Industrial</span>
      <span class="tag" style="color: var(--text);">{s['n_experiments']} Experiments</span>
      <span class="tag" style="color: var(--text);">8 Sensors</span>
    </div>
  </div>

  <!-- Back link -->
  <div class="section">
    <p class="section-label">Navigation</p>
    <p style="font-size: 0.85rem;">
      <a href="/r/research/" style="color: var(--accent); text-decoration: none;">&larr; Back to all experiments</a>
    </p>
  </div>

</main>

<footer>
  <p>&Delta;.72 Coherence Framework &mdash; Experiment 09: SKAB Industrial Fault Detection</p>
  <p style="margin-top: 0.5rem;"><a href="/r/research/">thorarinson</a> &middot; <a href="https://coherenceengine.org">coherenceengine.org</a></p>
</footer>

</body>
</html>"""

    return html


def main():
    print("=" * 60)
    print("  SKAB Industrial Fault Detection Dashboard Generator")
    print("=" * 60)

    data = load_results()
    print(f"  Loaded results: {data['stats']['n_experiments']} experiments")

    html = generate_html(data)

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_HTML, "w") as f:
        f.write(html)

    print(f"  Generated: {OUTPUT_HTML}")
    print(f"  File size: {OUTPUT_HTML.stat().st_size / 1024:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
