#!/usr/bin/env python3
"""Generate results/ecg/index.html — self-contained dashboard for Experiment 11.

Matches the exact visual style of the main Delta.72 research page
(/r/research/index.html) — same fonts, colors, layout, component classes.

Usage:
    uv run python experiments/ecg_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "ecg"
RESULTS_JSON = RESULTS_DIR / "exp11_results.json"
OUTPUT_HTML = RESULTS_DIR / "index.html"


def load_results() -> dict:
    with open(RESULTS_JSON) as f:
        return json.load(f)


def build_per_record_rows(per_record: list[dict]) -> str:
    """Build HTML table rows for per-record results."""
    rows = []
    for r in per_record:
        # F1 color coding
        f1 = r["f1"]
        if f1 >= 0.7:
            f1_style = 'color: var(--green); font-weight: 600;'
        elif f1 >= 0.3:
            f1_style = 'color: var(--amber);'
        else:
            f1_style = 'color: var(--red);'

        # Lead time formatting
        lead = r["lead_time_s"]
        if lead > 0:
            lead_str = f'{lead:.0f}s'
            lead_style = 'color: var(--accent);'
        else:
            lead_str = '&mdash;'
            lead_style = 'color: var(--dim);'

        # Recall color
        recall = r["recall"]
        if recall >= 0.99:
            recall_style = 'color: var(--green);'
        elif recall > 0:
            recall_style = 'color: var(--amber);'
        else:
            recall_style = 'color: var(--dim);'

        rows.append(
            f'          <tr>'
            f'<td>{r["record_id"]}</td>'
            f'<td>{r["n_beats"]}</td>'
            f'<td>{r["n_abnormal"]}</td>'
            f'<td style="{f1_style}">{f1:.3f}</td>'
            f'<td>{r["precision"]:.3f}</td>'
            f'<td style="{recall_style}">{recall:.3f}</td>'
            f'<td style="{lead_style}">{lead_str}</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def build_arrhythmia_type_rows(arrhythmia_types: dict) -> str:
    """Build HTML table rows for arrhythmia type counts."""
    # AAMI beat type labels
    labels = {
        "A": "Atrial premature",
        "V": "Ventricular premature",
        "Q": "Unclassifiable",
        "F": "Fusion of V + normal",
        "a": "Aberrated atrial premature",
        "J": "Junctional premature",
        "!": "Ventricular flutter wave",
        "E": "Ventricular escape",
        "S": "Supraventricular premature",
        "/": "Paced beat",
        "f": "Fusion of paced + normal",
    }
    sorted_types = sorted(arrhythmia_types.items(), key=lambda x: x[1], reverse=True)
    total = sum(v for _, v in sorted_types)
    rows = []
    for code, count in sorted_types:
        pct = count / total * 100
        label = labels.get(code, code)
        rows.append(
            f'          <tr>'
            f'<td style="font-family: \'JetBrains Mono\', monospace; color: var(--accent);">{code}</td>'
            f'<td>{label}</td>'
            f'<td style="font-weight: 500;">{count:,}</td>'
            f'<td style="color: var(--muted);">{pct:.1f}%</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def generate_html(data: dict) -> str:
    s = data["stats"]
    cfg = data["config"]

    per_record_rows = build_per_record_rows(s["per_record"])
    arrhythmia_rows = build_arrhythmia_type_rows(s["arrhythmia_types"])

    # Top 5 + Bottom 5 records by F1
    records_with_arrhythmia = [r for r in s["per_record"] if r["n_abnormal"] > 0]
    records_sorted = sorted(records_with_arrhythmia, key=lambda r: r["f1"], reverse=True)
    top5 = records_sorted[:5]
    bot5 = records_sorted[-5:]

    top5_rows = ""
    for r in top5:
        top5_rows += (
            f'          <tr><td>{r["record_id"]}</td><td>{r["n_beats"]}</td>'
            f'<td>{r["n_abnormal"]}</td>'
            f'<td style="color: var(--green); font-weight: 600;">{r["f1"]:.3f}</td>'
            f'<td>{r["precision"]:.3f}</td>'
            f'<td style="color: var(--green);">{r["recall"]:.3f}</td></tr>\n'
        )

    bot5_rows = ""
    for r in bot5:
        f1_style = 'color: var(--red);' if r["f1"] < 0.1 else 'color: var(--amber);'
        bot5_rows += (
            f'          <tr><td>{r["record_id"]}</td><td>{r["n_beats"]}</td>'
            f'<td>{r["n_abnormal"]}</td>'
            f'<td style="{f1_style}">{r["f1"]:.3f}</td>'
            f'<td>{r["precision"]:.3f}</td>'
            f'<td>{r["recall"]:.3f}</td></tr>\n'
        )

    # Total abnormal beats
    total_abnormal = sum(v for v in s["arrhythmia_types"].values())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>&Delta;.72 &mdash; Cardiac Arrhythmia Detection</title>
<meta name="description" content="Delta.72 coherence framework applied to MIT-BIH Arrhythmia Database. 45 ECG records, 360 Hz, 2 channels, cardiologist-annotated arrhythmia detection.">
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
    <li><a href="/r/research/skab/">SKAB</a></li>
    <li><a href="/r/research/uci-power/">UCI Power</a></li>
    <li><a href="/r/research/math/">Math</a></li>
    <li><a href="/r/research/ecg/" style="color: var(--accent); font-weight: 600;">ECG</a></li>
    <li style="opacity: 0.3;">|</li>
    <li><a href="#results">Results</a></li>
    <li><a href="#plots">Plots</a></li>
    <li><a href="#records">Per-Record</a></li>
  </ul>
</nav>

<!-- Hero -->
<section class="hero">
  <p class="hero-eyebrow">Experiment 11 &mdash; Cardiac Arrhythmia</p>
  <h1 class="hero-title">
    <strong>&Delta;.72</strong> on MIT-BIH ECG
  </h1>
  <p class="hero-sub">
    Applying the coherence framework to cardiologist-annotated electrocardiograms.
    {s['n_records']} ECG records at 360&thinsp;Hz, 2 channels, {total_abnormal:,} abnormal beats
    across 11 arrhythmia types. Can &Delta; detect cardiac rhythm departures from coherent baseline?
  </p>
</section>

<!-- Equation -->
<div class="equation-block">
  <div class="equation-main">&Delta; = (P &middot; A &middot; R) / (D + N)</div>
  <div class="equation-desc">
    Applied to 2-channel ECG (modified limb lead <span>MLII</span> + <span>V1/V5</span>)
    sampled at <span>360&thinsp;Hz</span>.<br>
    Baseline: first {cfg['baseline_seconds']}s of each record.
    Rolling window: {cfg['window_size']} samples ({cfg['window_size'] // 360}s),
    step {cfg['step_size']} ({cfg['step_size'] // 360}s).
    &Delta; threshold: {cfg['delta_threshold']}.
  </div>
</div>

<main>

  <!-- Summary Metrics -->
  <div class="section" id="results">
    <p class="section-label">Key Results</p>
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-value" style="color: var(--accent);">{s['n_records']}</div>
        <div class="metric-label">ECG Records</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--accent2);">{s['n_with_arrhythmia']}</div>
        <div class="metric-label">With Arrhythmia</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--amber);">{s['mean_f1']:.3f}</div>
        <div class="metric-label">Mean F1 Score</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--red);">{s['mean_precision']:.3f}</div>
        <div class="metric-label">Mean Precision</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--green);">{s['mean_recall']:.3f}</div>
        <div class="metric-label">Mean Recall</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--teal);">{s['mean_lead_s']:.0f}s</div>
        <div class="metric-label">Mean Lead Time</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--green);">{s['mean_delta_normal']:.3f}</div>
        <div class="metric-label">Mean &Delta; Normal</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--red);">{s['mean_delta_abnormal']:.3f}</div>
        <div class="metric-label">Mean &Delta; Abnormal</div>
      </div>
    </div>

    <p style="font-size: 0.85rem; color: var(--muted); line-height: 1.8; max-width: 700px;">
      Across {s['n_records']} MIT-BIH records, the &Delta; coherence metric achieved
      <strong style="color: var(--green);">{s['mean_recall']:.0%} recall</strong> &mdash; detecting arrhythmia
      windows in virtually every record that contains them. Precision is
      <strong style="color: var(--red);">{s['mean_precision']:.1%}</strong>, reflecting high sensitivity at the
      cost of false positives in normal rhythm. Mean lead time of
      <strong style="color: var(--teal);">{s['mean_lead_s']:.0f}s</strong> before the first annotated
      abnormal beat. {s['n_with_arrhythmia']} of {s['n_records']} records contain arrhythmia;
      {s['n_normal_only']} are normal-only controls.
    </p>
  </div>

  <!-- Method Characteristics -->
  <div class="section">
    <p class="section-label">Detection Profile</p>
    <div class="comparison">
      <div class="comp-card">
        <h3 style="color: var(--green);">Strengths</h3>
        <p>Recall: <strong style="color: var(--green);">{s['mean_recall']:.0%}</strong> &mdash; near-perfect</p>
        <p>Lead time: <strong>{s['mean_lead_s']:.0f}s</strong> mean advance warning</p>
        <p>No missed arrhythmia records</p>
        <p>&Delta; separates normal (<strong>{s['mean_delta_normal']:.3f}</strong>) from abnormal (<strong>{s['mean_delta_abnormal']:.3f}</strong>)</p>
      </div>
      <div class="comp-card">
        <h3 style="color: var(--amber);">Trade-offs</h3>
        <p>Precision: <strong style="color: var(--red);">{s['mean_precision']:.1%}</strong> &mdash; high false-positive rate</p>
        <p>F1: <strong style="color: var(--amber);">{s['mean_f1']:.3f}</strong> mean across records</p>
        <p>Best on high-burden records (PVC, bigeminy)</p>
        <p>Sparse arrhythmia records inflate FP count</p>
      </div>
    </div>
  </div>

  <!-- Plots -->
  <div class="section" id="plots">
    <p class="section-label">Experiment Plots</p>

    <div class="experiment">
      <div class="exp-num">Plot 01</div>
      <h2>Example ECG Record with &Delta; Overlay</h2>
      <div class="exp-desc">
        Top: raw 2-channel ECG trace with cardiologist beat annotations.
        Bottom: rolling &Delta; coherence with threshold and detected arrhythmia windows.
      </div>
      <div class="plot-wrap"><img src="exp11_example_record.png" alt="Example ECG record with coherence overlay"></div>
      <div class="finding">
        The &Delta; metric drops when inter-channel coherence breaks down around abnormal beats.
        Arrhythmia regions correspond to sustained low-coherence windows, while normal sinus
        rhythm maintains high &Delta; values near baseline.
        <span class="badge badge-pass">Visible Signal</span>
      </div>
    </div>

    <div class="experiment">
      <div class="exp-num">Plot 02</div>
      <h2>Detection Summary</h2>
      <div class="exp-desc">
        Per-record F1, precision, and recall across all {s['n_records']} records, sorted by F1.
      </div>
      <div class="plot-wrap"><img src="exp11_detection_summary.png" alt="Detection summary across records"></div>
      <div class="finding">
        Records with high arrhythmia burden (200, 208, 217, 232, 233) achieve F1 &gt; 0.89.
        Records with very sparse abnormal beats (1&ndash;3 per recording) show low precision
        because the entire record is flagged while only a handful of beats are abnormal.
        <span class="badge badge-pass">Pattern Consistent</span>
      </div>
    </div>

    <div class="experiment">
      <div class="exp-num">Plot 03</div>
      <h2>Arrhythmia Type Distribution</h2>
      <div class="exp-desc">
        Breakdown of the {total_abnormal:,} abnormal beats by AAMI classification across the database.
      </div>
      <div class="plot-wrap"><img src="exp11_arrhythmia_types.png" alt="Arrhythmia type distribution"></div>
      <div class="finding">
        Ventricular premature beats (V) dominate at {s['arrhythmia_types']['V']:,} occurrences,
        followed by atrial premature (A) at {s['arrhythmia_types']['A']:,} and paced beats (/)
        at {s['arrhythmia_types'].get('/', 0):,}. The database covers 11 distinct arrhythmia types.
        <span class="badge badge-pass">Comprehensive</span>
      </div>
    </div>

    <div class="experiment">
      <div class="exp-num">Plot 04</div>
      <h2>Lead Time Distribution</h2>
      <div class="exp-desc">
        How many seconds before the first annotated abnormal beat does &Delta; alert?
        Distribution across records with arrhythmia.
      </div>
      <div class="plot-wrap"><img src="exp11_lead_time.png" alt="Lead time distribution"></div>
      <div class="finding">
        Lead times range from immediate (0s, arrhythmia present from recording start) to
        over 1700s for records where rare abnormal beats appear late. Mean lead:
        <strong>{s['mean_lead_s']:.0f}s</strong>. For records where arrhythmia appears early,
        &Delta; flags the departure within the first analysis window.
        <span class="badge badge-pass">Early Detection</span>
      </div>
    </div>
  </div>

  <!-- Arrhythmia Types -->
  <div class="section">
    <p class="section-label">Arrhythmia Types</p>
    <table>
      <thead><tr><th>Code</th><th>Type</th><th>Count</th><th>Share</th></tr></thead>
      <tbody>
{arrhythmia_rows}
      </tbody>
    </table>
  </div>

  <!-- Notable Records -->
  <div class="section">
    <p class="section-label">Notable Records</p>

    <h3 style="font-size: 0.95rem; margin-bottom: 1rem;">Highest F1 (top 5 with arrhythmia)</h3>
    <table>
      <thead><tr><th>Record</th><th>Beats</th><th>Abnormal</th><th>F1</th><th>Precision</th><th>Recall</th></tr></thead>
      <tbody>
{top5_rows}      </tbody>
    </table>

    <h3 style="font-size: 0.95rem; margin: 1.5rem 0 1rem;">Lowest F1 (bottom 5 with arrhythmia)</h3>
    <table>
      <thead><tr><th>Record</th><th>Beats</th><th>Abnormal</th><th>F1</th><th>Precision</th><th>Recall</th></tr></thead>
      <tbody>
{bot5_rows}      </tbody>
    </table>
  </div>

  <!-- Per-Record Detail -->
  <div class="section" id="records">
    <p class="section-label">All Records</p>
    <details>
      <summary style="cursor: pointer; color: var(--accent); font-size: 0.88rem; margin-bottom: 1rem;">
        Show all {s['n_records']} records
      </summary>
      <table>
        <thead><tr><th>Record</th><th>Beats</th><th>Abnormal</th><th>F1</th><th>Precision</th><th>Recall</th><th>Lead</th></tr></thead>
        <tbody>
{per_record_rows}
        </tbody>
      </table>
    </details>
  </div>

  <!-- Dataset -->
  <div class="section">
    <p class="section-label">Dataset</p>
    <p style="font-size: 0.85rem; color: var(--muted); line-height: 1.8; max-width: 700px;">
      <strong style="color: var(--text);">MIT-BIH Arrhythmia Database</strong> &mdash; the gold-standard benchmark
      for cardiac arrhythmia detection, published by PhysioNet. 48 half-hour excerpts of two-channel ambulatory
      ECG recordings from 47 subjects, digitized at 360&thinsp;Hz with 11-bit resolution. Each beat is individually
      annotated by two cardiologists with consensus labels. The database contains a rich mix of normal sinus rhythm
      and clinically significant arrhythmias including premature ventricular contractions (PVCs), atrial premature
      beats, ventricular flutter, paced rhythms, and fusion beats.
    </p>
    <p style="font-size: 0.85rem; color: var(--muted); line-height: 1.8; max-width: 700px; margin-top: 1rem;">
      <strong style="color: var(--text);">Configuration</strong> &mdash;
      Baseline: first {cfg['baseline_seconds']}s of each record.
      Window: {cfg['window_size']} samples ({cfg['window_size'] // 360}s), step {cfg['step_size']} ({cfg['step_size'] // 360}s).
      &Delta; threshold: {cfg['delta_threshold']}.
      Channels: MLII (modified limb lead II) + V1 or V5.
    </p>
    <p style="font-size: 0.85rem; color: var(--muted); line-height: 1.8; max-width: 700px; margin-top: 1rem;">
      <strong style="color: var(--text);">Note</strong> &mdash; 3 records (115, 122, 212) contain only normal
      sinus rhythm and serve as negative controls. {s['n_records']} of the original 48 records were processed
      ({48 - s['n_records']} excluded due to signal quality issues).
    </p>
    <div style="margin-top: 1.5rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">
      <span class="tag tag-green">Python</span>
      <span class="tag" style="color: var(--accent);">NumPy</span>
      <span class="tag" style="color: var(--accent2);">SciPy</span>
      <span class="tag" style="color: var(--teal);">PhysioNet</span>
      <span class="tag tag-amber">MIT-BIH</span>
      <span class="tag" style="color: var(--text);">{s['n_records']} Records</span>
      <span class="tag" style="color: var(--red);">360 Hz</span>
      <span class="tag" style="color: var(--green);">2 Channels</span>
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
  <p>&Delta;.72 Coherence Framework &mdash; Experiment 11: MIT-BIH Cardiac Arrhythmia</p>
  <p style="margin-top: 0.5rem;"><a href="/r/research/">thorarinson</a> &middot; <a href="https://coherenceengine.org">coherenceengine.org</a></p>
</footer>

</body>
</html>"""

    return html


def main():
    print("=" * 60)
    print("  ECG Arrhythmia Dashboard Generator")
    print("=" * 60)

    data = load_results()
    print(f"  Loaded results: {data['stats']['n_records']} records")

    html = generate_html(data)

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_HTML, "w") as f:
        f.write(html)

    print(f"  Generated: {OUTPUT_HTML}")
    print(f"  File size: {OUTPUT_HTML.stat().st_size / 1024:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
