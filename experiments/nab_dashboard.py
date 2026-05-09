#!/usr/bin/env python3
"""Generate self-contained HTML dashboard for NAB benchmark results.

Matches the visual style of the main research page at
jthorvaldur.github.io/r/research/index.html (Outfit + JetBrains Mono,
dark theme, card layout, nav bar).

Usage:
    uv run python experiments/nab_dashboard.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "nab"
OUTPUT_PATH = RESULTS_DIR / "index.html"


def img_to_base64(path: Path) -> str:
    """Encode image file to base64 data URI."""
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"


def pct(v: float) -> str:
    """Format float as percentage string."""
    return f"{v * 100:.1f}%"


def f3(v: float) -> str:
    """Format float to 3 decimal places."""
    return f"{v:.3f}"


def badge(value: float, thresholds: tuple[float, float] = (0.3, 0.5)) -> str:
    """Return a badge HTML span based on value."""
    low, high = thresholds
    if value >= high:
        return f'<span class="badge badge-pass">{f3(value)}</span>'
    elif value >= low:
        return f'<span class="badge badge-warn">{f3(value)}</span>'
    else:
        return f'<span class="badge badge-fail">{f3(value)}</span>'


def generate_dashboard():
    results_path = RESULTS_DIR / "nab_results.json"
    if not results_path.exists():
        print(f"No results found at {results_path}. Run exp_nab.py first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    overall = results.get("overall", {})
    categories = results.get("categories", {})
    config = results.get("config", {})
    elapsed = results.get("elapsed_s", 0)

    # Encode plot images
    img_f1_cat = img_to_base64(RESULTS_DIR / "nab_f1_by_category.png")
    img_pr = img_to_base64(RESULTS_DIR / "nab_precision_recall.png")
    img_examples = img_to_base64(RESULTS_DIR / "nab_example_timeseries.png")
    img_summary = img_to_base64(RESULTS_DIR / "nab_summary.png")

    # --- Category rows ---
    cat_rows = ""
    for cat, data in categories.items():
        cat_display = cat.replace("artificial", "artificial ").replace("real", "real ")
        cf1 = data["coherence_f1"]
        vf1 = data["variance_f1"]
        winner = "coherence" if cf1 >= vf1 else "variance"
        cat_rows += f"""
        <tr>
          <td>{cat_display}</td>
          <td>{data['n_files']}</td>
          <td style="font-weight:600;">{badge(data['coherence_precision'])}</td>
          <td style="font-weight:600;">{badge(data['coherence_recall'])}</td>
          <td style="font-weight:600;">{badge(cf1)}</td>
          <td>{badge(data['variance_precision'])}</td>
          <td>{badge(data['variance_recall'])}</td>
          <td>{badge(vf1)}</td>
          <td><span class="badge {'badge-pass' if winner == 'coherence' else 'badge-fail'}">{winner}</span></td>
        </tr>"""

    # --- Per-file detail rows ---
    file_detail_rows = ""
    for cat, data in categories.items():
        for pf in data.get("per_file", []):
            cf1 = pf["coherence_f1"]
            vf1 = pf["variance_f1"]
            file_detail_rows += f"""
        <tr>
          <td style="color:var(--muted); font-size:0.72rem;">{cat}</td>
          <td>{pf['filename'].replace('.csv','')}</td>
          <td>{pf['n_points']:,}</td>
          <td>{pf['n_anomaly_windows']}</td>
          <td style="font-weight:600;">{badge(pf['coherence_precision'])}</td>
          <td style="font-weight:600;">{badge(pf['coherence_recall'])}</td>
          <td style="font-weight:600;">{badge(cf1)}</td>
          <td>{badge(vf1)}</td>
        </tr>"""

    # Coherence wins count
    total_files = overall.get("n_files", 0)
    coh_wins = 0
    var_wins = 0
    ties = 0
    for cat, data in categories.items():
        for pf in data.get("per_file", []):
            if pf["coherence_f1"] > pf["variance_f1"]:
                coh_wins += 1
            elif pf["coherence_f1"] < pf["variance_f1"]:
                var_wins += 1
            else:
                ties += 1

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>&Delta;.72 &mdash; NAB Benchmark Results</title>
<meta name="description" content="Delta.72 coherence framework benchmarked against {total_files} Numenta Anomaly Benchmark time series. Precision, recall, F1 comparison vs variance-based detection.">
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
    color: var(--muted); max-width: 600px; margin-left: auto; margin-right: auto;
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
    <li><a href="/r/research/">Exps 1&ndash;7</a></li>
    <li><a href="/r/research/nab/" style="color: var(--accent); font-weight: 600;">NAB</a></li>
    <li><a href="/r/research/nasa/">NASA</a></li>
    <li><a href="/r/research/skab/">SKAB</a></li>
    <li style="opacity: 0.3;">|</li>
    <li><a href="#results">Results</a></li>
    <li><a href="#categories">Categories</a></li>
    <li><a href="#examples">Examples</a></li>
    <li><a href="#detail">Detail</a></li>
  </ul>
</nav>

<!-- Hero -->
<section class="hero">
  <p class="hero-eyebrow">Experiment 08 &mdash; NAB Benchmark</p>
  <h1 class="hero-title">
    <strong>&Delta;.72</strong> vs Numenta Anomaly Benchmark
  </h1>
  <p class="hero-sub">
    Validating the coherence framework against {total_files} time series across {len(categories)}
    NAB categories. Point-level precision, recall, and F1 comparison against a variance-based detector.
  </p>
</section>

<!-- Equation -->
<div class="equation-block">
  <div class="equation-main">&Delta; = (P &middot; A &middot; R) / (D + N)</div>
  <div class="equation-desc">
    Coherence-based anomaly detection: ALERT when &Delta; &lt; {config.get('delta_threshold', 0.3)},
    <span>M</span> &lt; {config.get('memory_threshold', 0.4)},
    <span>W</span> &lt; {config.get('recovery_threshold', 0.4)}<br>
    Rolling window: {config.get('window_size', 168)} points, step {config.get('step_size', 24)}
  </div>
</div>

<main>

  <!-- Summary Metrics -->
  <div class="section" id="results">
    <p class="section-label">Key Results</p>
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-value" style="color: var(--accent);">{f3(overall.get('coherence_f1', 0))}</div>
        <div class="metric-label">&Delta; F1 Score</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--red);">{f3(overall.get('variance_f1', 0))}</div>
        <div class="metric-label">Variance F1 Score</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--green);">{f3(overall.get('coherence_precision', 0))}</div>
        <div class="metric-label">&Delta; Precision</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--teal);">{f3(overall.get('coherence_recall', 0))}</div>
        <div class="metric-label">&Delta; Recall</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--accent2);">{total_files}</div>
        <div class="metric-label">Time Series</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--amber);">{coh_wins}/{total_files}</div>
        <div class="metric-label">&Delta; Wins (F1)</div>
      </div>
    </div>

    <div class="comparison">
      <div class="comp-card">
        <h3 style="color: var(--accent);">&Delta; Coherence</h3>
        <p>Precision: <strong>{f3(overall.get('coherence_precision', 0))}</strong></p>
        <p>Recall: <strong>{f3(overall.get('coherence_recall', 0))}</strong></p>
        <p>F1: <strong style="color: var(--green);">{f3(overall.get('coherence_f1', 0))}</strong></p>
        <p>Wins on <strong>{coh_wins}</strong> of {total_files} files</p>
      </div>
      <div class="comp-card">
        <h3 style="color: var(--red);">Variance Detector</h3>
        <p>Precision: <strong>{f3(overall.get('variance_precision', 0))}</strong></p>
        <p>Recall: <strong>{f3(overall.get('variance_recall', 0))}</strong></p>
        <p>F1: <strong style="color: var(--red);">{f3(overall.get('variance_f1', 0))}</strong></p>
        <p>Wins on <strong>{var_wins}</strong> of {total_files} files</p>
      </div>
    </div>
  </div>

  <!-- Overall Summary Plot -->
  <div class="section">
    <p class="section-label">Overall Performance</p>
    <div class="experiment">
      <div class="exp-num">Experiment 08</div>
      <h2>NAB Benchmark &mdash; Overall Summary</h2>
      <div class="exp-desc">
        Macro-averaged precision, recall, and F1 across all {total_files} NAB time series.
        The coherence framework uses gated detection (&Delta; + M + W), while the variance
        detector flags windows where rolling variance exceeds {config.get('var_zscore', 2.5)}x the global variance.
      </div>
      {"<div class='plot-wrap'><img src='" + img_summary + "' alt='NAB Summary'></div>" if img_summary else ""}
    </div>
  </div>

  <!-- Category Breakdown -->
  <div class="section" id="categories">
    <p class="section-label">Results by Category</p>

    <div class="experiment">
      <h2>F1 Score Comparison by NAB Category</h2>
      <div class="exp-desc">
        Macro-averaged F1 per category. Each category contains real or synthetic time series
        with different anomaly characteristics.
      </div>
      {"<div class='plot-wrap'><img src='" + img_f1_cat + "' alt='F1 by Category'></div>" if img_f1_cat else ""}

      <table>
        <thead>
          <tr>
            <th>Category</th><th>Files</th>
            <th>Coh. P</th><th>Coh. R</th><th>Coh. F1</th>
            <th>Var. P</th><th>Var. R</th><th>Var. F1</th>
            <th>Winner</th>
          </tr>
        </thead>
        <tbody>{cat_rows}</tbody>
      </table>
    </div>

    <div class="experiment">
      <h2>Precision &amp; Recall Breakdown</h2>
      <div class="exp-desc">
        Side-by-side precision and recall by category. Coherence detectors tend toward
        higher recall (fewer missed anomalies) at the cost of precision.
      </div>
      {"<div class='plot-wrap'><img src='" + img_pr + "' alt='Precision & Recall'></div>" if img_pr else ""}
    </div>
  </div>

  <!-- Example Time Series -->
  <div class="section" id="examples">
    <p class="section-label">Example Detections</p>

    <div class="experiment">
      <h2>Representative Time Series with Alerts vs Labels</h2>
      <div class="exp-desc">
        Selected examples from different categories showing coherence alerts (blue shading)
        vs NAB ground-truth anomaly windows (red shading). Signal in blue, rolling-mean baseline in white.
      </div>
      {"<div class='plot-wrap'><img src='" + img_examples + "' alt='Example Time Series'></div>" if img_examples else ""}
    </div>
  </div>

  <!-- Full Per-File Detail -->
  <div class="section" id="detail">
    <p class="section-label">Full Per-File Results</p>

    <div class="experiment">
      <h2>All {total_files} Time Series</h2>
      <div class="exp-desc">
        Point-level precision, recall, and F1 for each NAB file. Sorted by category.
      </div>
      <div style="overflow-x: auto;">
        <table>
          <thead>
            <tr>
              <th>Category</th><th>File</th><th>Points</th><th>Anomalies</th>
              <th>Coh. P</th><th>Coh. R</th><th>Coh. F1</th><th>Var. F1</th>
            </tr>
          </thead>
          <tbody>{file_detail_rows}</tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Configuration -->
  <div class="section">
    <p class="section-label">Configuration</p>
    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem;">
      <span class="tag" style="color: var(--accent);">window={config.get('window_size', 168)}</span>
      <span class="tag" style="color: var(--accent2);">step={config.get('step_size', 24)}</span>
      <span class="tag" style="color: var(--green);">&Delta;&lt;{config.get('delta_threshold', 0.3)}</span>
      <span class="tag" style="color: var(--amber);">M&lt;{config.get('memory_threshold', 0.4)}</span>
      <span class="tag" style="color: var(--teal);">W&lt;{config.get('recovery_threshold', 0.4)}</span>
      <span class="tag" style="color: var(--red);">var_z={config.get('var_zscore', 2.5)}</span>
      <span class="tag" style="color: var(--muted);">{elapsed:.1f}s runtime</span>
    </div>
    <p style="font-size: 0.85rem; color: var(--muted); line-height: 1.8; max-width: 700px;">
      The coherence detector uses the full gated alert: &Delta; &lt; threshold AND
      M (Memory-of-Attractor) &lt; threshold AND W (Windowed Recovery) &lt; threshold.
      Baseline is a {config.get('baseline_window', 168)}-point centered rolling mean.
      The variance comparator flags windows where rolling variance exceeds
      {config.get('var_zscore', 2.5)}x the global residual variance.
    </p>
  </div>

</main>

<footer>
  <p>&Delta;.72 Coherence Framework &mdash; NAB Benchmark (Experiment 08)</p>
  <p style="margin-top: 0.5rem;"><a href="/r/research/">Back to main results</a> &middot; <a href="/">thorarinson</a></p>
</footer>

</body>
</html>"""

    OUTPUT_PATH.write_text(html)
    print(f"Dashboard written to {OUTPUT_PATH}")
    print(f"  ({len(html):,} bytes, self-contained with embedded images)")


if __name__ == "__main__":
    generate_dashboard()
