#!/usr/bin/env python3
"""Generate results/credit_card/index.html — self-contained dashboard for Experiment 12.

Matches the exact visual style of the main Delta.72 research page
(/r/research/index.html) — same fonts, colors, layout, component classes.

Usage:
    uv run python experiments/credit_card_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "credit_card"
RESULTS_JSON = RESULTS_DIR / "exp12_results.json"
OUTPUT_HTML = RESULTS_DIR / "index.html"


def load_results() -> dict:
    with open(RESULTS_JSON) as f:
        return json.load(f)


def generate_html(data: dict) -> str:
    s = data["stats"]
    cfg = data["config"]

    top_features = ", ".join(f"<span>{f}</span>" for f in s["top_features"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>&Delta;.72 &mdash; Credit Card Fraud Detection</title>
<meta name="description" content="Delta.72 coherence framework applied to credit card fraud detection. 284,807 transactions, 30 PCA features, 0.17% fraud rate.">
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
    <li><a href="/r/research/ecg/">ECG</a></li>
    <li><a href="/r/research/credit-card/" style="color: var(--accent); font-weight: 600;">Fraud</a></li>
    <li style="opacity: 0.3;">|</li>
    <li><a href="#results">Results</a></li>
    <li><a href="#plots">Plots</a></li>
    <li><a href="#dataset">Dataset</a></li>
  </ul>
</nav>

<!-- Hero -->
<section class="hero">
  <p class="hero-eyebrow">Experiment 12 &mdash; Financial Fraud</p>
  <h1 class="hero-title">
    <strong>&Delta;.72</strong> on Credit Card Transactions
  </h1>
  <p class="hero-sub">
    Applying the coherence framework to real financial fraud detection.
    {s['n_transactions']:,} transactions, {s['n_fraud']} fraudulent ({s['fraud_rate']:.2%} fraud rate),
    30 PCA-transformed features. Can &Delta; separate fraud from legitimate activity
    where variance fails?
  </p>
</section>

<!-- Equation -->
<div class="equation-block">
  <div class="equation-main">&Delta; = (P &middot; A &middot; R) / (D + N)</div>
  <div class="equation-desc">
    Applied to {cfg['n_features_used']} fraud-sensitive features: {top_features}.<br>
    Baseline: first {cfg['baseline_n']:,} transactions. Rolling window: {cfg['window_size']} transactions, step {cfg['step_size']}.
  </div>
</div>

<main>

  <!-- Summary Metrics -->
  <div class="section" id="results">
    <p class="section-label">Key Results</p>
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-value" style="color: var(--accent);">{s['n_transactions']:,}</div>
        <div class="metric-label">Transactions</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--red);">{s['n_fraud']}</div>
        <div class="metric-label">Fraud Cases</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--amber);">{s['fraud_rate']:.2%}</div>
        <div class="metric-label">Fraud Rate</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--green);">{s['coherence_f1']:.3f}</div>
        <div class="metric-label">&Delta; F1 Score</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--accent2);">{s['coherence_precision']:.3f}</div>
        <div class="metric-label">&Delta; Precision</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--teal);">{s['coherence_recall']:.3f}</div>
        <div class="metric-label">&Delta; Recall</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--red);">{s['variance_f1']:.3f}</div>
        <div class="metric-label">Variance F1</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--red);">{s['variance_precision']:.3f}</div>
        <div class="metric-label">Variance Precision</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color: var(--red);">{s['variance_recall']:.3f}</div>
        <div class="metric-label">Variance Recall</div>
      </div>
    </div>

    <p style="font-size: 0.85rem; color: var(--muted); line-height: 1.8; max-width: 700px;">
      Across {s['n_transactions']:,} credit card transactions, the &Delta; coherence metric achieved
      an F1 score of <strong style="color: var(--green);">{s['coherence_f1']:.3f}</strong> with
      <strong style="color: var(--teal);">{s['coherence_recall']:.1%} recall</strong> &mdash; catching the vast majority of
      fraud cases. Variance-based detection effectively failed, achieving only
      <strong style="color: var(--red);">{s['variance_recall']:.1%} recall</strong> and an F1 of {s['variance_f1']:.3f}.
      Mean &Delta; for fraud transactions was <strong style="color: var(--accent);">{s['mean_delta_fraud']:.3f}</strong>
      vs <strong style="color: var(--muted);">{s['mean_delta_legit']:.3f}</strong> for legitimate &mdash; a clear
      separation signal.
    </p>
  </div>

  <!-- Method Comparison -->
  <div class="section">
    <p class="section-label">Method Comparison</p>
    <div class="comparison">
      <div class="comp-card">
        <h3 style="color: var(--accent);">&Delta; Coherence</h3>
        <p>F1 Score: <strong style="color: var(--green);">{s['coherence_f1']:.3f}</strong></p>
        <p>Precision: <strong>{s['coherence_precision']:.3f}</strong></p>
        <p>Recall: <strong style="color: var(--teal);">{s['coherence_recall']:.1%}</strong></p>
        <p>Mean &Delta; (fraud): <strong>{s['mean_delta_fraud']:.3f}</strong></p>
        <p>Mean &Delta; (legit): <strong>{s['mean_delta_legit']:.3f}</strong></p>
      </div>
      <div class="comp-card">
        <h3 style="color: var(--red);">Variance</h3>
        <p>F1 Score: <strong style="color: var(--red);">{s['variance_f1']:.3f}</strong></p>
        <p>Precision: <strong>{s['variance_precision']:.3f}</strong></p>
        <p>Recall: <strong style="color: var(--red);">{s['variance_recall']:.1%}</strong></p>
        <p>Z-score threshold: <strong>{cfg['variance_zscore']}</strong></p>
        <p>Status: <span class="badge badge-fail">Near-zero detection</span></p>
      </div>
    </div>
  </div>

  <!-- Plots -->
  <div class="section" id="plots">
    <p class="section-label">Experiment Plots</p>

    <div class="experiment">
      <div class="exp-num">Plot 01</div>
      <h2>Fraud Detection Overview</h2>
      <div class="exp-desc">
        Transaction-level &Delta; coherence across the full dataset. Fraud transactions highlighted
        against the legitimate baseline. Shows how coherence departs from normal patterns during
        fraudulent activity.
      </div>
      <div class="plot-wrap"><img src="exp12_overview.png" alt="Fraud detection overview with coherence overlay"></div>
      <div class="finding">
        Fraudulent transactions produce a measurably higher &Delta; signal ({s['mean_delta_fraud']:.3f}) compared
        to legitimate transactions ({s['mean_delta_legit']:.3f}). The coherence framework detects structural
        departure from baseline spending patterns without requiring labeled training data.
        <span class="badge badge-pass">Signal Detected</span>
      </div>
    </div>

    <div class="experiment">
      <div class="exp-num">Plot 02</div>
      <h2>Precision&ndash;Recall Analysis</h2>
      <div class="exp-desc">
        Precision-recall trade-off for &Delta; coherence vs variance-based detection across
        different threshold settings. Demonstrates the fundamental advantage of coherence
        in extreme class-imbalance scenarios.
      </div>
      <div class="plot-wrap"><img src="exp12_precision_recall.png" alt="Precision-recall curves"></div>
      <div class="finding">
        &Delta; achieves <strong>{s['coherence_recall']:.1%} recall</strong> at {s['coherence_precision']:.1%} precision &mdash;
        a strong result for unsupervised fraud detection on 0.17% class imbalance.
        Variance collapses entirely, unable to distinguish fraud from normal fluctuation.
        <span class="badge badge-pass">Robust</span>
      </div>
    </div>

    <div class="experiment">
      <div class="exp-num">Plot 03</div>
      <h2>Feature Importance</h2>
      <div class="exp-desc">
        Contribution of each PCA component and the Amount feature to the overall &Delta; coherence
        signal. Ranked by discriminative power between fraud and legitimate transactions.
      </div>
      <div class="plot-wrap"><img src="exp12_feature_importance.png" alt="Feature importance for fraud detection"></div>
      <div class="finding">
        The top contributing features &mdash; <strong>Amount</strong>, <strong>V1</strong>, <strong>V2</strong>,
        <strong>V3</strong> &mdash; align with known fraud indicators in the PCA-transformed space.
        The coherence framework naturally weights features by their structural stability, surfacing
        the most discriminative signals without supervised feature selection.
        <span class="badge badge-pass">Interpretable</span>
      </div>
    </div>
  </div>

  <!-- Dataset -->
  <div class="section" id="dataset">
    <p class="section-label">Dataset</p>
    <p style="font-size: 0.85rem; color: var(--muted); line-height: 1.8; max-width: 700px;">
      <strong style="color: var(--text);">Credit Card Fraud Detection</strong> &mdash; European cardholders,
      September 2013. {s['n_transactions']:,} transactions over two days, of which {s['n_fraud']}
      ({s['fraud_rate']:.2%}) are fraud. Features V1&ndash;V28 are PCA components (original features
      withheld for confidentiality), plus <span style="color: var(--text); font-weight: 500;">Time</span>
      and <span style="color: var(--text); font-weight: 500;">Amount</span>. Extreme class imbalance
      makes this a challenging benchmark for anomaly detection.
    </p>
    <p style="font-size: 0.85rem; color: var(--muted); line-height: 1.8; max-width: 700px; margin-top: 1rem;">
      <strong style="color: var(--text);">Configuration</strong> &mdash;
      Baseline: first {cfg['baseline_n']:,} transactions.
      Window: {cfg['window_size']} transactions, step {cfg['step_size']}.
      &Delta; threshold: {cfg['delta_threshold']}.
      Variance z-score: {cfg['variance_zscore']}.
      Features: {', '.join(s['top_features'])}.
    </p>
    <div style="margin-top: 1.5rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">
      <span class="tag tag-green">Python</span>
      <span class="tag" style="color: var(--accent);">NumPy</span>
      <span class="tag" style="color: var(--accent2);">SciPy</span>
      <span class="tag" style="color: var(--teal);">Kaggle</span>
      <span class="tag tag-amber">PCA</span>
      <span class="tag" style="color: var(--text);">284K Txns</span>
      <span class="tag tag-red">0.17% Fraud</span>
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
  <p>&Delta;.72 Coherence Framework &mdash; Experiment 12: Credit Card Fraud Detection</p>
  <p style="margin-top: 0.5rem;"><a href="/r/research/">thorarinson</a> &middot; <a href="https://coherenceengine.org">coherenceengine.org</a></p>
</footer>

</body>
</html>"""

    return html


def main():
    print("=" * 60)
    print("  Credit Card Fraud Dashboard Generator")
    print("=" * 60)

    data = load_results()
    print(f"  Loaded results: {data['stats']['n_transactions']:,} transactions, {data['stats']['n_fraud']} fraud")

    html = generate_html(data)

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_HTML, "w") as f:
        f.write(html)

    print(f"  Generated: {OUTPUT_HTML}")
    print(f"  File size: {OUTPUT_HTML.stat().st_size / 1024:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
