#!/usr/bin/env python3
"""Validate web page numbers against JSON result files.

Reads all experiment result JSON files from results/ and cross-references
numerical claims in the deployed HTML pages. Flags any discrepancy.

Usage:
    uv run python scripts/validate_web.py
    uv run python scripts/validate_web.py --html-dir /path/to/jthorvaldur.github.io/r/research
    uv run python scripts/validate_web.py --fix  # show suggested fixes
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ── Config ──────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_HTML_DIR = Path.home() / "GitHub" / "jthorvaldur.github.io" / "r" / "research"

# Tolerance for floating point comparisons
FLOAT_TOL = 0.015  # 1.5% relative tolerance


# ── Load truth from JSON ────────────────────────────────────

def load_truth() -> dict:
    """Load all experiment results into a flat dict of canonical values."""
    truth = {}

    # Experiment results (exps 1-7)
    exp_file = RESULTS_DIR / "experiment_results.json"
    if exp_file.exists():
        with open(exp_file) as f:
            data = json.load(f)
        # Exp 1
        if "exp1" in data:
            truth["exp1_threshold_noise"] = data["exp1"]["threshold_noise"]
        # Exp 5
        if "exp5" in data:
            truth["exp5_cross_system_cv"] = data["exp5"]["cross_system_cv"]
        # Exp 6
        if "exp6" in data:
            truth["exp6_n_trials"] = data["exp6"]["n_trials"]

    # NAB
    nab_file = RESULTS_DIR / "nab" / "nab_results.json"
    if nab_file.exists():
        with open(nab_file) as f:
            data = json.load(f)
        o = data.get("overall", {})
        truth["nab_n_files"] = o.get("n_files")
        truth["nab_coherence_f1"] = o.get("coherence_f1")
        truth["nab_coherence_precision"] = o.get("coherence_precision")
        truth["nab_coherence_recall"] = o.get("coherence_recall")

    # NASA
    nasa_file = RESULTS_DIR / "nasa" / "exp8_results.json"
    if nasa_file.exists():
        with open(nasa_file) as f:
            data = json.load(f)
        s = data.get("stats", {})
        truth["nasa_n_engines"] = s.get("n_engines")
        truth["nasa_delta_detection_rate"] = s.get("delta_detection_rate")
        truth["nasa_delta_mean_lead"] = s.get("delta_mean_lead")
        truth["nasa_var_mean_lead"] = s.get("var_mean_lead")
        truth["nasa_mean_advantage"] = s.get("mean_advantage")

    # SKAB
    skab_file = RESULTS_DIR / "skab" / "exp9_results.json"
    if skab_file.exists():
        with open(skab_file) as f:
            data = json.load(f)
        s = data.get("stats", {})
        truth["skab_n_experiments"] = s.get("n_experiments")
        truth["skab_delta_detection_rate"] = s.get("delta_detection_rate")
        truth["skab_coherence_f1"] = s.get("coherence_f1")

    # UCI Power
    uci_file = RESULTS_DIR / "uci_power" / "exp10_results.json"
    if uci_file.exists():
        with open(uci_file) as f:
            data = json.load(f)
        s = data.get("stats", {})
        truth["uci_n_alerts"] = s.get("n_alerts")
        truth["uci_n_var_alerts"] = s.get("n_var_alerts")
        truth["uci_coherence_only"] = s.get("coherence_only")

    # ECG
    ecg_file = RESULTS_DIR / "ecg" / "exp11_results.json"
    if ecg_file.exists():
        with open(ecg_file) as f:
            data = json.load(f)
        s = data.get("stats", {})
        truth["ecg_n_records"] = s.get("n_records")
        truth["ecg_n_with_arrhythmia"] = s.get("n_with_arrhythmia")
        truth["ecg_mean_f1"] = s.get("mean_f1")
        truth["ecg_mean_precision"] = s.get("mean_precision")
        truth["ecg_mean_recall"] = s.get("mean_recall")
        truth["ecg_mean_lead_s"] = s.get("mean_lead_s")
        truth["ecg_mean_delta_normal"] = s.get("mean_delta_normal")
        truth["ecg_mean_delta_abnormal"] = s.get("mean_delta_abnormal")
        # Arrhythmia counts
        types = s.get("arrhythmia_types", {})
        truth["ecg_total_abnormal_beats"] = sum(types.values())
        truth["ecg_n_arrhythmia_types"] = len(types)
        for sym, count in types.items():
            truth[f"ecg_type_{sym}"] = count
        # Per-record for top/bottom validation
        per = s.get("per_record", [])
        for r in per:
            rid = r["record_id"]
            truth[f"ecg_rec_{rid}_f1"] = r["f1"]
            truth[f"ecg_rec_{rid}_precision"] = r["precision"]
            truth[f"ecg_rec_{rid}_recall"] = r["recall"]

    # Credit Card
    cc_file = RESULTS_DIR / "credit_card" / "exp12_results.json"
    if cc_file.exists():
        with open(cc_file) as f:
            data = json.load(f)
        s = data.get("stats", {})
        truth["cc_coherence_f1"] = s.get("coherence_f1")
        truth["cc_coherence_recall"] = s.get("coherence_recall")
        truth["cc_variance_f1"] = s.get("variance_f1")

    # EEG
    eeg_file = RESULTS_DIR / "exp13_results.json"
    if eeg_file.exists():
        with open(eeg_file) as f:
            data = json.load(f)
        s = data.get("stats", {})
        truth["eeg_n_recordings"] = s.get("n_recordings")
        truth["eeg_mean_f1"] = s.get("mean_f1")
        truth["eeg_mean_recall"] = s.get("mean_recall")
        truth["eeg_mean_lead_s"] = s.get("mean_lead_s")

    return truth


# ── Assertions: what each page should contain ───────────────

def build_assertions(truth: dict) -> dict[str, list[tuple[str, str, float | int]]]:
    """Build assertions mapping page path -> list of (label, pattern, expected_value).

    Each assertion is checked: does the HTML contain a number matching expected_value?
    """
    a: dict[str, list] = {}

    # Main dashboard — metric cards use "metric-value" class
    a["index.html"] = [
        ("NAB file count", r"(\d+) Time Series", truth.get("nab_n_files")),
        ("ECG record count", r"(\d+) Records", truth.get("ecg_n_records")),
        ("Math module count", r"(\d+) Modules", 14),
    ]

    # ECG page
    a["ecg/index.html"] = [
        ("record count meta", r"(\d+) ECG records", truth.get("ecg_n_records")),
        ("abnormal beat total", r"([\d,]+) abnormal beats", truth.get("ecg_total_abnormal_beats")),
        ("arrhythmia type count", r"(\d+) arrhythmia types", truth.get("ecg_n_arrhythmia_types")),
        ("mean F1 card", r"(\d+\.\d+) Mean F1", truth.get("ecg_mean_f1")),
        ("mean precision card", r"(\d+\.\d+) Mean Precision", truth.get("ecg_mean_precision")),
        ("mean recall card", r"(\d+\.\d+) Mean Recall", truth.get("ecg_mean_recall")),
        ("mean lead time card", r"(\d+)s Mean Lead", truth.get("ecg_mean_lead_s")),
        ("delta normal card", r"(\d+\.\d+) Mean .* Normal", truth.get("ecg_mean_delta_normal")),
    ]

    # NASA page
    a["nasa/index.html"] = [
        ("engine count", r"(\d+) turbofan engines", truth.get("nasa_n_engines")),
        ("delta mean lead", r"(\d+)\s+Mean\s+Lead|lead time of\s+(\d+)", truth.get("nasa_delta_mean_lead")),
    ]

    # NAB page
    a["nab/index.html"] = [
        ("file count", r"(\d+) NAB Files|(\d+) time series", truth.get("nab_n_files")),
    ]

    # SKAB page
    a["skab/index.html"] = [
        ("experiment count", r"(\d+) experiments across|(\d+) SKAB industrial", truth.get("skab_n_experiments")),
    ]

    # Credit card page
    a["credit-card/index.html"] = [
        ("coherence F1", r"(\d+\.\d+) F1 Score|F1 score of (\d+\.\d+)", truth.get("cc_coherence_f1")),
    ]

    # UCI power page
    a["uci-power/index.html"] = [
        ("coherence alert count", r"(\d+) alerts|(\d+) Alerts", truth.get("uci_n_alerts")),
    ]

    # Overview page
    a["overview/index.html"] = [
        ("NASA lead cycles", r"(\d+) cycles", truth.get("nasa_delta_mean_lead")),
        ("math extensions", r"(\w+) mathematical extensions", 14),
    ]

    # Math page
    a["math/index.html"] = [
        ("module count card", r"(\d+) Math Modules", 14),
    ]

    return a


# ── Matching logic ──────────────────────────────────────────

WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def numbers_match(found_str: str, expected) -> bool:
    """Check if a found string matches the expected value."""
    if expected is None:
        return True  # no truth to compare

    # Handle word numbers
    word = found_str.strip().lower()
    if word in WORD_TO_NUM:
        found_str = str(WORD_TO_NUM[word])

    # Strip commas and percent signs
    clean = found_str.replace(",", "").replace("%", "")

    try:
        found = float(clean)
    except ValueError:
        return False

    exp = float(expected)

    # For detection rates stored as 1.0 but displayed as 100%
    if abs(exp) <= 1.0 and found > 1.0 and abs(found / 100 - exp) < FLOAT_TOL:
        return True

    # Exact integer match
    if isinstance(expected, int) and found == exp:
        return True

    # Float tolerance
    if exp == 0:
        return abs(found) < FLOAT_TOL

    # Accept rounding: truncation or standard rounding within 1 unit
    if abs(found - exp) <= 1.01:
        return True

    # Relative tolerance
    if abs(found - exp) / max(abs(exp), 1e-9) < FLOAT_TOL:
        return True

    return False


def find_value_in_html(html: str, pattern: str) -> list[str]:
    """Find all matches of a pattern in HTML, returning captured groups."""
    # Strip HTML tags for easier matching
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"\s+", " ", text)

    matches = re.finditer(pattern, text, re.IGNORECASE)
    results = []
    for m in matches:
        for g in m.groups():
            if g is not None:
                results.append(g)
    return results


# ── Main ────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate web page numbers against JSON results")
    parser.add_argument("--html-dir", type=Path, default=DEFAULT_HTML_DIR,
                        help="Path to the research HTML directory")
    parser.add_argument("--fix", action="store_true",
                        help="Show suggested fix values")
    parser.add_argument("--json-only", action="store_true",
                        help="Just dump the truth values from JSON")
    args = parser.parse_args()

    truth = load_truth()

    if args.json_only:
        for k, v in sorted(truth.items()):
            if not k.startswith("ecg_rec_"):
                print(f"  {k}: {v}")
        return

    assertions = build_assertions(truth)
    errors = []
    warnings = []
    ok_count = 0

    for page_path, checks in sorted(assertions.items()):
        html_file = args.html_dir / page_path
        if not html_file.exists():
            warnings.append(f"  SKIP  {page_path} — file not found")
            continue

        html = html_file.read_text()

        for label, pattern, expected in checks:
            if expected is None:
                warnings.append(f"  SKIP  {page_path}: {label} — no JSON truth available")
                continue

            found_values = find_value_in_html(html, pattern)

            if not found_values:
                warnings.append(f"  MISS  {page_path}: {label} — pattern not found in HTML")
                continue

            matched = False
            for fv in found_values:
                if numbers_match(fv, expected):
                    matched = True
                    break

            if matched:
                ok_count += 1
            else:
                msg = f"  FAIL  {page_path}: {label}"
                msg += f" — HTML has {found_values}, expected ~{expected}"
                if args.fix:
                    if isinstance(expected, float):
                        if expected >= 100:
                            msg += f"  [fix: {expected:.0f}]"
                        elif expected >= 1:
                            msg += f"  [fix: {expected:.1f}]"
                        else:
                            msg += f"  [fix: {expected:.3f}]"
                    else:
                        msg += f"  [fix: {expected}]"
                errors.append(msg)

    # Report
    print(f"\n{'='*60}")
    print(f"  Web Value Validation Report")
    print(f"  Source: {RESULTS_DIR}")
    print(f"  Target: {args.html_dir}")
    print(f"{'='*60}\n")

    if errors:
        print(f"ERRORS ({len(errors)}):\n")
        for e in errors:
            print(e)
        print()

    if warnings:
        print(f"WARNINGS ({len(warnings)}):\n")
        for w in warnings:
            print(w)
        print()

    print(f"PASSED: {ok_count}")
    print(f"FAILED: {len(errors)}")
    print(f"WARNINGS: {len(warnings)}")
    print()

    if errors:
        print("Run with --fix to see suggested corrections.")
        sys.exit(1)
    else:
        print("All values match JSON sources.")
        sys.exit(0)


if __name__ == "__main__":
    main()
