#!/usr/bin/env python3
"""Experiment 13 — CHB-MIT EEG Seizure Detection (GPU).

Uses direct HTTP download of EDF files + pyedflib to read them.
"""
import json, sys, time, os, urllib.request, tempfile
sys.path.insert(0, os.path.expanduser("~/delta72"))
import numpy as np
from engine import coherence_score

try:
    import pyedflib
except ImportError:
    os.system("pip install -q pyedflib")
    import pyedflib

OUTPUT_DIR = os.path.expanduser("~/results/eeg")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = "https://physionet.org/files/chbmit/1.0.0"
SUBJECTS = ["chb01", "chb02", "chb03", "chb05", "chb08"]
WINDOW_SEC = 10
STEP_SEC = 5
BASELINE_SEC = 120
DELTA_THRESHOLD = 0.3
N_CHANNELS = 6
MAX_DURATION_SEC = 1800  # cap at 30 min per recording for speed


def normalize(v):
    vmin, vmax = v.min(), v.max()
    return (v - vmin) / (vmax - vmin) if vmax - vmin > 1e-12 else np.zeros_like(v)


def get_seizure_annotations(subject):
    url = f"{BASE_URL}/{subject}/{subject}-summary.txt"
    try:
        resp = urllib.request.urlopen(url, timeout=30)
        text = resp.read().decode()
    except Exception:
        return {}
    seizures = {}
    current_file = None
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("File Name:"):
            current_file = line.split(":", 1)[1].strip()
        elif "Seizure" in line and "Start" in line:
            try:
                val = line.split(":", 1)[1].strip().split()[0]
                seizures.setdefault(current_file, []).append({"start": int(val)})
            except (ValueError, IndexError):
                pass
        elif "Seizure" in line and "End" in line:
            try:
                val = line.split(":", 1)[1].strip().split()[0]
                if current_file and seizures.get(current_file):
                    seizures[current_file][-1]["end"] = int(val)
            except (ValueError, IndexError):
                pass
    return seizures


def download_edf(subject, filename):
    """Download EDF file to temp dir, return path."""
    url = f"{BASE_URL}/{subject}/{filename}"
    tmp = os.path.join(tempfile.gettempdir(), f"chbmit_{subject}_{filename}")
    if not os.path.exists(tmp):
        print(f"      Downloading {filename}...", end="", flush=True)
        urllib.request.urlretrieve(url, tmp)
        size_mb = os.path.getsize(tmp) / 1e6
        print(f" {size_mb:.0f}MB", flush=True)
    return tmp


def analyze_edf(subject, filename, seizure_times):
    """Download and analyze a single EDF recording."""
    try:
        edf_path = download_edf(subject, filename)
    except Exception as e:
        print(f"      Download failed: {e}", flush=True)
        return None

    try:
        f = pyedflib.EdfReader(edf_path)
    except Exception as e:
        print(f"      Read failed: {e}", flush=True)
        return None

    n_sig = min(N_CHANNELS, f.signals_in_file)
    fs = int(f.getSampleFrequency(0))
    n_samples = f.getNSamples()[0]
    duration = n_samples / fs

    # Cap duration
    max_samples = int(MAX_DURATION_SEC * fs)
    n_use = min(n_samples, max_samples)

    window = int(WINDOW_SEC * fs)
    step = int(STEP_SEC * fs)
    baseline_end = int(BASELINE_SEC * fs)

    if n_use < baseline_end + window:
        f.close()
        return None

    # Read channels
    signals = np.zeros((n_use, n_sig))
    for ch in range(n_sig):
        raw = f.readSignal(ch)
        signals[:, ch] = normalize(raw[:n_use])
    f.close()

    # Baselines
    baselines = [signals[:baseline_end, ch].copy() for ch in range(n_sig)]

    # Seizure mask
    sz_mask = np.zeros(n_use, dtype=bool)
    for sz in seizure_times:
        s = int(sz["start"] * fs)
        e = int(sz.get("end", sz["start"] + 60) * fs)
        sz_mask[max(0, s):min(n_use, e)] = True

    # Rolling coherence
    deltas = []
    for start in range(0, n_use - window, step):
        end = start + window
        ch_deltas = []
        for ch in range(n_sig):
            sig_win = signals[start:end, ch]
            base_win = np.tile(baselines[ch], (window // len(baselines[ch])) + 1)[:window]
            scores = coherence_score(sig_win, base_win)
            ch_deltas.append(scores["delta"])
        mid = start + window // 2
        deltas.append({
            "time_s": mid / fs,
            "delta": float(np.mean(ch_deltas)),
            "is_seizure": bool(sz_mask[start:end].any()),
        })

    # Normalize
    d_arr = np.array([d["delta"] for d in deltas])
    if len(d_arr) > 5 and d_arr[:5].mean() > 0:
        d_norm = np.clip(d_arr / d_arr[:5].mean(), 0, 2.0)
    else:
        d_norm = d_arr
    for i, dn in enumerate(d_norm):
        deltas[i]["delta_norm"] = float(dn)

    pred = np.array([d["delta_norm"] < DELTA_THRESHOLD for d in deltas])
    truth = np.array([d["is_seizure"] for d in deltas])

    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    first_sz = next((i for i, d in enumerate(deltas) if d["is_seizure"]), None)
    first_alert = next((i for i, d in enumerate(deltas) if d["delta_norm"] < DELTA_THRESHOLD), None)
    lead = 0
    if first_sz and first_alert and first_alert < first_sz:
        lead = deltas[first_sz]["time_s"] - deltas[first_alert]["time_s"]

    # Cleanup temp file
    try:
        os.remove(edf_path)
    except Exception:
        pass

    return {
        "subject": subject,
        "filename": filename,
        "duration_s": round(duration, 0),
        "n_channels": n_sig,
        "fs": fs,
        "has_seizure": bool(sz_mask.any()),
        "n_seizures": len(seizure_times),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "lead_time_s": round(lead, 1),
        "n_windows": len(deltas),
        "n_alerts": int(pred.sum()),
    }


def main():
    print("=" * 60, flush=True)
    print("  Experiment 13 — CHB-MIT EEG Seizure Detection", flush=True)
    print("=" * 60, flush=True)
    t0 = time.time()

    all_results = []

    for subject in SUBJECTS:
        print(f"\n  Subject: {subject}", flush=True)
        seizures = get_seizure_annotations(subject)
        sz_files = list(seizures.keys())
        print(f"    Seizure files: {sz_files[:5]}", flush=True)

        # Get all EDF files for this subject
        url = f"{BASE_URL}/{subject}/RECORDS"
        try:
            resp = urllib.request.urlopen(url, timeout=30)
            all_files = [r.strip().split("/")[-1] for r in resp.read().decode().split("\n") if r.strip()]
        except Exception:
            # Fallback: construct filenames
            all_files = [f"{subject}_{i:02d}.edf" for i in range(1, 20)]

        normal_files = [f for f in all_files if f not in sz_files]

        # Analyze: up to 2 seizure files + 1 normal per subject
        analyzed = 0
        for edf_name in sz_files[:2] + normal_files[:1]:
            if analyzed >= 3:
                break
            sz_times = seizures.get(edf_name, [])
            print(f"    {edf_name} (sz={len(sz_times)})...", flush=True)

            result = analyze_edf(subject, edf_name, sz_times)
            if result:
                all_results.append(result)
                analyzed += 1
                status = f"SZ={result['n_seizures']}" if result["has_seizure"] else "normal"
                print(f"      {result['duration_s']:.0f}s, {status}, F1={result['f1']:.3f}, lead={result['lead_time_s']:.0f}s", flush=True)

    sz_recs = [r for r in all_results if r["has_seizure"]]
    normal_recs = [r for r in all_results if not r["has_seizure"]]

    mean_f1 = np.mean([r["f1"] for r in sz_recs]) if sz_recs else 0
    mean_rec = np.mean([r["recall"] for r in sz_recs]) if sz_recs else 0
    mean_prec = np.mean([r["precision"] for r in sz_recs]) if sz_recs else 0
    leads = [r["lead_time_s"] for r in sz_recs if r["lead_time_s"] > 0]
    mean_lead = np.mean(leads) if leads else 0

    elapsed = time.time() - t0

    print(f"\n{'='*60}", flush=True)
    print(f"  {len(all_results)} recordings ({len(sz_recs)} seizure, {len(normal_recs)} normal)", flush=True)
    print(f"  F1={mean_f1:.3f} P={mean_prec:.3f} R={mean_rec:.3f} lead={mean_lead:.1f}s", flush=True)
    print(f"  Elapsed: {elapsed:.1f}s", flush=True)

    result = {
        "experiment": 13,
        "name": "CHB-MIT EEG Seizure Detection",
        "dataset": f"CHB-MIT — {len(SUBJECTS)} subjects, 256 Hz, {N_CHANNELS} channels",
        "config": {"window_sec": WINDOW_SEC, "step_sec": STEP_SEC, "baseline_sec": BASELINE_SEC, "delta_threshold": DELTA_THRESHOLD},
        "stats": {
            "n_recordings": len(all_results),
            "n_seizure": len(sz_recs),
            "n_normal": len(normal_recs),
            "mean_f1": round(mean_f1, 3),
            "mean_precision": round(mean_prec, 3),
            "mean_recall": round(mean_rec, 3),
            "mean_lead_s": round(mean_lead, 1),
            "per_recording": all_results,
        },
        "elapsed_s": round(elapsed, 1),
    }

    out_path = os.path.join(OUTPUT_DIR, "exp13_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
