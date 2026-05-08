#!/usr/bin/env python3
"""Delta.72 GPU workload — runs full experiment suite on Vast.ai GPU instance.

Designed for gpu-workers pattern:
    gpuw upload ./src ./experiments ./workloads /workspace/delta72/
    gpuw run delta72_gpu.py --n-monte-carlo 1000 --output /workspace/results/
    gpuw download /workspace/results/ ./results/

Can also run locally:
    python workloads/delta72_gpu.py --n-monte-carlo 500 --output results/
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def setup_env():
    """Install dependencies if running on fresh GPU instance."""
    try:
        import numpy
        import matplotlib
        import scipy
    except ImportError:
        print("Installing dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "numpy", "scipy", "matplotlib", "torch",
        ])


def main():
    parser = argparse.ArgumentParser(description="Delta.72 GPU Workload")
    parser.add_argument("--n-monte-carlo", type=int, default=1000,
                       help="Monte Carlo trials (default: 1000 for GPU, 500 for CPU)")
    parser.add_argument("--output", type=str, default="/workspace/results",
                       help="Output directory for results and plots")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    args = parser.parse_args()

    output_dir = Path(args.output)

    print("=" * 60)
    print("  Delta.72 GPU Workload")
    print("=" * 60)
    print(f"  Monte Carlo trials: {args.n_monte_carlo}")
    print(f"  Output: {output_dir}")
    print(f"  GPU available: {_check_gpu()}")
    print("=" * 60)

    if args.dry_run:
        return

    setup_env()

    # Add experiment code to path
    workload_dir = Path(__file__).resolve().parent
    repo_root = workload_dir.parent
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root / "experiments"))

    # Run experiments
    from run_all import main as run_experiments
    sys.argv = [
        "run_all.py",
        "--n-monte-carlo", str(args.n_monte_carlo),
        "--output-dir", str(output_dir),
    ]

    t0 = time.time()
    run_experiments()
    elapsed = time.time() - t0

    # Generate dashboard
    from generate_dashboard import generate_dashboard
    dashboard_path = output_dir / "dashboard.html"
    generate_dashboard(output_dir, dashboard_path)

    print(f"\n  Total workload time: {elapsed:.1f}s")
    print(f"  Dashboard: {dashboard_path}")
    print(f"  Results: {output_dir / 'experiment_results.json'}")


def _check_gpu() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"Yes ({torch.cuda.get_device_name(0)})"
    except Exception:
        pass
    return "No (CPU mode)"


if __name__ == "__main__":
    main()
