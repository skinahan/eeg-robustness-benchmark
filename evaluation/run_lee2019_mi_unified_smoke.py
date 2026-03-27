#!/usr/bin/env python3
"""
Minimal end-to-end smoke test for Lee2019_MI via unified_experiment_runner.

Requires MOABB with Lee2019_MI, network on first run for data download, and
project dependencies (see requirements.txt).

Example:
  python evaluation/run_lee2019_mi_unified_smoke.py
"""

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    runner = repo_root / "evaluation" / "unified_experiment_runner.py"
    if not runner.is_file():
        print(f"Missing runner: {runner}", file=sys.stderr)
        return 1

    os.chdir(repo_root)
    cmd = [
        sys.executable,
        str(runner),
        "--model",
        "eegnet",
        "--dataset",
        "Lee2019_MI",
        "--subjects",
        "1",
        "--mode",
        "test_perturb",
        "--eval_mode",
        "CrossSession",
        "--seed",
        "42",
        "--noise_type",
        "gaussian",
        "--intensity",
        "10.0",
        "--test_perturb_gaussian_only",
        "--noise_perturbation_num_steps",
        "3",
        "--overwrite",
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
