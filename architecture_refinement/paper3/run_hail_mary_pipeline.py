"""
Hail Mary orchestration:

  Analysis only (default):
    python -m architecture_refinement.paper3.run_hail_mary_pipeline [--seeds ...] [--no-overwrite]

  Training + analysis:
    python -m architecture_refinement.paper3.run_hail_mary_pipeline --experiment [--overwrite] [--dry-run] ...

  Build panel only:
    python -m architecture_refinement.paper3.run_hail_mary_build_panel

`--overwrite` with `--experiment` forwards to evaluation.unified_experiment_runner (re-run cached folds).
`--no-overwrite` skips analysis CSVs that already exist (passed to learnability/stability/sensitivity/aggregate).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hail Mary: optional experiment dispatch, then post-hoc analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Run run_hail_mary_experiment before analysis (unified_experiment_runner jobs).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="With --experiment: pass --overwrite to unified_experiment_runner (re-run even if cached).",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Pass --no-overwrite to analysis steps (skip writing if output CSVs exist).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --experiment: pass --dry-run to run_hail_mary_experiment (no jobs).",
    )
    parser.add_argument(
        "--topology-panel-dir",
        type=str,
        default="architecture_refinement/outputs/hail_mary/topology_panel",
    )
    parser.add_argument(
        "--experiment-output-dir",
        type=str,
        default="architecture_refinement/outputs/hail_mary/runs",
        help="Output dir for hail_mary_run_manifest.json when using --experiment",
    )
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument("--eval-mode", type=str, default="CrossSession")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=42)
    args = parser.parse_args()

    py = args.python
    panel = Path(args.topology_panel_dir)
    if not panel.is_absolute():
        panel = _REPO_ROOT / panel
    exp_out = Path(args.experiment_output_dir)
    if not exp_out.is_absolute():
        exp_out = _REPO_ROOT / exp_out

    if args.experiment:
        exp_cmd = [
            py,
            "-m",
            "architecture_refinement.paper3.run_hail_mary_experiment",
            "--topology-panel-dir",
            str(panel),
            "--output-dir",
            str(exp_out),
            "--dataset",
            args.dataset,
            "--eval-mode",
            args.eval_mode,
            "--num-seeds",
            str(args.num_seeds),
            "--seed-start",
            str(args.seed_start),
        ]
        if args.overwrite:
            exp_cmd.append("--overwrite")
        if args.dry_run:
            exp_cmd.append("--dry-run")
        print("Running:", " ".join(exp_cmd))
        r = subprocess.run(exp_cmd, cwd=str(_REPO_ROOT))
        if r.returncode != 0:
            return int(r.returncode)

    analysis_steps = [
        "architecture_refinement.paper3.run_hail_mary_learnability",
        "architecture_refinement.paper3.run_hail_mary_stability",
        "architecture_refinement.paper3.run_hail_mary_sensitivity",
        "architecture_refinement.paper3.run_hail_mary_aggregate",
    ]
    cmd_base = [py, "-m"]
    for mod in analysis_steps:
        cmd = cmd_base + [mod]
        if args.seeds and "aggregate" not in mod:
            cmd.extend(["--seeds", *[str(s) for s in args.seeds]])
        if args.no_overwrite:
            cmd.append("--no-overwrite")
        print("Running:", " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(_REPO_ROOT))
        if r.returncode != 0:
            return int(r.returncode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
