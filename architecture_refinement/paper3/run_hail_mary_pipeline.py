"""
Hail Mary one-shot analysis pipeline (after training jobs complete):

  python -m architecture_refinement.paper3.run_hail_mary_learnability
  python -m architecture_refinement.paper3.run_hail_mary_stability
  python -m architecture_refinement.paper3.run_hail_mary_sensitivity
  python -m architecture_refinement.paper3.run_hail_mary_aggregate

Build panel only:
  python -m architecture_refinement.paper3.run_hail_mary_build_panel

Dispatch training:
  python -m architecture_refinement.paper3.run_hail_mary_experiment [--dry-run]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all Hail Mary post-hoc analysis steps")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()

    steps = [
        "architecture_refinement.paper3.run_hail_mary_learnability",
        "architecture_refinement.paper3.run_hail_mary_stability",
        "architecture_refinement.paper3.run_hail_mary_sensitivity",
        "architecture_refinement.paper3.run_hail_mary_aggregate",
    ]
    cmd_base = [args.python, "-m"]
    for mod in steps:
        cmd = cmd_base + [mod]
        if args.seeds and "aggregate" not in mod:
            cmd.extend(["--seeds", *[str(s) for s in args.seeds]])
        print("Running:", " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(_REPO_ROOT))
        if r.returncode != 0:
            return int(r.returncode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
