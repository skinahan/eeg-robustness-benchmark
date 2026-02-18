#!/usr/bin/env python3
"""
Compute TPE-guided proxy robustness distribution summary from a backed-up Optuna study.

Uses trial.values = (entropy, curvature) to reconstruct robustness_score:
  robustness_score = 0.5 * clip(entropy, 0, 1) + 0.5 * sigmoid(curvature)

Outputs: n, mean, std, 90th percentile, 95th percentile, max
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    optuna = None
    TrialState = None


def _robustness_from_trial_values(values) -> float:
    """Reconstruct robustness_score from (entropy, curvature) = (te, orc)."""
    if values is None or len(values) < 2:
        return None
    entropy = float(np.clip(values[0], 0.0, 1.0))
    orc_val = float(values[1])
    curvature_score = 1.0 / (1.0 + np.exp(-orc_val))
    return 0.5 * entropy + 0.5 * curvature_score


def compute_tpe_robustness_summary(
    study_path: Path,
    output_path: Path = None,
) -> dict:
    """
    Load Optuna study, compute robustness from trial values, return summary stats.
    """
    with open(study_path, "rb") as f:
        study = pickle.load(f)

    scores = []
    for t in study.trials:
        if getattr(t, "state", None) != TrialState.COMPLETE:
            continue
        v = getattr(t, "values", None)
        r = _robustness_from_trial_values(v)
        if r is not None:
            scores.append(r)

    if not scores:
        return {"n": 0, "error": "No valid trials with values"}

    scores_arr = np.array(scores)
    summary = {
        "n": len(scores),
        "mean": float(np.mean(scores_arr)),
        "std": float(np.std(scores_arr)),
        "90th_percentile": float(np.percentile(scores_arr, 90)),
        "95th_percentile": float(np.percentile(scores_arr, 95)),
        "max": float(np.max(scores_arr)),
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote: {output_path}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute TPE proxy robustness distribution from Optuna study"
    )
    _default_study = Path(__file__).resolve().parent / "outputs" / "optimization" / "step3_optuna_study.pkl"
    parser.add_argument(
        "--study",
        type=str,
        default=str(_default_study),
        help="Path to step3_optuna_study.pkl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/optimization/tpe_robustness_distribution_summary.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    study_path = Path(args.study)
    if not study_path.exists():
        print(f"Study not found: {study_path}", file=sys.stderr)
        return 1

    summary = compute_tpe_robustness_summary(
        study_path=study_path,
        output_path=Path(args.output),
    )

    print("TPE proxy robustness distribution summary:")
    for k, v in summary.items():
        if k != "error":
            print(f"  {k}: {v}")

    return 0 if "error" not in summary else 1


if __name__ == "__main__":
    sys.exit(main())
