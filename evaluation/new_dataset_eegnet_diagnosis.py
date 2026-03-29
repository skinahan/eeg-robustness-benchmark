#!/usr/bin/env python3
"""
Lightweight EEGNet diagnosis for newer benchmark datasets (Yang2025, Shin2017A, Lee2019_MI).

Runs the unified experiment stack in **baseline** mode (clean validation metrics only; no
perturbation grid) so each job trains EEGNet and reports validation accuracy / ROC-AUC.

Design constraints (defaults):
  - At most 2 subjects and 3 experimental seeds.
  - EEGNet only; benchmark defaults from ``config.get_paradigm`` / ``get_dataset_sampling_rate``.
  - Lee2019_MI / Shin2017A: training-side sliding-window augmentation enabled (same as the
    unified runner); disable with ``--no-lee2019-mi-train-sliding-window`` or
    ``--no-shin2017a-train-sliding-window`` respectively.

Optional ``--resample-hz`` (e.g. 128) overrides the MOABB MotorImagery resample target and
patches model ``sfreq`` plus sliding-window lengths so they stay consistent with the
resampled tensors.

Interpretation:
  - Binary chance accuracy / ROC-AUC ≈ 0.5. Within-session metrics above chance are the
    primary sanity check. Cross-session MI is often at or below chance for small *n*; treat
    failure there as a signal to revisit preprocessing, not necessarily a broken pipeline.

Example (repo root, conda env ``ncp_robustness_proj``):

  python evaluation/new_dataset_eegnet_diagnosis.py

  python evaluation/new_dataset_eegnet_diagnosis.py --resample-hz 128 --datasets Yang2025 Shin2017A
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

# Project imports (unified runner adds repo root to sys.path)
_CURRENT = Path(__file__).resolve().parent
_REPO_ROOT = _CURRENT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from config import get_dataset_sampling_rate, get_paradigm  # noqa: E402
from evaluation.unified_experiment_runner import (  # noqa: E402
    UnifiedExperimentRunner,
    _require_moabb_dataset_ctor,
)
try:
    from moabb.datasets import Yang2025  # type: ignore
except ImportError:
    Yang2025 = None  # type: ignore[misc, assignment]

DEFAULT_DATASETS = ("Yang2025", "Shin2017A", "Lee2019_MI")
DEFAULT_SEEDS = (42, 123, 456)
DEFAULT_SUBJECTS = (1, 2)
CHANCE = 0.5


def _apply_resample_override(runner: UnifiedExperimentRunner, resample_hz: float) -> None:
    """Align paradigm resample, EEGNet sfreq, and sliding-window sample counts with resample_hz."""
    ds = runner.dataset
    runner.paradigm = get_paradigm(resample=float(resample_hz), dataset=ds)
    runner._paradigm_resample_hz = float(resample_hz)

    def _patched_mfk(
        self: UnifiedExperimentRunner, n_chans: int, n_times: int, n_outputs: int
    ) -> Dict[str, Any]:
        return {
            "n_chans": n_chans,
            "n_times": n_times,
            "n_outputs": n_outputs,
            "sfreq": float(resample_hz),
        }

    runner._model_factory_kwargs = types.MethodType(_patched_mfk, runner)  # type: ignore[method-assign]


def _ensure_yang_available() -> None:
    _require_moabb_dataset_ctor("Yang2025", Yang2025)


def _run_one(
    dataset: str,
    eval_mode: str,
    seed: int,
    subjects: Sequence[int],
    resample_hz: Optional[float],
    lee2019_sliding: bool,
    shin2017a_sliding: bool,
    overwrite: bool,
) -> pd.DataFrame:
    runner = UnifiedExperimentRunner(
        model="eegnet",
        dataset=dataset,
        subjects=list(subjects),
        mode="baseline",
        eval_mode=eval_mode,
        seed=seed,
        noise_type=None,
        intensity=None,
        tune=False,
        overwrite=overwrite,
        lee2019_mi_train_sliding_window=lee2019_sliding,
        shin2017a_train_sliding_window=shin2017a_sliding,
    )
    if resample_hz is not None:
        _apply_resample_override(runner, float(resample_hz))
    out = runner.run_experiment()
    if out is None or out.empty:
        return pd.DataFrame()
    out = out.copy()
    out["diagnostic_resample_hz"] = resample_hz
    return out


def _summarize_block(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "mean_accuracy": float("nan"),
            "mean_roc_auc": float("nan"),
            "min_accuracy": float("nan"),
            "fraction_above_chance_acc": float("nan"),
        }
    acc_col = "validation_accuracy" if "validation_accuracy" in df.columns else None
    auc_col = "validation_roc_auc" if "validation_roc_auc" in df.columns else None
    acc = df[acc_col].astype(float) if acc_col else pd.Series(dtype=float)
    auc = df[auc_col].astype(float) if auc_col else pd.Series(dtype=float)
    frac = float((acc > CHANCE).mean()) if len(acc) else float("nan")
    return {
        "mean_accuracy": float(acc.mean()) if len(acc) else float("nan"),
        "mean_roc_auc": float(auc.mean()) if len(auc) else float("nan"),
        "min_accuracy": float(acc.min()) if len(acc) else float("nan"),
        "fraction_above_chance_acc": frac,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EEGNet diagnosis for Yang2025 / Shin2017A / Lee2019_MI")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help=f"Subset of {list(DEFAULT_DATASETS)}",
    )
    p.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=list(DEFAULT_SUBJECTS),
        help="At most 2 subjects recommended (default: 1 2).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Experimental seeds (default: 42 123 456; at most 3 recommended).",
    )
    p.add_argument(
        "--eval-modes",
        nargs="+",
        default=["WithinSession", "CrossSession"],
        choices=["WithinSession", "CrossSession"],
    )
    p.add_argument(
        "--resample-hz",
        type=float,
        default=None,
        help="If set, resample MOABB epochs to this rate (Hz) and match EEGNet sfreq / Lee windows.",
    )
    p.add_argument(
        "--no-lee2019-mi-train-sliding-window",
        action="store_true",
        help="Disable Lee2019_MI training sliding windows (benchmark default is ON).",
    )
    p.add_argument(
        "--no-shin2017a-train-sliding-window",
        action="store_true",
        help="Disable Shin2017A training sliding windows (unified default is ON).",
    )
    p.add_argument(
        "--respect-skip",
        action="store_true",
        help="If set, allow unified runner skip-if-results-exist (default: overwrite/retrain).",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Write concatenated long-form results CSV (default: evaluation/output/eegnet_new_dataset_diagnosis_<timestamp>.csv).",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write summary JSON next to CSV unless overridden.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.subjects) > 2:
        print("Warning: using more than 2 subjects is allowed but not recommended for a lightweight diagnosis.")
    if len(args.seeds) > 3:
        print("Warning: using more than 3 seeds is allowed but not recommended for a lightweight diagnosis.")

    if "Yang2025" in args.datasets:
        try:
            _ensure_yang_available()
        except ImportError as e:
            print(f"Skipping Yang2025: {e}")
            args.datasets = [d for d in args.datasets if d != "Yang2025"]

    if not args.datasets:
        print("No datasets left to run.")
        return 1

    lee_sliding = not args.no_lee2019_mi_train_sliding_window
    shin_sliding = not args.no_shin2017a_train_sliding_window
    all_frames: List[pd.DataFrame] = []
    summaries: List[Dict[str, Any]] = []

    for dataset in args.datasets:
        native_hz = float(get_dataset_sampling_rate(dataset))
        for eval_mode in args.eval_modes:
            for seed in args.seeds:
                label = f"{dataset} | {eval_mode} | seed={seed}"
                if args.resample_hz is not None:
                    label += f" | resample={args.resample_hz} Hz (native config {native_hz} Hz)"
                print(f"\n=== {label} ===")
                try:
                    df = _run_one(
                        dataset=dataset,
                        eval_mode=eval_mode,
                        seed=seed,
                        subjects=args.subjects,
                        resample_hz=args.resample_hz,
                        lee2019_sliding=lee_sliding,
                        shin2017a_sliding=shin_sliding,
                        overwrite=not args.respect_skip,
                    )
                except Exception as e:
                    print(f"[FAIL] {label}: {e}")
                    summaries.append(
                        {
                            "dataset": dataset,
                            "eval_mode": eval_mode,
                            "seed": seed,
                            "resample_hz": args.resample_hz,
                            "native_hz_config": native_hz,
                            "error": str(e),
                        }
                    )
                    continue
                if df.empty:
                    print(f"[WARN] No result rows for {label}")
                else:
                    all_frames.append(df)
                sm = _summarize_block(df)
                sm.update(
                    {
                        "dataset": dataset,
                        "eval_mode": eval_mode,
                        "seed": seed,
                        "resample_hz": args.resample_hz,
                        "native_hz_config": native_hz,
                        "lee2019_mi_train_sliding_window": lee_sliding if dataset == "Lee2019_MI" else None,
                    }
                )
                summaries.append(sm)
                print(
                    f"  mean acc={sm['mean_accuracy']:.4f}  mean ROC-AUC={sm['mean_roc_auc']:.4f}  "
                    f"frac(acc>{CHANCE})={sm['fraction_above_chance_acc']:.2f}"
                )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _REPO_ROOT / "evaluation" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_csv or (out_dir / f"eegnet_new_dataset_diagnosis_{ts}.csv")
    json_path = args.out_json or csv_path.with_suffix(".summary.json")

    if all_frames:
        long_df = pd.concat(all_frames, ignore_index=True)
        long_df.to_csv(csv_path, index=False)
        print(f"\nWrote long-form results: {csv_path}")
    else:
        print("\nNo non-empty result frames; CSV not written.")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"runs": summaries}, f, indent=2)
    print(f"Wrote summary: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
