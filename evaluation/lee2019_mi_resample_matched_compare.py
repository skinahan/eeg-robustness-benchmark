#!/usr/bin/env python3
"""
Matched baseline runs on **Lee2019_MI** at two paradigm resample rates (default: **128 Hz** vs **1000 Hz**).

Same protocol otherwise: ``UnifiedExperimentRunner`` defaults, Lee training sliding windows unless
disabled, identical ``subjects`` / ``seeds`` / ``eval_modes`` across resample conditions.

Use to compare models (e.g. ``eegnet`` vs ``branched_wiredcfc_arch4``) under the same splits; only
``pipeline_resample_hz`` changes (see ``config.resolve_paradigm_resample_hz`` precedence).

Examples (repo root, conda env ``ncp_robustness_proj``):

  $env:KMP_DUPLICATE_LIB_OK='TRUE'
  conda run -n ncp_robustness_proj python evaluation/lee2019_mi_resample_matched_compare.py \\
      --model branched_wiredcfc_arch4 --subjects 1 2 --seeds 42 --eval-modes WithinSession CrossSession --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from config import add_branched_wiredcfc_architecture, get_model_registry  # noqa: E402
from evaluation.unified_experiment_runner import (  # noqa: E402
    UnifiedExperimentRunner,
    _load_pipeline_from_experiment_config,
    _resolve_lee2019_mi_train_sliding_window,
)
from project_paths import resolve_architecture_json_path  # noqa: E402


def _ensure_branched_registered(architecture_json: Path) -> None:
    if "branched_wiredcfc_arch4" in get_model_registry():
        return
    if not architecture_json.is_file():
        raise FileNotFoundError(f"Missing architecture JSON: {architecture_json}")
    if not add_branched_wiredcfc_architecture("branched_wiredcfc_arch4", str(architecture_json)):
        raise RuntimeError("Failed to register branched_wiredcfc_arch4")


def _run_one(
    *,
    model: str,
    resample_hz: float,
    eval_mode: str,
    seed: int,
    subjects: Sequence[int],
    pipeline: Optional[Dict[str, Any]],
    lee_sliding: bool,
    overwrite: bool,
) -> pd.DataFrame:
    runner = UnifiedExperimentRunner(
        model=model,
        dataset="Lee2019_MI",
        subjects=list(subjects),
        mode="baseline",
        eval_mode=eval_mode,
        seed=seed,
        noise_type=None,
        intensity=None,
        tune=False,
        overwrite=overwrite,
        lee2019_mi_train_sliding_window=lee_sliding,
        pipeline=pipeline,
        pipeline_resample_hz=float(resample_hz),
    )
    df = runner.run_experiment()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["paradigm_resample_hz_run"] = float(resample_hz)
    return df


def main() -> int:
    p = argparse.ArgumentParser(description="Lee2019_MI: matched 128 Hz vs 1000 Hz baseline runs")
    p.add_argument("--model", type=str, default="branched_wiredcfc_arch4")
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument(
        "--eval-modes",
        nargs="+",
        default=["WithinSession", "CrossSession"],
        choices=["WithinSession", "CrossSession"],
    )
    p.add_argument(
        "--resamples",
        type=float,
        nargs="+",
        default=[128.0, 1000.0],
        help="Paradigm resample (Hz); default 128 vs native-scale 1000.",
    )
    p.add_argument(
        "--experiment-config",
        type=str,
        default=None,
        help="Optional YAML; pipeline.train_sliding_window merged with --no-lee2019-mi-train-sliding-window",
    )
    p.add_argument("--no-lee2019-mi-train-sliding-window", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--architecture-json",
        type=Path,
        default=None,
        help="Required wiring JSON if model is branched_wiredcfc_arch4 (default: resolve_architecture_json_path).",
    )
    args = p.parse_args()

    if args.model == "branched_wiredcfc_arch4":
        arch = args.architecture_json or resolve_architecture_json_path()
        _ensure_branched_registered(Path(arch))

    pipe = _load_pipeline_from_experiment_config(args.experiment_config, "Lee2019_MI")
    lee_sw = _resolve_lee2019_mi_train_sliding_window(pipe, args.no_lee2019_mi_train_sliding_window)

    out_dir = args.out_dir or (_REPO / "evaluation" / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    frames: List[pd.DataFrame] = []
    for hz in args.resamples:
        for seed in args.seeds:
            for em in args.eval_modes:
                tag = f"Lee2019_MI | {args.model} | resample={hz} Hz | seed={seed} | {em}"
                print(f"\n=== {tag} ===")
                try:
                    df = _run_one(
                        model=args.model,
                        resample_hz=hz,
                        eval_mode=em,
                        seed=seed,
                        subjects=args.subjects,
                        pipeline=pipe,
                        lee_sliding=lee_sw,
                        overwrite=args.overwrite,
                    )
                except Exception as e:
                    print(f"[FAIL] {tag}: {e}")
                    import traceback

                    traceback.print_exc()
                    continue
                if df.empty:
                    print(f"[WARN] No rows: {tag}")
                    continue
                frames.append(df)

    if not frames:
        print("No results.")
        return 1

    all_df = pd.concat(frames, ignore_index=True)
    csv_path = out_dir / f"lee2019_mi_resample_matched_{args.model}_{ts}.csv"
    all_df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    meta = {
        "model": args.model,
        "dataset": "Lee2019_MI",
        "subjects": args.subjects,
        "seeds": args.seeds,
        "eval_modes": args.eval_modes,
        "resamples_hz": args.resamples,
        "lee2019_mi_train_sliding_window": lee_sw,
        "experiment_config": args.experiment_config,
    }
    json_path = out_dir / f"lee2019_mi_resample_matched_{args.model}_{ts}.summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {json_path}")

    # Quick pivot: mean val acc by resample and eval_mode
    if "validation_accuracy" in all_df.columns and "paradigm_resample_hz_run" in all_df.columns:
        sub = all_df.groupby(["paradigm_resample_hz_run", "eval_mode"], dropna=False)["validation_accuracy"].mean()
        print("\nMean validation_accuracy by resample_hz and eval_mode:")
        print(sub.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
