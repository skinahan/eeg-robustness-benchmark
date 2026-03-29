#!/usr/bin/env python3
"""
Compare MotorImagery time windows for Shin2017A + EEGNet (baseline mode).

MOABB provides 10 s trials; `config.get_paradigm("Shin2017A")` now defaults to [2, 6] s.
This script sweeps a few (tmin, tmax) presets so you can compare against the benchmark default
or the legacy full-10 s window.

Run from repo root (conda env ncp_robustness_proj):

  $env:KMP_DUPLICATE_LIB_OK='TRUE'
  conda run -n ncp_robustness_proj python evaluation/shin2017a_window_exploration.py

  conda run -n ncp_robustness_proj python evaluation/shin2017a_window_exploration.py --presets mid4 full10 --eval-modes WithinSession --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from moabb.paradigms import MotorImagery  # noqa: E402

from config import get_dataset_sampling_rate  # noqa: E402
from evaluation.unified_experiment_runner import UnifiedExperimentRunner  # noqa: E402


# (name, tmin, tmax) seconds relative to MI epoch start in MOABB
PRESETS: Dict[str, Tuple[float, float]] = {
    "full10": (0.0, 10.0),
    "first4": (0.0, 4.0),
    "mid4": (2.0, 6.0),
    "late4": (4.0, 8.0),
}


def _paradigm_shin(tmin: float, tmax: float, resample_hz: float) -> MotorImagery:
    return MotorImagery(
        events=["left_hand", "right_hand"],
        fmin=8,
        fmax=35,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        resample=resample_hz,
        n_classes=2,
    )


def _run(
    preset: str,
    tmin: float,
    tmax: float,
    eval_mode: str,
    subjects: Sequence[int],
    seed: int,
    *,
    shin2017a_train_sliding_window: bool = True,
    eegnet_lr: Optional[float] = None,
    eegnet_batch_size: Optional[int] = None,
    eegnet_weight_decay: Optional[float] = None,
) -> pd.DataFrame:
    resample_hz = float(get_dataset_sampling_rate("Shin2017A"))
    runner = UnifiedExperimentRunner(
        model="eegnet",
        dataset="Shin2017A",
        subjects=list(subjects),
        mode="baseline",
        eval_mode=eval_mode,
        seed=seed,
        noise_type=None,
        intensity=None,
        tune=False,
        overwrite=True,
        shin2017a_train_sliding_window=shin2017a_train_sliding_window,
        shin2017a_eegnet_optimizer_lr=eegnet_lr,
        shin2017a_eegnet_batch_size=eegnet_batch_size,
        shin2017a_eegnet_weight_decay=eegnet_weight_decay,
    )
    runner.paradigm = _paradigm_shin(tmin, tmax, resample_hz)
    df = runner.run_experiment()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["window_preset"] = preset
    df["window_tmin"] = tmin
    df["window_tmax"] = tmax
    df["shin2017a_eegnet_lr_override"] = eegnet_lr
    df["shin2017a_eegnet_batch_override"] = eegnet_batch_size
    df["shin2017a_eegnet_wd_override"] = eegnet_weight_decay
    return df


def _summarize(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or "validation_accuracy" not in df.columns:
        return {"mean_accuracy": float("nan"), "mean_roc_auc": float("nan")}
    return {
        "mean_accuracy": float(df["validation_accuracy"].astype(float).mean()),
        "mean_roc_auc": float(df["validation_roc_auc"].astype(float).mean())
        if "validation_roc_auc" in df.columns
        else float("nan"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Shin2017A EEGNet time-window sweep")
    p.add_argument(
        "--presets",
        nargs="+",
        default=["mid4", "first4", "full10"],
        choices=sorted(PRESETS.keys()),
        help="Named (tmin,tmax) windows to compare",
    )
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--eval-modes",
        nargs="+",
        default=["WithinSession", "CrossSession"],
        choices=["WithinSession", "CrossSession"],
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--eegnet-lr",
        type=float,
        default=None,
        help="Override AdamW lr (default: config.get_shin2017a_eegnet_factory_extras).",
    )
    p.add_argument(
        "--eegnet-batch-size",
        type=int,
        default=None,
        help="Override batch size (default: 32 from config for Shin2017A).",
    )
    p.add_argument(
        "--eegnet-weight-decay",
        type=float,
        default=None,
        help="Override AdamW weight_decay (default: 1e-4 from config).",
    )
    p.add_argument(
        "--no-shin2017a-train-sliding-window",
        action="store_true",
        help=(
            "Disable Lee-style training sliding windows (1s / 0.5s stride); "
            "validation/eval stay one fixed crop per trial. Default: sliding ON (matches unified runner)."
        ),
    )
    args = p.parse_args()

    out_dir = args.out_dir or (_REPO / "evaluation" / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows: List[Dict[str, Any]] = []
    frames: List[pd.DataFrame] = []

    for preset in args.presets:
        tmin, tmax = PRESETS[preset]
        for em in args.eval_modes:
            label = f"{preset} [{tmin},{tmax}]s | {em} | seed={args.seed}"
            print(f"\n=== {label} ===")
            try:
                df = _run(
                    preset,
                    tmin,
                    tmax,
                    em,
                    args.subjects,
                    args.seed,
                    shin2017a_train_sliding_window=not args.no_shin2017a_train_sliding_window,
                    eegnet_lr=args.eegnet_lr,
                    eegnet_batch_size=args.eegnet_batch_size,
                    eegnet_weight_decay=args.eegnet_weight_decay,
                )
            except Exception as e:
                print(f"[FAIL] {e}")
                rows.append(
                    {
                        "preset": preset,
                        "tmin": tmin,
                        "tmax": tmax,
                        "eval_mode": em,
                        "seed": args.seed,
                        "error": str(e),
                    }
                )
                continue
            sm = _summarize(df)
            sm.update(
                {
                    "preset": preset,
                    "tmin": tmin,
                    "tmax": tmax,
                    "eval_mode": em,
                    "seed": args.seed,
                }
            )
            rows.append(sm)
            print(f"  mean acc={sm['mean_accuracy']:.4f}  mean ROC-AUC={sm['mean_roc_auc']:.4f}")
            if not df.empty:
                frames.append(df)

    csv_path = out_dir / f"shin2017a_window_exploration_{ts}.csv"
    json_path = out_dir / f"shin2017a_window_exploration_{ts}.summary.json"
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(csv_path, index=False)
        print(f"\nWrote {csv_path}")
    else:
        print("\nNo result rows; CSV not written.")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "runs": rows,
                "presets_defined": PRESETS,
                "eegnet_lr_override": args.eegnet_lr,
                "eegnet_batch_size_override": args.eegnet_batch_size,
                "eegnet_weight_decay_override": args.eegnet_weight_decay,
                "shin2017a_train_sliding_window": not args.no_shin2017a_train_sliding_window,
            },
            f,
            indent=2,
        )
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
