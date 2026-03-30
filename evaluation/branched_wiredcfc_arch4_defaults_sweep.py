#!/usr/bin/env python3
"""
Grid search for training defaults (AdamW lr, batch size, weight decay) for ``branched_wiredcfc_arch4``
on **Lee2019_MI** and **Shin2017A** using the unified experiment stack in **baseline** mode.

This mirrors the spirit of ``tests/test_new_dataset_feasibility_config.py`` (dataset contracts) plus
``new_dataset_eegnet_diagnosis.py`` / ``shin2017a_window_exploration.py`` (lightweight empirical runs),
but targets the BranchedWiredCfC architecture and **trainer** hyperparameters.

**Protocol:** Same paradigms and sliding-window behavior as ``UnifiedExperimentRunner`` (Lee/Shin
sliding windows on by default; see ``--no-lee2019-mi-train-sliding-window`` /
``--no-shin2017a-train-sliding-window``). Trainer kwargs are merged into ``_model_factory_kwargs``
(``models.branched_wiredcfc.create_branched_wiredcfc_classifier`` accepts ``optimizer__lr``,
``batch_size``, ``optimizer__weight_decay``, etc.).

**Requires:** conda env ``ncp_robustness_proj``, MOABB data, and a wiring JSON (default:
``outputs/architectures/best_architecture_4_trial_178.json`` via ``project_paths.resolve_architecture_json_path``).

Examples (repo root):

  $env:KMP_DUPLICATE_LIB_OK='TRUE'
  conda run -n ncp_robustness_proj python evaluation/branched_wiredcfc_arch4_defaults_sweep.py --dry-run

  conda run -n ncp_robustness_proj python evaluation/branched_wiredcfc_arch4_defaults_sweep.py \\
      --subjects 1 2 --seeds 42 --eval-modes WithinSession
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    _resolve_shin2017a_train_sliding_window,
)
from project_paths import resolve_architecture_json_path  # noqa: E402

MODEL_NAME = "branched_wiredcfc_arch4"
DEFAULT_LRS = (1e-3, 5e-4, 1e-4)
DEFAULT_BATCH_SIZES = (32, 64)
DEFAULT_WEIGHT_DECAYS = (0.0, 1e-4)


def _ensure_model_registered(architecture_json: Path) -> None:
    if MODEL_NAME in get_model_registry():
        return
    if not architecture_json.is_file():
        raise FileNotFoundError(
            f"Architecture JSON not found: {architecture_json}\n"
            "Pass --architecture-json or place best_architecture_4_trial_178.json under outputs/architectures/."
        )
    ok = add_branched_wiredcfc_architecture(MODEL_NAME, str(architecture_json))
    if not ok:
        raise RuntimeError(f"Failed to register {MODEL_NAME} from {architecture_json}")


def _patch_model_factory_kwargs(runner: UnifiedExperimentRunner, trainer_kw: Dict[str, Any]) -> None:
    """Merge optimizer/batch/wd (and other factory kwargs) into the runner's model factory."""

    def _patched(
        self: UnifiedExperimentRunner,
        n_chans: int,
        n_times: int,
        n_outputs: int,
    ) -> Dict[str, Any]:
        base = UnifiedExperimentRunner._model_factory_kwargs(self, n_chans, n_times, n_outputs)
        out = dict(base)
        out.update(trainer_kw)
        return out

    runner._model_factory_kwargs = types.MethodType(_patched, runner)  # type: ignore[method-assign]


def _run_one(
    *,
    dataset: str,
    eval_mode: str,
    seed: int,
    subjects: Sequence[int],
    trainer_kw: Dict[str, Any],
    pipeline: Optional[Dict[str, Any]],
    lee_sliding: bool,
    shin_sliding: bool,
    overwrite: bool,
) -> pd.DataFrame:
    runner = UnifiedExperimentRunner(
        model=MODEL_NAME,
        dataset=dataset,
        subjects=list(subjects),
        mode="baseline",
        eval_mode=eval_mode,
        seed=seed,
        noise_type=None,
        intensity=None,
        tune=False,
        overwrite=overwrite,
        lee2019_mi_train_sliding_window=lee_sliding,
        shin2017a_train_sliding_window=shin_sliding,
        pipeline=pipeline,
    )
    _patch_model_factory_kwargs(runner, trainer_kw)
    df = runner.run_experiment()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    for k, v in trainer_kw.items():
        df[f"trainer_{k}"] = v
    return df


def _aggregate_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "validation_accuracy" not in df.columns:
        return pd.DataFrame()
    keys = [c for c in df.columns if c.startswith("trainer_")]
    g = df.groupby(["dataset"] + keys, dropna=False)
    out = g["validation_accuracy"].agg(["mean", "std", "count"]).reset_index()
    out = out.rename(columns={"mean": "mean_val_acc", "std": "std_val_acc", "count": "n_rows"})
    if "validation_roc_auc" in df.columns:
        auc = g["validation_roc_auc"].mean().reset_index(name="mean_val_roc_auc")
        out = out.merge(auc, on=["dataset"] + keys, how="left")
    return out.sort_values(
        ["dataset", "mean_val_acc", "mean_val_roc_auc"],
        ascending=[True, False, False],
        na_position="last",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Trainer hyperparameter sweep for {MODEL_NAME} on Lee2019_MI / Shin2017A")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["Lee2019_MI", "Shin2017A"],
        choices=["Lee2019_MI", "Shin2017A"],
    )
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument(
        "--eval-modes",
        nargs="+",
        default=["WithinSession", "CrossSession"],
        choices=["WithinSession", "CrossSession"],
    )
    p.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=list(DEFAULT_LRS),
        help="AdamW learning rates to try (default: 1e-3 5e-4 1e-4).",
    )
    p.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help="Minibatch sizes (default: 32 64).",
    )
    p.add_argument(
        "--weight-decays",
        type=float,
        nargs="+",
        default=list(DEFAULT_WEIGHT_DECAYS),
        help="AdamW weight decay (default: 0 1e-4).",
    )
    p.add_argument(
        "--experiment-config",
        type=str,
        default=None,
        help="Optional YAML with datasets.<name>.pipeline (resample, train_sliding_window).",
    )
    p.add_argument(
        "--architecture-json",
        type=Path,
        default=None,
        help="Wiring JSON for branched_wiredcfc_arch4 (default: project_paths.resolve_architecture_json_path).",
    )
    p.add_argument(
        "--no-lee2019-mi-train-sliding-window",
        action="store_true",
    )
    p.add_argument(
        "--no-shin2017a-train-sliding-window",
        action="store_true",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite cached results (recommended for sweeps).")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the grid and exit without training.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    grid: List[Tuple[float, int, float]] = list(
        itertools.product(args.learning_rates, args.batch_sizes, args.weight_decays)
    )

    if args.dry_run:
        arch_display = str(args.architecture_json) if args.architecture_json else "<default: resolve_architecture_json_path()>"
        print(f"[dry-run] architecture_json={arch_display}")
        print(f"[dry-run] datasets={args.datasets} subjects={args.subjects} seeds={args.seeds} eval_modes={args.eval_modes}")
        print(f"[dry-run] grid ({len(grid)} points): lr x batch_size x weight_decay")
        for lr, bs, wd in grid:
            print(f"  optimizer__lr={lr}  batch_size={bs}  optimizer__weight_decay={wd}")
        total_runs = len(grid) * len(args.datasets) * len(args.seeds) * len(args.eval_modes)
        print(f"[dry-run] total baseline runs: {total_runs}")
        return 0

    arch_path = args.architecture_json or resolve_architecture_json_path()
    _ensure_model_registered(Path(arch_path))

    pipeline_by_ds: Dict[str, Optional[Dict[str, Any]]] = {}
    for ds in args.datasets:
        pipeline_by_ds[ds] = _load_pipeline_from_experiment_config(args.experiment_config, ds)

    lee_sliding = _resolve_lee2019_mi_train_sliding_window(
        pipeline_by_ds.get("Lee2019_MI"),
        args.no_lee2019_mi_train_sliding_window,
    )
    shin_sliding = _resolve_shin2017a_train_sliding_window(
        pipeline_by_ds.get("Shin2017A"),
        args.no_shin2017a_train_sliding_window,
    )

    out_dir = args.out_dir or (_REPO / "evaluation" / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    frames: List[pd.DataFrame] = []
    for dataset in args.datasets:
        pipe = pipeline_by_ds.get(dataset)
        for seed in args.seeds:
            for eval_mode in args.eval_modes:
                for lr, bs, wd in grid:
                    trainer_kw = {
                        "optimizer__lr": float(lr),
                        "batch_size": int(bs),
                        "optimizer__weight_decay": float(wd),
                    }
                    label = f"{dataset} | {eval_mode} | seed={seed} | lr={lr} bs={bs} wd={wd}"
                    print(f"\n=== {label} ===")
                    try:
                        df = _run_one(
                            dataset=dataset,
                            eval_mode=eval_mode,
                            seed=seed,
                            subjects=args.subjects,
                            trainer_kw=trainer_kw,
                            pipeline=pipe,
                            lee_sliding=lee_sliding,
                            shin_sliding=shin_sliding,
                            overwrite=args.overwrite,
                        )
                    except Exception as e:
                        print(f"[FAIL] {label}: {e}")
                        continue
                    if df.empty:
                        print(f"[WARN] No rows for {label}")
                        continue
                    df["sweep_dataset"] = dataset
                    frames.append(df)

    if not frames:
        print("No results collected.")
        return 1

    all_df = pd.concat(frames, ignore_index=True)
    all_df["dataset"] = all_df["sweep_dataset"]
    csv_path = out_dir / f"branched_wiredcfc_arch4_defaults_sweep_{ts}.csv"
    all_df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    agg = _aggregate_scores(all_df)
    agg_path = out_dir / f"branched_wiredcfc_arch4_defaults_sweep_{ts}.summary.csv"
    agg.to_csv(agg_path, index=False)
    print(f"Wrote {agg_path}")

    best_by_dataset: Dict[str, Any] = {}
    for ds in args.datasets:
        sub = agg[agg["dataset"] == ds]
        if sub.empty:
            continue
        row = sub.iloc[0]
        best_by_dataset[ds] = {
            "optimizer__lr": float(row["trainer_optimizer__lr"]),
            "batch_size": int(row["trainer_batch_size"]),
            "optimizer__weight_decay": float(row["trainer_optimizer__weight_decay"]),
            "mean_val_acc": float(row["mean_val_acc"]),
            "mean_val_roc_auc": float(row["mean_val_roc_auc"]) if "mean_val_roc_auc" in row else None,
        }

    meta = {
        "model": MODEL_NAME,
        "architecture_json": str(arch_path),
        "datasets": args.datasets,
        "subjects": args.subjects,
        "seeds": args.seeds,
        "eval_modes": args.eval_modes,
        "grid": {"learning_rates": args.learning_rates, "batch_sizes": args.batch_sizes, "weight_decays": args.weight_decays},
        "lee2019_mi_train_sliding_window": lee_sliding,
        "shin2017a_train_sliding_window": shin_sliding,
        "experiment_config": args.experiment_config,
        "best_by_dataset": best_by_dataset,
    }
    json_path = out_dir / f"branched_wiredcfc_arch4_defaults_sweep_{ts}.summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {json_path}")
    if best_by_dataset:
        print("\nBest trainer config by dataset (by mean validation accuracy, then ROC-AUC):")
        for ds, cfg in best_by_dataset.items():
            print(f"  {ds}: {cfg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
