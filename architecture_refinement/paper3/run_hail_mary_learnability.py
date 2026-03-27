"""
Hail Mary Block A: extract learnability metrics from training_history/*.json.

Aggregates per (model_name, seed) across CrossSession folds (mean over history files).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

VAL_KEYS = ("valid_roc_auc", "validation_roc_auc", "valid_auc", "roc_auc", "val_roc_auc")
TRAIN_LOSS_KEYS = ("train_loss", "loss")


def _pick_scalar(epoch: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[float]:
    for k in keys:
        if k in epoch and epoch[k] is not None:
            v = epoch[k]
            if isinstance(v, (list, tuple)) and len(v) == 1:
                v = v[0]
            try:
                x = float(v)
                if np.isfinite(x):
                    return x
            except (TypeError, ValueError):
                continue
    return None


def _parse_history(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return data


def summarize_one_history(hist: List[Dict[str, Any]], checkpoint_epochs: List[int]) -> Dict[str, float]:
    if not hist:
        return {}

    vals = []
    trains = []
    for ep in hist:
        v = _pick_scalar(ep, VAL_KEYS)
        if v is not None:
            vals.append(v)
        t = _pick_scalar(ep, TRAIN_LOSS_KEYS)
        if t is not None:
            trains.append(t)

    out: Dict[str, float] = {}
    if vals:
        arr = np.array(vals, dtype=float)
        best_i = int(np.nanargmax(arr))
        out["best_val_roc_auc"] = float(arr[best_i])
        out["epoch_best_val"] = float(best_i + 1)
        out["final_val_roc_auc"] = float(vals[-1])
        for ce in checkpoint_epochs:
            idx = ce - 1
            if 0 <= idx < len(vals):
                out[f"val_roc_auc_epoch_{ce}"] = float(vals[idx])
            else:
                out[f"val_roc_auc_epoch_{ce}"] = float("nan")
    if trains:
        out["final_train_loss"] = float(trains[-1])
        out["mean_train_loss"] = float(np.nanmean(trains))
    return out


def collect_histories_for_run(
    results_root: Path,
    dataset: str,
    model_name: str,
    seed: int,
) -> List[Path]:
    paradigm = "MotorImagery"
    base = results_root / paradigm / dataset / model_name / "CrossSessionEvaluation" / str(seed)
    if not base.exists():
        return []
    return sorted(base.rglob("training_history/history*.json"))


def extract_learnability_longform(
    *,
    results_root: Path,
    dataset: str,
    model_names: List[str],
    seeds: List[int],
    checkpoint_epochs: List[int],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_name in model_names:
        for seed in seeds:
            paths = collect_histories_for_run(results_root, dataset, model_name, seed)
            if not paths:
                rows.append(
                    {
                        "model_name": model_name,
                        "seed": seed,
                        "n_history_files": 0,
                        "error": "no_training_history",
                    }
                )
                continue
            per_file_stats: List[Dict[str, float]] = []
            for p in paths:
                hist = _parse_history(p)
                per_file_stats.append(summarize_one_history(hist, checkpoint_epochs))
            # mean across folds
            keys = set()
            for s in per_file_stats:
                keys.update(s.keys())
            agg: Dict[str, Any] = {
                "model_name": model_name,
                "seed": seed,
                "n_history_files": len(paths),
            }
            for k in sorted(keys):
                vals = [float(s[k]) for s in per_file_stats if k in s and np.isfinite(s[k])]
                if vals:
                    agg[k] = float(np.mean(vals))
                else:
                    agg[k] = float("nan")
            rows.append(agg)
    return pd.DataFrame(rows)


def load_models_from_manifest(panel_dir: Path) -> List[str]:
    mj = panel_dir / "topology_manifest.json"
    if mj.exists():
        data = json.loads(mj.read_text(encoding="utf-8"))
        return [str(r["model_name"]) for r in data.get("manifest", [])]
    mc = panel_dir / "topology_manifest.csv"
    if mc.exists():
        df = pd.read_csv(mc)
        return df["model_name"].astype(str).tolist()
    raise FileNotFoundError(f"No topology_manifest in {panel_dir}")


def run_cli() -> int:
    parser = argparse.ArgumentParser(description="Hail Mary learnability extraction")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument(
        "--topology-panel-dir",
        type=str,
        default="architecture_refinement/outputs/hail_mary/topology_panel",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--checkpoint-epochs", type=str, default="5,10,20,50")
    parser.add_argument("--output-csv", type=str, default="architecture_refinement/outputs/hail_mary/analysis/learnability_longform.csv")
    args = parser.parse_args()

    rr = Path(args.results_root)
    if not rr.is_absolute():
        rr = _REPO_ROOT / rr
    panel = Path(args.topology_panel_dir)
    if not panel.is_absolute():
        panel = _REPO_ROOT / panel

    models = load_models_from_manifest(panel)
    seeds = list(args.seeds) if args.seeds else [42, 43, 44]
    ce = [int(x.strip()) for x in args.checkpoint_epochs.split(",") if x.strip()]

    df = extract_learnability_longform(
        results_root=rr,
        dataset=args.dataset,
        model_names=models,
        seeds=seeds,
        checkpoint_epochs=ce,
    )
    out = Path(args.output_csv)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
