"""
Hail Mary Block B: aggregate stability-related fields from training_history JSON.

Expects optional keys from HailMaryStabilityCallback: batch_train_loss_var, batch_train_loss_mean.
Also summarizes valid_loss epoch-to-epoch volatility when present.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_history(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _col(hist: List[Dict[str, Any]], key: str) -> np.ndarray:
    out = []
    for ep in hist:
        v = ep.get(key)
        if v is not None:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                pass
    return np.array(out, dtype=float)


def summarize_history_file(path: Path) -> Dict[str, float]:
    hist = _parse_history(path)
    if not hist:
        return {}
    out: Dict[str, float] = {}
    bvar = _col(hist, "batch_train_loss_var")
    if bvar.size:
        out["mean_batch_train_loss_var"] = float(np.nanmean(bvar))
        out["max_batch_train_loss_var"] = float(np.nanmax(bvar))
    vl = _col(hist, "valid_loss")
    if vl.size > 1:
        out["valid_loss_epoch_to_epoch_var"] = float(np.var(vl))
    tl = _col(hist, "train_loss")
    if tl.size > 1:
        out["train_loss_epoch_to_epoch_var"] = float(np.var(tl))
    return out


def collect_stability_longform(
    *,
    results_root: Path,
    dataset: str,
    model_names: List[str],
    seeds: List[int],
) -> pd.DataFrame:
    paradigm = "MotorImagery"
    rows: List[Dict[str, Any]] = []
    for model_name in model_names:
        for seed in seeds:
            base = results_root / paradigm / dataset / model_name / "CrossSessionEvaluation" / str(seed)
            paths = sorted(base.rglob("training_history/history*.json")) if base.exists() else []
            if not paths:
                rows.append({"model_name": model_name, "seed": seed, "n_history_files": 0})
                continue
            stats: List[Dict[str, float]] = [summarize_history_file(p) for p in paths]
            keys = set()
            for s in stats:
                keys.update(s.keys())
            row: Dict[str, Any] = {"model_name": model_name, "seed": seed, "n_history_files": len(paths)}
            for k in sorted(keys):
                vals = [float(s[k]) for s in stats if k in s and np.isfinite(s[k])]
                row[k] = float(np.mean(vals)) if vals else float("nan")
            rows.append(row)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Hail Mary stability aggregation")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument(
        "--topology-panel-dir",
        type=str,
        default="architecture_refinement/outputs/hail_mary/topology_panel",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--output-csv", type=str, default="architecture_refinement/outputs/hail_mary/analysis/stability_longform.csv")
    args = parser.parse_args()

    rr = Path(args.results_root)
    if not rr.is_absolute():
        rr = _REPO_ROOT / rr
    panel = Path(args.topology_panel_dir)
    if not panel.is_absolute():
        panel = _REPO_ROOT / panel

    models = load_models_from_manifest(panel)
    seeds = list(args.seeds) if args.seeds else [42, 43, 44]

    df = collect_stability_longform(results_root=rr, dataset=args.dataset, model_names=models, seeds=seeds)
    out = Path(args.output_csv)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
