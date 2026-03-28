"""
Hail Mary Block C: collect clean / low / moderate test ROC-AUC from test_perturb CSVs;
compute relative degradation RD and mini-AUPC (trapezoid over alpha grid).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architecture_refinement.paper3.hail_mary_cli import add_overwrite_arguments, can_write_output, results_model_segment


def _find_perturb_csvs(run_dir: Path) -> List[Path]:
    if not run_dir.exists():
        return []
    out: List[Path] = []
    for p in run_dir.rglob("*.csv"):
        name_l = p.name.lower()
        path_l = str(p).lower()
        # Short-path outputs use .../test_perturb/tp_<id>_s###_seed#.csv (no "perturb" in filename).
        if "test_perturb" in path_l or "perturb" in name_l or name_l.startswith("tp_"):
            out.append(p)
    return sorted(set(out))


def _pick_numeric(df: pd.DataFrame, col: str, default: float = np.nan) -> float:
    if col not in df.columns:
        return float(default)
    s = pd.to_numeric(df[col], errors="coerce")
    if s.notna().any():
        return float(s.dropna().iloc[0])
    return float(default)


def summarize_one_session_csv(path: Path, noise_type: str) -> Optional[Dict[str, Any]]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "noise_type" not in df.columns:
        return None
    sub = df[df["noise_type"].astype(str) == noise_type].copy()
    if sub.empty:
        return None
    sub["alpha"] = pd.to_numeric(sub.get("alpha"), errors="coerce")
    sub["corrupted_roc_auc"] = pd.to_numeric(sub.get("corrupted_roc_auc"), errors="coerce")
    clean = _pick_numeric(sub, "clean_roc_auc")
    if not np.isfinite(clean) and "clean_score" in sub.columns:
        clean = _pick_numeric(sub, "clean_score")

    # Prefer rows sorted by alpha
    sub = sub.sort_values("alpha", na_position="last")
    by_alpha: Dict[float, float] = {}
    for _, row in sub.iterrows():
        a = row.get("alpha")
        c = row.get("corrupted_roc_auc")
        if pd.notna(a) and pd.notna(c):
            by_alpha[float(a)] = float(c)
    if not by_alpha:
        return None

    a0 = clean if np.isfinite(clean) else by_alpha.get(0.0)
    if a0 is None or not np.isfinite(a0):
        return None

    def auc_for_alpha(target: float) -> float:
        if target in by_alpha:
            return by_alpha[target]
        alphas = sorted(by_alpha.keys())
        if not alphas:
            return float("nan")
        nearest = min(alphas, key=lambda x: abs(x - target))
        return by_alpha[nearest]

    a_low = auc_for_alpha(0.25)
    a_mod = auc_for_alpha(0.5)

    RD = lambda a: (float(a0) - a) / float(a0) if float(a0) > 0 else float("nan")
    rd_low = RD(a_low)
    rd_mod = RD(a_mod)

    alphas_x = np.array([0.0, 0.25, 0.5], dtype=float)
    rds = np.array([0.0, rd_low if np.isfinite(rd_low) else 0.0, rd_mod if np.isfinite(rd_mod) else 0.0], dtype=float)
    mini_aupc_rd = float(np.trapz(rds, alphas_x)) if np.all(np.isfinite(rds)) else float("nan")

    return {
        "clean_test_roc_auc": float(a0),
        "perturbed_roc_auc_low": float(a_low),
        "perturbed_roc_auc_moderate": float(a_mod),
        "RD_low": float(rd_low),
        "RD_moderate": float(rd_mod),
        "mini_AUPC_RD_trapz_alpha_0_0p25_0p5": mini_aupc_rd,
        "source_csv": str(path),
    }


def collect_sensitivity_longform(
    *,
    results_root: Path,
    dataset: str,
    model_names: List[str],
    seeds: List[int],
    noise_type: str = "gaussian",
) -> pd.DataFrame:
    paradigm = "MotorImagery"
    rows: List[Dict[str, Any]] = []
    for model_name in model_names:
        for seed in seeds:
            model_segment = results_model_segment(model_name)
            base = results_root / paradigm / dataset / model_segment / "CrossSessionEvaluation" / str(seed)
            csvs = _find_perturb_csvs(base)
            if not csvs:
                rows.append({"model_name": model_name, "seed": seed, "error": "no_csv"})
                continue
            per_session: List[Dict[str, Any]] = []
            for p in csvs:
                s = summarize_one_session_csv(p, noise_type=noise_type)
                if s:
                    per_session.append(s)
            if not per_session:
                rows.append({"model_name": model_name, "seed": seed, "error": "no_rows"})
                continue
            agg: Dict[str, Any] = {"model_name": model_name, "seed": seed, "n_csv": len(per_session)}
            keys = [k for k in per_session[0].keys() if k != "source_csv"]
            for k in keys:
                vals = [float(d[k]) for d in per_session if k in d and np.isfinite(d[k])]
                if vals:
                    agg[k] = float(np.mean(vals))
            rows.append(agg)
    return pd.DataFrame(rows)


def load_models_from_manifest(panel_dir: Path) -> List[str]:
    mj = panel_dir / "topology_manifest.json"
    if mj.exists():
        import json

        data = json.loads(mj.read_text(encoding="utf-8"))
        return [str(r["model_name"]) for r in data.get("manifest", [])]
    mc = panel_dir / "topology_manifest.csv"
    if mc.exists():
        df = pd.read_csv(mc)
        return df["model_name"].astype(str).tolist()
    raise FileNotFoundError(f"No topology_manifest in {panel_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Hail Mary sensitivity (Block C) collector")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument(
        "--topology-panel-dir",
        type=str,
        default="architecture_refinement/outputs/hail_mary/topology_panel",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--noise-type", type=str, default="gaussian")
    parser.add_argument("--output-csv", type=str, default="architecture_refinement/outputs/hail_mary/analysis/sensitivity_longform.csv")
    add_overwrite_arguments(parser)
    args = parser.parse_args()

    rr = Path(args.results_root)
    if not rr.is_absolute():
        rr = _REPO_ROOT / rr
    panel = Path(args.topology_panel_dir)
    if not panel.is_absolute():
        panel = _REPO_ROOT / panel

    models = load_models_from_manifest(panel)
    seeds = list(args.seeds) if args.seeds else [42, 43, 44]

    df = collect_sensitivity_longform(
        results_root=rr,
        dataset=args.dataset,
        model_names=models,
        seeds=seeds,
        noise_type=args.noise_type,
    )
    out = Path(args.output_csv)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    if not can_write_output(out, overwrite=args.overwrite):
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
