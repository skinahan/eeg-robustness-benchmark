"""
Paper 3 Experiment 3 (Plot 3): Proxy-plane organization + hit-rate.

Uses trained results from Experiment 2. Computes:
- Proxy-plane robustness map (te_hat, orc_hat) vs RD_max
- Accuracy-robustness scatter (a_0 vs RD_max)
- Hit-rate: P(robust | G1) vs P(robust | G2) with bootstrap CI
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

from architecture_refinement.metrics_te_orc import compute_paper3_proxies
from architecture_refinement.paper3.arch_graph_utils import graph_from_architecture
from utils import short_run_id


def _collect_perturb_results(
    repo_root: Path,
    dataset: str,
    model_name: str,
    seed: int,
    noise_type: str,
) -> Optional[Dict[str, Any]]:
    """Collect clean_roc_auc, RD_max for one (model, seed)."""
    paradigm = "MotorImagery" if "BNCI" in dataset else "SSVEP"
    base = repo_root / "results" / paradigm / dataset
    for stem in [short_run_id(model_name), model_name]:
        path = base / stem / "CrossSessionEvaluation" / str(seed)
        if not path.exists():
            continue
        for p in path.rglob("*.csv"):
            if "test_perturb" not in str(p):
                continue
            try:
                df = pd.read_csv(p)
                if "noise_type" not in df.columns or noise_type not in df["noise_type"].astype(str).values:
                    continue
                sub = df[df["noise_type"].astype(str) == noise_type].copy()
                if sub.empty:
                    continue
                sub["corrupted_roc_auc"] = pd.to_numeric(sub["corrupted_roc_auc"], errors="coerce")
                sub["clean_roc_auc"] = pd.to_numeric(
                    sub.get("clean_roc_auc", sub.get("clean_score", np.nan)), errors="coerce"
                )
                clean = float(sub["clean_roc_auc"].iloc[0]) if sub["clean_roc_auc"].notna().any() else float("nan")
                roc_vals = sub["corrupted_roc_auc"].to_numpy()
                r_t = roc_vals / clean if np.isfinite(clean) and clean > 0 else np.full_like(roc_vals, np.nan)
                rd_max = float(np.nanmax(1.0 - r_t)) if np.isfinite(r_t).any() else float("nan")
                return {"clean_roc_auc": clean, "RD_max": rd_max}
            except Exception:
                pass
    return None


def _bootstrap_ci_diff(
    a: List[float],
    b: List[float],
    n_boot: int = 1000,
) -> Tuple[float, float, float]:
    """Bootstrap CI for mean(a) - mean(b). Returns (diff, lo, hi)."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    if len(a_arr) < 2 or len(b_arr) < 2:
        diff = float(np.mean(a_arr) - np.mean(b_arr)) if (len(a_arr) and len(b_arr)) else float("nan")
        return diff, float("nan"), float("nan")
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n_boot):
        sa = rng.choice(a_arr, size=len(a_arr), replace=True)
        sb = rng.choice(b_arr, size=len(b_arr), replace=True)
        diffs.append(float(np.mean(sa) - np.mean(sb)))
    diffs = np.array(diffs)
    return float(np.mean(a_arr) - np.mean(b_arr)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run_experiment3(
    experiment2_dir: Path,
    experiment1_dir: Path,
    output_dir: Path,
    dataset: str = "BNCI2014_001",
    robust_percentile: float = 20.0,
) -> Dict[str, Any]:
    """
    Run Experiment 3 analysis: proxy-plane map, hit-rate, scatter.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    exp2_dir = Path(experiment2_dir)
    exp1_dir = Path(experiment1_dir)
    if not exp2_dir.is_absolute():
        exp2_dir = _REPO_ROOT / exp2_dir
    if not exp1_dir.is_absolute():
        exp1_dir = _REPO_ROOT / exp1_dir

    manifest_path = exp2_dir / "experiment2_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Experiment 2 manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    groups = manifest.get("groups", {})
    g1_models = groups.get("G1", [])
    g2_models = groups.get("G2", [])
    g3_models = groups.get("G3", [])
    g4_models = groups.get("G4", [])
    g5_models = groups.get("G5", [])
    seeds = list(range(42, 42 + manifest.get("S", 5)))

    # Load proxy values for G1 and G2 from architecture JSONs
    arch_dir = exp2_dir / "experiment2_pilot" / "selected_architectures"
    if not arch_dir.exists():
        arch_dir = exp1_dir / "selected_architectures"

    def _get_proxy(model: str) -> Tuple[float, float]:
        for d in [arch_dir, exp1_dir / "selected_architectures"]:
            p = d / f"{model}.json"
            if p.exists():
                a = json.loads(p.read_text())
                te = float(a.get("te_hat", np.nan))
                oc = float(a.get("orc_hat", np.nan))
                if np.isfinite(te) and np.isfinite(oc):
                    return te, oc
                G = graph_from_architecture(a)
                if G is not None:
                    try:
                        te, oc = compute_paper3_proxies(G)
                        return te, oc
                    except Exception:
                        pass
                return te, oc
        return float("nan"), float("nan")

    # Collect results per (model, seed) for all groups G1–G5
    def _group_for_model(m: str) -> str:
        if m in g1_models:
            return "G1"
        if m in g2_models:
            return "G2"
        if m in g3_models:
            return "G3"
        if m in g4_models:
            return "G4"
        if m in g5_models:
            return "G5"
        return "unknown"

    all_models = g1_models + g2_models + g3_models + g4_models + g5_models
    rows: List[Dict[str, Any]] = []
    for model_name in all_models:
        te, oc = _get_proxy(model_name)
        group = _group_for_model(model_name)
        for seed in seeds:
            res = _collect_perturb_results(_REPO_ROOT, dataset, model_name, seed, "ar1_drift")
            if res:
                rows.append({
                    "model": model_name,
                    "group": group,
                    "seed": seed,
                    "te_hat": te,
                    "orc_hat": oc,
                    "clean_roc_auc": res["clean_roc_auc"],
                    "RD_max": res["RD_max"],
                })

    df = pd.DataFrame(rows)
    if df.empty:
        summary = {"error": "No results collected", "n_rows": 0}
        (output_dir / "experiment3_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    # Hit-rate: robust = lowest 20% RD_max among WS-Flex runs (G1 + G2 only)
    ws_flex = df[df["group"].isin(["G1", "G2"])]
    rd_ws = ws_flex["RD_max"].dropna()
    g1_total = len(df[df["group"] == "G1"])
    g2_total = len(df[df["group"] == "G2"])
    if len(rd_ws) >= 2 and g1_total > 0 and g2_total > 0:
        threshold = float(np.percentile(rd_ws, robust_percentile))
        df["robust"] = df["RD_max"] <= threshold
        g1_robust = df[df["group"] == "G1"]["robust"].sum()
        g2_robust = df[df["group"] == "G2"]["robust"].sum()
        hit_proxy = g1_robust / g1_total
        hit_uniform = g2_robust / g2_total
        # Bootstrap CI for hit-rate difference
        g1_binary = df[df["group"] == "G1"]["robust"].astype(float).tolist()
        g2_binary = df[df["group"] == "G2"]["robust"].astype(float).tolist()
        diff, lo, hi = _bootstrap_ci_diff(g1_binary, g2_binary)
    else:
        hit_proxy = hit_uniform = float("nan")
        diff = lo = hi = float("nan")
        threshold = float("nan")

    summary = {
        "n_rows": len(df),
        "robust_percentile": robust_percentile,
        "robust_threshold_RD_max": threshold,
        "hit_rate_proxy": hit_proxy,
        "hit_rate_uniform": hit_uniform,
        "hit_rate_diff": diff,
        "hit_rate_diff_ci": [lo, hi],
        "g1_n": g1_total,
        "g2_n": g2_total,
    }

    df.to_csv(output_dir / "experiment3_results.csv", index=False)
    (output_dir / "experiment3_summary.json").write_text(json.dumps(summary, indent=2))

    # Binned proxy-plane aggregation (2D grid)
    B = 6
    grid: Dict[Tuple[int, int], List[float]] = {}
    for _, r in df.iterrows():
        if np.isfinite(r["te_hat"]) and np.isfinite(r["orc_hat"]) and np.isfinite(r["RD_max"]):
            bi = min(int(r["te_hat"] * B), B - 1) if r["te_hat"] < 1.0 else B - 1
            bj = min(int(r["orc_hat"] * B), B - 1) if r["orc_hat"] < 1.0 else B - 1
            grid.setdefault((bi, bj), []).append(r["RD_max"])
    bin_means = {str(k): float(np.mean(v)) for k, v in grid.items()}
    summary["proxy_plane_bin_means"] = bin_means

    (output_dir / "experiment3_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Exp3] Hit-rate proxy={hit_proxy:.3f}, uniform={hit_uniform:.3f}, diff={diff:.3f} [{lo:.3f}, {hi:.3f}]")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Paper 3 Experiment 3: Proxy plane + hit-rate")
    parser.add_argument("--experiment2-dir", type=str, required=True)
    parser.add_argument("--experiment1-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="architecture_refinement/outputs/paper3_experiment3")
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument("--robust-percentile", type=float, default=20.0)
    args = parser.parse_args()

    exp2_dir = Path(args.experiment2_dir)
    exp1_dir = Path(args.experiment1_dir)
    if not exp2_dir.is_absolute():
        exp2_dir = _REPO_ROOT / exp2_dir
    if not exp1_dir.is_absolute():
        exp1_dir = _REPO_ROOT / exp1_dir
    out_dir = _REPO_ROOT / args.output_dir
    run_experiment3(
        experiment2_dir=exp2_dir,
        experiment1_dir=exp1_dir,
        output_dir=out_dir,
        dataset=args.dataset,
        robust_percentile=args.robust_percentile,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
