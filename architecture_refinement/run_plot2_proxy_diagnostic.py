"""
Phase 2: Objective proxy sanity – TE/ORC vs density (Plot_2_Investigation.txt).

Tests whether TE and ORC monotonically track k/density (which would explain selection collapse).
Input: Phase 1 report JSON (or CSV of accepted graphs). Computes correlations and Pareto width.
Gate: PASS if Pareto set includes multiple regimes; FAIL if >80% of Pareto points in one regime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architecture_refinement.pareto_hv import pareto_front_2d


def _pareto_membership_2d(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Max-max Pareto membership; True if point is on the front."""
    n = int(xs.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=bool)
    order = np.lexsort((-ys, -xs))
    best_y = -np.inf
    on_pf = np.zeros((n,), dtype=bool)
    for idx in order:
        y = float(ys[idx])
        if y > best_y:
            on_pf[idx] = True
            best_y = y
    return on_pf


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 2 Phase 2: TE/ORC vs density proxy diagnostic")
    parser.add_argument("--phase1_report", type=str, default=None, help="Path to phase1_report.json from Phase 1")
    parser.add_argument("--accepted_csv", type=str, default=None, help="Alternative: path to CSV with columns k, density, TE_raw, ORC_raw, regime")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (default: same as phase1_report or cwd)")
    parser.add_argument("--pareto_regime_frac_threshold", type=float, default=0.8, help="FAIL if > this fraction of Pareto points in one regime (default 0.8)")
    args = parser.parse_args()

    if args.phase1_report:
        with open(args.phase1_report, encoding="utf-8") as f:
            report = json.load(f)
        dists = report.get("metric_distributions_by_regime", {})
        # Rebuild flat arrays: k, density, TE_raw, ORC_raw, regime (label)
        k_list, density_list, te_list, orc_list, regime_list = [], [], [], [], []
        for reg, d in dists.items():
            ks = d.get("k", [])
            dens = d.get("density", [])
            tes = d.get("TE_raw", [])
            orcs = d.get("ORC_raw", [])
            n = max(len(ks), len(dens), len(tes), len(orcs))
            for i in range(n):
                k_list.append(ks[i] if i < len(ks) else None)
                density_list.append(dens[i] if i < len(dens) else None)
                te_list.append(tes[i] if i < len(tes) else None)
                orc_list.append(orcs[i] if i < len(orcs) else None)
                regime_list.append(reg)
        # Drop rows with missing TE or ORC
        valid = [i for i in range(len(te_list)) if te_list[i] is not None and orc_list[i] is not None]
        k_arr = np.array([k_list[i] for i in valid], dtype=float)
        density_arr = np.array([density_list[i] if density_list[i] is not None else np.nan for i in valid], dtype=float)
        te_arr = np.array([te_list[i] for i in valid], dtype=float)
        orc_arr = np.array([orc_list[i] for i in valid], dtype=float)
        regime_arr = np.array([regime_list[i] for i in valid], dtype=object)
        out_dir = Path(args.out_dir) if args.out_dir else Path(args.phase1_report).parent
    elif args.accepted_csv:
        import pandas as pd
        df = pd.read_csv(args.accepted_csv)
        for col in ["k", "TE_raw", "ORC_raw"]:
            if col not in df.columns:
                raise ValueError(f"CSV must have column {col}")
        df = df.dropna(subset=["TE_raw", "ORC_raw"])
        k_arr = df["k"].to_numpy(dtype=float)
        density_arr = df["density"].to_numpy(dtype=float) if "density" in df.columns else np.full(len(df), np.nan)
        te_arr = df["TE_raw"].to_numpy(dtype=float)
        orc_arr = df["ORC_raw"].to_numpy(dtype=float)
        regime_arr = df["regime"].to_numpy(dtype=object) if "regime" in df.columns else np.array(["unknown"] * len(df), dtype=object)
        out_dir = Path(args.out_dir) if args.out_dir else Path(args.accepted_csv).parent
    else:
        parser.error("Provide either --phase1_report or --accepted_csv")

    n = len(te_arr)
    if n < 2:
        print("Too few accepted points for correlation/Pareto analysis.")
        sys.exit(1)

    # Correlations (drop NaN for each pair)
    def corr(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 2:
            return float("nan")
        return float(np.corrcoef(x[mask], y[mask])[0, 1])

    correlations = {
        "TE_raw_vs_k": corr(te_arr, k_arr),
        "TE_raw_vs_density": corr(te_arr, density_arr) if np.any(np.isfinite(density_arr)) else float("nan"),
        "ORC_raw_vs_k": corr(orc_arr, k_arr),
        "ORC_raw_vs_density": corr(orc_arr, density_arr) if np.any(np.isfinite(density_arr)) else float("nan"),
        "TE_raw_vs_ORC_raw": corr(te_arr, orc_arr),
    }

    # Pareto front (max-max TE, ORC)
    pareto_mask = _pareto_membership_2d(te_arr, orc_arr)
    pareto_points = np.sum(pareto_mask)
    pareto_regimes = regime_arr[pareto_mask]
    unique_regimes = list(np.unique(regime_arr))
    pareto_per_regime = {str(r): int(np.sum(pareto_regimes == r)) for r in unique_regimes}
    max_regime_frac = (max(pareto_per_regime.values()) / pareto_points) if pareto_points > 0 else 0.0
    gate_pass = max_regime_frac <= args.pareto_regime_frac_threshold and len([c for c in pareto_per_regime.values() if c > 0]) >= 2

    report = {
        "schema_version": 1,
        "n_accepted": int(n),
        "correlations": correlations,
        "pareto_width": {
            "n_pareto_total": int(pareto_points),
            "n_pareto_by_regime": pareto_per_regime,
            "max_regime_fraction": float(max_regime_frac),
        },
        "gate": {
            "pareto_regime_frac_threshold": args.pareto_regime_frac_threshold,
            "pass": bool(gate_pass),
            "reason": "Pareto set includes multiple regimes" if gate_pass else f">{args.pareto_regime_frac_threshold:.0%} of Pareto points in one regime",
        },
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "phase2_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Phase 2 report written to {out_json}")
    print("Correlations:", correlations)
    print(f"Pareto total: {pareto_points}, by regime: {pareto_per_regime}")
    print(f"Gate PASS: {gate_pass}")


if __name__ == "__main__":
    main()
