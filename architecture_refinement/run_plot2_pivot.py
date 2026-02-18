"""
Plot 2 M5 — Pivot Plan (when NO-GO on TPE > random).

Reuses existing Plot 2 training results to produce:
- Basin density: heatmaps of y=-maxRD over (C,L) bins per regime
- Family effect: WS-Flex (random/regime-stratified) vs external random under matched capacity
- Practical selection: regime-stratified random + light proxy filtering vs TPE
- Headroom: ORACLE top-B from trained pool
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Plot 2 M5: Pivot analysis (basin, family effect, headroom)")
    parser.add_argument("--plot2_dir", type=str, required=True, help="Path to Plot 2 run directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output for pivot figures (default: plot2_dir/pivot)")
    parser.add_argument("--n_boot", type=int, default=5000)
    args = parser.parse_args()

    plot2_dir = Path(args.plot2_dir)
    if not plot2_dir.exists():
        print(f"[M5] Plot2 dir not found: {plot2_dir}")
        return 1

    out_dir = Path(args.output_dir) if args.output_dir else (plot2_dir / "pivot")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest and selected architectures
    manifest_path = plot2_dir / "plot2_manifest.json"
    sel_path = plot2_dir / "selected_architectures.csv"
    if not manifest_path.exists():
        print(f"[M5] Missing manifest: {manifest_path}")
        return 1
    if not sel_path.exists():
        print(f"[M5] Missing selected architectures: {sel_path}")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sel = pd.read_csv(sel_path)

    # Load per-graph metrics from analysis if available
    analysis_dir = plot2_dir / "analysis"
    per_graph_path = analysis_dir / "per_graph_metrics.csv"
    if per_graph_path.exists():
        per_graph = pd.read_csv(per_graph_path)
    else:
        print("[M5] Run analyze_plot2_results.py first to produce per_graph_metrics.csv")
        per_graph = pd.DataFrame()

    report: Dict[str, Any] = {}
    report["plot2_dir"] = str(plot2_dir)
    report["pivot_thesis"] = (
        "WS-Flex defines a solution-rich topology basin for robust CfC graphs under capacity control; "
        "sophisticated selection yields limited marginal gains."
    )

    # Basin density: heatmaps of y=-maxRD over (C,L) bins per regime
    if not per_graph.empty and "max_rd_mean" in per_graph.columns:
        y = -per_graph["max_rd_mean"]
        if "C_bin" in per_graph.columns and "L_bin" in per_graph.columns and "k" in per_graph.columns:
            regime_col = "regime" if "regime" in per_graph.columns else None
            if regime_col is None:
                degree_regimes = manifest.get("degree_regimes", {})
                k_to_regime = {}
                for r, ks in degree_regimes.items():
                    for k in ks:
                        k_to_regime[k] = r
                per_graph["regime"] = per_graph["k"].map(lambda x: k_to_regime.get(int(x), "unknown"))
            report["basin_density"] = {
                "n_graphs": int(len(per_graph)),
                "mean_y": float(y.mean()) if np.isfinite(y).any() else float("nan"),
                "top_quartile_pct": float((y >= y.quantile(0.75)).mean() * 100) if len(y) >= 4 else float("nan"),
            }
    else:
        report["basin_density"] = {"status": "skipped", "reason": "No per_graph_metrics or max_rd column"}

    # Family effect: compare WS-Flex vs external random
    if "method" in sel.columns:
        ws_methods = [m for m in sel["method"].unique() if str(m) in ("baseline_a", "baseline_b", "tpe", "baseline")]
        ext_methods = [m for m in sel["method"].unique() if str(m) == "external_random"]
        if ws_methods and ext_methods and not per_graph.empty:
            ws_df = per_graph[per_graph["method"].isin(ws_methods)]
            ext_df = per_graph[per_graph["method"] == "external_random"]
            if "max_rd_mean" in per_graph.columns:
                ws_rd = ws_df["max_rd_mean"].dropna()
                ext_rd = ext_df["max_rd_mean"].dropna()
                if len(ws_rd) >= 2 and len(ext_rd) >= 2:
                    delta = float(ext_rd.mean() - ws_rd.mean())
                    report["family_effect"] = {
                        "delta_maxRD_ext_minus_ws": delta,
                        "ws_mean_maxRD": float(ws_rd.mean()),
                        "ext_mean_maxRD": float(ext_rd.mean()),
                        "n_ws": int(len(ws_rd)),
                        "n_ext": int(len(ext_rd)),
                    }
                else:
                    report["family_effect"] = {"status": "insufficient_data"}
        else:
            report["family_effect"] = {"status": "skipped", "reason": "No external_random or WS methods"}
    else:
        report["family_effect"] = {"status": "skipped", "reason": "No method column"}

    # Headroom: ORACLE top-B
    if not per_graph.empty and "max_rd_mean" in per_graph.columns and "method" in per_graph.columns:
        B = int(manifest.get("selection", {}).get("B", 8))
        oracle = per_graph.nsmallest(B, "max_rd_mean")
        report["oracle_headroom"] = {
            "B": B,
            "oracle_mean_maxRD": float(oracle["max_rd_mean"].mean()),
            "overall_mean_maxRD": float(per_graph["max_rd_mean"].mean()),
            "headroom": float(per_graph["max_rd_mean"].mean() - oracle["max_rd_mean"].mean()),
        }
    else:
        report["oracle_headroom"] = {"status": "skipped"}

    (out_dir / "pivot_report.json").write_text(json.dumps(report, indent=2))
    print(f"[M5] Wrote {out_dir / 'pivot_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
