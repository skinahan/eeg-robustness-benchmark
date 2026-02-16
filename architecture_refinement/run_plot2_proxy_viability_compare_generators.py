"""
Plot 2 Proxy Viability: Generator Mode Comparison (G1 ws_flex vs G2 modular_ws_flex).

Runs proxy viability gates V1–V5 separately for ws_flex (G1) and modular_ws_flex (G2),
using bin edges derived from G0 (ws_flex only). Outputs side-by-side comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architecture_refinement.metrics_te_orc import compute_te_residual, compute_orc_residual
from architecture_refinement.run_plot2_proxy_viability import (
    DEGREE_REGIMES_DEFAULT,
    PARETO_MIN_REGIMES,
    PARETO_MIN_CELLS,
    _sample_g0_reference,
    _sample_g1_targeted,
    _sample_g2_targeted,
    _compute_frozen_bin_edges,
    _compute_mu_by_k,
    _assign_bins,
    _gate_v1,
    _gate_v2,
    _gate_v3,
    _gate_v4,
    _gate_v5,
    _pareto_membership_2d,
)


def _compute_occupancy(df: pd.DataFrame, degree_regimes: Dict[str, List[int]]) -> Dict[str, Dict[str, int]]:
    """Occupancy counts per (regime, C_bin, L_bin)."""
    occ: Dict[str, Dict[str, int]] = {}
    for reg in degree_regimes:
        sub = df[df["regime"] == reg]
        cell_counts: Dict[str, int] = {}
        for _, r in sub.iterrows():
            cb, lb = r.get("C_bin"), r.get("L_bin")
            if cb and lb and cb != "unknown" and lb != "unknown":
                key = f"{cb}_{lb}"
                cell_counts[key] = cell_counts.get(key, 0) + 1
        occ[str(reg)] = cell_counts
    return occ


def _compute_pareto_width(df: pd.DataFrame) -> int:
    """Number of points on Pareto front in (TE_res, sigma)."""
    te_res = df["TE_res"].values if "TE_res" in df.columns else df["TE"].values
    sigma = df["sigma"].values
    valid = np.isfinite(te_res) & np.isfinite(sigma)
    if valid.sum() < 2:
        return 0
    xs = np.asarray(te_res, dtype=float)
    ys = np.asarray(sigma, dtype=float)
    pf_mask = _pareto_membership_2d(xs, ys) & valid
    return int(np.sum(pf_mask))


def _compute_redundancy_correlations(df: pd.DataFrame) -> Dict[str, float]:
    """Correlations for redundancy check."""
    out: Dict[str, float] = {}
    for P in ["TE_res", "sigma"]:
        if P not in df.columns:
            continue
        if "k" in df.columns:
            c = df[[P, "k"]].corr().loc[P, "k"]
            out[f"corr_{P}_k"] = float(c) if np.isfinite(c) else float("nan")
        if "density" in df.columns:
            c = df[[P, "density"]].corr().loc[P, "density"]
            out[f"corr_{P}_density"] = float(c) if np.isfinite(c) else float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot 2 Proxy Viability: Compare ws_flex (G1) vs modular_ws_flex (G2) using G0 bin edges."
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--M_ref", type=int, default=2000, help="Samples per mode (G0, G1, G2)")
    parser.add_argument("--seed", type=int, default=202600)
    parser.add_argument("--seed_mod_params", type=int, default=202607)
    parser.add_argument(
        "--relaxed_v2",
        action="store_true",
        help="Use relaxed V2 (min 2 regimes, 5 cells on Pareto) for both modes.",
    )
    parser.add_argument(
        "--relaxed_v4",
        action="store_true",
        help="Use relaxed V4 for near_dense (5/9 cells) for both modes.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    H = int(args.H)
    M_ref = int(args.M_ref)
    seed = int(args.seed)
    seed_mod_params = int(args.seed_mod_params)
    degree_regimes = DEGREE_REGIMES_DEFAULT

    rng = np.random.default_rng(seed)

    # G0: ws_flex neutral reference
    print("[COMPARE] Sampling G0 reference (ws_flex, uniform k,p)...")
    g0_rows, g0_hashes = _sample_g0_reference(H, degree_regimes, M_ref, rng)
    mu_orc_by_k = _compute_mu_by_k(g0_rows, "ORC")
    mu_te_by_k = _compute_mu_by_k(g0_rows, "TE")
    for r in g0_rows:
        r["TE_res"] = compute_te_residual(r["TE"], int(r["k"]), mu_te_by_k)
        r["ORC_res"] = compute_orc_residual(r["ORC"], int(r["k"]), mu_orc_by_k)
    bin_edges = _compute_frozen_bin_edges(g0_rows, degree_regimes)
    _assign_bins(g0_rows, bin_edges, degree_regimes)

    # G1: ws_flex targeted
    print("[COMPARE] Sampling G1 (ws_flex targeted)...")
    g1_rows, g1_hashes = _sample_g1_targeted(
        H, degree_regimes, M_ref, rng, bin_edges, mu_te_by_k, mu_orc_by_k
    )
    df_g1 = pd.DataFrame(g1_rows)

    # G2: modular_ws_flex targeted
    print("[COMPARE] Sampling G2 (modular_ws_flex targeted)...")
    g2_rows, g2_hashes = _sample_g2_targeted(
        H, degree_regimes, M_ref, rng, bin_edges, mu_te_by_k, mu_orc_by_k,
        seed_mod_params=seed_mod_params,
    )
    df_g2 = pd.DataFrame(g2_rows)

    # Gates per mode (relaxed_v2/v4 apply to both ws_flex and modular_ws_flex)
    v2_min_regimes = 2 if args.relaxed_v2 else PARETO_MIN_REGIMES
    v2_min_cells = 5 if args.relaxed_v2 else PARETO_MIN_CELLS
    v4_min_cells_near_dense = 5 if args.relaxed_v4 else None

    v1_g1, v1_reasons_g1 = _gate_v1(df_g1)
    v1_g2, v1_reasons_g2 = _gate_v1(df_g2)
    v2_g1, v2_reasons_g1 = _gate_v2(df_g1, min_regimes=v2_min_regimes, min_cells=v2_min_cells)
    v2_g2, v2_reasons_g2 = _gate_v2(df_g2, min_regimes=v2_min_regimes, min_cells=v2_min_cells)
    v3_g1, v3_reasons_g1 = _gate_v3(df_g1, degree_regimes)
    v3_g2, v3_reasons_g2 = _gate_v3(df_g2, degree_regimes)
    v4_g1, v4_reasons_g1 = _gate_v4(df_g1, degree_regimes, min_cells_near_dense=v4_min_cells_near_dense)
    v4_g2, v4_reasons_g2 = _gate_v4(df_g2, degree_regimes, min_cells_near_dense=v4_min_cells_near_dense)
    v5_g1, v5_reasons_g1 = _gate_v5(g0_hashes, g1_hashes, None)
    v5_g2, v5_reasons_g2 = _gate_v5(g0_hashes, g2_hashes, None)

    # Occupancy and Pareto
    occ_g1 = _compute_occupancy(df_g1, degree_regimes)
    occ_g2 = _compute_occupancy(df_g2, degree_regimes)
    n_occupied_g1 = sum(len(cells) for cells in occ_g1.values())
    n_occupied_g2 = sum(len(cells) for cells in occ_g2.values())
    pareto_g1 = _compute_pareto_width(df_g1)
    pareto_g2 = _compute_pareto_width(df_g2)
    corr_g1 = _compute_redundancy_correlations(df_g1)
    corr_g2 = _compute_redundancy_correlations(df_g2)

    # GO2 comparison
    occupancy_increase = (n_occupied_g2 - n_occupied_g1) / max(1, n_occupied_g1)
    pareto_increase_ok = pareto_g2 >= 2 * pareto_g1 if pareto_g1 > 0 else pareto_g2 >= 10

    report = {
        "schema_version": 1,
        "H": H,
        "M_ref": M_ref,
        "seed": seed,
        "seed_mod_params": seed_mod_params,
        "relaxed_v2": args.relaxed_v2,
        "relaxed_v4": args.relaxed_v4,
        "bin_edge_source": "G0_neutral_reference",
        "gates_per_mode": {
            "ws_flex": {
                "V1_redundancy": {"pass": v1_g1, "reasons": v1_reasons_g1},
                "V2_pareto_width": {"pass": v2_g1, "reasons": v2_reasons_g1},
                "V3_cell_feasibility": {"pass": v3_g1, "reasons": v3_reasons_g1},
                "V4_cell_occupancy": {"pass": v4_g1, "reasons": v4_reasons_g1},
                "V5_overlap": {"pass": v5_g1, "reasons": v5_reasons_g1},
            },
            "modular_ws_flex": {
                "V1_redundancy": {"pass": v1_g2, "reasons": v1_reasons_g2},
                "V2_pareto_width": {"pass": v2_g2, "reasons": v2_reasons_g2},
                "V3_cell_feasibility": {"pass": v3_g2, "reasons": v3_reasons_g2},
                "V4_cell_occupancy": {"pass": v4_g2, "reasons": v4_reasons_g2},
                "V5_overlap": {"pass": v5_g2, "reasons": v5_reasons_g2},
            },
        },
        "occupancy_per_mode": {
            "ws_flex": {k: dict(v) for k, v in occ_g1.items()},
            "modular_ws_flex": {k: dict(v) for k, v in occ_g2.items()},
        },
        "n_occupied_cells_total": {"ws_flex": n_occupied_g1, "modular_ws_flex": n_occupied_g2},
        "pareto_width_per_mode": {"ws_flex": pareto_g1, "modular_ws_flex": pareto_g2},
        "redundancy_correlations_per_mode": {"ws_flex": corr_g1, "modular_ws_flex": corr_g2},
        "go_comparison": {
            "occupancy_increase": float(occupancy_increase),
            "occupancy_increase_ok": occupancy_increase >= 0.25,
            "pareto_increase_ok": pareto_increase_ok,
            "go2_pass": occupancy_increase >= 0.25 and pareto_increase_ok,
        },
    }

    (output_dir / "generator_comparison_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    df_g1.to_csv(output_dir / "g1_ws_flex_metrics.csv", index=False)
    df_g2.to_csv(output_dir / "g2_modular_ws_flex_metrics.csv", index=False)

    print("[COMPARE] Report written to", output_dir / "generator_comparison_report.json")
    print("  ws_flex:      V1={} V2={} V3={} V4={} V5={} | occupied={} pareto={}".format(
        v1_g1, v2_g1, v3_g1, v4_g1, v5_g1, n_occupied_g1, pareto_g1))
    print("  modular_ws_flex: V1={} V2={} V3={} V4={} V5={} | occupied={} pareto={}".format(
        v1_g2, v2_g2, v3_g2, v4_g2, v5_g2, n_occupied_g2, pareto_g2))
    print("  GO2: occupancy_increase={:.2%} (need >=25%), pareto_ok={}".format(
        occupancy_increase, pareto_increase_ok))


if __name__ == "__main__":
    main()
