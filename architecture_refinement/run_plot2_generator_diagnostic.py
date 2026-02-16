"""
Phase 1: Generator feasibility and diversity diagnostic (Plot_2_Investigation.txt).

Training-free. Generates N graphs from the configured generator (plain WS or WS-Flex),
records validity, rejection reason, and derived metrics; reports feasibility by regime,
rejection histograms, and metric distributions. Gate: each regime must have >= 5% feasibility
or document which constraints make it infeasible.

For the full Stage 0 (plot2_revision) including N_random + N_tpe pools, coverage scores,
correlation matrices, and GO/NO-GO gate, use run_plot2_mini_diagnostic.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architecture_refinement.config import default_config
from architecture_refinement.topology_analyzer import TopologyAnalyzer
from architecture_refinement.metrics_te_orc import compute_te_orc
from architecture_refinement.ws_flex_generator import (
    make_ws_flex_graph,
    WSFlexParams,
    sample_modular_params,
    DEFAULT_M_VALUES,
    DEFAULT_P_OUT_LO,
    DEFAULT_P_OUT_HI,
    DEFAULT_R_OUT_LO,
    DEFAULT_R_OUT_HI,
)
from architecture_refinement.capacity_utils import capacity_filter

# Default k-regime bins (spec: super_sparse [2,6], sparse [7,12], moderate [13,18], near_dense [19,26])
DEFAULT_REGIME_BINS = {
    "super_sparse": (2, 6),
    "sparse": (7, 12),
    "moderate": (13, 18),
    "near_dense": (19, 26),
}


def _k_to_regime(k: int, regime_bins: Dict[str, Tuple[int, int]]) -> Optional[str]:
    k = int(k)
    for name, (lo, hi) in regime_bins.items():
        if lo <= k <= hi:
            return name
    return None


def _make_ws_graph(H: int, k: int, p: float, seed: int) -> nx.Graph:
    return nx.watts_strogatz_graph(int(H), int(k), float(p), seed=int(seed))


def _make_graph(
    H: int,
    k: int,
    p: float,
    seed: int,
    generator_mode: str,
    *,
    seed_mod_params: int = 202607,
    M_values: Tuple[int, ...] = DEFAULT_M_VALUES,
    p_out_lo: float = DEFAULT_P_OUT_LO,
    p_out_hi: float = DEFAULT_P_OUT_HI,
    r_out_lo: float = DEFAULT_R_OUT_LO,
    r_out_hi: float = DEFAULT_R_OUT_HI,
    use_r_out: bool = True,
    k_out: Optional[int] = None,
    sample_idx: int = 0,
) -> Tuple[nx.Graph, Optional[WSFlexParams]]:
    """Make graph by mode; returns (G, params) for metrics, params=None for plain."""
    if generator_mode == "modular_ws_flex":
        sm_seed = seed_mod_params + sample_idx
        M, p_out, r_out = sample_modular_params(
            H, sm_seed, M_values=M_values, p_out_lo=p_out_lo, p_out_hi=p_out_hi,
            r_out_lo=r_out_lo, r_out_hi=r_out_hi,
        )
        G, params = make_ws_flex_graph(
            H, k, p, seed,
            generator_mode="modular_ws_flex",
            M=M, p_out=p_out, r_out=r_out if use_r_out else None,
            k_out=k_out if not use_r_out else None,
        )
        return G, params
    G = _make_ws_graph(H, k, p, seed)
    return G, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 2 Phase 1: generator feasibility diagnostic")
    parser.add_argument("--H", type=int, default=32, help="Number of nodes (hidden size)")
    parser.add_argument("--N", type=int, default=2000, help="Number of graphs to attempt")
    parser.add_argument("--k_min", type=int, default=2, help="Minimum k (inclusive)")
    parser.add_argument("--k_max", type=int, default=26, help="Maximum k (inclusive)")
    parser.add_argument("--p_min", type=float, default=0.0)
    parser.add_argument("--p_max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=202600, help="RNG seed for sampling k, p")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (default: architecture_refinement/outputs/plot2_phase1)")
    parser.add_argument("--regime_bins", type=str, default=None, help="JSON object of regime -> [k_lo, k_hi], e.g. {\"super_sparse\":[2,6], ...}")
    parser.add_argument(
        "--generator_mode",
        type=str,
        default="plain_ws_flex",
        choices=["plain_ws_flex", "modular_ws_flex"],
        help="Generator mode: plain_ws_flex (default) or modular_ws_flex (Plot2_revision3).",
    )
    parser.add_argument(
        "--capacity_filter",
        action="store_true",
        help="Reject graphs whose E_active falls outside regime band (Plot2_revision3 Step C).",
    )
    parser.add_argument(
        "--M_values",
        type=str,
        default="2,4,8",
        help="Comma-separated module counts for modular_ws_flex (default: 2,4,8).",
    )
    parser.add_argument("--p_out_min", type=float, default=DEFAULT_P_OUT_LO)
    parser.add_argument("--p_out_max", type=float, default=DEFAULT_P_OUT_HI)
    parser.add_argument("--r_out_min", type=float, default=DEFAULT_R_OUT_LO)
    parser.add_argument("--r_out_max", type=float, default=DEFAULT_R_OUT_HI)
    parser.add_argument(
        "--use_r_out",
        action="store_true",
        default=True,
        help="Use r_out for inter-module edges (default: True).",
    )
    parser.add_argument(
        "--no_use_r_out",
        action="store_false",
        dest="use_r_out",
        help="Use k_out instead of r_out.",
    )
    parser.add_argument("--k_out", type=int, default=2, help="Inter-module edges when use_r_out=False.")
    parser.add_argument("--seed_mod_params", type=int, default=202607, help="RNG seed for modular param sampling.")
    args = parser.parse_args()

    H = max(2, int(args.H))
    N = max(1, int(args.N))
    k_min = max(2, int(args.k_min))
    k_max = min(H - 1, int(args.k_max))
    if k_max < k_min:
        k_max = k_min
    p_min = float(args.p_min)
    p_max = float(args.p_max)
    rng = np.random.default_rng(int(args.seed))

    if args.regime_bins:
        regime_bins = json.loads(args.regime_bins)
        regime_bins = {k: (int(v[0]), int(v[1])) for k, v in regime_bins.items()}
    else:
        regime_bins = dict(DEFAULT_REGIME_BINS)

    out_dir = Path(args.out_dir) if args.out_dir else _THIS_DIR / "outputs" / "plot2_phase1"
    out_dir.mkdir(parents=True, exist_ok=True)

    analyzer = TopologyAnalyzer(default_config, logger=None)

    # Collect all attempts
    attempts: List[Dict[str, Any]] = []
    k_values = list(range(k_min, k_max + 1))

    generator_mode = getattr(args, "generator_mode", "plain_ws_flex")
    M_values = tuple(int(x.strip()) for x in str(args.M_values).split(",") if x.strip())
    if not M_values:
        M_values = DEFAULT_M_VALUES

    for i in range(N):
        k = int(rng.choice(k_values))
        p = float(rng.uniform(p_min, p_max))
        graph_seed = int(rng.integers(0, 2**31 - 1))
        G, wsf_params = _make_graph(
            H, k, p, graph_seed, generator_mode,
            seed_mod_params=int(getattr(args, "seed_mod_params", 202607)),
            M_values=M_values,
            p_out_lo=float(getattr(args, "p_out_min", DEFAULT_P_OUT_LO)),
            p_out_hi=float(getattr(args, "p_out_max", DEFAULT_P_OUT_HI)),
            r_out_lo=float(getattr(args, "r_out_min", DEFAULT_R_OUT_LO)),
            r_out_hi=float(getattr(args, "r_out_max", DEFAULT_R_OUT_HI)),
            use_r_out=bool(getattr(args, "use_r_out", True)),
            k_out=int(getattr(args, "k_out", 2)) if not getattr(args, "use_r_out", True) else None,
            sample_idx=i,
        )
        regime = _k_to_regime(k, regime_bins)

        valid = nx.is_connected(G)
        rejection = None if valid else "disconnected"
        E_active = None
        if valid:
            cap_ok, cap_reason, E_active = capacity_filter(G, k, graph_seed, regime_bins, H)
            if getattr(args, "capacity_filter", False) and not cap_ok:
                valid = False
                rejection = cap_reason

        row = {
            "attempt": i + 1,
            "k": k,
            "p": p,
            "graph_seed": graph_seed,
            "regime": regime,
            "valid": valid,
            "rejection_reason": rejection,
            "density": None,
            "clustering": None,
            "path_length": None,
            "spectral_radius": None,
            "TE_raw": None,
            "ORC_raw": None,
            "generator_mode": generator_mode,
        }
        if wsf_params:
            row["M"] = wsf_params.M
            row["k_out"] = wsf_params.k_out
            row["p_out"] = wsf_params.p_out
            row["r_out"] = wsf_params.r_out
        if E_active is not None:
            row["E_active"] = E_active
        if valid:
            topo = analyzer.analyze_graph(G)
            row["density"] = float(topo.get("density", float("nan")))
            row["clustering"] = float(topo.get("clustering_coefficient", float("nan")))
            row["path_length"] = float(topo.get("avg_path_length", float("nan")))
            row["spectral_radius"] = float(topo.get("spectral_radius", float("nan"))) if "spectral_radius" in topo else None
            te, orc, _ = compute_te_orc(G, orc_alpha=0.5)
            row["TE_raw"] = float(te) if np.isfinite(te) else None
            row["ORC_raw"] = float(orc) if np.isfinite(orc) else None
        attempts.append(row)

    n_valid = sum(1 for r in attempts if r["valid"])
    feasibility_overall = n_valid / len(attempts) if attempts else 0.0

    # By regime (based on attempted k)
    regime_attempts: Dict[str, int] = {r: 0 for r in regime_bins}
    regime_accepts: Dict[str, int] = {r: 0 for r in regime_bins}
    unknown_attempts = 0
    unknown_accepts = 0
    for r in attempts:
        reg = r.get("regime")
        if reg is None:
            unknown_attempts += 1
            if r["valid"]:
                unknown_accepts += 1
        else:
            regime_attempts[reg] = regime_attempts.get(reg, 0) + 1
            if r["valid"]:
                regime_accepts[reg] = regime_accepts.get(reg, 0) + 1

    feasibility_by_regime = {}
    for reg in regime_bins:
        att = regime_attempts.get(reg, 0)
        acc = regime_accepts.get(reg, 0)
        feasibility_by_regime[reg] = (acc / att) if att > 0 else 0.0

    # Rejection reason histogram (overall and by regime)
    rejection_overall: Dict[str, int] = {}
    for r in attempts:
        reason = r.get("rejection_reason") or "accepted"
        rejection_overall[reason] = rejection_overall.get(reason, 0) + 1
    rejection_by_regime: Dict[str, Dict[str, int]] = {reg: {} for reg in regime_bins}
    for r in attempts:
        reg = r.get("regime")
        if reg is None:
            continue
        reason = r.get("rejection_reason") or "accepted"
        rejection_by_regime[reg][reason] = rejection_by_regime[reg].get(reason, 0) + 1

    # Metric distributions by regime (accepted only; same-length lists per regime)
    accepted = [r for r in attempts if r["valid"]]
    metric_distributions: Dict[str, Dict[str, List[float]]] = {}
    for reg in regime_bins:
        sub = [r for r in accepted if r.get("regime") == reg and r.get("TE_raw") is not None and r.get("ORC_raw") is not None]
        metric_distributions[reg] = {
            "k": [r["k"] for r in sub],
            "TE_raw": [r["TE_raw"] for r in sub],
            "ORC_raw": [r["ORC_raw"] for r in sub],
            "clustering": [r["clustering"] for r in sub if r.get("clustering") is not None],
            "path_length": [r["path_length"] for r in sub if r.get("path_length") is not None],
            "density": [r["density"] for r in sub if r.get("density") is not None],
            "spectral_radius": [r["spectral_radius"] for r in sub if r.get("spectral_radius") is not None and np.isfinite(r.get("spectral_radius", float("nan")))],
        }

    # Gate: each regime >= 5% feasibility
    min_feasibility = 0.05
    gate_pass = all(feasibility_by_regime.get(reg, 0) >= min_feasibility for reg in regime_bins)
    gate_detail = {}
    for reg in regime_bins:
        f = feasibility_by_regime.get(reg, 0)
        gate_detail[reg] = {"feasibility": f, "pass": f >= min_feasibility}

    report = {
        "schema_version": 1,
        "config": {
            "H": H,
            "N": N,
            "k_min": k_min,
            "k_max": k_max,
            "p_min": p_min,
            "p_max": p_max,
            "seed": args.seed,
            "regime_bins": {k: list(v) for k, v in regime_bins.items()},
            "generator_mode": generator_mode,
            "modular_params": {
                "M_values": list(M_values),
                "p_out_min": getattr(args, "p_out_min", DEFAULT_P_OUT_LO),
                "p_out_max": getattr(args, "p_out_max", DEFAULT_P_OUT_HI),
                "r_out_min": getattr(args, "r_out_min", DEFAULT_R_OUT_LO),
                "r_out_max": getattr(args, "r_out_max", DEFAULT_R_OUT_HI),
                "use_r_out": getattr(args, "use_r_out", True),
                "seed_mod_params": getattr(args, "seed_mod_params", 202607),
            } if generator_mode == "modular_ws_flex" else None,
        },
        "feasibility_rate_overall": feasibility_overall,
        "feasibility_rate_by_regime": feasibility_by_regime,
        "rejection_reason_histogram_overall": rejection_overall,
        "rejection_reason_histogram_by_regime": rejection_by_regime,
        "metric_distributions_by_regime": {
            k: {kk: vv for kk, vv in v.items() if vv}  # drop empty lists for brevity
            for k, v in metric_distributions.items()
        },
        "gate": {
            "min_feasibility": min_feasibility,
            "pass": gate_pass,
            "by_regime": gate_detail,
        },
        "n_attempted": len(attempts),
        "n_accepted": n_valid,
    }

    out_json = out_dir / "phase1_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Phase 1 report written to {out_json}")

    # Write accepted_graphs.csv for Phase 2/3 (only accepted rows with full metrics)
    import csv
    accepted_for_csv = [r for r in attempts if r["valid"] and r.get("TE_raw") is not None and r.get("ORC_raw") is not None]
    if accepted_for_csv:
        out_csv = out_dir / "accepted_graphs.csv"
        base_fields = ["k", "p", "graph_seed", "regime", "density", "clustering", "path_length", "spectral_radius", "TE_raw", "ORC_raw"]
        extras = [c for c in ["generator_mode", "M", "k_out", "p_out", "r_out"] if any(r.get(c) is not None for r in accepted_for_csv)]
        fieldnames = base_fields + extras
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(accepted_for_csv)
        print(f"Accepted graphs CSV written to {out_csv} ({len(accepted_for_csv)} rows)")
    print(f"Feasibility overall: {feasibility_overall:.2%}")
    for reg in regime_bins:
        print(f"  {reg}: {feasibility_by_regime.get(reg, 0):.2%}")
    print(f"Gate PASS: {gate_pass} (each regime >= {min_feasibility:.0%})")
    if not gate_pass:
        print("  Failing regimes:", [r for r, d in gate_detail.items() if not d["pass"]])


if __name__ == "__main__":
    main()
