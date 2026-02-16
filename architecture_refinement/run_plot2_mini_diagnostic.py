"""
Plot 2 Diagnostic Mini Run — Coverage & Saturation Check (mini_diagnostic_spec.md).

Training-free, graph-only: generates random and TPE WS-Flex pools, computes
(C, L) coverage per regime, and compares TPE vs random to determine whether
the generator produces sufficient diversity to justify proxy-guided search.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import pandas as pd
from scipy import stats

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architecture_refinement.config import default_config
from architecture_refinement.topology_analyzer import TopologyAnalyzer
from architecture_refinement.metrics_te_orc import compute_te_orc
from architecture_refinement.ws_flex_generator import (
    make_ws_flex_graph,
    sample_modular_params,
    DEFAULT_M_VALUES,
    DEFAULT_P_OUT_LO,
    DEFAULT_P_OUT_HI,
    DEFAULT_R_OUT_LO,
    DEFAULT_R_OUT_HI,
)
from architecture_refinement.capacity_utils import capacity_filter

# Degree regimes: even k only (spec: super_sparse [2,6], sparse [7,12], moderate [13,18], near_dense [19,26])
DEGREE_REGIMES: Dict[str, List[int]] = {
    "super_sparse": [2, 4, 6],
    "sparse": [8, 10, 12],
    "moderate": [14, 16, 18],
    "near_dense": [20, 22, 24, 26],
}

# For _k_to_regime: (lo, hi) per regime
REGIME_BINS = {
    "super_sparse": (2, 6),
    "sparse": (7, 12),
    "moderate": (13, 18),
    "near_dense": (19, 26),
}


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
) -> nx.Graph:
    """Make graph by mode."""
    if generator_mode == "modular_ws_flex":
        sm_seed = seed_mod_params + sample_idx
        M, p_out, r_out = sample_modular_params(
            H, sm_seed, M_values=M_values, p_out_lo=p_out_lo, p_out_hi=p_out_hi,
            r_out_lo=r_out_lo, r_out_hi=r_out_hi,
        )
        G, _ = make_ws_flex_graph(
            H, k, p, seed,
            generator_mode="modular_ws_flex",
            M=M, p_out=p_out, r_out=r_out if use_r_out else None,
            k_out=k_out if not use_r_out else None,
        )
        return G
    return _make_ws_graph(H, k, p, seed)


def _k_to_regime(k: int, regime_bins: Optional[Dict[str, Tuple[int, int]]] = None) -> Optional[str]:
    regime_bins = regime_bins or REGIME_BINS
    k = int(k)
    for name, (lo, hi) in regime_bins.items():
        if lo <= k <= hi:
            return name
    return None


def _compute_row(
    G: nx.Graph,
    k: int,
    p: float,
    analyzer: TopologyAnalyzer,
) -> Dict[str, Any]:
    topo = analyzer.analyze_graph(G)
    te, orc, _ = compute_te_orc(G, orc_alpha=0.5)
    rho = float(topo.get("spectral_radius", float("nan")))
    return {
        "k": k,
        "regime": _k_to_regime(k),
        "density": float(topo.get("density", float("nan"))),
        "clustering": float(topo.get("clustering_coefficient", float("nan"))),
        "path_length": float(topo.get("avg_path_length", float("nan"))),
        "spectral_radius": rho if np.isfinite(rho) else float("nan"),
        "TE": float(te) if np.isfinite(te) else float("nan"),
        "ORC": float(orc) if np.isfinite(orc) else float("nan"),
    }


def run_random_pool(
    H: int,
    N_random: int,
    seed: int,
    analyzer: TopologyAnalyzer,
    max_attempts_factor: int = 50,
    generator_mode: str = "plain_ws_flex",
    capacity_filter_on: bool = False,
    *,
    seed_mod_params: int = 202607,
    M_values: Tuple[int, ...] = DEFAULT_M_VALUES,
    p_out_lo: float = DEFAULT_P_OUT_LO,
    p_out_hi: float = DEFAULT_P_OUT_HI,
    r_out_lo: float = DEFAULT_R_OUT_LO,
    r_out_hi: float = DEFAULT_R_OUT_HI,
    use_r_out: bool = True,
    k_out: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    k_values = sorted({k for ks in DEGREE_REGIMES.values() for k in ks})
    rows: List[Dict[str, Any]] = []
    max_attempts = N_random * max_attempts_factor
    attempts = 0
    while len(rows) < N_random:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Random pool: failed to get {N_random} connected graphs within {max_attempts} attempts (got {len(rows)})."
            )
        k = int(rng.choice(k_values))
        p = float(rng.uniform(0.0, 1.0))
        graph_seed = int(rng.integers(0, 2**31 - 1))
        G = _make_graph(
            H, k, p, graph_seed, generator_mode,
            seed_mod_params=seed_mod_params,
            M_values=M_values, p_out_lo=p_out_lo, p_out_hi=p_out_hi,
            r_out_lo=r_out_lo, r_out_hi=r_out_hi,
            use_r_out=use_r_out, k_out=k_out,
            sample_idx=len(rows),
        )
        if not nx.is_connected(G):
            continue
        if capacity_filter_on:
            cap_ok, _, _ = capacity_filter(G, k, 0, REGIME_BINS, H)
            if not cap_ok:
                continue
        row = _compute_row(G, k, p, analyzer)
        row["p"] = p
        row["graph_seed"] = graph_seed
        rows.append(row)
    return rows


def run_tpe_pool(
    H: int,
    N_tpe: int,
    seed: int,
    analyzer: TopologyAnalyzer,
    generator_mode: str = "plain_ws_flex",
    capacity_filter_on: bool = False,
    *,
    sample_idx_offset: int = 0,
    seed_mod_params: int = 202607,
    M_values: Tuple[int, ...] = DEFAULT_M_VALUES,
    p_out_lo: float = DEFAULT_P_OUT_LO,
    p_out_hi: float = DEFAULT_P_OUT_HI,
    r_out_lo: float = DEFAULT_R_OUT_LO,
    r_out_hi: float = DEFAULT_R_OUT_HI,
    use_r_out: bool = True,
    k_out: Optional[int] = None,
) -> List[Dict[str, Any]]:
    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            f"Optuna is required for TPE pool: {e}. Install with: pip install optuna"
        ) from e

    k_values = sorted({k for ks in DEGREE_REGIMES.values() for k in ks})
    tpe_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)

    def objective(trial: "optuna.Trial") -> Tuple[float, float]:
        k = int(trial.suggest_categorical("k", k_values))
        p = float(trial.suggest_float("p", 0.0, 1.0))
        graph_seed = int(rng.integers(0, 2**31 - 1))
        G = _make_graph(
            H, k, p, graph_seed, generator_mode,
            seed_mod_params=seed_mod_params,
            M_values=M_values, p_out_lo=p_out_lo, p_out_hi=p_out_hi,
            r_out_lo=r_out_lo, r_out_hi=r_out_hi,
            use_r_out=use_r_out, k_out=k_out,
            sample_idx=sample_idx_offset + len(tpe_rows),
        )
        if not nx.is_connected(G):
            raise optuna.TrialPruned()
        if capacity_filter_on:
            cap_ok, _, _ = capacity_filter(G, k, 0, REGIME_BINS, H)
            if not cap_ok:
                raise optuna.TrialPruned()
        row = _compute_row(G, k, p, analyzer)
        row["p"] = p
        row["graph_seed"] = graph_seed
        tpe_rows.append(row)
        return (float(row["TE"]), float(row["ORC"]))

    def stop_when_enough(study: "optuna.Study", trial: "optuna.FrozenTrial") -> None:
        if len(tpe_rows) >= N_tpe:
            study.stop()

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler)
    # Many trials can be pruned (disconnected); run until we have N_tpe completed
    study.optimize(
        objective,
        n_trials=max(N_tpe * 25, 2000),
        callbacks=[stop_when_enough],
        show_progress_bar=False,
    )
    if len(tpe_rows) < N_tpe:
        raise RuntimeError(
            f"TPE pool: got only {len(tpe_rows)} connected trials (target {N_tpe}). "
            "Try increasing n_trials or seed."
        )
    return tpe_rows[:N_tpe]


def _tertile_bins(series: pd.Series) -> Tuple[float, float]:
    """Return (q33, q66) for tertile boundaries (low < q33 <= mid < q66 <= high)."""
    q = series.dropna()
    if len(q) < 2:
        return float("nan"), float("nan")
    return float(q.quantile(1.0 / 3.0)), float(q.quantile(2.0 / 3.0))


def _assign_bin(val: float, q33: float, q66: float) -> str:
    if np.isnan(val) or (np.isnan(q33) and np.isnan(q66)):
        return "unknown"
    if np.isnan(q33) or np.isnan(q66):
        return "mid"
    if val <= q33:
        return "low"
    if val < q66:
        return "mid"
    return "high"


def _compute_z_rho_bin(rows: List[Dict[str, Any]], eps: float = 1e-8) -> None:
    """
    Add spectral_radius_norm and z_rho_bin to each row (in-place).
    z_rho = (ρ - median(ρ_bin)) / (MAD(ρ_bin) + ε) within regime × (C,L) bin.
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    if "spectral_radius" not in df.columns or df["spectral_radius"].isna().all():
        for r in rows:
            r["spectral_radius_norm"] = float("nan")
            r["z_rho_bin"] = float("nan")
        return
    bin_stats: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
    tertiles_by_regime: Dict[str, Tuple[float, float, float, float]] = {}
    for reg in df["regime"].dropna().unique():
        sub = df[df["regime"] == reg]
        if len(sub) < 2:
            continue
        q33_c, q66_c = _tertile_bins(sub["clustering"])
        q33_l, q66_l = _tertile_bins(sub["path_length"])
        tertiles_by_regime[str(reg)] = (q33_c, q66_c, q33_l, q66_l)
        sub = sub.copy()
        sub["C_bin"] = sub["clustering"].apply(lambda v: _assign_bin(v, q33_c, q66_c))
        sub["L_bin"] = sub["path_length"].apply(lambda v: _assign_bin(v, q33_l, q66_l))
        for (c_bin, l_bin), grp in sub.groupby(["C_bin", "L_bin"]):
            if c_bin == "unknown" or l_bin == "unknown":
                continue
            rhos = grp["spectral_radius"].dropna()
            if len(rhos) < 1:
                continue
            med = float(np.median(rhos))
            mad = float(np.median(np.abs(rhos - med))) if len(rhos) > 1 else eps
            bin_stats[(str(reg), c_bin, l_bin)] = (med, mad + eps)
    for r in rows:
        r["z_rho_bin"] = float("nan")
        r["spectral_radius_norm"] = float("nan")
        reg = r.get("regime")
        if reg is None or reg not in tertiles_by_regime:
            continue
        q33_c, q66_c, q33_l, q66_l = tertiles_by_regime[str(reg)]
        c_bin = _assign_bin(r.get("clustering", float("nan")), q33_c, q66_c)
        l_bin = _assign_bin(r.get("path_length", float("nan")), q33_l, q66_l)
        key = (str(reg), c_bin, l_bin)
        if key not in bin_stats or c_bin == "unknown" or l_bin == "unknown":
            continue
        rho = r.get("spectral_radius", float("nan"))
        if not np.isfinite(rho):
            continue
        med, denom = bin_stats[key]
        z = (rho - med) / denom
        r["z_rho_bin"] = float(z)
        r["spectral_radius_norm"] = float(z)


def compute_coverage_and_occupancy(
    df: pd.DataFrame,
    regime: str,
) -> Tuple[float, int, Dict[Tuple[str, str], int]]:
    sub = df[df["regime"] == regime].copy()
    if len(sub) < 2:
        return 0.0, 0, {}
    q33_c, q66_c = _tertile_bins(sub["clustering"])
    q33_l, q66_l = _tertile_bins(sub["path_length"])
    sub = sub.copy()
    sub["C_bin"] = sub["clustering"].apply(lambda v: _assign_bin(v, q33_c, q66_c))
    sub["L_bin"] = sub["path_length"].apply(lambda v: _assign_bin(v, q33_l, q66_l))
    occupancy: Dict[Tuple[str, str], int] = {}
    for _, r in sub.iterrows():
        key = (r["C_bin"], r["L_bin"])
        if key[0] == "unknown" or key[1] == "unknown":
            continue
        occupancy[key] = occupancy.get(key, 0) + 1
    occupied_bins = len(occupancy)
    coverage_score = occupied_bins / 9.0 if occupied_bins else 0.0
    return coverage_score, occupied_bins, occupancy


def run_analysis(
    random_path: Path,
    tpe_path: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    df_r = pd.read_csv(random_path)
    df_t = pd.read_csv(tpe_path)
    regime_names = list(DEGREE_REGIMES.keys())

    summary: Dict[str, Any] = {
        "coverage_by_regime_and_pool": {},
        "occupied_bins_by_regime_and_pool": {},
        "occupancy_by_regime_and_pool": {},
        "correlations_random": {},
        "correlations_tpe": {},
        "ks_tests_by_regime": {},
    }

    table_rows: List[Dict[str, Any]] = []

    for reg in regime_names:
        cov_r, occ_r, hist_r = compute_coverage_and_occupancy(df_r, reg)
        cov_t, occ_t, hist_t = compute_coverage_and_occupancy(df_t, reg)
        summary["coverage_by_regime_and_pool"][reg] = {"random": cov_r, "tpe": cov_t}
        summary["occupied_bins_by_regime_and_pool"][reg] = {"random": occ_r, "tpe": occ_t}
        summary["occupancy_by_regime_and_pool"][reg] = {
            "random": {f"{c}_{l}": cnt for (c, l), cnt in hist_r.items()},
            "tpe": {f"{c}_{l}": cnt for (c, l), cnt in hist_t.items()},
        }

        sub_r = df_r[df_r["regime"] == reg]
        sub_t = df_t[df_t["regime"] == reg]
        dominant_r = ""
        dominant_t = ""
        if hist_r:
            dominant_r = max(hist_r.items(), key=lambda x: x[1])[0]
            dominant_r = f"{dominant_r[0]}_{dominant_r[1]}"
        if hist_t:
            dominant_t = max(hist_t.items(), key=lambda x: x[1])[0]
            dominant_t = f"{dominant_t[0]}_{dominant_t[1]}"

        ks_c_stat, ks_c_pval = float("nan"), float("nan")
        ks_l_stat, ks_l_pval = float("nan"), float("nan")
        notes = []
        if sub_r["clustering"].notna().sum() >= 2 and sub_t["clustering"].notna().sum() >= 2:
            ks_c = stats.ks_2samp(sub_r["clustering"].dropna(), sub_t["clustering"].dropna())
            ks_c_stat, ks_c_pval = ks_c.statistic, ks_c.pvalue
            notes.append(f"KS_C={ks_c_stat:.3f}")
        if sub_r["path_length"].notna().sum() >= 2 and sub_t["path_length"].notna().sum() >= 2:
            ks_l = stats.ks_2samp(sub_r["path_length"].dropna(), sub_t["path_length"].dropna())
            ks_l_stat, ks_l_pval = ks_l.statistic, ks_l.pvalue
            notes.append(f"KS_L={ks_l_stat:.3f}")
        summary["ks_tests_by_regime"][reg] = {
            "KS_C_statistic": ks_c_stat,
            "KS_C_pvalue": ks_c_pval,
            "KS_L_statistic": ks_l_stat,
            "KS_L_pvalue": ks_l_pval,
        }

        table_rows.append({
            "Regime": reg,
            "Pool": "random",
            "Coverage": f"{cov_r:.3f}",
            "Occupied_bins": occ_r,
            "Dominant_bins": dominant_r,
            "Notes": "; ".join(notes) if notes else "",
        })
        table_rows.append({
            "Regime": reg,
            "Pool": "tpe",
            "Coverage": f"{cov_t:.3f}",
            "Occupied_bins": occ_t,
            "Dominant_bins": dominant_t,
            "Notes": "; ".join(notes) if notes else "",
        })

    def corr_series(a: pd.Series, b: pd.Series) -> float:
        mask = a.notna() & b.notna()
        if mask.sum() < 2:
            return float("nan")
        return float(a[mask].corr(b[mask]))

    for label, df in [("random", df_r), ("tpe", df_t)]:
        summary[f"correlations_{label}"] = {
            "k_clustering": corr_series(df["k"], df["clustering"]),
            "k_path_length": corr_series(df["k"], df["path_length"]),
            "ORC_clustering": corr_series(df["ORC"], df["clustering"]),
            "ORC_path_length": corr_series(df["ORC"], df["path_length"]),
        }

    return summary, table_rows


def write_csv(rows: List[Dict[str, Any]], path: Path, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    import logging
    logging.getLogger("optuna").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Plot 2 Mini Diagnostic: coverage & saturation check (training-free, graph-only)"
    )
    parser.add_argument("--H", type=int, default=32, help="Number of nodes (hidden size)")
    parser.add_argument("--N_random", type=int, default=2000, help="Random WS-Flex pool size")
    parser.add_argument("--N_tpe", type=int, default=2000, help="TPE WS-Flex trials (pool size)")
    parser.add_argument("--seed", type=int, default=202600, help="RNG seed")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: architecture_refinement/outputs/plot2_mini_diagnostic)",
    )
    parser.add_argument(
        "--exit_no_go",
        action="store_true",
        help="Exit with code 1 if Stage 0 gate is NO-GO (for automation).",
    )
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
    parser.add_argument("--M_values", type=str, default="2,4,8")
    parser.add_argument("--p_out_min", type=float, default=DEFAULT_P_OUT_LO)
    parser.add_argument("--p_out_max", type=float, default=DEFAULT_P_OUT_HI)
    parser.add_argument("--r_out_min", type=float, default=DEFAULT_R_OUT_LO)
    parser.add_argument("--r_out_max", type=float, default=DEFAULT_R_OUT_HI)
    parser.add_argument("--use_r_out", action="store_true", default=True)
    parser.add_argument("--no_use_r_out", action="store_false", dest="use_r_out")
    parser.add_argument("--k_out", type=int, default=2)
    parser.add_argument("--seed_mod_params", type=int, default=202607)
    args = parser.parse_args()

    H = max(2, int(args.H))
    N_random = max(1, int(args.N_random))
    N_tpe = max(1, int(args.N_tpe))
    seed = int(args.seed)

    out_dir = Path(args.out_dir) if args.out_dir else _THIS_DIR / "outputs" / "plot2_mini_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    analyzer = TopologyAnalyzer(default_config, logger=None)
    generator_mode = getattr(args, "generator_mode", "plain_ws_flex")
    fieldnames = ["k", "p", "graph_seed", "regime", "density", "clustering", "path_length", "spectral_radius", "spectral_radius_norm", "z_rho_bin", "TE", "ORC"]

    capacity_filter_on = getattr(args, "capacity_filter", False)
    M_values = tuple(int(x.strip()) for x in str(getattr(args, "M_values", "2,4,8")).split(",") if x.strip())
    if not M_values:
        M_values = DEFAULT_M_VALUES
    mod_kw = {
        "seed_mod_params": int(getattr(args, "seed_mod_params", 202607)),
        "M_values": M_values,
        "p_out_lo": float(getattr(args, "p_out_min", DEFAULT_P_OUT_LO)),
        "p_out_hi": float(getattr(args, "p_out_max", DEFAULT_P_OUT_HI)),
        "r_out_lo": float(getattr(args, "r_out_min", DEFAULT_R_OUT_LO)),
        "r_out_hi": float(getattr(args, "r_out_max", DEFAULT_R_OUT_HI)),
        "use_r_out": bool(getattr(args, "use_r_out", True)),
        "k_out": int(getattr(args, "k_out", 2)) if not getattr(args, "use_r_out", True) else None,
    }
    print(f"Generating random WS-Flex pool (mode={generator_mode}, capacity_filter={capacity_filter_on})...")
    random_rows = run_random_pool(
        H, N_random, seed, analyzer,
        generator_mode=generator_mode, capacity_filter_on=capacity_filter_on,
        **mod_kw,
    )
    print(f"  Got {len(random_rows)} random graphs")

    print("Generating TPE WS-Flex pool...")
    tpe_rows = run_tpe_pool(
        H, N_tpe, seed + 1, analyzer,
        generator_mode=generator_mode, capacity_filter_on=capacity_filter_on,
        sample_idx_offset=N_random,
        **mod_kw,
    )
    print(f"  Got {len(tpe_rows)} TPE graphs")

    combined = random_rows + tpe_rows
    _compute_z_rho_bin(combined)

    random_path = out_dir / "random_ws_flex_metrics.csv"
    write_csv(random_rows, random_path, fieldnames)
    print(f"  Wrote {len(random_rows)} rows to {random_path}")

    tpe_path = out_dir / "tpe_ws_flex_metrics.csv"
    write_csv(tpe_rows, tpe_path, fieldnames)
    print(f"  Wrote {len(tpe_rows)} rows to {tpe_path}")

    print("Running analysis...")
    summary, table_rows = run_analysis(random_path, tpe_path)

    # Stage 0 GO/NO-GO (Plot2_revision2): GO if >=2 regimes have >=5/9 (C,L) bins populated (random or TPE) OR TPE distinct (KS p < 0.05)
    cov_r = summary["coverage_by_regime_and_pool"]
    occ_r = summary["occupied_bins_by_regime_and_pool"]
    n_ok_random = sum(1 for r in cov_r if occ_r[r]["random"] >= 5)
    n_ok_tpe = sum(1 for r in cov_r if occ_r[r]["tpe"] >= 5)
    coverage_ok = n_ok_random >= 2 or n_ok_tpe >= 2
    tpe_distinct = False
    for reg, ks_data in summary.get("ks_tests_by_regime", {}).items():
        p_c = ks_data.get("KS_C_pvalue", float("nan"))
        p_l = ks_data.get("KS_L_pvalue", float("nan"))
        if (np.isfinite(p_c) and p_c < 0.05) or (np.isfinite(p_l) and p_l < 0.05):
            tpe_distinct = True
            break
    n_regimes = len(cov_r)
    n_low_both = sum(1 for r in cov_r if occ_r[r]["random"] <= 4 and occ_r[r]["tpe"] <= 4)
    no_go_condition = n_low_both >= max(1, n_regimes // 2) and not tpe_distinct
    stage0_go = coverage_ok or tpe_distinct
    stage0_gate = {
        "pass": stage0_go,
        "reason": "GO" if stage0_go else "NO-GO",
        "coverage_ok": coverage_ok,
        "n_regimes_ge_5_bins_random": n_ok_random,
        "n_regimes_ge_5_bins_tpe": n_ok_tpe,
        "tpe_distinct_from_random": tpe_distinct,
        "no_go_condition": no_go_condition,
        "n_regimes_low_both": n_low_both,
    }
    summary["stage0_gate"] = stage0_gate

    summary_path = out_dir / "diagnostic_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote {summary_path}")

    table_path = out_dir / "diagnostic_summary_table.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("Regime\tPool\tCoverage\tOccupied_bins\tDominant_bins\tNotes\n")
        for r in table_rows:
            f.write(
                f"{r['Regime']}\t{r['Pool']}\t{r['Coverage']}\t{r['Occupied_bins']}\t{r['Dominant_bins']}\t{r['Notes']}\n"
            )
    print(f"  Wrote {table_path}")

    print(f"Stage 0 gate: {stage0_gate['reason']} (pass={stage0_go})")
    if stage0_go:
        print("  GO: at least 2 regimes have ≥5/9 (C,L) bins populated (random or TPE) or TPE statistically distinct from random.")
    else:
        print("  NO-GO: fewer than 2 regimes with ≥5 bins in both pools and TPE ≈ random. Do not proceed to Stage 1.")
    if n_ok_random >= 2 and tpe_distinct:
        print("Outcome hint: Generator sufficient (≥5/9 bins in ≥2 regimes; TPE shows distinct (C,L) occupancy).")
    elif not stage0_go:
        print("Outcome hint: Generator insufficient. Expand search space before Stage 1.")
    else:
        print("Outcome hint: Mixed — some regimes OK. Consider coverage-aware selection + regime-specific tweaks.")

    if not stage0_go and getattr(args, "exit_no_go", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
