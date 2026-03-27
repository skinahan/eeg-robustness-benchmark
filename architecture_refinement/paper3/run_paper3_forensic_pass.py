"""
Paper 3 Forensic Pass: mine existing experiment3_results (and optional stratified pilot)
to answer variance-vs-topology, regime effects, WS-Flex graph diversity, and fragility regions.

Outputs: forensic_pass_summary.json, CSV tables, optional PDF figures.
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

from architecture_refinement.config import default_config
from architecture_refinement.metrics_te_orc import compute_paper3_proxies
from architecture_refinement.paper3.arch_graph_utils import graph_from_architecture
from architecture_refinement.paper3.run_paper3_analysis_followups import (
    _load_architecture_metadata,
    analyze_stratified_by_degree,
)
from architecture_refinement.paper3.run_paper3_q3_stratified_analysis import (
    merge_plan_topology_meta,
    merge_stratified_longform_for_analysis,
    topology_table_from_longform,
)
from architecture_refinement.topology_analyzer import TopologyAnalyzer
from utils import short_run_id

DEFAULT_K_BINS = [(2, 4), (6, 8), (10, 12), (14, 16), (18, 20), (22, 24)]


def _resolve_paths(
    experiment3_csv: Path,
    experiment2_dir: Path,
) -> Tuple[Path, Path, Path]:
    exp3 = Path(experiment3_csv)
    if not exp3.is_absolute():
        exp3 = _REPO_ROOT / exp3
    e2 = Path(experiment2_dir)
    if not e2.is_absolute():
        e2 = _REPO_ROOT / e2
    arch_dir = e2 / "experiment2_pilot" / "selected_architectures"
    if not arch_dir.exists():
        arch_dir = e2.parent / "experiment1" / "selected_architectures"
    return exp3, e2, arch_dir


def _load_manifest(exp2_dir: Path) -> Dict[str, Any]:
    p = exp2_dir / "experiment2_manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"experiment2_manifest.json not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _proxy_row(row: pd.Series) -> float:
    te = float(row.get("te_hat", np.nan))
    oc = float(row.get("orc_hat", np.nan))
    if not (np.isfinite(te) and np.isfinite(oc)):
        return float("nan")
    return 0.5 * (te + abs(oc))


def analyze_variance_decomposition_longform(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Public API: ICC(1) and variance ratios from long-form columns model, seed, RD_max.
    Used by forensic targeted reruns (Option A).
    """
    sub = df.dropna(subset=["RD_max", "model"]).copy()
    if sub.empty:
        return {"error": "no rows"}
    lab, inv = np.unique(sub["model"].astype(str).values, return_inverse=True)
    y = sub["RD_max"].to_numpy(dtype=float)
    icc_block = _icc_one_way_random(y, inv)

    per = sub.groupby("model", as_index=False).agg(
        RD_mean=("RD_max", "mean"),
        RD_var=("RD_max", lambda x: float(np.var(x, ddof=1)) if len(x) > 1 else float("nan")),
        n_seeds=("RD_max", "count"),
    )
    means = per["RD_mean"].to_numpy(dtype=float)
    between_var = float(np.var(means, ddof=1)) if len(means) > 1 else float("nan")
    wvars = per["RD_var"].to_numpy(dtype=float)
    wmask = per["n_seeds"].to_numpy() > 1
    mean_within = float(np.nanmean(wvars[wmask])) if wmask.any() else float("nan")
    ratio = (
        float(between_var / mean_within)
        if np.isfinite(between_var) and np.isfinite(mean_within) and mean_within > 1e-15
        else float("nan")
    )

    return {
        "icc_one_way_random": icc_block,
        "between_topology_variance_of_means": between_var,
        "mean_within_topology_variance": mean_within,
        "variance_ratio_between_over_mean_within": ratio,
        "n_topologies": int(len(per)),
        "per_topology": per.to_dict(orient="records"),
    }


def _icc_one_way_random(y: np.ndarray, group_ids: np.ndarray) -> Dict[str, Any]:
    """
    ICC(1) single measure, one-way random effects (Shrout & Fleiss style).
    y: observations; group_ids: same-length integer labels for group membership.
    """
    y = np.asarray(y, dtype=float).ravel()
    g = np.asarray(group_ids, dtype=int).ravel()
    mask = np.isfinite(y)
    y, g = y[mask], g[mask]
    if len(y) < 4:
        return {"icc": float("nan"), "note": "n<4"}

    uniq = np.unique(g)
    k = len(uniq)
    if k < 2:
        return {"icc": float("nan"), "note": "single group"}

    n_list = []
    means = []
    for u in uniq:
        sub = y[g == u]
        n_list.append(len(sub))
        means.append(float(np.mean(sub)))

    N = int(np.sum(n_list))
    grand = float(np.mean(y))

    ss_between = sum(n_list[i] * (means[i] - grand) ** 2 for i in range(k))
    ss_within = 0.0
    for i, u in enumerate(uniq):
        sub = y[g == u]
        ss_within += float(np.sum((sub - means[i]) ** 2))

    df_b = k - 1
    df_w = N - k
    if df_b <= 0 or df_w <= 0:
        return {"icc": float("nan"), "note": "df error"}

    msb = ss_between / df_b
    msw = ss_within / df_w if df_w > 0 else float("nan")

    n0 = (N - sum(ni**2 for ni in n_list) / N) / (k - 1) if k > 1 else float(N / k)
    denom = msb + (n0 - 1) * msw
    icc = (msb - msw) / denom if np.isfinite(denom) and denom > 1e-15 else float("nan")

    return {
        "icc": float(icc) if np.isfinite(icc) else float("nan"),
        "ms_between": float(msb),
        "ms_within": float(msw),
        "n_groups": k,
        "n_total": N,
        "n0": float(n0),
        "group_sizes": n_list,
    }


def block_a_topology_vs_seed(
    df: pd.DataFrame,
    *,
    groups: Tuple[str, ...] = ("G2", "G1"),
) -> Dict[str, Any]:
    """Variance within vs between topologies; proxy bands; ICC."""
    out: Dict[str, Any] = {}
    df = df.dropna(subset=["RD_max", "model", "seed"]).copy()
    df["proxy_score_row"] = df.apply(_proxy_row, axis=1)

    for gname in groups:
        sub = df[df["group"] == gname].copy()
        if sub.empty:
            out[gname] = {"error": "no rows"}
            continue

        per_seed = sub.groupby(["model"], as_index=False).agg(
            RD_mean=("RD_max", "mean"),
            RD_var=("RD_max", lambda x: float(np.var(x, ddof=1)) if len(x) > 1 else float("nan")),
            RD_std=("RD_max", "std"),
            n_seeds=("RD_max", "count"),
            te_hat=("te_hat", "first"),
            orc_hat=("orc_hat", "first"),
        )
        per_seed["proxy_topo"] = 0.5 * (
            per_seed["te_hat"].astype(float) + np.abs(per_seed["orc_hat"].astype(float))
        )

        means = per_seed["RD_mean"].to_numpy(dtype=float)
        between_var = float(np.var(means, ddof=1)) if len(means) > 1 else float("nan")

        wvars = per_seed["RD_var"].to_numpy(dtype=float)
        wmask = np.isfinite(wvars) & (per_seed["n_seeds"].to_numpy() > 1)
        mean_within_var = float(np.nanmean(wvars[wmask])) if wmask.any() else float("nan")

        ratio = (
            float(between_var / mean_within_var)
            if (np.isfinite(between_var) and np.isfinite(mean_within_var) and mean_within_var > 1e-15)
            else float("nan")
        )

        # ICC: long-form y and group index
        lab, inv = np.unique(sub["model"].astype(str).values, return_inverse=True)
        icc_block = _icc_one_way_random(sub["RD_max"].to_numpy(dtype=float), inv)

        domination = False
        if np.isfinite(mean_within_var) and np.isfinite(between_var):
            domination = mean_within_var >= between_var or (np.isfinite(ratio) and ratio < 1.0)

        # Proxy quintiles (topology-level proxy)
        band_rows: List[Dict[str, Any]] = []
        pq = per_seed.dropna(subset=["proxy_topo"])
        if len(pq) >= 5:
            try:
                pq = pq.copy()
                pq["proxy_band"] = pd.qcut(
                    pq["proxy_topo"], q=5, labels=False, duplicates="drop"
                )
                for band in sorted(pq["proxy_band"].dropna().unique()):
                    bsub = pq[pq["proxy_band"] == band]
                    mstd = bsub["RD_std"].to_numpy(dtype=float)
                    band_rows.append({
                        "proxy_quintile_index": int(band),
                        "n_topologies": int(len(bsub)),
                        "mean_seed_std_RD_max": float(np.nanmean(mstd)),
                        "mean_within_var_RD_max": float(
                            np.nanmean(bsub["RD_var"].to_numpy(dtype=float))
                        ),
                    })
            except Exception as e:
                band_rows.append({"error": str(e)})

        out[gname] = {
            "n_topologies": int(len(per_seed)),
            "between_topology_variance_of_means": between_var,
            "mean_within_topology_variance_across_seeds": mean_within_var,
            "variance_ratio_between_over_mean_within": ratio,
            "interpretation_optimization_stochasticity_dominates": domination,
            "icc_one_way_random": icc_block,
            "proxy_quintile_bands": band_rows,
            "per_topology_summary": per_seed.to_dict(orient="records"),
        }

    return out


def _topology_table_main_exp3(
    df: pd.DataFrame,
    groups: Tuple[str, ...] = ("G2", "G1"),
) -> pd.DataFrame:
    """Mean RD per (model, group), std across seeds, first te/orc."""
    sub = df[df["group"].isin(groups)].copy()
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby(["model", "group"], as_index=False).agg(
        RD_max_per_topo=("RD_max", "mean"),
        RD_max_std_seeds=("RD_max", "std"),
        n_seeds=("seed", "count"),
        te_hat=("te_hat", "first"),
        orc_hat=("orc_hat", "first"),
    )
    g["proxy_score"] = 0.5 * (g["te_hat"].astype(float) + np.abs(g["orc_hat"].astype(float)))
    return g


def _bootstrap_mean_ci(vals: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(x, size=len(x), replace=True))) for _ in range(n_boot)]
    return float(np.mean(x)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def block_b_regime_effects(
    topo_df: pd.DataFrame,
    arch_dir: Path,
    *,
    group: str = "G2",
    heatmap_bins: int = 5,
    k_bins: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """TE/ORC deciles, 2D heatmap, TE×k table."""
    k_bins = k_bins or list(DEFAULT_K_BINS)
    sub = topo_df[topo_df["group"] == group].dropna(subset=["RD_max_per_topo", "te_hat", "orc_hat"]).copy()
    if sub.empty:
        return {"error": f"no {group} topology rows"}

    sub["orc_abs"] = np.abs(sub["orc_hat"].astype(float))

    # k, p from arch
    k_list: List[int] = []
    p_list: List[float] = []
    for m in sub["model"].astype(str):
        meta = _load_architecture_metadata(arch_dir, m)
        if meta is None:
            for stem in [m, short_run_id(m)]:
                pth = arch_dir / f"{stem}.json"
                if pth.exists():
                    meta = json.loads(pth.read_text(encoding="utf-8"))
                    break
        if meta:
            k_list.append(int(meta.get("k", -1)))
            p_list.append(float(meta.get("p", -1.0)))
        else:
            k_list.append(-1)
            p_list.append(-1.0)
    sub["k"] = k_list
    sub["p"] = p_list

    def _stratum_for_k(k: int) -> str:
        if k < 0:
            return "k_unknown"
        for lo, hi in k_bins:
            if lo <= k <= hi:
                return f"k_{lo}-{hi}"
        return "k_other"

    sub["k_stratum"] = sub["k"].apply(_stratum_for_k)

    te_deciles: List[Dict[str, Any]] = []
    orc_deciles: List[Dict[str, Any]] = []
    n_dec = min(10, max(3, len(sub) // 3))
    try:
        sub["_te_d"] = pd.qcut(sub["te_hat"], q=n_dec, labels=False, duplicates="drop")
        for d in sorted(sub["_te_d"].dropna().unique()):
            b = sub[sub["_te_d"] == d]
            m, lo, hi = _bootstrap_mean_ci(b["RD_max_per_topo"].to_numpy(dtype=float))
            te_deciles.append({
                "te_decile": int(d),
                "n_topologies": int(len(b)),
                "mean_RD_max": m,
                "ci95_lo": lo,
                "ci95_hi": hi,
            })
    except Exception as e:
        te_deciles.append({"error": str(e)})

    try:
        sub["_oc_d"] = pd.qcut(sub["orc_abs"], q=n_dec, labels=False, duplicates="drop")
        for d in sorted(sub["_oc_d"].dropna().unique()):
            b = sub[sub["_oc_d"] == d]
            m, lo, hi = _bootstrap_mean_ci(b["RD_max_per_topo"].to_numpy(dtype=float))
            orc_deciles.append({
                "orc_abs_decile": int(d),
                "n_topologies": int(len(b)),
                "mean_RD_max": m,
                "ci95_lo": lo,
                "ci95_hi": hi,
            })
    except Exception as e:
        orc_deciles.append({"error": str(e)})

    # 2D heatmap bins
    heatmap: Dict[str, Any] = {}
    try:
        sub["_te_h"] = pd.qcut(sub["te_hat"], q=heatmap_bins, labels=False, duplicates="drop")
        sub["_or_h"] = pd.qcut(sub["orc_abs"], q=heatmap_bins, labels=False, duplicates="drop")
        cells: List[Dict[str, Any]] = []
        for i in sorted(sub["_te_h"].dropna().unique()):
            for j in sorted(sub["_or_h"].dropna().unique()):
                b = sub[(sub["_te_h"] == i) & (sub["_or_h"] == j)]
                if b.empty:
                    continue
                cells.append({
                    "te_bin": int(i),
                    "orc_bin": int(j),
                    "n": int(len(b)),
                    "mean_RD_max": float(np.mean(b["RD_max_per_topo"])),
                    "mean_seed_std": float(np.nanmean(b["RD_max_std_seeds"].to_numpy(dtype=float))),
                })
        heatmap["cells"] = cells
        heatmap["n_bins"] = heatmap_bins
    except Exception as e:
        heatmap["error"] = str(e)

    # TE decile × k_stratum
    te_k_table: List[Dict[str, Any]] = []
    if "_te_d" in sub.columns:
        for d in sorted(sub["_te_d"].dropna().unique()):
            for ks in sorted(sub["k_stratum"].unique()):
                b = sub[(sub["_te_d"] == d) & (sub["k_stratum"] == ks)]
                if b.empty:
                    continue
                te_k_table.append({
                    "te_decile": int(d),
                    "k_stratum": str(ks),
                    "n": int(len(b)),
                    "mean_RD_max": float(np.mean(b["RD_max_per_topo"])),
                })

    sub_out = sub.drop(columns=[c for c in sub.columns if str(c).startswith("_")], errors="ignore")
    return {
        "group": group,
        "te_decile_table": te_deciles,
        "orc_abs_decile_table": orc_deciles,
        "te_orc_heatmap": heatmap,
        "te_decile_by_k_stratum": te_k_table,
        "topology_rows": sub_out,
    }


def block_c_ws_flex_diversity(
    manifest: Dict[str, Any],
    arch_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Graph metrics for G1+G2 architectures."""
    groups = manifest.get("groups", {})
    models = list(groups.get("G1", [])) + list(groups.get("G2", []))
    analyzer = TopologyAnalyzer(default_config)
    rows: List[Dict[str, Any]] = []

    for model in models:
        arch_path = None
        arch = None
        for stem in [model, short_run_id(model)]:
            p = arch_dir / f"{stem}.json"
            if p.exists():
                arch_path = p
                arch = json.loads(p.read_text(encoding="utf-8"))
                break
        if arch is None:
            rows.append({"model": model, "error": "arch_not_found"})
            continue

        G = graph_from_architecture(arch)
        if G is None:
            rows.append({"model": model, "error": "graph_from_architecture failed"})
            continue

        te_hat, orc_hat = compute_paper3_proxies(G)
        metrics = analyzer.analyze_graph(G)
        row: Dict[str, Any] = {
            "model": model,
            "n_nodes": int(G.number_of_nodes()),
            "n_edges": int(G.number_of_edges()),
            "density": float(nx_density_safe(G)),
            "avg_degree": float(np.mean([d for _, d in G.degree()])),
            "te_hat_recomputed": float(te_hat),
            "orc_hat_recomputed": float(orc_hat),
            "clustering_coefficient": metrics.get("clustering_coefficient", float("nan")),
            "avg_path_length": metrics.get("avg_path_length", float("nan")),
            "algebraic_connectivity": metrics.get("algebraic_connectivity", float("nan")),
            "k": int(arch.get("k", -1)),
            "p": float(arch.get("p", -1.0)),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    summary: Dict[str, Any] = {"n_graphs": len(df), "metrics": {}}
    numeric_cols = [
        "density",
        "avg_degree",
        "te_hat_recomputed",
        "orc_hat_recomputed",
        "clustering_coefficient",
        "avg_path_length",
        "algebraic_connectivity",
    ]
    for c in numeric_cols:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        if v.empty:
            continue
        summary["metrics"][c] = {
            "min": float(v.min()),
            "max": float(v.max()),
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "iqr": float(v.quantile(0.75) - v.quantile(0.25)),
            "cv": float(v.std() / v.mean()) if abs(float(v.mean())) > 1e-12 else float("nan"),
        }

    return df, summary


def nx_density_safe(G: Any) -> float:
    import networkx as nx

    try:
        return float(nx.density(G))
    except Exception:
        return float("nan")


def block_d_fragility(
    topo_df: pd.DataFrame,
    *,
    group: str = "G2",
    high_rd_quantile: float = 0.75,
    high_std_quantile: float = 0.75,
    bottom_decile: float = 0.1,
) -> Dict[str, Any]:
    """Flag fragile cells (high mean RD + high seed std); list worst decile topologies."""
    sub = topo_df[topo_df["group"] == group].dropna(subset=["RD_max_per_topo"]).copy()
    if sub.empty:
        return {"error": f"no {group} rows"}

    rd = sub["RD_max_per_topo"].to_numpy(dtype=float)
    st = sub["RD_max_std_seeds"].to_numpy(dtype=float)
    if not np.isfinite(st).all():
        st = np.nan_to_num(st, nan=0.0)

    thr_rd = float(np.quantile(rd, high_rd_quantile))
    thr_st = float(np.quantile(st, high_std_quantile))
    fragile_mask = (rd >= thr_rd) & (st >= thr_st)
    fragile = sub.loc[fragile_mask, ["model", "te_hat", "orc_hat", "RD_max_per_topo", "RD_max_std_seeds"]].copy()

    cut = float(np.quantile(rd, bottom_decile))
    worst = sub[sub["RD_max_per_topo"] <= cut].sort_values("RD_max_per_topo")
    worst_list = worst[
        ["model", "te_hat", "orc_hat", "k", "p", "RD_max_per_topo"]
    ].to_dict(orient="records") if all(c in worst.columns for c in ["k", "p"]) else worst.to_dict(orient="records")

    return {
        "group": group,
        "fragility_thresholds": {
            "high_RD_max_q": high_rd_quantile,
            "high_seed_std_q": high_std_quantile,
            "rd_threshold": thr_rd,
            "seed_std_threshold": thr_st,
        },
        "n_fragile_topologies": int(fragile_mask.sum()),
        "fragile_topologies": fragile.to_dict(orient="records"),
        "bottom_decile_fraction": bottom_decile,
        "rd_cut_bottom_decile": cut,
        "worst_topologies": worst_list,
    }


def _merge_k_p_onto_topo(topo_df: pd.DataFrame, arch_dir: Path) -> pd.DataFrame:
    out = topo_df.copy()
    ks, ps = [], []
    for m in out["model"].astype(str):
        meta = _load_architecture_metadata(arch_dir, m)
        if meta is None:
            for stem in [m, short_run_id(m)]:
                pth = arch_dir / f"{stem}.json"
                if pth.exists():
                    meta = json.loads(pth.read_text(encoding="utf-8"))
                    break
        ks.append(int(meta.get("k", -1)) if meta else -1)
        ps.append(float(meta.get("p", -1.0)) if meta else -1.0)
    out["k"] = ks
    out["p"] = ps
    return out


def _build_conclusions(
    block_a: Dict[str, Any],
    block_c_summary: Dict[str, Any],
    block_d: Dict[str, Any],
    family: Optional[Dict[str, Any]],
) -> List[str]:
    bullets: List[str] = []
    g2 = block_a.get("G2", {})
    if g2.get("interpretation_optimization_stochasticity_dominates"):
        bullets.append(
            "G2: Within-topology (seed) variance is comparable to or larger than between-topology "
            "variance of mean RD_max — optimization stochasticity may dominate fine-grained topology effects."
        )
    icc = (g2.get("icc_one_way_random") or {}).get("icc")
    if icc is not None and np.isfinite(icc) and icc < 0.05:
        bullets.append(
            f"G2: ICC of RD_max across topologies is very low ({icc:.4f}), indicating weak repeatability of "
            "mean robustness across distinct topologies relative to within-topology noise."
        )

    met = block_c_summary.get("metrics", {})
    for key in ("te_hat_recomputed", "orc_hat_recomputed", "avg_path_length", "clustering_coefficient"):
        m = met.get(key, {})
        if m and "iqr" in m and "mean" in m:
            rel = m["iqr"] / (abs(m["mean"]) + 1e-9)
            if rel < 0.15:
                bullets.append(
                    f"WS-Flex (G1+G2) pool shows narrow spread for {key} (IQR/mean≈{rel:.3f}); "
                    "within-family dynamical diversity may be limited."
                )
                break

    if block_d.get("n_fragile_topologies", 0) > 0:
        bullets.append(
            f"Identified {block_d['n_fragile_topologies']} topologies in high-RD and high seed-std regime "
            "(fragile / unstable training outcomes)."
        )

    if family and family.get("group_means"):
        bullets.append(
            "See family_contrast for coarse G1–G5 topology-class means of RD_max (out-of-family narrative)."
        )

    if not bullets:
        bullets.append(
            "Review block JSON tables and figures; no automatic strong narrative was inferred (mixed or weak signals)."
        )
    return bullets


def analyze_family_contrast_g1_g5(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Coarse group-level contrast (Option B): mean RD_max per topology, then summarize by group G1–G5.
    """
    sub = df.dropna(subset=["RD_max", "model", "group"])
    per = sub.groupby(["group", "model"], as_index=False)["RD_max"].mean()
    rows: Dict[str, Any] = {}
    for g in ["G1", "G2", "G3", "G4", "G5"]:
        v = per[per["group"] == g]["RD_max"].to_numpy(dtype=float)
        if len(v) == 0:
            continue
        rows[g] = {
            "n_topologies": int(len(v)),
            "mean_of_topology_means": float(np.mean(v)),
            "std_across_topologies": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "min": float(np.min(v)),
            "max": float(np.max(v)),
        }
    return {"group_means": rows, "note": "Uses same topology-mean aggregation as selection comparison."}


def run_forensic_pass(
    *,
    experiment3_csv: Path,
    experiment2_dir: Path,
    output_dir: Path,
    stratified_pilot_root: Optional[Path] = None,
    stratified_csv: Optional[Path] = None,
    figures: bool = False,
) -> Dict[str, Any]:
    exp3_csv, exp2_dir, arch_dir = _resolve_paths(experiment3_csv, experiment2_dir)
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = _REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(exp3_csv)
    manifest = _load_manifest(exp2_dir)

    topo_main = _topology_table_main_exp3(df, groups=("G1", "G2", "G3", "G4", "G5"))
    topo_main = _merge_k_p_onto_topo(topo_main, arch_dir)

    block_a = block_a_topology_vs_seed(df, groups=("G2", "G1"))

    block_b = block_b_regime_effects(topo_main, arch_dir, group="G2")
    block_b_df = block_b.pop("topology_rows", None)
    if block_b_df is not None and isinstance(block_b_df, pd.DataFrame):
        block_b_df.to_csv(output_dir / "forensic_block_b_topology_rows.csv", index=False)

    div_df, div_summary = block_c_ws_flex_diversity(manifest, arch_dir)
    div_df.to_csv(output_dir / "forensic_ws_flex_graph_metrics.csv", index=False)

    topo_g2 = topo_main[topo_main["group"] == "G2"].copy()
    block_d = block_d_fragility(topo_g2.assign(group="G2"), group="G2")

    family = analyze_family_contrast_g1_g5(df)

    # Stratified longform (optional): extra regime plots data
    strat_note = None
    if stratified_pilot_root and stratified_csv:
        sp = Path(stratified_pilot_root)
        sc = Path(stratified_csv)
        if not sp.is_absolute():
            sp = _REPO_ROOT / sp
        if not sc.is_absolute():
            sc = _REPO_ROOT / sc
        if sp.exists() and sc.exists():
            try:
                long_df = merge_stratified_longform_for_analysis(sp, sc, exp3_csv)
                long_df.to_csv(output_dir / "forensic_stratified_longform.csv", index=False)
                topo_s = topology_table_from_longform(long_df)
                topo_s = merge_plan_topology_meta(topo_s, sp)
                topo_s.to_csv(output_dir / "forensic_stratified_topology_table.csv", index=False)
                strat_note = "merged stratified pilot longform + topology table"
            except Exception as e:
                strat_note = f"stratified merge failed: {e}"

    combined: Dict[str, Any] = {
        "block_a_variance_seed_vs_topology": block_a,
        "block_b_regime_effects": block_b,
        "block_c_ws_flex_diversity": div_summary,
        "block_d_fragility": block_d,
        "family_contrast_g1_g5": family,
        "paths": {
            "experiment3_csv": str(exp3_csv),
            "experiment2_dir": str(exp2_dir),
            "arch_dir": str(arch_dir),
        },
        "stratified_merge": strat_note,
        "conclusions_auto": _build_conclusions(block_a, div_summary, block_d, family),
        "targeted_rerun_notes": {
            "option_a_variance_decomposition": (
                "After selecting 6–10 diverse topologies from the heatmap or q3_stratified_sample_plan.json, "
                "copy forensic_rerun_seeds_manifest.example.json, set model_names and source_architectures_dir, "
                "then: python -m architecture_refinement.paper3.run_paper3_forensic_rerun train --manifest <path>. "
                "Collect RD_max per (model, seed) into a CSV and run: ... analyze --csv <path> --output-json variance_decomposition.json"
            ),
            "option_b_out_of_family": (
                "Use family_contrast_g1_g5 in this summary and run_paper3_q3_stratified_analysis analyze_selection_comparison "
                "for coarse G1–G5 bars. If within WS-Flex is flat but groups differ, prefer a family-level claim. "
                "Minimal new runs: add matched-n baselines in experiment2 groups rather than more WS-Flex NAS."
            ),
        },
    }

    (output_dir / "forensic_pass_summary.json").write_text(
        json.dumps(combined, indent=2, default=str),
        encoding="utf-8",
    )

    # Degree stratification (reuse experiment3 df + arch)
    _, deg_summary = analyze_stratified_by_degree(df, arch_dir)
    (output_dir / "forensic_stratified_by_degree_summary.json").write_text(
        json.dumps({"stratified_by_degree": deg_summary}, indent=2),
        encoding="utf-8",
    )

    if figures:
        _plot_figures(
            output_dir=output_dir,
            block_b=block_b,
            div_df=div_df,
            topo_main=topo_main,
            group="G2",
        )

    return combined


def _plot_figures(
    *,
    output_dir: Path,
    block_b: Dict[str, Any],
    div_df: pd.DataFrame,
    topo_main: pd.DataFrame,
    group: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Decile line plot: TE
    te_t = block_b.get("te_decile_table") or []
    if te_t and "te_decile" in (te_t[0] or {}):
        xs = [r["te_decile"] for r in te_t if "te_decile" in r]
        ys = [r["mean_RD_max"] for r in te_t if "mean_RD_max" in r]
        if xs and ys:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(xs, ys, "o-", color="#0072B2")
            ax.set_xlabel("TE decile (low to high)")
            ax.set_ylabel("Mean RD_max (topology mean)")
            ax.set_title(f"Forensic: RD_max vs TE decile ({group})")
            fig.tight_layout()
            fig.savefig(output_dir / "forensic_te_deciles_rd_max.pdf", dpi=300, bbox_inches="tight")
            plt.close()

    # 2D heatmap
    hm = block_b.get("te_orc_heatmap") or {}
    cells = hm.get("cells") or []
    if cells:
        nb = int(hm.get("n_bins", 5))
        grid = np.full((nb, nb), np.nan, dtype=float)
        cnt = np.zeros((nb, nb), dtype=float)
        for c in cells:
            i, j = int(c["te_bin"]), int(c["orc_bin"])
            if 0 <= i < nb and 0 <= j < nb:
                grid[i, j] = c["mean_RD_max"]
                cnt[i, j] = c["n"]
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(grid.T, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xlabel("TE bin")
        ax.set_ylabel("|ORC| bin")
        ax.set_title(f"Mean RD_max ({group})")
        for i in range(nb):
            for j in range(nb):
                if cnt[i, j] > 0:
                    ax.text(i, j, int(cnt[i, j]), ha="center", va="center", color="w", fontsize=7)
        plt.colorbar(im, ax=ax, label="mean RD_max")
        fig.tight_layout()
        fig.savefig(output_dir / "forensic_te_orc_heatmap_rd_max.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    # Histograms for diversity
    cols = ["avg_degree", "clustering_coefficient", "avg_path_length", "te_hat_recomputed"]
    avail = [c for c in cols if c in div_df.columns and div_df[c].notna().any()]
    if avail:
        fig, axes = plt.subplots(2, 2, figsize=(7, 6))
        axes = axes.ravel()
        for ax, c in zip(axes, avail[:4]):
            v = pd.to_numeric(div_df[c], errors="coerce").dropna()
            ax.hist(v, bins=min(20, max(5, len(v) // 2)), color="#4A90A4", edgecolor="black", linewidth=0.5)
            ax.set_title(c)
        fig.suptitle("WS-Flex (G1+G2) graph metric distributions")
        fig.tight_layout()
        fig.savefig(output_dir / "forensic_ws_flex_metric_histograms.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    sub = topo_main[topo_main["group"] == group]
    if not sub.empty and len(sub) > 2:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.scatter(
            sub["proxy_score"],
            sub["RD_max_per_topo"],
            c="#0072B2",
            alpha=0.75,
            edgecolors="k",
            linewidths=0.5,
        )
        ax.set_xlabel("Proxy score")
        ax.set_ylabel("RD_max per topology")
        ax.set_title(f"Forensic: proxy vs robustness ({group})")
        fig.tight_layout()
        fig.savefig(output_dir / "forensic_proxy_vs_rd_scatter.pdf", dpi=300, bbox_inches="tight")
        plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper 3 Forensic Pass (existing results)")
    parser.add_argument("--experiment3-csv", type=str, required=True)
    parser.add_argument("--experiment2-dir", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="architecture_refinement/outputs/paper3/forensic_pass",
    )
    parser.add_argument(
        "--stratified-pilot-root",
        type=str,
        default=None,
        help="Optional: q3_stratified_pilot directory for merged longform",
    )
    parser.add_argument(
        "--stratified-csv",
        type=str,
        default=None,
        help="Optional: q3_stratified_experiment3_results.csv",
    )
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()

    sp = Path(args.stratified_pilot_root) if args.stratified_pilot_root else None
    sc = Path(args.stratified_csv) if args.stratified_csv else None

    out = run_forensic_pass(
        experiment3_csv=Path(args.experiment3_csv),
        experiment2_dir=Path(args.experiment2_dir),
        output_dir=Path(args.output_dir),
        stratified_pilot_root=sp,
        stratified_csv=sc,
        figures=args.figures,
    )
    print(json.dumps({"conclusions_auto": out.get("conclusions_auto"), "output": args.output_dir}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
