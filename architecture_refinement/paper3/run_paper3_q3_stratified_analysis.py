"""
Paper 3 Q3 follow-up analysis: stratified proxy–robustness scatter (3a) and selection comparison bars (3b).

Reads q3_stratified_sample_plan.json + stratified CSV + main experiment3_results.csv for reuse rows.

Outputs include plot3_q3_metric_robustness_alignment_latest.pdf: same visual language as the original
plot3_q3_metric_robustness scatter (G1–G5), with the stratified pilot overlaid and correlation
statistics computed on the stratified sample only (Experiment A). Also plot3a_q3_te_orc_vs_robustness.pdf
(three panels: combined proxy vs RD_max, TE-only vs RD_max, |ORC|-only vs RD_max) and
q3_stratified_correlation_stats.json with te_hat_vs_RD_max and orc_hat_abs_vs_RD_max blocks.

Stratified sampling (train script) uses a 2D grid on (te_hat, orc_hat); q3_stratified_sample_plan.json
records te_bin, orc_bin per item when regenerated.

Experiment A strengthening (beyond raw Pearson/Spearman): `q3_stratified_experiment_a_strengthening.json`
adds proxy-decile bin means, top vs bottom proxy quartile comparison, partial correlations controlling
WS parameters (k, p), proxy diagnostics (TE vs ORC balance), and mean RD_max variance across training seeds.
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


def _merge_stratified_longform(
    pilot_root: Path,
    stratified_csv: Path,
    main_exp3_csv: Path,
) -> pd.DataFrame:
    """One row per (model, seed, group) for all stratified-design topologies."""
    plan_path = pilot_root / "q3_stratified_sample_plan.json"
    manifest_path = pilot_root / "q3_stratified_manifest.json"
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    seeds = list(json.loads(manifest_path.read_text(encoding="utf-8")).get("seeds", [42, 43, 44]))

    main_df = pd.read_csv(main_exp3_csv) if main_exp3_csv.exists() else pd.DataFrame()
    strat_df = pd.read_csv(stratified_csv) if stratified_csv.exists() else pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for it in plan["items"]:
        model = it["model_name"]
        te_hat = float(it["te_hat"])
        orc_hat = float(it["orc_hat"])
        if it["source"] == "train":
            sub = strat_df[strat_df["model"] == model] if not strat_df.empty else pd.DataFrame()
        else:
            sub = main_df[main_df["model"] == model] if not main_df.empty else pd.DataFrame()
            sub = sub[sub["seed"].isin(seeds)]
        if sub.empty:
            for seed in seeds:
                rows.append({
                    "model": model,
                    "group": "G_strat",
                    "seed": seed,
                    "te_hat": te_hat,
                    "orc_hat": orc_hat,
                    "RD_max": np.nan,
                    "clean_roc_auc": np.nan,
                    "source": it["source"],
                })
            continue
        for _, r in sub.iterrows():
            rows.append({
                "model": model,
                "group": "G_strat",
                "seed": int(r["seed"]),
                "te_hat": te_hat,
                "orc_hat": orc_hat,
                "RD_max": float(r["RD_max"]),
                "clean_roc_auc": float(r.get("clean_roc_auc", np.nan)),
                "source": it["source"],
            })

    return pd.DataFrame(rows)


def _topology_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """Mean RD_max per model (and std across seeds); proxy from first row."""
    if long_df.empty:
        return pd.DataFrame()
    g = long_df.groupby("model", as_index=False).agg(
        RD_max_per_topo=("RD_max", "mean"),
        RD_max_std_seeds=("RD_max", "std"),
        n_seeds=("seed", "count"),
        te_hat=("te_hat", "first"),
        orc_hat=("orc_hat", "first"),
        source=("source", "first"),
    )
    g["proxy_score"] = 0.5 * (g["te_hat"] + np.abs(g["orc_hat"]))
    return g


def _merge_plan_topology(topo_df: pd.DataFrame, pilot_root: Path) -> pd.DataFrame:
    """Attach k, p, stratification bins, graph_seed from q3_stratified_sample_plan.json."""
    plan_path = pilot_root / "q3_stratified_sample_plan.json"
    if not plan_path.exists() or topo_df.empty:
        return topo_df
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = []
    for it in plan.get("items", []):
        rows.append({
            "model": it["model_name"],
            "k": int(it["k"]),
            "p": float(it["p"]),
            "te_bin": int(it.get("te_bin", -1)),
            "orc_bin": int(it.get("orc_bin", -1)),
            "graph_seed": int(it.get("graph_seed", 0)),
        })
    meta = pd.DataFrame(rows)
    return topo_df.merge(meta, on="model", how="left")


def _partial_pearson_r_p(
    x: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
) -> Tuple[float, float]:
    """Partial Pearson r between x and y controlling columns of Z (numpy OLS; no extra deps)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = len(x)
    if n < 4 or Z.shape[0] != n:
        return float("nan"), float("nan")
    Zc = np.column_stack([np.ones(n), Z])
    try:
        bx, _, _, _ = np.linalg.lstsq(Zc, x, rcond=None)
        by, _, _, _ = np.linalg.lstsq(Zc, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    rx = x - Zc @ bx
    ry = y - Zc @ by
    denom = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
    if denom <= 1e-15:
        return float("nan"), float("nan")
    r = float(np.dot(rx, ry) / denom)
    k_ctrl = Z.shape[1]
    df = n - 2 - k_ctrl
    if df <= 0 or abs(r) >= 1.0 - 1e-15:
        return r, float("nan")
    try:
        from scipy.stats import t as student_t
        t_stat = abs(r) * np.sqrt(df / max(1e-15, 1.0 - r**2))
        p = float(2.0 * (1.0 - student_t.cdf(t_stat, df)))
    except ImportError:
        p = float("nan")
    return r, p


def analyze_experiment_a_strengthening(
    topo_df: pd.DataFrame,
    pilot_root: Path,
) -> Dict[str, Any]:
    """
    Supplement correlation with: bin means, top/bottom proxy quartiles, partial r (k,p),
    proxy diagnostics, mean seed-noise across topologies.
    """
    out: Dict[str, Any] = {
        "n_topologies": int(len(topo_df)),
        "notes": [],
    }
    df = topo_df.dropna(subset=["RD_max_per_topo", "proxy_score"]).copy()
    if df.empty:
        out["notes"].append("No complete RD_max / proxy rows")
        return out

    out["seed_variance"] = {
        "mean_std_RD_max_across_seeds": float(np.nanmean(df["RD_max_std_seeds"].to_numpy(dtype=float)))
        if "RD_max_std_seeds" in df.columns and df["RD_max_std_seeds"].notna().any()
        else None,
        "interpretation": "Lower is less noise across seeds per topology; fix seeds or average seeds to reduce this.",
    }

    te = df["te_hat"].to_numpy(dtype=float)
    oc = np.abs(df["orc_hat"].to_numpy(dtype=float))
    out["proxy_diagnostics"] = {
        "te_hat_std_sample": float(np.nanstd(te)),
        "orc_hat_abs_std_sample": float(np.nanstd(oc)),
        "ratio_std_orc_over_te": float(np.nanstd(oc) / (np.nanstd(te) + 1e-12)),
        "note": "orc_hat in the pool is nonnegative (|ORC|/10 clipped in compute_paper3_proxies); proxy_score = 0.5*(te_hat+|orc_hat|).",
    }
    try:
        from scipy.stats import pearsonr
        r_te_oc, p_te_oc = pearsonr(te, oc)
        out["proxy_diagnostics"]["pearson_te_vs_abs_orc"] = {"r": float(r_te_oc), "pvalue": float(p_te_oc)}
    except Exception:
        pass

    try:
        from scipy.stats import pearsonr
        rd = df["RD_max_per_topo"].to_numpy(dtype=float)
        r_rd_te, p_rd_te = pearsonr(rd, te)
        r_rd_oc, p_rd_oc = pearsonr(rd, oc)
        out["proxy_diagnostics"]["marginal_r_RD_vs_te"] = {"r": float(r_rd_te), "pvalue": float(p_rd_te)}
        out["proxy_diagnostics"]["marginal_r_RD_vs_abs_orc"] = {"r": float(r_rd_oc), "pvalue": float(p_rd_oc)}
    except Exception:
        pass

    n = len(df)
    nq = min(10, max(3, n // 5))
    try:
        df["_qbin"] = pd.qcut(df["proxy_score"], q=nq, duplicates="drop")
        bin_agg = (
            df.groupby("_qbin", observed=True)
            .agg(
                mean_RD_max=("RD_max_per_topo", "mean"),
                n=("RD_max_per_topo", "count"),
                proxy_min=("proxy_score", "min"),
                proxy_max=("proxy_score", "max"),
            )
            .reset_index()
        )
        bin_agg["_qbin"] = bin_agg["_qbin"].astype(str)
        out["bin_analysis_proxy_quantiles"] = {
            "n_bins": int(len(bin_agg)),
            "bins": bin_agg.to_dict(orient="records"),
        }
    except Exception as e:
        out["notes"].append(f"bin_analysis skipped: {e}")

    df2 = df.dropna(subset=["k", "p"])
    if len(df2) >= 5:
        Z = np.column_stack([df2["k"].to_numpy(dtype=float), df2["p"].to_numpy(dtype=float)])
        x = df2["proxy_score"].to_numpy(dtype=float)
        y = df2["RD_max_per_topo"].to_numpy(dtype=float)
        pr, pp = _partial_pearson_r_p(x, y, Z)
        out["partial_correlations"] = {
            "controls": ["k", "p"],
            "description": "Partial Pearson r between proxy_score and RD_max_per_topo after linear adjustment for WS k and p.",
            "proxy_vs_RD_partial_r": float(pr),
            "pvalue": float(pp) if pp == pp else None,
        }
        pte, _ = _partial_pearson_r_p(df2["te_hat"].to_numpy(dtype=float), y, Z)
        poc, _ = _partial_pearson_r_p(np.abs(df2["orc_hat"]).to_numpy(dtype=float), y, Z)
        out["partial_correlations"]["te_vs_RD_partial_r"] = float(pte)
        out["partial_correlations"]["abs_orc_vs_RD_partial_r"] = float(poc)
    else:
        out["partial_correlations"] = {"note": "Need k,p and n>=5 for partial correlations"}

    # Top 25% proxy vs bottom 25% by proxy_score
    hi = df["proxy_score"].quantile(0.75)
    lo = df["proxy_score"].quantile(0.25)
    top_rd = df.loc[df["proxy_score"] >= hi, "RD_max_per_topo"].dropna().to_numpy(dtype=float)
    bot_rd = df.loc[df["proxy_score"] <= lo, "RD_max_per_topo"].dropna().to_numpy(dtype=float)
    qblock: Dict[str, Any] = {
        "proxy_top_quartile_threshold": float(hi),
        "proxy_bottom_quartile_threshold": float(lo),
        "n_top": int(len(top_rd)),
        "n_bottom": int(len(bot_rd)),
    }
    if len(top_rd) >= 1 and len(bot_rd) >= 1:
        qblock["mean_RD_top"] = float(np.mean(top_rd))
        qblock["mean_RD_bottom"] = float(np.mean(bot_rd))
        qblock["mean_diff_top_minus_bottom"] = float(np.mean(top_rd) - np.mean(bot_rd))
        try:
            from scipy.stats import mannwhitneyu
            stat, p_mw = mannwhitneyu(top_rd, bot_rd, alternative="two-sided")
            qblock["mannwhitney_statistic"] = float(stat)
            qblock["mannwhitney_pvalue"] = float(p_mw)
        except Exception:
            pass
        rng = np.random.default_rng(42)
        boots = []
        for _ in range(2000):
            a = rng.choice(top_rd, size=len(top_rd), replace=True)
            b = rng.choice(bot_rd, size=len(bot_rd), replace=True)
            boots.append(float(np.mean(a) - np.mean(b)))
        boots = np.array(boots)
        qblock["bootstrap_diff_mean_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    out["top_bottom_proxy_quartiles"] = qblock

    return out


def plot_experiment_a_proxy_bins(
    topo_df: pd.DataFrame,
    strengthening: Dict[str, Any],
    output_path: Path,
) -> None:
    """Bar chart: mean RD_max per proxy quantile bin (if bin table present)."""
    bins = (strengthening.get("bin_analysis_proxy_quantiles") or {}).get("bins")
    if not bins:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.arange(len(bins))
    means = [float(b["mean_RD_max"]) for b in bins]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(xs, means, color="#4A90A4", edgecolor="black", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{i+1}" for i in range(len(bins))], fontsize=8)
    ax.set_xlabel("Proxy score quantile bin (1 = lowest proxy, …)")
    ax.set_ylabel("Mean RD_max (per topology, then mean in bin)")
    ax.set_title("Experiment A: mean robustness by proxy score bin")
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[3a-strengthen] Saved {output_path}")


def _corr_pearson_spearman(x: pd.Series, y: pd.Series) -> Optional[Dict[str, Any]]:
    """Pearson + Spearman for aligned series (same index); drops NaN pairwise."""
    try:
        from scipy.stats import pearsonr, spearmanr
    except ImportError:
        return None
    m = pd.concat([x, y], axis=1).dropna()
    if len(m) < 2:
        return None
    a, b = m.iloc[:, 0].to_numpy(dtype=float), m.iloc[:, 1].to_numpy(dtype=float)
    r_p, p_p = pearsonr(a, b)
    r_s, p_s = spearmanr(a, b)
    return {
        "n": int(len(m)),
        "pearson_r": float(r_p),
        "pearson_pvalue": float(p_p),
        "spearman_rho": float(r_s),
        "spearman_pvalue": float(p_s),
    }


def _correlation_stats(topo_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Robustness (RD_max per topology, mean over seeds) vs combined proxy score, vs TE alone,
    vs |ORC| alone (orc_hat in CSV is the nonnegative Paper 3 ORC proxy).
    """
    sub = topo_df.dropna(subset=["proxy_score", "RD_max_per_topo"])
    out: Dict[str, Any] = {"n": len(sub)}
    if len(sub) < 2:
        out["note"] = "Insufficient points for correlation"
        return out
    try:
        from scipy.stats import pearsonr, spearmanr
        r_p, p_p = pearsonr(sub["proxy_score"], sub["RD_max_per_topo"])
        r_s, p_s = spearmanr(sub["proxy_score"], sub["RD_max_per_topo"])
        out["pearson_r"] = float(r_p)
        out["pearson_pvalue"] = float(p_p)
        out["spearman_rho"] = float(r_s)
        out["spearman_pvalue"] = float(p_s)
    except ImportError:
        out["note"] = "scipy not installed"
        return out

    te_block = _corr_pearson_spearman(topo_df["te_hat"], topo_df["RD_max_per_topo"])
    if te_block is not None:
        out["te_hat_vs_RD_max"] = te_block

    _orc = topo_df.assign(orc_abs=lambda d: np.abs(d["orc_hat"].astype(float)))
    orc_block = _corr_pearson_spearman(_orc["orc_abs"], _orc["RD_max_per_topo"])
    if orc_block is not None:
        out["orc_hat_abs_vs_RD_max"] = {
            **orc_block,
            "description": "|ORC| proxy (Paper 3 pool convention; nonnegative)",
        }

    return out


# Okabe–Ito palette (match run_paper3_analysis_followups._plot_q3_metric_robustness_alignment)
_PLOT3_GROUP_COLORS = {
    "G1": "#E69F00",
    "G2": "#0072B2",
    "G3": "#009E73",
    "G4": "#D55E00",
    "G5": "#CC79A7",
}
_PLOT3_GROUP_LABELS = {
    "G1": "G1 (Proxy WS-Flex)",
    "G2": "G2 (Uniform WS-Flex)",
    "G3": "G3 (Dense CfC)",
    "G4": "G4 (Random Sparse)",
    "G5": "G5 (NCP)",
}
_STRAT_TRAIN_COLOR = "#3182BD"
_STRAT_REUSE_COLOR = "#E69F00"
_STRAT_TRAIN_LABEL = "Stratified WS-Flex (pilot, new)"
_STRAT_REUSE_LABEL = "Stratified WS-Flex (pilot, G1 overlap)"


def _arch_dir_from_main_exp3_csv(main_exp3_csv: Path) -> Path:
    """.../paper3/experiment3/experiment3_results.csv -> .../experiment2/.../selected_architectures"""
    return main_exp3_csv.parent.parent / "experiment2" / "experiment2_pilot" / "selected_architectures"


def _reuse_g1_model_names(pilot_root: Path) -> set:
    plan_path = pilot_root / "q3_stratified_sample_plan.json"
    if not plan_path.exists():
        return set()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    return {it["model_name"] for it in plan.get("items", []) if it.get("source") == "reuse_g1"}


def plot_plot3_style_alignment_latest(
    *,
    main_exp3_csv: Path,
    pilot_root: Path,
    stratified_topo_df: pd.DataFrame,
    stats_stratified: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Original plot3 look (G1–G5 scatter + grey regression line) with main Experiment 3 context,
    overlaid with the latest stratified pilot (dense proxy coverage). Correlation text uses the
    stratified sample only (Experiment A), per advisor spec.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from architecture_refinement.paper3.run_paper3_analysis_followups import analyze_q3_metric_robustness_alignment

    main_df = pd.read_csv(main_exp3_csv)
    arch_dir = _arch_dir_from_main_exp3_csv(main_exp3_csv)
    main_topo, _ = analyze_q3_metric_robustness_alignment(main_df, arch_dir=arch_dir if arch_dir.is_dir() else None)
    if main_topo.empty:
        print("[plot3-latest] No main experiment3 topology rows; skip combined plot3-style figure.")
        return

    reuse_models = _reuse_g1_model_names(pilot_root)
    if reuse_models:
        mask_dup = (main_topo["group"] == "G1") & (main_topo["model"].isin(reuse_models))
        main_topo = main_topo.loc[~mask_dup].copy()

    strat = stratified_topo_df.dropna(subset=["proxy_score", "RD_max_per_topo"])
    if strat.empty:
        print("[plot3-latest] No stratified points; skip combined figure.")
        return

    fig, ax = plt.subplots(figsize=(5, 4))

    for group in ["G1", "G2", "G3", "G4", "G5"]:
        sub = main_topo[main_topo["group"] == group]
        if not sub.empty:
            ax.scatter(
                sub["proxy_score"],
                sub["RD_max_per_topo"],
                c=_PLOT3_GROUP_COLORS.get(group, "gray"),
                s=56,
                edgecolors="black",
                linewidths=0.6,
                alpha=0.45,
                zorder=2,
                label=_PLOT3_GROUP_LABELS.get(group, group),
            )

    for source, label, c in (
        ("train", _STRAT_TRAIN_LABEL, _STRAT_TRAIN_COLOR),
        ("reuse_g1", _STRAT_REUSE_LABEL, _STRAT_REUSE_COLOR),
    ):
        ssub = strat[strat["source"] == source]
        if not ssub.empty:
            ax.scatter(
                ssub["proxy_score"],
                ssub["RD_max_per_topo"],
                c=c,
                s=72,
                edgecolors="black",
                linewidths=1.0,
                alpha=0.95,
                zorder=5,
                label=label,
            )

    if len(strat) >= 2:
        coef = np.polyfit(strat["proxy_score"], strat["RD_max_per_topo"], 1)
        x_line = np.linspace(strat["proxy_score"].min(), strat["proxy_score"].max(), 100)
        ax.plot(x_line, np.polyval(coef, x_line), linestyle="--", color="#cccccc", linewidth=1, zorder=3, label=None)

    ax.set_xlabel(r"Proxy score ($\frac{1}{2}(\hat{TE} + |\widehat{ORC}|)$)")
    ax.set_ylabel("RD_max under AR(1) drift")

    all_x = pd.concat([main_topo["proxy_score"], strat["proxy_score"]], ignore_index=True)
    all_y = pd.concat([main_topo["RD_max_per_topo"], strat["RD_max_per_topo"]], ignore_index=True)
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    pad_x = max(0.02 * (x_max - x_min), 0.001) if x_max > x_min else 0.03
    pad_y = max(0.02 * (y_max - y_min), 0.001) if y_max > y_min else 0.03
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    txt_lines = [
        "Stratified sample (Experiment A):",
        f"n = {stats_stratified.get('n', len(strat))}",
    ]
    if "pearson_r" in stats_stratified:
        txt_lines.append(
            f"Combined proxy: r = {stats_stratified['pearson_r']:.3f}, p = {stats_stratified['pearson_pvalue']:.3g}"
        )
    if "spearman_rho" in stats_stratified:
        txt_lines.append(
            f"Combined proxy: ρ = {stats_stratified['spearman_rho']:.3f}, p = {stats_stratified['spearman_pvalue']:.3g}"
        )
    te_s = stats_stratified.get("te_hat_vs_RD_max") or {}
    orc_s = stats_stratified.get("orc_hat_abs_vs_RD_max") or {}
    if te_s.get("pearson_r") is not None:
        txt_lines.append(
            f"TE only: r = {te_s['pearson_r']:.3f}, p = {te_s['pearson_pvalue']:.3g}"
        )
    if orc_s.get("pearson_r") is not None:
        txt_lines.append(
            f"|ORC| only: r = {orc_s['pearson_r']:.3f}, p = {orc_s['pearson_pvalue']:.3g}"
        )
    ax.text(
        0.02, 0.98, "\n".join(txt_lines),
        transform=ax.transAxes,
        fontsize=7.5,
        verticalalignment="top",
        family="monospace",
        zorder=6,
    )

    ax.legend(loc="lower right", frameon=False, fontsize=7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot3-latest] Saved {output_path}")


def plot_stratified_correlation(
    topo_df: pd.DataFrame,
    stats: Dict[str, Any],
    output_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = topo_df.dropna(subset=["proxy_score", "RD_max_per_topo"])
    if sub.empty:
        print("[3a] No valid points; skip figure")
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    colors = {"train": "#0072B2", "reuse_g1": "#E69F00"}
    for src in sub["source"].unique():
        s = sub[sub["source"] == src]
        ax.scatter(
            s["proxy_score"],
            s["RD_max_per_topo"],
            c=colors.get(src, "#333333"),
            s=55,
            edgecolors="black",
            linewidths=0.75,
            alpha=0.85,
            label="New train" if src == "train" else "Reuse G1",
        )

    if len(sub) >= 2:
        coef = np.polyfit(sub["proxy_score"], sub["RD_max_per_topo"], 1)
        x_line = np.linspace(sub["proxy_score"].min(), sub["proxy_score"].max(), 100)
        ax.plot(x_line, np.polyval(coef, x_line), linestyle="--", color="#cccccc", linewidth=1)

    ax.set_xlabel(r"Proxy score ($\frac{1}{2}(\hat{TE} + |\widehat{ORC}|)$)")
    ax.set_ylabel("RD_max under AR(1) drift")
    ax.legend(loc="best", frameon=False)

    txt_lines = [f"n = {stats.get('n', len(sub))}"]
    if "pearson_r" in stats:
        txt_lines.append(f"Pearson r = {stats['pearson_r']:.3f}, p = {stats['pearson_pvalue']:.3g}")
    if "spearman_rho" in stats:
        txt_lines.append(f"Spearman ρ = {stats['spearman_rho']:.3f}, p = {stats['spearman_pvalue']:.3g}")
    ax.text(
        0.03, 0.97, "\n".join(txt_lines),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        family="monospace",
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[3a] Saved {output_path}")


def _format_corr_annotation(block: Optional[Dict[str, Any]]) -> List[str]:
    if not block or "pearson_r" not in block:
        return ["(insufficient data)"]
    return [
        f"n = {block.get('n', '?')}",
        f"Pearson r = {block['pearson_r']:.3f}, p = {block['pearson_pvalue']:.3g}",
        f"Spearman ρ = {block['spearman_rho']:.3f}, p = {block['spearman_pvalue']:.3g}",
    ]


def _stats_block_for_panel(stats: Dict[str, Any], *, panel: str) -> Optional[Dict[str, Any]]:
    if panel == "combined":
        keys = ("n", "pearson_r", "pearson_pvalue", "spearman_rho", "spearman_pvalue")
        b = {k: stats[k] for k in keys if k in stats}
        return b if "pearson_r" in b else None
    key = "te_hat_vs_RD_max" if panel == "te" else "orc_hat_abs_vs_RD_max"
    raw = stats.get(key)
    if not isinstance(raw, dict):
        return None
    return {k: v for k, v in raw.items() if k != "description"}


def plot_stratified_te_orc_panels(
    topo_df: pd.DataFrame,
    stats: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Three panels: combined proxy score, TE-only, |ORC|-only vs RD_max (stratified sample).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = topo_df.copy()
    df["orc_abs"] = np.abs(df["orc_hat"].astype(float))

    panel_specs: List[Tuple[str, str, str, str]] = [
        ("proxy_score", r"Proxy score $\frac{1}{2}(\hat{TE} + |\widehat{ORC}|)$", "Combined", "combined"),
        ("te_hat", r"$\hat{TE}$ only", "TE", "te"),
        ("orc_abs", r"$|\widehat{ORC}|$ only", "|ORC|", "orc"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=True)
    colors = {"train": "#0072B2", "reuse_g1": "#E69F00"}

    for ax, (xcol, xlabel, title_suffix, panel_id) in zip(axes, panel_specs):
        sub = df.dropna(subset=[xcol, "RD_max_per_topo"])
        if sub.empty:
            ax.set_title(title_suffix)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        for src in sub["source"].unique():
            s = sub[sub["source"] == src]
            ax.scatter(
                s[xcol],
                s["RD_max_per_topo"],
                c=colors.get(str(src), "#333333"),
                s=50,
                edgecolors="black",
                linewidths=0.65,
                alpha=0.85,
                label="New train" if src == "train" else "Reuse G1",
            )
        if len(sub) >= 2:
            coef = np.polyfit(sub[xcol].to_numpy(dtype=float), sub["RD_max_per_topo"].to_numpy(dtype=float), 1)
            x_line = np.linspace(float(sub[xcol].min()), float(sub[xcol].max()), 100)
            ax.plot(x_line, np.polyval(coef, x_line), linestyle="--", color="#cccccc", linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_title(title_suffix)
        ann = _stats_block_for_panel(stats, panel=panel_id)
        lines = _format_corr_annotation(ann)
        ax.text(
            0.03,
            0.97,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            family="monospace",
        )
        ax.legend(loc="best", frameon=False, fontsize=7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)

    axes[0].set_ylabel("RD_max under AR(1) drift")
    fig.suptitle("Stratified sample: robustness vs proxy components", fontsize=11, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[3a-te-orc] Saved {output_path}")


def _bootstrap_group_mean_ci(
    topo_means: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for mean of topology-level means."""
    x = np.asarray(topo_means, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(x) == 1:
        m = float(x[0])
        return m, m, m
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s = rng.choice(x, size=len(x), replace=True)
        boots.append(float(np.mean(s)))
    boots = np.array(boots)
    return float(np.mean(x)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def analyze_selection_comparison(
    main_exp3_csv: Path,
    output_fig: Path,
    stats_json: Path,
) -> Dict[str, Any]:
    df = pd.read_csv(main_exp3_csv)
    df = df.dropna(subset=["RD_max", "group", "model"])
    summary: Dict[str, Any] = {"groups": {}, "g1_vs_g2": {}}

    GROUP_ORDER = ["G1", "G2", "G3", "G4", "G5"]
    GROUP_LABELS = {
        "G1": "G1 Proxy WS-Flex",
        "G2": "G2 Uniform WS-Flex",
        "G3": "G3 Dense CfC",
        "G4": "G4 Random sparse",
        "G5": "G5 NCP",
    }

    per_topo = df.groupby(["group", "model"], as_index=False)["RD_max"].mean()
    heights: List[float] = []
    errs_lo: List[float] = []
    errs_hi: List[float] = []
    labels: List[str] = []
    ns: List[int] = []

    for g in GROUP_ORDER:
        sub = per_topo[per_topo["group"] == g]
        vals = sub["RD_max"].to_numpy(dtype=float)
        n_topo = len(vals)
        if n_topo == 0:
            continue
        if g == "G5" and n_topo == 1:
            seed_vals = df[df["group"] == g]["RD_max"].to_numpy(dtype=float)
            m = float(np.mean(seed_vals))
            sem = float(np.std(seed_vals, ddof=1) / np.sqrt(len(seed_vals))) if len(seed_vals) > 1 else 0.0
            heights.append(m)
            errs_lo.append(m - 1.96 * sem)
            errs_hi.append(m + 1.96 * sem)
            summary["groups"][g] = {
                "n_topologies": 1,
                "n_seed_runs": int(len(seed_vals)),
                "mean_RD_max": m,
                "ci_low": m - 1.96 * sem,
                "ci_high": m + 1.96 * sem,
                "note": "Error bar: ±1.96×SEM across seeds (single wiring)",
            }
        else:
            m, lo, hi = _bootstrap_group_mean_ci(vals)
            heights.append(m)
            errs_lo.append(lo)
            errs_hi.append(hi)
            summary["groups"][g] = {
                "n_topologies": n_topo,
                "mean_of_topology_means": m,
                "bootstrap_ci95": [lo, hi],
            }
        labels.append(GROUP_LABELS[g])
        ns.append(n_topo)

    g1_vals = per_topo[per_topo["group"] == "G1"]["RD_max"].to_numpy(dtype=float)
    g2_vals = per_topo[per_topo["group"] == "G2"]["RD_max"].to_numpy(dtype=float)
    g1_vs_g2: Dict[str, Any] = {}
    if len(g1_vals) >= 2 and len(g2_vals) >= 2:
        rng = np.random.default_rng(123)
        diffs = []
        for _ in range(2000):
            a = rng.choice(g1_vals, size=len(g1_vals), replace=True)
            b = rng.choice(g2_vals, size=len(g2_vals), replace=True)
            diffs.append(float(np.mean(a) - np.mean(b)))
        diffs = np.array(diffs)
        g1_vs_g2 = {
            "mean_g1": float(np.mean(g1_vals)),
            "mean_g2": float(np.mean(g2_vals)),
            "diff_mean_g1_minus_g2": float(np.mean(g1_vals) - np.mean(g2_vals)),
            "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        }
        try:
            from scipy.stats import mannwhitneyu
            stat, pval = mannwhitneyu(g1_vals, g2_vals, alternative="two-sided")
            g1_vs_g2["mannwhitney_statistic"] = float(stat)
            g1_vs_g2["mannwhitney_pvalue"] = float(pval)
        except (ImportError, ValueError):
            pass
    summary["g1_vs_g2"] = g1_vs_g2

    stats_json.parent.mkdir(parents=True, exist_ok=True)
    stats_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not heights:
        print("[3b] No group data in main experiment3 CSV; skip bar figure")
        return summary

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(heights))
    yerr = np.array([[h - lo for h, lo in zip(heights, errs_lo)], [hi - h for h, hi in zip(heights, errs_hi)]])
    fig, ax = plt.subplots(figsize=(7.5, 4))
    colors = ["#E69F00", "#0072B2", "#009E73", "#D55E00", "#CC79A7"]
    ax.bar(x, heights, yerr=yerr, capsize=4, color=colors[: len(heights)], edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean RD_max (topology mean, then group mean)")
    ax.set_title("Selection-method comparison (main Experiment 3)")
    for i, n in enumerate(ns):
        ax.text(i, errs_hi[i] + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0] + 1e-6), f"n={n}", ha="center", fontsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    fig.tight_layout()
    fig.savefig(output_fig, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[3b] Saved {output_fig}")
    return summary


def run_analysis(
    *,
    pilot_root: Path,
    stratified_csv: Path,
    main_exp3_csv: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    long_df = _merge_stratified_longform(pilot_root, stratified_csv, main_exp3_csv)
    long_df.to_csv(output_dir / "q3_stratified_merged_longform.csv", index=False)

    topo_df = _topology_table(long_df)
    topo_merged = _merge_plan_topology(topo_df, pilot_root)
    topo_merged.to_csv(output_dir / "q3_stratified_topology_table.csv", index=False)

    stats = _correlation_stats(topo_merged)
    stats_path = output_dir / "q3_stratified_correlation_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    strengthening = analyze_experiment_a_strengthening(topo_merged, pilot_root)
    (output_dir / "q3_stratified_experiment_a_strengthening.json").write_text(
        json.dumps(strengthening, indent=2, default=str),
        encoding="utf-8",
    )
    plot_experiment_a_proxy_bins(
        topo_merged,
        strengthening,
        output_dir / "plot3a_experiment_a_proxy_bins.pdf",
    )

    plot_stratified_correlation(
        topo_merged,
        stats,
        output_dir / "plot3a_q3_proxy_robustness_stratified.pdf",
    )
    plot_stratified_te_orc_panels(
        topo_merged,
        stats,
        output_dir / "plot3a_q3_te_orc_vs_robustness.pdf",
    )

    if main_exp3_csv.exists():
        plot_plot3_style_alignment_latest(
            main_exp3_csv=main_exp3_csv,
            pilot_root=pilot_root,
            stratified_topo_df=topo_merged,
            stats_stratified=stats,
            output_path=output_dir / "plot3_q3_metric_robustness_alignment_latest.pdf",
        )
        analyze_selection_comparison(
            main_exp3_csv,
            output_dir / "plot3b_q3_selection_comparison.pdf",
            output_dir / "q3_selection_comparison_stats.json",
        )
    else:
        print("[plot3-latest / 3b] Skipped: main experiment3_results.csv not found")


def merge_stratified_longform_for_analysis(
    pilot_root: Path,
    stratified_csv: Path,
    main_exp3_csv: Path,
) -> pd.DataFrame:
    """Public wrapper for forensic pass and external tools (same as internal merge)."""
    return _merge_stratified_longform(pilot_root, stratified_csv, main_exp3_csv)


def topology_table_from_longform(long_df: pd.DataFrame) -> pd.DataFrame:
    """Public wrapper for mean RD per model + proxy from long-form rows."""
    return _topology_table(long_df)


def merge_plan_topology_meta(topo_df: pd.DataFrame, pilot_root: Path) -> pd.DataFrame:
    """Public wrapper: attach k, p, te_bin, orc_bin from q3_stratified_sample_plan.json."""
    return _merge_plan_topology(topo_df, pilot_root)


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper 3 Q3 stratified + selection figures")
    parser.add_argument(
        "--pilot-root",
        type=str,
        default="architecture_refinement/outputs/paper3/q3_stratified_pilot",
    )
    parser.add_argument(
        "--stratified-csv",
        type=str,
        default="architecture_refinement/outputs/paper3/q3_stratified_pilot/q3_stratified_experiment3_results.csv",
    )
    parser.add_argument(
        "--main-experiment3-csv",
        type=str,
        default="architecture_refinement/outputs/paper3/experiment3/experiment3_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="architecture_refinement/outputs/paper3/analysis_followups",
    )
    parser.add_argument(
        "--collect-first",
        action="store_true",
        help="Run q3 stratified result collection before plotting (needs completed training jobs)",
    )
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument("--results-root", type=str, default=None)
    args = parser.parse_args()

    pilot = Path(args.pilot_root)
    scsv = Path(args.stratified_csv)
    main_csv = Path(args.main_experiment3_csv)
    out = Path(args.output_dir)
    pilot = _REPO_ROOT / pilot if not pilot.is_absolute() else pilot
    scsv = _REPO_ROOT / scsv if not scsv.is_absolute() else scsv
    main_csv = _REPO_ROOT / main_csv if not main_csv.is_absolute() else main_csv
    out = _REPO_ROOT / out if not out.is_absolute() else out

    if args.collect_first:
        from architecture_refinement.paper3.run_paper3_q3_stratified_collect import collect_stratified_results

        root = Path(args.results_root) if args.results_root else _REPO_ROOT
        collect_stratified_results(pilot, scsv, root, dataset=args.dataset)

    run_analysis(
        pilot_root=pilot,
        stratified_csv=scsv,
        main_exp3_csv=main_csv,
        output_dir=out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
