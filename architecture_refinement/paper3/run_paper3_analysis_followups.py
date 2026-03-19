"""
Paper 3 Analysis Follow-ups: Post-hoc analyses using existing experiment results.

No new training required. Implements:
1. Topology-controlled robustness variance (G1 vs G2 across topologies)
2. Stratification by k/E_active to control for degree
3. Subject-level robustness decomposition
4. Q3 metric-robustness alignment (proxy score vs RD_max scatter)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import short_run_id


def _load_experiment3_results(exp3_dir: Path) -> pd.DataFrame:
    """Load experiment3_results.csv."""
    path = exp3_dir / "experiment3_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Experiment 3 results not found: {path}")
    return pd.read_csv(path)


def _load_manifest(exp2_dir: Path) -> Dict[str, Any]:
    """Load experiment2_manifest.json."""
    path = exp2_dir / "experiment2_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Experiment 2 manifest not found: {path}")
    return json.loads(path.read_text())


def _load_architecture_metadata(arch_dir: Path, model_name: str) -> Optional[Dict[str, Any]]:
    """Load k, E_active from architecture JSON."""
    for stem in [model_name, short_run_id(model_name)]:
        p = arch_dir / f"{stem}.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return None


def analyze_topology_variance(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare variability of RD_max across topologies (not seeds) in G1 vs G2.
    For each model: RD_max_per_topo = mean(RD_max over seeds).
    Compare std(RD_max_per_topo) for G1 vs G2.
    """
    ws_flex = df[df["group"].isin(["G1", "G2"])].copy()
    if ws_flex.empty:
        return {"error": "No G1/G2 data", "n_g1": 0, "n_g2": 0}

    per_topo = (
        ws_flex.groupby(["model", "group"])["RD_max"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    per_topo = per_topo.rename(columns={"mean": "RD_max_per_topo", "std": "RD_max_seed_std"})

    g1 = per_topo[per_topo["group"] == "G1"]["RD_max_per_topo"].dropna()
    g2 = per_topo[per_topo["group"] == "G2"]["RD_max_per_topo"].dropna()

    summary: Dict[str, Any] = {
        "n_g1_topologies": len(g1),
        "n_g2_topologies": len(g2),
        "g1_RD_max_per_topo_mean": float(g1.mean()) if len(g1) > 0 else float("nan"),
        "g1_RD_max_per_topo_std": float(g1.std()) if len(g1) > 1 else float("nan"),
        "g1_RD_max_per_topo_iqr": float(g1.quantile(0.75) - g1.quantile(0.25)) if len(g1) > 0 else float("nan"),
        "g2_RD_max_per_topo_mean": float(g2.mean()) if len(g2) > 0 else float("nan"),
        "g2_RD_max_per_topo_std": float(g2.std()) if len(g2) > 1 else float("nan"),
        "g2_RD_max_per_topo_iqr": float(g2.quantile(0.75) - g2.quantile(0.25)) if len(g2) > 0 else float("nan"),
    }

    if len(g1) >= 2 and len(g2) >= 2:
        try:
            from scipy.stats import levene
            stat, pval = levene(g1, g2)
            summary["levene_statistic"] = float(stat)
            summary["levene_pvalue"] = float(pval)
        except ImportError:
            summary["levene_note"] = "scipy not available for Levene test"

    return summary


def analyze_stratified_by_degree(
    df: pd.DataFrame,
    arch_dir: Path,
    k_bins: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Stratify G2 by k (or E_active) and compare RD_max within strata vs G1.
    """
    if k_bins is None:
        k_bins = [(2, 4), (6, 8), (10, 12), (14, 16), (18, 20), (22, 24)]

    ws_flex = df[df["group"].isin(["G1", "G2"])].copy()
    if ws_flex.empty:
        return pd.DataFrame(), {"error": "No G1/G2 data"}

    models = ws_flex["model"].unique()
    meta: Dict[str, Dict[str, Any]] = {}
    for m in models:
        arch = _load_architecture_metadata(arch_dir, m)
        if arch is None:
            continue
        k = arch.get("k", -1)
        e = arch.get("E_active", -1)
        if k >= 0:
            meta[m] = {"k": int(k), "E_active": int(e)}

    ws_flex["k"] = ws_flex["model"].map(lambda x: meta.get(x, {}).get("k", -1))
    ws_flex["E_active"] = ws_flex["model"].map(lambda x: meta.get(x, {}).get("E_active", -1))
    ws_flex = ws_flex[ws_flex["k"] >= 0]

    def _stratum_for_k(k: int) -> str:
        for lo, hi in k_bins:
            if lo <= k <= hi:
                return f"k_{lo}-{hi}"
        return "k_other"

    ws_flex["stratum"] = ws_flex["k"].apply(_stratum_for_k)

    rows: List[Dict[str, Any]] = []
    for stratum in sorted(ws_flex["stratum"].unique()):
        sub = ws_flex[ws_flex["stratum"] == stratum]
        g1_sub = sub[sub["group"] == "G1"]["RD_max"].dropna()
        g2_sub = sub[sub["group"] == "G2"]["RD_max"].dropna()
        n_g1 = len(g1_sub)
        n_g2 = len(g2_sub)
        mean_g1 = float(g1_sub.mean()) if n_g1 > 0 else float("nan")
        mean_g2 = float(g2_sub.mean()) if n_g2 > 0 else float("nan")
        diff = mean_g1 - mean_g2 if (np.isfinite(mean_g1) and np.isfinite(mean_g2)) else float("nan")

        if n_g1 >= 2 and n_g2 >= 2:
            rng = np.random.default_rng(42)
            diffs = []
            for _ in range(500):
                s1 = rng.choice(g1_sub, size=len(g1_sub), replace=True)
                s2 = rng.choice(g2_sub, size=len(g2_sub), replace=True)
                diffs.append(float(np.mean(s1) - np.mean(s2)))
            ci_lo = float(np.percentile(diffs, 2.5))
            ci_hi = float(np.percentile(diffs, 97.5))
        else:
            ci_lo = ci_hi = float("nan")

        rows.append({
            "stratum": stratum,
            "n_G1": n_g1,
            "n_G2": n_g2,
            "mean_RD_G1": mean_g1,
            "mean_RD_G2": mean_g2,
            "diff_G1_minus_G2": diff,
            "diff_ci_lo": ci_lo,
            "diff_ci_hi": ci_hi,
        })

    stratified_df = pd.DataFrame(rows)
    summary = {"n_strata": len(rows), "k_bins": k_bins}
    return stratified_df, summary


def analyze_q3_metric_robustness_alignment(
    df: pd.DataFrame,
    arch_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Q3: How do TE and |ORC| organize trained robustness?

    Aggregates experiment3 results to topology-level: one point per (model, group)
    with RD_max_per_topo = mean(RD_max) over seeds and proxy_score = 0.5*(te_hat + |orc_hat|).
    Only topologies with valid proxy_score and RD_max are included.

    When arch_dir is provided and te_hat/orc_hat are NaN (e.g. for NCP), attempts
    to compute proxies from the architecture JSON.
    """
    if "te_hat" not in df.columns or "orc_hat" not in df.columns:
        return pd.DataFrame(), {"error": "experiment3_results missing te_hat/orc_hat"}

    per_topo = (
        df.groupby(["model", "group"])
        .agg(
            RD_max=("RD_max", "mean"),
            te_hat=("te_hat", "first"),
            orc_hat=("orc_hat", "first"),
        )
        .reset_index()
    )
    per_topo = per_topo.rename(columns={"RD_max": "RD_max_per_topo"})

    # Fallback: compute proxies from arch for rows with NaN (e.g. G5 NCP)
    if arch_dir is not None:
        from architecture_refinement.metrics_te_orc import compute_paper3_proxies
        from architecture_refinement.paper3.arch_graph_utils import graph_from_architecture

        arch_dir = Path(arch_dir)
        for idx, row in per_topo.iterrows():
            if pd.isna(row["te_hat"]) or pd.isna(row["orc_hat"]):
                model = row["model"]
                for stem in [model, short_run_id(model)]:
                    arch_path = arch_dir / f"{stem}.json"
                    if arch_path.exists():
                        try:
                            arch = json.loads(arch_path.read_text())
                            G = graph_from_architecture(arch)
                            if G is not None:
                                te, oc = compute_paper3_proxies(G)
                                per_topo.at[idx, "te_hat"] = te
                                per_topo.at[idx, "orc_hat"] = oc
                                if row["group"] == "G5":
                                    print(f"[Analysis 4] Filled NCP proxy for {model}: te_hat={te:.4f}, orc_hat={oc:.4f}")
                                break
                        except Exception as e:
                            if row["group"] == "G5":
                                print(f"[Analysis 4] Could not compute NCP proxy for {model}: {e}")

    per_topo["proxy_score"] = 0.5 * (per_topo["te_hat"] + per_topo["orc_hat"].abs())
    topology_df = per_topo.dropna(subset=["proxy_score", "RD_max_per_topo"])

    summary: Dict[str, Any] = {
        "n_topologies": len(topology_df),
    }
    if len(topology_df) >= 2:
        try:
            from scipy.stats import pearsonr, spearmanr
            r_pearson, p_pearson = pearsonr(
                topology_df["proxy_score"], topology_df["RD_max_per_topo"]
            )
            r_spearman, p_spearman = spearmanr(
                topology_df["proxy_score"], topology_df["RD_max_per_topo"]
            )
            summary["pearson_r"] = float(r_pearson)
            summary["pearson_pvalue"] = float(p_pearson)
            summary["spearman_rho"] = float(r_spearman)
            summary["spearman_pvalue"] = float(p_spearman)
        except ImportError:
            summary["correlation_note"] = "scipy not available for Pearson/Spearman"

    return topology_df, summary


def _collect_perturb_results_by_subject(
    repo_root: Path,
    dataset: str,
    model_name: str,
    seed: int,
    noise_type: str,
) -> List[Dict[str, Any]]:
    """
    Collect RD_max per (model, seed, subject) from test_perturb CSVs.
    Infers subject from path (sub-001 -> 1). Aggregates across sessions per subject (mean).
    """
    paradigm = "MotorImagery" if "BNCI" in dataset else "SSVEP"
    base = repo_root / "results" / paradigm / dataset
    raw: List[Dict[str, Any]] = []

    for stem in [short_run_id(model_name), model_name]:
        path = base / stem / "CrossSessionEvaluation" / str(seed)
        if not path.exists():
            continue
        for p in path.rglob("*.csv"):
            if "test_perturb" not in str(p):
                continue
            match = re.search(r"sub-(\d+)", str(p))
            subject = int(match.group(1)) if match else None
            if subject is None:
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
                raw.append({"subject": subject, "RD_max": rd_max, "clean_roc_auc": clean})
            except Exception:
                pass

    if not raw:
        return []
    agg_df = pd.DataFrame(raw).groupby("subject").agg({"RD_max": "mean", "clean_roc_auc": "first"}).reset_index()
    return agg_df.to_dict("records")


def analyze_subject_level_robustness(
    repo_root: Path,
    manifest: Dict[str, Any],
    dataset: str = "BNCI2014_001",
    noise_type: str = "ar1_drift",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute per-subject RD_max and aggregate by group.
    """
    groups = manifest.get("groups", {})
    seeds = list(range(42, 42 + manifest.get("S", 5)))
    all_models = (
        groups.get("G1", []) + groups.get("G2", []) +
        groups.get("G3", []) + groups.get("G4", []) + groups.get("G5", [])
    )

    rows: List[Dict[str, Any]] = []
    for model_name in all_models:
        group = next((g for g in ["G1", "G2", "G3", "G4", "G5"] if model_name in groups.get(g, [])), "unknown")
        for seed in seeds:
            subj_results = _collect_perturb_results_by_subject(
                repo_root, dataset, model_name, seed, noise_type
            )
            for r in subj_results:
                rows.append({
                    "model": model_name,
                    "group": group,
                    "seed": seed,
                    "subject": r["subject"],
                    "RD_max": r["RD_max"],
                    "clean_roc_auc": r["clean_roc_auc"],
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, {"error": "No subject-level data collected", "note": "Check that test_perturb CSVs exist under results/.../sub-XXX/..."}

    per_subject_group = (
        df.groupby(["group", "subject"])["RD_max"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary: Dict[str, Any] = {}
    for g in ["G1", "G2", "G3", "G4", "G5"]:
        sub = per_subject_group[per_subject_group["group"] == g]
        if sub.empty:
            continue
        vals = sub["mean"].dropna()
        summary[f"{g}_n_subjects"] = len(vals)
        summary[f"{g}_mean_RD_per_subject"] = float(vals.mean()) if len(vals) > 0 else float("nan")
        summary[f"{g}_std_RD_per_subject"] = float(vals.std()) if len(vals) > 1 else float("nan")
        summary[f"{g}_min_RD_per_subject"] = float(vals.min()) if len(vals) > 0 else float("nan")
        summary[f"{g}_max_RD_per_subject"] = float(vals.max()) if len(vals) > 0 else float("nan")

    return df, summary


def _plot_topology_variance(df: pd.DataFrame, output_dir: Path) -> None:
    """Boxplot of RD_max_per_topo by group (G1 vs G2)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ws = df[df["group"].isin(["G1", "G2"])]
    if ws.empty:
        return
    per_topo = ws.groupby(["model", "group"])["RD_max"].mean().reset_index()
    per_topo = per_topo.rename(columns={"RD_max": "RD_max_per_topo"})
    fig, ax = plt.subplots(figsize=(4, 3))
    groups = ["G1", "G2"]
    data = [per_topo[per_topo["group"] == g]["RD_max_per_topo"].dropna() for g in groups]
    ax.boxplot(data, labels=groups)
    ax.set_title("RD_max per topology (mean over seeds) by group")
    ax.set_ylabel("RD_max")
    fig.tight_layout()
    fig.savefig(output_dir / "topology_variance_boxplot.pdf", bbox_inches="tight")
    plt.close()


def _plot_stratified_by_degree(stratified_df: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bar chart: mean RD by stratum for G1 vs G2."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if stratified_df.empty:
        return
    x = range(len(stratified_df))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - w / 2 for i in x], stratified_df["mean_RD_G1"], w, label="G1")
    ax.bar([i + w / 2 for i in x], stratified_df["mean_RD_G2"], w, label="G2")
    ax.set_xticks(x)
    ax.set_xticklabels(stratified_df["stratum"], rotation=45, ha="right")
    ax.set_ylabel("Mean RD_max")
    ax.set_title("RD_max by k stratum (G1 vs G2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "stratified_by_degree.pdf", bbox_inches="tight")
    plt.close()


def _plot_q3_metric_robustness_alignment(topology_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Q3 Metric-Robustness Alignment: scatter of topology-level RD_max vs proxy score.
    Color by group (G1-G5), light grey dashed regression line.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if topology_df.empty or len(topology_df) < 1:
        print("[Plot 3 Q3] No topology data with valid proxy score; skipping figure.")
        return

    # Okabe-Ito colorblind-safe palette (approximates spec hues: orange, blue, green, red, purple)
    GROUP_COLORS = {
        "G1": "#E69F00",   # orange
        "G2": "#0072B2",   # blue
        "G3": "#009E73",   # bluish-green (distinct from red)
        "G4": "#D55E00",   # vermillion (distinct from green)
        "G5": "#CC79A7",   # reddish-purple
    }
    GROUP_LABELS = {
        "G1": "G1 (Proxy WS-Flex)",
        "G2": "G2 (Uniform WS-Flex)",
        "G3": "G3 (Dense CfC)",
        "G4": "G4 (Random Sparse)",
        "G5": "G5 (NCP)",
    }

    fig, ax = plt.subplots(figsize=(5, 4))
    for group in ["G1", "G2", "G3", "G4", "G5"]:
        sub = topology_df[topology_df["group"] == group]
        if not sub.empty:
            ax.scatter(
                sub["proxy_score"],
                sub["RD_max_per_topo"],
                c=GROUP_COLORS.get(group, "gray"),
                s=60,
                edgecolors="black",
                linewidths=0.75,
                alpha=0.8,
                label=GROUP_LABELS.get(group, group),
            )

    if len(topology_df) >= 2:
        coef = np.polyfit(topology_df["proxy_score"], topology_df["RD_max_per_topo"], 1)
        x_line = np.linspace(
            topology_df["proxy_score"].min(),
            topology_df["proxy_score"].max(),
            100,
        )
        y_line = np.polyval(coef, x_line)
        ax.plot(x_line, y_line, linestyle="--", color="#cccccc", linewidth=1)

    ax.set_xlabel(r"Proxy score ($\frac{1}{2}(\hat{TE} + |\widehat{ORC}|)$)")
    ax.set_ylabel("RD_max under AR(1) drift")
    x_min, x_max = topology_df["proxy_score"].min(), topology_df["proxy_score"].max()
    y_min, y_max = topology_df["RD_max_per_topo"].min(), topology_df["RD_max_per_topo"].max()
    pad_x = max(0.02 * (x_max - x_min), 0.001) if x_max > x_min else 0.03
    pad_y = max(0.02 * (y_max - y_min), 0.001) if y_max > y_min else 0.03
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.legend(loc="upper right", frameon=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    fig.tight_layout()
    out_path = output_dir / "plot3_q3_metric_robustness.pdf"
    fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot 3 Q3] Saved {out_path}")


def _plot_subject_level_robustness(subject_df: pd.DataFrame, output_dir: Path) -> None:
    """Boxplot of RD_max by group with subjects as points."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    per_subj = subject_df.groupby(["group", "subject"])["RD_max"].mean().reset_index()
    groups = sorted(per_subj["group"].unique())
    data = [per_subj[per_subj["group"] == g]["RD_max"].dropna() for g in groups]
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(data, labels=groups, patch_artist=True)
    for i, (g, d) in enumerate(zip(groups, data)):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.5, s=20)
    ax.set_ylabel("RD_max (mean per subject)")
    ax.set_title("Subject-level robustness by group")
    fig.tight_layout()
    fig.savefig(output_dir / "subject_level_robustness.pdf", bbox_inches="tight")
    plt.close()


def run_all_analyses(
    experiment3_dir: Path,
    experiment2_dir: Path,
    output_dir: Path,
    results_root: Optional[Path] = None,
    dataset: str = "BNCI2014_001",
    figures: bool = False,
) -> Dict[str, Any]:
    """Run all four follow-up analyses and write outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exp3_dir = Path(experiment3_dir)
    exp2_dir = Path(experiment2_dir)
    if not exp3_dir.is_absolute():
        exp3_dir = _REPO_ROOT / exp3_dir
    if not exp2_dir.is_absolute():
        exp2_dir = _REPO_ROOT / exp2_dir
    repo_root = results_root or _REPO_ROOT
    arch_dir = exp2_dir / "experiment2_pilot" / "selected_architectures"
    exp1_dir = exp2_dir.parent / "experiment1"
    if not arch_dir.exists():
        arch_dir = exp1_dir / "selected_architectures"

    df = _load_experiment3_results(exp3_dir)
    manifest = _load_manifest(exp2_dir)

    combined_summary: Dict[str, Any] = {}

    # Analysis 1: Topology variance
    var_summary = analyze_topology_variance(df)
    combined_summary["topology_variance"] = var_summary
    (output_dir / "topology_variance_summary.json").write_text(json.dumps(var_summary, indent=2))
    print("[Analysis 1] Topology variance:", json.dumps(var_summary, indent=2))
    if figures:
        _plot_topology_variance(df, output_dir)

    # Analysis 2: Stratified by degree
    stratified_df, strat_summary = analyze_stratified_by_degree(df, arch_dir)
    combined_summary["stratified_by_degree"] = strat_summary
    if not stratified_df.empty:
        stratified_df.to_csv(output_dir / "stratified_by_degree.csv", index=False)
        print("[Analysis 2] Stratified by degree:", output_dir / "stratified_by_degree.csv")
        if figures:
            _plot_stratified_by_degree(stratified_df, output_dir)
    else:
        print("[Analysis 2] No G1/G2 data with k metadata")

    # Analysis 3: Subject-level
    subject_df, subject_summary = analyze_subject_level_robustness(
        repo_root, manifest, dataset=dataset
    )
    combined_summary["subject_level"] = subject_summary
    if not subject_df.empty:
        subject_df.to_csv(output_dir / "subject_level_robustness.csv", index=False)
        print("[Analysis 3] Subject-level:", output_dir / "subject_level_robustness.csv")
        if figures:
            _plot_subject_level_robustness(subject_df, output_dir)
    else:
        print("[Analysis 3] No subject-level data (check test_perturb path structure)")

    # Analysis 4: Q3 metric-robustness alignment
    q3_topology_df, q3_summary = analyze_q3_metric_robustness_alignment(df, arch_dir=arch_dir)
    combined_summary["q3_metric_robustness"] = q3_summary
    if not q3_topology_df.empty:
        q3_topology_df.to_csv(output_dir / "q3_metric_robustness_topologies.csv", index=False)
        print("[Analysis 4] Q3 metric-robustness:", json.dumps(q3_summary, indent=2))
        if figures:
            _plot_q3_metric_robustness_alignment(q3_topology_df, output_dir)
    else:
        print("[Analysis 4] No Q3 data (missing te_hat/orc_hat or no valid proxy scores)")

    (output_dir / "analysis_followups_summary.json").write_text(json.dumps(combined_summary, indent=2))
    return combined_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper 3 Analysis Follow-ups")
    parser.add_argument("--experiment3-dir", type=str, required=True)
    parser.add_argument("--experiment2-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="architecture_refinement/outputs/paper3/analysis_followups")
    parser.add_argument("--results-root", type=str, default=None, help="Root for results/ (default: repo root)")
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument("--figures", action="store_true", help="Generate optional figures (boxplot, bar chart, Q3 scatter)")
    parser.add_argument("--dry-run-subject-check", action="store_true", help="Only check if subject-level data is available (no full analysis)")
    args = parser.parse_args()

    out_dir = _REPO_ROOT / args.output_dir
    results_root = Path(args.results_root) if args.results_root else None

    if args.dry_run_subject_check:
        exp2_dir = Path(args.experiment2_dir)
        if not exp2_dir.is_absolute():
            exp2_dir = _REPO_ROOT / exp2_dir
        manifest = _load_manifest(exp2_dir)
        groups = manifest.get("groups", {})
        all_models = groups.get("G1", []) + groups.get("G2", [])[:2]
        seeds = list(range(42, 42 + manifest.get("S", 5)))[:1]
        root = results_root or _REPO_ROOT
        found = 0
        for m in all_models:
            for s in seeds:
                r = _collect_perturb_results_by_subject(root, args.dataset, m, s, "ar1_drift")
                found += len(r)
        print(f"Subject-level check: found {found} (model,seed,subject) records from sample models/seeds.")
        print("If 0: test_perturb CSVs may not exist or path structure lacks sub-XXX.")
        return 0

    run_all_analyses(
        experiment3_dir=Path(args.experiment3_dir),
        experiment2_dir=Path(args.experiment2_dir),
        output_dir=out_dir,
        results_root=results_root,
        dataset=args.dataset,
        figures=args.figures,
    )
    print(f"\nOutputs: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
