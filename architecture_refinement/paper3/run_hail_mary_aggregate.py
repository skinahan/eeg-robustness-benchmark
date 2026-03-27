"""
Hail Mary: merge topology manifest + learnability + stability + sensitivity tables;
optional Kruskal–Wallis, Spearman; export §11-style figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_topology_manifest(panel_dir: Path) -> pd.DataFrame:
    mj = panel_dir / "topology_manifest.json"
    if mj.exists():
        data = json.loads(mj.read_text(encoding="utf-8"))
        return pd.DataFrame(data.get("manifest", []))
    mc = panel_dir / "topology_manifest.csv"
    return pd.read_csv(mc)


def merge_tables(
    topo: pd.DataFrame,
    learn: pd.DataFrame,
    stab: Optional[pd.DataFrame],
    sens: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if learn is not None and not learn.empty:
        m = topo.merge(learn, on="model_name", how="left")
    else:
        m = topo.copy()
    if stab is not None and not stab.empty and "seed" in m.columns:
        m = m.merge(stab, on=["model_name", "seed"], how="left")
    if sens is not None and not sens.empty and "seed" in m.columns:
        m = m.merge(sens, on=["model_name", "seed"], how="left")
    return m


def per_topology_summary(df: pd.DataFrame, group_col: str = "topology_id") -> pd.DataFrame:
    if group_col not in df.columns:
        return df
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    agg_cols = [c for c in num_cols if c not in (group_col, "seed", "graph_seed", "k")]
    agg_cols = [c for c in agg_cols if c in df.columns]
    if not agg_cols:
        return df
    return df.groupby(group_col, dropna=False)[agg_cols].mean().reset_index()


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return float("nan"), float("nan")

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    r, p = spearmanr(x[mask], y[mask])
    return float(r), float(p)


def _kruskal_groups(df: pd.DataFrame, value_col: str, group_col: str = "topology_id") -> Dict[str, Any]:
    try:
        from scipy.stats import kruskal
    except ImportError:
        return {"statistic": float("nan"), "pvalue": float("nan"), "n_groups": 0, "error": "scipy_not_installed"}

    groups = []
    for _, g in df.groupby(group_col):
        v = pd.to_numeric(g[value_col], errors="coerce").dropna().to_numpy()
        if len(v) > 0:
            groups.append(v)
    if len(groups) < 2:
        return {"statistic": float("nan"), "pvalue": float("nan"), "n_groups": len(groups)}
    stat, p = kruskal(*groups)
    return {"statistic": float(stat), "pvalue": float(p), "n_groups": len(groups)}


def run_stats(merged: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "te_hat" in merged.columns and "best_val_roc_auc" in merged.columns:
        r, p = _spearman(
            merged["te_hat"].to_numpy(),
            merged["best_val_roc_auc"].to_numpy(),
        )
        out["spearman_te_hat_vs_best_val_roc_auc"] = {"r": r, "p": p}
    if "orc_hat" in merged.columns and "RD_moderate" in merged.columns:
        r, p = _spearman(merged["orc_hat"].to_numpy(), merged["RD_moderate"].to_numpy())
        out["spearman_orc_hat_vs_RD_moderate"] = {"r": r, "p": p}
    if "topology_id" in merged.columns and "best_val_roc_auc" in merged.columns:
        out["kruskal_best_val_roc_auc_by_topology"] = _kruskal_groups(
            merged.dropna(subset=["best_val_roc_auc"]),
            "best_val_roc_auc",
            "topology_id",
        )
    return out


def plot_figures(
    merged: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # Early learnability bar chart (mean best_val per topology)
    if "topology_id" in summary.columns and "best_val_roc_auc" in summary.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        s = summary.sort_values("topology_id")
        ax.bar(range(len(s)), s["best_val_roc_auc"].to_numpy(), color="#0072B2")
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels(s["topology_id"].astype(str), rotation=45, ha="right")
        ax.set_ylabel("Mean best val ROC-AUC (across seeds)")
        ax.set_title("Hail Mary: learnability by topology")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_learnability_best_val_by_topology.pdf", dpi=200)
        plt.close()

    # TE vs early val
    if "te_hat" in merged.columns and "val_roc_auc_epoch_10" in merged.columns:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(merged["te_hat"], merged["val_roc_auc_epoch_10"], alpha=0.7, c="#D55E00")
        ax.set_xlabel("TE_hat")
        ax.set_ylabel("Val ROC-AUC (epoch 10)")
        ax.set_title("Exploratory: TE vs early learnability")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_exploratory_te_vs_epoch10.pdf", dpi=200)
        plt.close()

    # Sensitivity: clean vs perturbed (if columns exist)
    cols = ["clean_test_roc_auc", "perturbed_roc_auc_low", "perturbed_roc_auc_moderate"]
    if all(c in summary.columns for c in cols) and "topology_id" in summary.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        s = summary.sort_values("topology_id")
        x = np.arange(len(s))
        w = 0.25
        ax.bar(x - w, s["clean_test_roc_auc"], width=w, label="clean", color="#009E73")
        ax.bar(x, s["perturbed_roc_auc_low"], width=w, label="low", color="#0072B2")
        ax.bar(x + w, s["perturbed_roc_auc_moderate"], width=w, label="moderate", color="#CC79A7")
        ax.set_xticks(x)
        ax.set_xticklabels(s["topology_id"].astype(str), rotation=45, ha="right")
        ax.set_ylabel("Test ROC-AUC")
        ax.legend()
        ax.set_title("Hail Mary: clean vs perturbed (mean)")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_sensitivity_clean_vs_perturbed.pdf", dpi=200)
        plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Hail Mary aggregate analysis")
    parser.add_argument(
        "--topology-panel-dir",
        type=str,
        default="architecture_refinement/outputs/hail_mary/topology_panel",
    )
    parser.add_argument("--learnability-csv", type=str, default="architecture_refinement/outputs/hail_mary/analysis/learnability_longform.csv")
    parser.add_argument("--stability-csv", type=str, default="architecture_refinement/outputs/hail_mary/analysis/stability_longform.csv")
    parser.add_argument("--sensitivity-csv", type=str, default="architecture_refinement/outputs/hail_mary/analysis/sensitivity_longform.csv")
    parser.add_argument("--output-dir", type=str, default="architecture_refinement/outputs/hail_mary/analysis")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    panel = Path(args.topology_panel_dir)
    if not panel.is_absolute():
        panel = _REPO_ROOT / panel

    learn_path = Path(args.learnability_csv)
    if not learn_path.is_absolute():
        learn_path = _REPO_ROOT / learn_path
    stab_path = Path(args.stability_csv)
    if not stab_path.is_absolute():
        stab_path = _REPO_ROOT / stab_path
    sens_path = Path(args.sensitivity_csv)
    if not sens_path.is_absolute():
        sens_path = _REPO_ROOT / sens_path
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir

    topo = load_topology_manifest(panel)
    learn = pd.read_csv(learn_path) if learn_path.exists() else pd.DataFrame()
    stab = pd.read_csv(stab_path) if stab_path.exists() else None
    sens = pd.read_csv(sens_path) if sens_path.exists() else None

    merged = merge_tables(topo, learn, stab, sens)
    merged_path = out_dir / "merged_run_table.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_path, index=False)

    summary = per_topology_summary(merged) if "topology_id" in merged.columns else merged
    summary_path = out_dir / "topology_summary_table.csv"
    summary.to_csv(summary_path, index=False)

    stats = run_stats(merged)
    stats_path = out_dir / "inferential_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    if not args.no_figures:
        try:
            plot_figures(merged, summary, out_dir / "figures")
        except Exception as e:
            print(f"[WARN] figures skipped: {e}")

    print(f"Wrote {merged_path}, {summary_path}, {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
