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

from architecture_refinement.paper3.hail_mary_cli import add_overwrite_arguments, can_write_output


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


def _primary_learnability_col(df: pd.DataFrame) -> Optional[str]:
    if "best_val_roc_auc" in df.columns and df["best_val_roc_auc"].notna().any():
        return "best_val_roc_auc"
    if "best_val_balanced_accuracy" in df.columns and df["best_val_balanced_accuracy"].notna().any():
        return "best_val_balanced_accuracy"
    if "best_val_accuracy" in df.columns and df["best_val_accuracy"].notna().any():
        return "best_val_accuracy"
    return None


def per_topology_learnability_seed_table(merged: pd.DataFrame, learn_col: str) -> pd.DataFrame:
    if "topology_id" not in merged.columns or learn_col not in merged.columns:
        return pd.DataFrame()
    sub = merged.dropna(subset=[learn_col])
    if sub.empty:
        return pd.DataFrame()
    t = (
        sub.groupby("topology_id", dropna=False)
        .agg(
            te_hat=("te_hat", "first"),
            mean_learnability=(learn_col, "mean"),
            std_learnability_across_seeds=(learn_col, "std"),
            n_seeds=(learn_col, "count"),
        )
        .reset_index()
    )
    return t


def run_stats(merged: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "notes": [
            "Within-epoch batch_train_loss_var is often 0 when only one training batch fits per epoch; prefer epoch-to-epoch loss metrics (valid_loss_epoch_to_epoch_var, train_loss_epoch_to_epoch_var, mean_train_loss_epoch_abs_delta) for stability comparisons.",
            "If valid_roc_auc is logged but every epoch is NaN, learnability summaries use valid_acc (see run_hail_mary_learnability summarize_one_history).",
        ]
    }
    learn_col = _primary_learnability_col(merged)
    if learn_col and "te_hat" in merged.columns:
        r, p = _spearman(merged["te_hat"].to_numpy(), merged[learn_col].to_numpy())
        out["spearman_te_hat_vs_primary_learnability"] = {
            "learnability_column": learn_col,
            "r": r,
            "p": p,
        }
    if "te_hat" in merged.columns and "best_val_roc_auc" in merged.columns:
        r, p = _spearman(
            merged["te_hat"].to_numpy(),
            merged["best_val_roc_auc"].to_numpy(),
        )
        out["spearman_te_hat_vs_best_val_roc_auc"] = {"r": r, "p": p}
    if "orc_hat" in merged.columns and "RD_moderate" in merged.columns:
        r, p = _spearman(merged["orc_hat"].to_numpy(), merged["RD_moderate"].to_numpy())
        out["spearman_orc_hat_vs_RD_moderate"] = {"r": r, "p": p}
    if "topology_id" in merged.columns and learn_col:
        out[f"kruskal_{learn_col}_by_topology"] = _kruskal_groups(
            merged.dropna(subset=[learn_col]),
            learn_col,
            "topology_id",
        )
    if "te_hat" in merged.columns and "mean_train_loss_epoch_abs_delta" in merged.columns:
        r, p = _spearman(
            merged["te_hat"].to_numpy(),
            merged["mean_train_loss_epoch_abs_delta"].to_numpy(),
        )
        out["spearman_te_hat_vs_mean_train_loss_epoch_abs_delta"] = {"r": r, "p": p}
    if "te_hat" in merged.columns and "valid_loss_epoch_to_epoch_var" in merged.columns:
        r, p = _spearman(
            merged["te_hat"].to_numpy(),
            merged["valid_loss_epoch_to_epoch_var"].to_numpy(),
        )
        out["spearman_te_hat_vs_valid_loss_epoch_to_epoch_var"] = {"r": r, "p": p}
    if "te_hat" in merged.columns and "train_loss_epoch_to_epoch_var" in merged.columns:
        r, p = _spearman(
            merged["te_hat"].to_numpy(),
            merged["train_loss_epoch_to_epoch_var"].to_numpy(),
        )
        out["spearman_te_hat_vs_train_loss_epoch_to_epoch_var"] = {"r": r, "p": p}
    if "topology_id" in merged.columns and "mean_train_loss_epoch_abs_delta" in merged.columns:
        out["kruskal_mean_train_loss_epoch_abs_delta_by_topology"] = _kruskal_groups(
            merged.dropna(subset=["mean_train_loss_epoch_abs_delta"]),
            "mean_train_loss_epoch_abs_delta",
            "topology_id",
        )
    if "topology_id" in merged.columns and "valid_loss_epoch_to_epoch_var" in merged.columns:
        out["kruskal_valid_loss_epoch_to_epoch_var_by_topology"] = _kruskal_groups(
            merged.dropna(subset=["valid_loss_epoch_to_epoch_var"]),
            "valid_loss_epoch_to_epoch_var",
            "topology_id",
        )
    if learn_col:
        pt = per_topology_learnability_seed_table(merged, learn_col)
        if not pt.empty and len(pt) >= 3:
            r, p = _spearman(pt["te_hat"].to_numpy(), pt["std_learnability_across_seeds"].to_numpy())
            out["spearman_te_hat_vs_std_learnability_across_seeds"] = {
                "learnability_column": learn_col,
                "r": r,
                "p": p,
                "n_topologies": int(len(pt)),
            }
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

    learn_col = _primary_learnability_col(summary)
    # Early learnability bar chart (mean primary learnability per topology)
    if learn_col and "topology_id" in summary.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        s = summary.sort_values("topology_id")
        ax.bar(range(len(s)), s[learn_col].to_numpy(), color="#0072B2")
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels(s["topology_id"].astype(str), rotation=45, ha="right")
        ylab = {
            "best_val_roc_auc": "Mean best val ROC-AUC (across seeds)",
            "best_val_balanced_accuracy": "Mean best val balanced accuracy (across seeds)",
            "best_val_accuracy": "Mean best validation accuracy (across seeds)",
        }.get(learn_col, "Mean learnability (across seeds)")
        ax.set_ylabel(ylab)
        ax.set_title("Hail Mary: learnability by topology")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_learnability_best_val_by_topology.pdf", dpi=200)
        plt.close()

    # TE vs early val (ROC preferred; else validation accuracy at epoch 10)
    early_col = None
    y_label = ""
    if "val_roc_auc_epoch_10" in merged.columns and merged["val_roc_auc_epoch_10"].notna().any():
        early_col, y_label = "val_roc_auc_epoch_10", "Val ROC-AUC (epoch 10)"
    elif "val_accuracy_epoch_10" in merged.columns and merged["val_accuracy_epoch_10"].notna().any():
        early_col, y_label = "val_accuracy_epoch_10", "Val accuracy (epoch 10)"
    if early_col and "te_hat" in merged.columns:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(merged["te_hat"], merged[early_col], alpha=0.7, c="#D55E00")
        ax.set_xlabel("TE_hat")
        ax.set_ylabel(y_label)
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

    # Stability: epoch-to-epoch validation loss variance (preferred over batch variance when batch var is 0)
    if "topology_id" in summary.columns and "valid_loss_epoch_to_epoch_var" in summary.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        s = summary.sort_values("topology_id")
        ax.bar(range(len(s)), s["valid_loss_epoch_to_epoch_var"].to_numpy(), color="#882255")
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels(s["topology_id"].astype(str), rotation=45, ha="right")
        ax.set_ylabel("Mean valid loss epoch-to-epoch variance")
        ax.set_title("Hail Mary: validation loss volatility by topology (mean across seeds)")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_stability_valid_loss_epoch_var_by_topology.pdf", dpi=200)
        plt.close()

    if "topology_id" in summary.columns and "mean_train_loss_epoch_abs_delta" in summary.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        s = summary.sort_values("topology_id")
        ax.bar(range(len(s)), s["mean_train_loss_epoch_abs_delta"].to_numpy(), color="#44AA99")
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels(s["topology_id"].astype(str), rotation=45, ha="right")
        ax.set_ylabel("Mean |Δ train loss| epoch-to-epoch")
        ax.set_title("Hail Mary: train loss step size by topology (mean across seeds)")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_stability_train_loss_abs_delta_by_topology.pdf", dpi=200)
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
    add_overwrite_arguments(parser)
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

    merged_path = out_dir / "merged_run_table.csv"
    if not can_write_output(merged_path, overwrite=args.overwrite):
        return 0

    topo = load_topology_manifest(panel)
    learn = pd.read_csv(learn_path) if learn_path.exists() else pd.DataFrame()
    stab = pd.read_csv(stab_path) if stab_path.exists() else None
    sens = pd.read_csv(sens_path) if sens_path.exists() else None

    merged = merge_tables(topo, learn, stab, sens)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_path, index=False)

    summary = per_topology_summary(merged) if "topology_id" in merged.columns else merged
    summary_path = out_dir / "topology_summary_table.csv"
    summary.to_csv(summary_path, index=False)

    stats = run_stats(merged)
    stats_path = out_dir / "inferential_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    learn_col = _primary_learnability_col(merged)
    seed_var_path = out_dir / "learnability_seed_variance_by_topology.csv"
    if learn_col:
        pt = per_topology_learnability_seed_table(merged, learn_col)
        if not pt.empty and can_write_output(seed_var_path, overwrite=args.overwrite):
            pt.to_csv(seed_var_path, index=False)

    if not args.no_figures:
        try:
            plot_figures(merged, summary, out_dir / "figures")
        except Exception as e:
            print(f"[WARN] figures skipped: {e}")

    extra = f", {seed_var_path}" if learn_col and seed_var_path.exists() else ""
    print(f"Wrote {merged_path}, {summary_path}, {stats_path}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
