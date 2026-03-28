"""
Paper 3 — Chapter 5 figure generation (Architecture Refinement).

Aligned with ``Figure Specifications_Architecture_Refinement_Paper3_Latest.md`` (revised 4-figure set).

**Inputs**
  - ``experiment3_results.csv`` — run-level rows (model, group, seed, te_hat, orc_hat,
    clean_roc_auc, RD_max, optional robust).
  - ``experiment3_summary.json`` — optional ``proxy_plane_bin_means`` (legacy proxy-bins figure).
  - ``--forensic-dir`` — outputs from ``run_paper3_forensic_pass``:
    ``forensic_ws_flex_graph_metrics.csv`` (Figure 2A). Figure 2B uses G2 topology means from
    ``experiment3_results.csv`` (and optional ``--experiment2-dir`` for TE/ORC fill from architectures).
  - Optional ``--stratified-topology-csv`` — topology-level stratified pilot table for **Figure 1**
    (both panels; preferred). Without it, Figure 1 uses G1+G2 topology means only.
  - Optional ``--experiment1-dir`` — ``proxy_pool.csv`` for legacy ``coverage`` figure.

**Outputs (main spec filenames)**
  - ``chapter5_topology_means.csv``
  - ``figure1_proxy_signal_failure.pdf`` — Figure 1 (TE or proxy vs ``RD_max``; $|\widehat{\mathrm{ORC}}|$ vs ``RD_max``, same topologies).
  - ``figure2_ws_flex_structural_limits.pdf`` — Figure 2 (TE distribution + G1+G2 run-level TE vs.\ $|\\widehat{\\mathrm{ORC}}|$ scatter, $\\mathrm{RD}_{\\max}$ viridis, 2D KDE).
  - ``figure3_out_of_family_rdmax.pdf`` — Figure 3 (family comparison: seaborn box + strip; writes ``option_b_family_contrast.json``).
  - ``figure4_clean_vs_rdmax_optional.pdf`` — Figure 4 optional (Clean ROC-AUC vs.\ RD_max).

**Default** ``--only``: ``fig1``, ``fig2``, ``fig3``. Add ``fig4`` for the optional clean-accuracy figure.

**Legacy keys** (prior pipeline): ``legacy_te_orc_space``, ``legacy_metrics_two_panel``,
``legacy_runlevel_all_groups``, ``legacy_proxy_bins``, ``coverage``.

Additional regime-only plots remain in ``run_paper3_forensic_pass.py --figures``.

Primary robustness axis: ``RD_max`` (lower is better).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architecture_refinement.paper3.run_paper3_forensic_pass import _topology_table_main_exp3
from architecture_refinement.paper3.run_paper3_option_b_family_contrast import (
    GROUP_LABELS,
    GROUP_ORDER,
    run_option_b_analysis,
)
from utils import short_run_id

FIG_FORMAT = "pdf"
FIG_DPI = 300

GROUP_COLORS = {
    "G1": "#E69F00",
    "G2": "#0072B2",
    "G3": "#009E73",
    "G4": "#D55E00",
    "G5": "#CC79A7",
}

GROUP_MARKERS = {
    "G1": "o",
    "G2": "s",
    "G3": "^",
    "G4": "D",
    "G5": "v",
}

# Match run_paper3_q3_stratified_analysis.plot_plot3_style_alignment_latest
_STRAT_TRAIN_COLOR = "#3182BD"
_STRAT_REUSE_COLOR = "#E69F00"
_STRAT_TRAIN_LABEL = "Stratified WS-Flex (pilot, new)"
_STRAT_REUSE_LABEL = "Stratified WS-Flex (pilot, G1 overlap)"

DEFAULT_FIGS = frozenset({"fig1", "fig2", "fig3", "fig4"})
LEGACY_KEYS = frozenset({
    "legacy_te_orc_space",
    "legacy_metrics_two_panel",
    "legacy_runlevel_all_groups",
    "legacy_proxy_bins",
})
VALID_ONLY = DEFAULT_FIGS | {"fig4"} | LEGACY_KEYS | {"coverage"}


def apply_figure_style() -> None:
    """
    Match ``analysis/analyze_results.py`` conventions: seaborn ``whitegrid``, sans-serif,
    axis labels 12pt, subtle grid (alpha 0.3), 300 DPI PDFs with white background.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl

    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid")
    except ImportError:
        pass

    mpl.rcParams.update({
        "figure.dpi": FIG_DPI,
        "savefig.dpi": FIG_DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans", "sans-serif"],
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.2,
        "lines.markersize": 5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "-",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
        "pdf.fonttype": 42,
    })


def _axes_grid(ax: Any) -> None:
    """Explicit grid matching ``analyze_results`` ``ax.grid(True, alpha=0.3)``."""
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def _arch_fill_te_orc(per_topo: pd.DataFrame, arch_dir: Optional[Path]) -> None:
    if arch_dir is None:
        return
    from architecture_refinement.metrics_te_orc import compute_paper3_proxies
    from architecture_refinement.paper3.arch_graph_utils import graph_from_architecture

    arch_dir = Path(arch_dir)
    for idx, row in per_topo.iterrows():
        if pd.notna(row["te_hat"]) and pd.notna(row["orc_hat"]):
            continue
        model = row["model"]
        for stem in [str(model), short_run_id(str(model))]:
            arch_path = arch_dir / f"{stem}.json"
            if not arch_path.exists():
                continue
            try:
                arch = json.loads(arch_path.read_text(encoding="utf-8"))
                G = graph_from_architecture(arch)
                if G is not None:
                    te, oc = compute_paper3_proxies(G)
                    per_topo.at[idx, "te_hat"] = te
                    per_topo.at[idx, "orc_hat"] = oc
                    break
            except Exception:
                continue


def build_chapter5_topology_means(
    df: pd.DataFrame,
    arch_dir: Optional[Path] = None,
    summary_json: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    One row per (group, model) with columns aligned to the chapter spec:
    group, model, te_hat, orc_hat, clean_roc_auc_mean, clean_roc_auc_sd,
    RD_max_mean, RD_max_sd, robust_rate, n_seeds.
    """
    sub = df.dropna(subset=["RD_max", "group", "model"]).copy()
    if sub.empty:
        return pd.DataFrame()

    gb = sub.groupby(["model", "group"], as_index=False)
    if "clean_roc_auc" in sub.columns:
        per_topo = gb.agg(
            RD_max_mean=("RD_max", "mean"),
            RD_max_sd=("RD_max", "std"),
            n_seeds=("RD_max", "count"),
            te_hat=("te_hat", "first"),
            orc_hat=("orc_hat", "first"),
            clean_roc_auc_mean=("clean_roc_auc", "mean"),
            clean_roc_auc_sd=("clean_roc_auc", "std"),
        )
    else:
        per_topo = gb.agg(
            RD_max_mean=("RD_max", "mean"),
            RD_max_sd=("RD_max", "std"),
            n_seeds=("RD_max", "count"),
            te_hat=("te_hat", "first"),
            orc_hat=("orc_hat", "first"),
        )
        per_topo["clean_roc_auc_mean"] = np.nan
        per_topo["clean_roc_auc_sd"] = np.nan

    _arch_fill_te_orc(per_topo, arch_dir)

    threshold: Optional[float] = None
    if summary_json is not None:
        t = summary_json.get("robust_threshold_RD_max")
        if t is not None and np.isfinite(float(t)):
            threshold = float(t)

    robust_rates: List[float] = []
    for _, row in per_topo.iterrows():
        m, g = row["model"], row["group"]
        part = sub[(sub["model"] == m) & (sub["group"] == g)]
        if "robust" in part.columns and part["robust"].notna().any():
            rr = part["robust"]
            if rr.dtype == object:
                rr = rr.map(lambda x: x is True or x == "True" or x == 1.0)
            robust_rates.append(float(rr.mean()))
        elif threshold is not None:
            robust_rates.append(float((part["RD_max"] <= threshold).mean()))
        else:
            robust_rates.append(float("nan"))
    per_topo["robust_rate"] = robust_rates

    # Spec column order; rename RD_* already correct
    cols = [
        "group",
        "model",
        "te_hat",
        "orc_hat",
        "clean_roc_auc_mean",
        "clean_roc_auc_sd",
        "RD_max_mean",
        "RD_max_sd",
        "robust_rate",
        "n_seeds",
    ]
    out = per_topo[cols]
    return out


def _savefig(path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        path,
        format=FIG_FORMAT,
        dpi=FIG_DPI,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()


def _add_linear_correlation_annotation(ax, xx: np.ndarray, yy: np.ndarray) -> None:
    mask = np.isfinite(xx) & np.isfinite(yy)
    if mask.sum() < 2:
        return
    try:
        from scipy.stats import pearsonr, spearmanr

        pr, pp = pearsonr(xx[mask], yy[mask])
        sr, sp = spearmanr(xx[mask], yy[mask])
        coef = np.polyfit(xx[mask], yy[mask], 1)
        xs = np.linspace(float(np.nanmin(xx[mask])), float(np.nanmax(xx[mask])), 80)
        ax.plot(xs, np.polyval(coef, xs), color="#888888", linestyle="--", linewidth=1.0, zorder=0)
        ax.text(
            0.02,
            0.98,
            f"Pearson $r={pr:.3f}$ ($p={pp:.3g}$)\nSpearman $\\rho={sr:.3f}$ ($p={sp:.3g}$)",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            zorder=10,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": "0.72",
                "linewidth": 0.7,
                "alpha": 0.94,
            },
        )
    except Exception:
        pass


def plot_figure1_spec(
    topo: pd.DataFrame,
    out_path: Path,
    *,
    stratified_topo: Optional[pd.DataFrame] = None,
    fig1_metric: str = "te",
) -> None:
    """Figure 1 (spec): Panel A TE or proxy vs RD_max; Panel B |ORC| vs RD_max (same topology units)."""
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.8, 3.9))

    # --- Panel A: stratified topology means or G1/G2 fallback ---
    xlab = r"Topological entropy ($\widehat{\mathrm{TE}}$)"
    if fig1_metric == "proxy":
        xlab = r"Proxy score $\frac{1}{2}(\widehat{\mathrm{TE}} + |\widehat{\mathrm{ORC}}|)$"
    panel_a_note = ""
    xx = np.array([], dtype=float)
    yy = np.array([], dtype=float)

    if stratified_topo is not None and not stratified_topo.empty:
        st = stratified_topo.copy()
        st["orc_abs"] = st["orc_hat"].abs()
        st["_x"] = 0.5 * (st["te_hat"] + st["orc_abs"]) if fig1_metric == "proxy" else st["te_hat"]
        ax_a.scatter(
            st["_x"],
            st["_rd"],
            c="#333333",
            s=42,
            alpha=0.82,
            edgecolors="k",
            linewidths=0.45,
            zorder=2,
            label="Topology mean",
        )
        xx = st["_x"].to_numpy(dtype=float)
        yy = st["_rd"].to_numpy(dtype=float)
        panel_a_note = "Stratified pilot (topology means)"
    else:
        ws = topo[topo["group"].isin(["G1", "G2"])].dropna(subset=["te_hat", "orc_hat", "RD_max_mean"])
        if ws.empty:
            ax_a.text(0.5, 0.5, "No G1/G2 topology data", ha="center", va="center", transform=ax_a.transAxes)
        else:
            ws = ws.copy()
            ws["orc_abs"] = ws["orc_hat"].abs()
            ws["_x"] = (
                0.5 * (ws["te_hat"].astype(float) + ws["orc_abs"].astype(float))
                if fig1_metric == "proxy"
                else ws["te_hat"].astype(float)
            )
            for g in ["G1", "G2"]:
                sub = ws[ws["group"] == g]
                if sub.empty:
                    continue
                ax_a.scatter(
                    sub["_x"],
                    sub["RD_max_mean"],
                    c=GROUP_COLORS.get(g, "gray"),
                    s=45,
                    alpha=0.85,
                    edgecolors="k",
                    linewidths=0.5,
                    label=GROUP_LABELS.get(g, g),
                )
            xx = ws["_x"].to_numpy(dtype=float)
            yy = ws["RD_max_mean"].to_numpy(dtype=float)
            panel_a_note = "Main study WS-Flex (G1/G2) topology means"

    _add_linear_correlation_annotation(ax_a, xx, yy)

    ax_a.set_xlabel(xlab)
    ax_a.set_ylabel(r"$\mathrm{RD}_{\max}$ (topology mean)")
    if fig1_metric == "proxy":
        ax_a.set_title("A: Proxy score vs $\\mathrm{RD}_{\\max}$")
    else:
        ax_a.set_title("A: $\\widehat{\\mathrm{TE}}$ vs $\\mathrm{RD}_{\\max}$")
    if panel_a_note:
        ax_a.text(0.98, 0.02, panel_a_note, transform=ax_a.transAxes, ha="right", va="bottom", fontsize=8, style="italic")
    h, l = ax_a.get_legend_handles_labels()
    if h:
        ax_a.legend(loc="best", frameon=False, fontsize=10)

    # --- Panel B: |ORC| vs RD_max (same topology-level sample as A) ---
    xb = np.array([], dtype=float)
    yb = np.array([], dtype=float)
    if stratified_topo is not None and not stratified_topo.empty:
        stb = stratified_topo.copy()
        stb["orc_abs"] = pd.to_numeric(stb["orc_hat"], errors="coerce").abs()
        ax_b.scatter(
            stb["orc_abs"],
            stb["_rd"],
            c="#333333",
            s=42,
            alpha=0.82,
            edgecolors="k",
            linewidths=0.45,
            zorder=2,
            label="Topology mean",
        )
        xb = stb["orc_abs"].to_numpy(dtype=float)
        yb = stb["_rd"].to_numpy(dtype=float)
        ax_b.text(0.98, 0.02, panel_a_note, transform=ax_b.transAxes, ha="right", va="bottom", fontsize=8, style="italic")
    else:
        ws_b = topo[topo["group"].isin(["G1", "G2"])].dropna(subset=["te_hat", "orc_hat", "RD_max_mean"])
        if ws_b.empty:
            ax_b.text(0.5, 0.5, "No G1/G2 topology data", ha="center", va="center", transform=ax_b.transAxes)
        else:
            ws_b = ws_b.copy()
            ws_b["orc_abs"] = ws_b["orc_hat"].astype(float).abs()
            for g in ["G1", "G2"]:
                sub = ws_b[ws_b["group"] == g]
                if sub.empty:
                    continue
                ax_b.scatter(
                    sub["orc_abs"],
                    sub["RD_max_mean"],
                    c=GROUP_COLORS.get(g, "gray"),
                    s=45,
                    alpha=0.85,
                    edgecolors="k",
                    linewidths=0.5,
                    label=GROUP_LABELS.get(g, g),
                )
            xb = ws_b["orc_abs"].to_numpy(dtype=float)
            yb = ws_b["RD_max_mean"].to_numpy(dtype=float)
            ax_b.text(0.98, 0.02, panel_a_note, transform=ax_b.transAxes, ha="right", va="bottom", fontsize=8, style="italic")

    _add_linear_correlation_annotation(ax_b, xb, yb)
    ax_b.set_xlabel(r"Ollivier--Ricci magnitude ($\widehat{|\mathrm{ORC}|}$)")
    ax_b.set_ylabel(r"$\mathrm{RD}_{\max}$ (topology mean)")
    ax_b.set_title("B: $|\widehat{\mathrm{ORC}}|$ vs $\\mathrm{RD}_{\\max}$")
    hb, lb = ax_b.get_legend_handles_labels()
    if hb:
        ax_b.legend(loc="best", frameon=False, fontsize=10)

    _axes_grid(ax_a)
    _axes_grid(ax_b)
    fig.suptitle("Proxy Signal Failure under Controlled Sampling", y=1.03, fontsize=12)
    fig.tight_layout()
    _savefig(out_path)


def plot_legacy_te_orc_space(topo: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    t = topo.dropna(subset=["te_hat", "orc_hat", "RD_max_mean"])
    if t.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        _savefig(out_path)
        return

    rmin = float(t["RD_max_mean"].min())
    rmax = float(t["RD_max_mean"].max())
    if not np.isfinite(rmin) or not np.isfinite(rmax) or abs(rmax - rmin) < 1e-12:
        rmin, rmax = rmin - 1e-6, rmax + 1e-6
    norm = mcolors.Normalize(vmin=rmin, vmax=rmax)
    cmap = plt.cm.viridis
    for g in GROUP_ORDER:
        sub = t[t["group"] == g]
        if sub.empty:
            continue
        c = cmap(norm(sub["RD_max_mean"].to_numpy(dtype=float)))
        ax.scatter(
            sub["te_hat"],
            sub["orc_hat"].abs(),
            c=c,
            s=70,
            marker=GROUP_MARKERS.get(g, "o"),
            edgecolors="black",
            linewidths=0.75,
            alpha=0.9,
            label=GROUP_LABELS.get(g, g),
        )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\mathrm{RD}_{\max}$ (mean over seeds)")
    ax.set_xlabel(r"Topological entropy ($\widehat{\mathrm{TE}}$)")
    ax.set_ylabel(r"Ollivier--Ricci magnitude ($\widehat{|\mathrm{ORC}|}$)")
    ax.set_title("Trained topologies in proxy metric space")
    ax.legend(loc="best", frameon=False)
    _axes_grid(ax)
    fig.tight_layout()
    _savefig(out_path)


def _plot_te_orc_heatmap_rd_max(ax, hm: Dict[str, Any]) -> None:
    """Draw TE×|ORC| binned mean RD_max (same layout as forensic_te_orc_heatmap_rd_max.pdf)."""
    import matplotlib.pyplot as plt

    cells = hm.get("cells") or []
    nb = int(hm.get("n_bins", 5))
    if not cells:
        ax.text(0.5, 0.5, "No heatmap cells", ha="center", va="center", transform=ax.transAxes)
        return
    grid = np.full((nb, nb), np.nan, dtype=float)
    cnt = np.zeros((nb, nb), dtype=float)
    for c in cells:
        i, j = int(c["te_bin"]), int(c["orc_bin"])
        if 0 <= i < nb and 0 <= j < nb:
            grid[i, j] = float(c["mean_RD_max"])
            cnt[i, j] = float(c.get("n", 0))
    im = ax.imshow(grid.T, origin="lower", aspect="auto", cmap="viridis")
    for i in range(nb):
        for j in range(nb):
            if cnt[i, j] > 0:
                ax.text(i, j, int(cnt[i, j]), ha="center", va="center", color="w", fontsize=7)
    ax.set_xlabel(r"$\widehat{\mathrm{TE}}$ bin (low to high)")
    ax.set_ylabel(r"$|\widehat{\mathrm{ORC}}|$ bin (low to high)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"mean $\mathrm{RD}_{\max}$")


def _plot_figure2_panel_b_te_orc(
    ax: Any,
    df: pd.DataFrame,
    arch_dir: Optional[Path],
    forensic_metrics_path: Optional[Path] = None,
) -> None:
    """G1+G2 run-level: each point = one trained run; TE vs |ORC|; viridis RD_max; marker encodes family; KDE overlay."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if df.empty or "group" not in df.columns:
        ax.text(0.5, 0.5, "No experiment3_results rows", ha="center", va="center", transform=ax.transAxes)
        return

    runs = df[df["group"].isin(["G1", "G2"])].copy()
    if runs.empty:
        ax.text(0.5, 0.5, "No G1/G2 run rows in experiment3_results", ha="center", va="center", transform=ax.transAxes)
        return

    for col in ("te_hat", "orc_hat", "RD_max"):
        if col not in runs.columns:
            runs[col] = np.nan
        else:
            runs[col] = pd.to_numeric(runs[col], errors="coerce")

    if forensic_metrics_path is not None and Path(forensic_metrics_path).exists():
        div_df = pd.read_csv(forensic_metrics_path)
        if "model" in div_df.columns and "te_hat_recomputed" in div_df.columns:
            mcols = ["model", "te_hat_recomputed"]
            if "orc_hat_recomputed" in div_df.columns:
                mcols.append("orc_hat_recomputed")
            mrg = div_df[mcols].drop_duplicates(subset=["model"])
            runs = runs.merge(mrg, on="model", how="left")
            runs["te_hat"] = runs["te_hat"].fillna(pd.to_numeric(runs["te_hat_recomputed"], errors="coerce"))
            if "orc_hat_recomputed" in runs.columns:
                runs["orc_hat"] = runs["orc_hat"].fillna(
                    pd.to_numeric(runs["orc_hat_recomputed"], errors="coerce")
                )
            for drop_c in ("te_hat_recomputed", "orc_hat_recomputed"):
                if drop_c in runs.columns:
                    runs.drop(columns=[drop_c], inplace=True)

    fill_df = (
        runs.groupby(["model", "group"], as_index=False).first()[["model", "group", "te_hat", "orc_hat"]].copy()
    )
    fill_df["RD_max_mean"] = 0.0
    _arch_fill_te_orc(fill_df, arch_dir)
    merged = fill_df[["model", "group", "te_hat", "orc_hat"]].rename(
        columns={"te_hat": "_te_fill", "orc_hat": "_oc_fill"}
    )
    runs = runs.merge(merged, on=["model", "group"], how="left")
    runs["te_hat"] = runs["te_hat"].fillna(runs["_te_fill"])
    runs["orc_hat"] = runs["orc_hat"].fillna(runs["_oc_fill"])
    runs.drop(columns=["_te_fill", "_oc_fill"], inplace=True)

    runs["orc_abs"] = runs["orc_hat"].abs()
    sub = runs.dropna(subset=["te_hat", "orc_abs", "RD_max"])
    if sub.empty:
        ax.text(0.5, 0.5, "No valid TE / |ORC| / RD_max for G1/G2 runs", ha="center", va="center", transform=ax.transAxes)
        return

    x = sub["te_hat"].to_numpy(dtype=float)
    y = sub["orc_abs"].to_numpy(dtype=float)
    rd = sub["RD_max"].to_numpy(dtype=float)
    rmin, rmax = float(np.nanmin(rd)), float(np.nanmax(rd))
    if not np.isfinite(rmin) or not np.isfinite(rmax) or abs(rmax - rmin) < 1e-12:
        rmin, rmax = rmin - 1e-6, rmax + 1e-6
    norm = mcolors.Normalize(vmin=rmin, vmax=rmax)
    cmap = plt.cm.viridis

    # 2D KDE contours (behind points), pooled over G1+G2
    if len(x) >= 3:
        try:
            from scipy.stats import gaussian_kde

            kde2 = gaussian_kde(np.vstack([x, y]))
            pad_x = 0.03 * (float(np.max(x)) - float(np.min(x)) + 1e-9)
            pad_y = 0.03 * (float(np.max(y)) - float(np.min(y)) + 1e-9)
            xi = np.linspace(float(np.min(x)) - pad_x, float(np.max(x)) + pad_x, 100)
            yi = np.linspace(float(np.min(y)) - pad_y, float(np.max(y)) + pad_y, 100)
            X, Y = np.meshgrid(xi, yi)
            Z = kde2(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            ax.contour(
                X,
                Y,
                Z,
                levels=8,
                colors="0.55",
                linewidths=0.65,
                alpha=0.9,
                zorder=1,
            )
        except Exception:
            pass

    sc_last = None
    for g in ("G1", "G2"):
        sg = sub[sub["group"] == g]
        if sg.empty:
            continue
        sc_last = ax.scatter(
            sg["te_hat"].to_numpy(dtype=float),
            sg["orc_abs"].to_numpy(dtype=float),
            c=sg["RD_max"].to_numpy(dtype=float),
            cmap=cmap,
            norm=norm,
            marker=GROUP_MARKERS.get(g, "o"),
            s=16,
            alpha=0.7,
            edgecolors="none",
            linewidths=0,
            zorder=3,
            rasterized=True,
        )

    if sc_last is not None:
        cbar = plt.colorbar(sc_last, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative Degradation (RD_max)")
    cbar.ax.tick_params(labelsize=9)

    leg_handles: List[Any] = []
    for g in ("G1", "G2"):
        if sub[sub["group"] == g].empty:
            continue
        leg_handles.append(
            Line2D(
                [0],
                [0],
                marker=GROUP_MARKERS.get(g, "o"),
                linestyle="None",
                color="none",
                markerfacecolor="0.45",
                markeredgecolor="none",
                markersize=4.0,
                label=GROUP_LABELS.get(g, g),
            )
        )
    if leg_handles:
        ax.legend(handles=leg_handles, loc="best", frameon=False, fontsize=10)

    ax.set_xlabel(r"Topological entropy ($\widehat{\mathrm{TE}}$)")
    ax.set_ylabel(r"$|\widehat{\mathrm{ORC}}|$")
    ax.set_title("B: G1 + G2 trained runs (topology $\\times$ seed)")


def plot_figure2_spec(
    out_path: Path,
    *,
    forensic_dir: Optional[Path],
    df: pd.DataFrame,
    arch_dir: Optional[Path],
) -> None:
    """Figure 2 (spec): WS-Flex TE distribution + G1+G2 run-level TE×|ORC| scatter with KDE overlay."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))

    fd: Optional[Path] = None
    if forensic_dir:
        fd = Path(forensic_dir)
        if not fd.is_absolute():
            fd = _REPO_ROOT / fd

    metrics_path = fd / "forensic_ws_flex_graph_metrics.csv" if fd else None
    if metrics_path and metrics_path.exists():
        div_df = pd.read_csv(metrics_path)
        col = "te_hat_recomputed" if "te_hat_recomputed" in div_df.columns else None
        if col:
            te = pd.to_numeric(div_df[col], errors="coerce").dropna().to_numpy(dtype=float)
            if te.size > 0:
                if te.size >= 2:
                    try:
                        from scipy.stats import gaussian_kde

                        xs = np.linspace(float(np.min(te)), float(np.max(te)), 200)
                        kde = gaussian_kde(te)
                        ax1.plot(xs, kde(xs), color="#0072B2", linewidth=1.2, zorder=1)
                        ax1.fill_between(xs, kde(xs), alpha=0.25, color="#0072B2", zorder=1)
                    except Exception:
                        ax1.hist(
                            te,
                            bins=min(20, max(5, max(1, len(te) // 2))),
                            density=True,
                            color="#4A90A4",
                            edgecolor="black",
                            alpha=0.85,
                        )
                else:
                    ax1.hist(te, bins=1, density=True, color="#4A90A4", edgecolor="black", alpha=0.85)
                q1, med, q3 = np.percentile(te, [25, 50, 75])
                ax1.axvline(q1, color="#555555", linestyle="--", linewidth=0.95, zorder=4)
                ax1.axvline(med, color="#1a1a1a", linestyle="-", linewidth=1.05, zorder=4)
                ax1.axvline(q3, color="#555555", linestyle="--", linewidth=0.95, zorder=4)
                mean_te = float(np.mean(te))
                iqr = float(np.percentile(te, 75) - np.percentile(te, 25))
                ax1.text(
                    0.02,
                    0.98,
                    f"mean $\\approx$ {mean_te:.3f}\nIQR $\\approx$ {iqr:.3f}",
                    transform=ax1.transAxes,
                    va="top",
                fontsize=9,
            )
            ax1.set_xlabel(r"Topological entropy ($\widehat{\mathrm{TE}}$)")
            ax1.set_ylabel("Density")
            ax1.set_title("A: WS-Flex TE distribution")
        else:
            ax1.text(0.5, 0.5, "Column te_hat_recomputed missing", ha="center", va="center", transform=ax1.transAxes)
    else:
        ax1.text(
            0.5,
            0.5,
            "Pass --forensic-dir with forensic_ws_flex_graph_metrics.csv\n(from run_paper3_forensic_pass)",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize=9,
        )
        print("[figure2] Missing forensic_ws_flex_graph_metrics.csv — Panel A is a stub.")

    try:
        _plot_figure2_panel_b_te_orc(
            ax2,
            df,
            arch_dir,
            forensic_metrics_path=metrics_path if metrics_path and metrics_path.exists() else None,
        )
    except Exception as e:
        ax2.text(0.5, 0.5, f"Panel B error:\n{e}", ha="center", va="center", transform=ax2.transAxes, fontsize=9)
        print(f"[figure2] Panel B failed: {e}")

    _axes_grid(ax1)
    _axes_grid(ax2)
    fig.suptitle("Structural Limitations of WS-Flex Space", y=1.02, fontsize=12)
    fig.tight_layout()
    _savefig(out_path)


def _resolve_under_repo(p: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else _REPO_ROOT / p


def _reuse_g1_models_from_plan(pilot_root: Optional[Path]) -> Set[str]:
    if pilot_root is None:
        return set()
    plan_path = _resolve_under_repo(pilot_root) / "q3_stratified_sample_plan.json"
    if not plan_path.exists():
        return set()
    plan_obj = json.loads(plan_path.read_text(encoding="utf-8"))
    return {it["model_name"] for it in plan_obj.get("items", []) if it.get("source") == "reuse_g1"}


def load_stratified_topology_table(csv_path: Path) -> pd.DataFrame:
    """Topology-level rows from Q3 stratified analysis (te_hat, orc_hat, RD_max_per_topo)."""
    path = _resolve_under_repo(csv_path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "te_hat" not in df.columns or "orc_hat" not in df.columns:
        return pd.DataFrame()
    rd_col = (
        "RD_max_per_topo"
        if "RD_max_per_topo" in df.columns
        else ("RD_max_mean" if "RD_max_mean" in df.columns else None)
    )
    if rd_col is None:
        return pd.DataFrame()
    out = df.copy()
    out["_rd"] = pd.to_numeric(out[rd_col], errors="coerce")
    out["te_hat"] = pd.to_numeric(out["te_hat"], errors="coerce")
    out["orc_hat"] = pd.to_numeric(out["orc_hat"], errors="coerce")
    if "source" not in out.columns:
        out["source"] = "train"
    return out.dropna(subset=["te_hat", "orc_hat", "_rd"])


def plot_fig3(
    topo: pd.DataFrame,
    out_path: Path,
    *,
    stratified_topo: Optional[pd.DataFrame] = None,
    reuse_g1_models: Optional[Set[str]] = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    t = topo.dropna(subset=["RD_max_mean", "te_hat", "orc_hat"]).copy()
    if reuse_g1_models:
        dup = (t["group"] == "G1") & (t["model"].isin(reuse_g1_models))
        t = t.loc[~dup].copy()

    main_alpha = 0.85
    for g in GROUP_ORDER:
        sub = t[t["group"] == g]
        if sub.empty:
            continue
        c = GROUP_COLORS.get(g, "gray")
        m = GROUP_MARKERS.get(g, "o")
        ax1.scatter(
            sub["te_hat"],
            sub["RD_max_mean"],
            c=c,
            marker=m,
            s=55,
            edgecolors="k",
            linewidths=0.5,
            label=GROUP_LABELS.get(g, g),
            alpha=main_alpha,
            zorder=2,
        )
        ax2.scatter(
            sub["orc_hat"].abs(),
            sub["RD_max_mean"],
            c=c,
            marker=m,
            s=55,
            edgecolors="k",
            linewidths=0.5,
            label=GROUP_LABELS.get(g, g),
            alpha=main_alpha,
            zorder=2,
        )

    strat = stratified_topo if stratified_topo is not None else pd.DataFrame()
    if not strat.empty:
        for source, label, c in (
            ("train", _STRAT_TRAIN_LABEL, _STRAT_TRAIN_COLOR),
            ("reuse_g1", _STRAT_REUSE_LABEL, _STRAT_REUSE_COLOR),
        ):
            ssub = strat[strat["source"] == source]
            if ssub.empty:
                continue
            ax1.scatter(
                ssub["te_hat"],
                ssub["_rd"],
                c=c,
                marker="P",
                s=68,
                edgecolors="black",
                linewidths=0.75,
                alpha=0.95,
                zorder=5,
                label=label,
            )
            ax2.scatter(
                ssub["orc_hat"].abs(),
                ssub["_rd"],
                c=c,
                marker="P",
                s=68,
                edgecolors="black",
                linewidths=0.75,
                alpha=0.95,
                zorder=5,
                label=label,
            )

    def _line_and_rho(ax, xx: np.ndarray, yy: np.ndarray, *, suffix: str) -> None:
        mask = np.isfinite(xx) & np.isfinite(yy)
        if mask.sum() < 2:
            return
        try:
            from scipy.stats import spearmanr

            rho, _ = spearmanr(xx[mask], yy[mask])
            coef = np.polyfit(xx[mask], yy[mask], 1)
            xs = np.linspace(float(np.nanmin(xx[mask])), float(np.nanmax(xx[mask])), 80)
            ax.plot(xs, np.polyval(coef, xs), color="#bbbbbb", linestyle="--", linewidth=1.0, zorder=0)
            extra = f" {suffix}" if suffix else ""
            ax.text(
                0.02,
                0.98,
                rf"Spearman $\rho \approx {rho:.3f}$" + extra,
                transform=ax.transAxes,
                va="top",
                fontsize=9,
            )
        except Exception:
            pass

    if not strat.empty and len(strat) >= 2:
        st_te = strat["te_hat"].to_numpy(dtype=float)
        st_rd = strat["_rd"].to_numpy(dtype=float)
        st_oc = strat["orc_hat"].abs().to_numpy(dtype=float)
        _line_and_rho(ax1, st_te, st_rd, suffix="(stratified pilot)")
        _line_and_rho(ax2, st_oc, st_rd, suffix="(stratified pilot)")
    else:
        _line_and_rho(
            ax1,
            t["te_hat"].to_numpy(dtype=float),
            t["RD_max_mean"].to_numpy(dtype=float),
            suffix="",
        )
        _line_and_rho(
            ax2,
            t["orc_hat"].abs().to_numpy(dtype=float),
            t["RD_max_mean"].to_numpy(dtype=float),
            suffix="",
        )

    ax1.set_xlabel(r"$\widehat{\mathrm{TE}}$")
    ax1.set_ylabel(r"$\mathrm{RD}_{\max}$ (topology mean)")
    ax2.set_xlabel(r"$\widehat{|\mathrm{ORC}|}$")
    ax2.set_ylabel(r"$\mathrm{RD}_{\max}$ (topology mean)")

    ax1.set_title("Robustness vs TE")
    ax2.set_title("Robustness vs |ORC|")
    ax2.legend(loc="best", frameon=False, fontsize=10)
    _axes_grid(ax1)
    _axes_grid(ax2)
    supt = "Static graph metrics vs trained robustness"
    if not strat.empty:
        supt += " (stratified pilot overlaid)"
    fig.suptitle(supt, y=1.02, fontsize=12)
    fig.tight_layout()
    _savefig(out_path)


def plot_fig4(topo: pd.DataFrame, out_path: Path, _repo_root: Path, df_run: pd.DataFrame) -> Dict[str, Any]:
    """Out-of-family RD_max: seaborn box + strip (aligned with ``analysis/analyze_results.py``)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    groups_present: List[str] = []
    for g in GROUP_ORDER:
        v = topo[topo["group"] == g]["RD_max_mean"].dropna().to_numpy()
        if len(v) == 0:
            continue
        groups_present.append(g)

    if not groups_present:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        _savefig(out_path)
        return {}

    long_rows: List[Dict[str, Any]] = []
    for g in groups_present:
        for val in topo.loc[topo["group"] == g, "RD_max_mean"].dropna():
            long_rows.append({"group": g, "RD_max": float(val)})
    long_df = pd.DataFrame(long_rows)
    long_df["group"] = pd.Categorical(long_df["group"], categories=groups_present, ordered=True)

    palette = [GROUP_COLORS[g] for g in groups_present]
    sns.boxplot(
        data=long_df,
        x="group",
        y="RD_max",
        order=groups_present,
        palette=palette,
        ax=ax,
        width=0.55,
        linewidth=1.0,
        fliersize=0,
        boxprops=dict(alpha=0.55, linewidth=1.0),
        medianprops=dict(color="0.15", linewidth=1.6),
        whiskerprops=dict(color="0.35", linewidth=0.9),
        capprops=dict(color="0.35", linewidth=0.9),
    )
    sns.stripplot(
        data=long_df,
        x="group",
        y="RD_max",
        order=groups_present,
        hue="group",
        palette=palette,
        ax=ax,
        dodge=False,
        jitter=0.12,
        size=4,
        alpha=0.82,
        linewidth=0.45,
        edgecolor="0.2",
        legend=False,
        zorder=3,
    )
    ax.set_xticklabels([GROUP_LABELS[g] for g in groups_present], rotation=22, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel(r"$\mathrm{RD}_{\max}$ (topology mean over seeds)")
    ax.set_title("Cross-Comparison: Topology Families")
    _axes_grid(ax)
    fig.tight_layout()
    _savefig(out_path)

    stats = run_option_b_analysis(df_run, manifest_meta=None)
    return stats


def plot_fig5(topo: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    t = topo.dropna(subset=["clean_roc_auc_mean", "RD_max_mean"])
    if t.empty:
        ax.text(0.5, 0.5, "Need clean_roc_auc in results", ha="center", va="center", transform=ax.transAxes)
        _savefig(out_path)
        return

    for g in GROUP_ORDER:
        sub = t[t["group"] == g]
        if sub.empty:
            continue
        ax.scatter(
            sub["clean_roc_auc_mean"],
            sub["RD_max_mean"],
            c=GROUP_COLORS.get(g, "gray"),
            marker=GROUP_MARKERS.get(g, "o"),
            s=60,
            edgecolors="black",
            linewidths=0.6,
            label=GROUP_LABELS.get(g, g),
            alpha=0.88,
        )
    ax.set_xlabel("Clean ROC-AUC (mean over seeds)")
    ax.set_ylabel(r"$\mathrm{RD}_{\max}$ (topology mean)")
    ax.set_title("Clean ROC-AUC vs. Relative Degradation")
    ax.legend(loc="best", frameon=False)
    _axes_grid(ax)
    fig.tight_layout()
    _savefig(out_path)


def plot_fig6_fixed(df: pd.DataFrame, out_path: Path) -> None:
    """Box + jitter by group order G1..G5 with stable positions."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sub = df.dropna(subset=["RD_max", "group"])
    positions: List[int] = []
    data: List[np.ndarray] = []
    labels: List[str] = []
    pos = 1
    for g in GROUP_ORDER:
        v = sub[sub["group"] == g]["RD_max"].dropna().to_numpy()
        if len(v) == 0:
            continue
        data.append(v)
        labels.append(GROUP_LABELS.get(g, g))
        positions.append(pos)
        pos += 1

    if not data:
        ax.text(0.5, 0.5, "No run-level data", ha="center", va="center", transform=ax.transAxes)
        _savefig(out_path)
        return

    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True)
    for i, g in enumerate([g for g in GROUP_ORDER if len(sub[sub["group"] == g]["RD_max"].dropna()) > 0]):
        bp["boxes"][i].set_facecolor(GROUP_COLORS.get(g, "#cccccc"))
        bp["boxes"][i].set_alpha(0.45)
        part = sub[sub["group"] == g]["RD_max"].dropna().to_numpy()
        x = np.random.normal(positions[i], 0.06, size=len(part))
        ax.scatter(x, part, alpha=0.4, s=22, c=GROUP_COLORS.get(g, "k"), edgecolors="none", zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel(r"$\mathrm{RD}_{\max}$ (per run)")
    ax.set_title("Run-level variability by family")
    _axes_grid(ax)
    fig.tight_layout()
    _savefig(out_path)


def plot_fig7(summary: Dict[str, Any], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    bins = summary.get("proxy_plane_bin_means") or {}
    if not bins:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "proxy_plane_bin_means not in summary", ha="center", va="center", transform=ax.transAxes)
        _savefig(out_path)
        return

    keys = sorted(bins.keys(), key=lambda s: str(s))
    vals = [float(bins[k]) for k in keys]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(range(len(keys)), vals, color="#4A90A4", edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(r"Mean $\mathrm{RD}_{\max}$ in bin")
    ax.set_title("Proxy-plane bin means (WS-Flex study)")
    _axes_grid(ax)
    fig.tight_layout()
    _savefig(out_path)


def plot_coverage(
    experiment1_dir: Path,
    topo: pd.DataFrame,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    pool_path = experiment1_dir / "proxy_pool.csv"
    if not pool_path.exists():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, f"Missing {pool_path.name}", ha="center", va="center", transform=ax.transAxes)
        _savefig(out_path)
        return

    df_pool = pd.read_csv(pool_path)
    oc_pool = pd.to_numeric(df_pool["orc_hat"], errors="coerce").abs() if "orc_hat" in df_pool.columns else pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(
        pd.to_numeric(df_pool["te_hat"], errors="coerce"),
        oc_pool,
        alpha=0.25,
        s=10,
        c="gray",
        label="Sampled candidates",
        rasterized=True,
    )
    t = topo.dropna(subset=["te_hat", "orc_hat"])
    if not t.empty:
        ax.scatter(
            t["te_hat"],
            t["orc_hat"].abs(),
            s=55,
            c="orange",
            edgecolors="black",
            linewidths=0.9,
            label="Evaluated trained topologies",
            zorder=4,
        )
    ax.set_xlabel(r"$\widehat{\mathrm{TE}}$")
    ax.set_ylabel(r"$\widehat{|\mathrm{ORC}|}$")
    ax.set_title("Metric space coverage (Experiment 1 pool + Experiment 3)")
    ax.legend(loc="lower right", frameon=False)
    _axes_grid(ax)
    fig.tight_layout()
    _savefig(out_path)


def _parse_only(s: Optional[str]) -> Optional[Set[str]]:
    if not s:
        return None
    parts = {x.strip().lower() for x in s.split(",") if x.strip()}
    bad = parts - VALID_ONLY
    if bad:
        raise SystemExit(f"Unknown --only keys: {bad}. Valid: {sorted(VALID_ONLY)}")
    return parts


def run(
    *,
    experiment3_dir: Path,
    experiment2_dir: Optional[Path],
    experiment1_dir: Optional[Path],
    output_dir: Path,
    write_topology_csv: bool,
    only: Optional[Set[str]],
    stratified_topology_csv: Optional[Path] = None,
    stratified_pilot_root: Optional[Path] = None,
    forensic_dir: Optional[Path] = None,
    fig1_metric: str = "te",
) -> None:
    apply_figure_style()
    exp3_dir = Path(experiment3_dir)
    if not exp3_dir.is_absolute():
        exp3_dir = _REPO_ROOT / exp3_dir
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = exp3_dir / "experiment3_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    summary_path = exp3_dir / "experiment3_summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    arch_dir: Optional[Path] = None
    if experiment2_dir:
        e2 = Path(experiment2_dir)
        if not e2.is_absolute():
            e2 = _REPO_ROOT / e2
        arch_dir = e2 / "experiment2_pilot" / "selected_architectures"
        if not arch_dir.exists():
            cand = e2.parent / "experiment1" / "selected_architectures"
            if cand.exists():
                arch_dir = cand

    topo = build_chapter5_topology_means(df, arch_dir=arch_dir, summary_json=summary)
    if write_topology_csv and not topo.empty:
        topo.to_csv(out_dir / "chapter5_topology_means.csv", index=False)

    want = only if only is not None else DEFAULT_FIGS

    strat_df = pd.DataFrame()
    reuse_g1: Set[str] = set()
    if stratified_topology_csv:
        strat_df = load_stratified_topology_table(Path(stratified_topology_csv))
        if strat_df.empty:
            print("[stratified] --stratified-topology-csv: no usable rows; Figure 1 falls back to G1/G2 topology means.")
        elif stratified_pilot_root:
            reuse_g1 = _reuse_g1_models_from_plan(Path(stratified_pilot_root))

    if "fig1" in want:
        plot_figure1_spec(
            topo,
            out_dir / "figure1_proxy_signal_failure.pdf",
            stratified_topo=strat_df if not strat_df.empty else None,
            fig1_metric=fig1_metric,
        )
    if "fig2" in want:
        plot_figure2_spec(
            out_dir / "figure2_ws_flex_structural_limits.pdf",
            forensic_dir=Path(forensic_dir) if forensic_dir else None,
            df=df,
            arch_dir=arch_dir,
        )
    if "fig3" in want:
        st = plot_fig4(topo, out_dir / "figure3_out_of_family_rdmax.pdf", _REPO_ROOT, df)
        if st:
            (out_dir / "option_b_family_contrast.json").write_text(json.dumps(st, indent=2, default=str), encoding="utf-8")
    if "fig4" in want:
        plot_fig5(topo, out_dir / "figure4_clean_vs_rdmax_optional.pdf")

    if "legacy_te_orc_space" in want:
        plot_legacy_te_orc_space(topo, out_dir / "legacy_te_orc_metric_space_trained.pdf")
    if "legacy_metrics_two_panel" in want:
        plot_fig3(
            topo,
            out_dir / "legacy_metrics_vs_rdmax_two_panel.pdf",
            stratified_topo=strat_df if not strat_df.empty else None,
            reuse_g1_models=reuse_g1 if reuse_g1 else None,
        )
    if "legacy_runlevel_all_groups" in want:
        plot_fig6_fixed(df, out_dir / "legacy_run_level_rdmax_by_family.pdf")
    if "legacy_proxy_bins" in want:
        plot_fig7(summary, out_dir / "legacy_proxy_plane_bin_means.pdf")
    if "coverage" in want:
        e1 = experiment1_dir
        if e1:
            e1p = Path(e1)
            if not e1p.is_absolute():
                e1p = _REPO_ROOT / e1p
            plot_coverage(e1p, topo, out_dir / "fig_coverage_te_orc.pdf")
        else:
            print("[coverage] Skipped: pass --experiment1-dir for proxy_pool.csv")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Paper 3 chapter figures (Experiment 3, revised spec)")
    parser.add_argument(
        "--experiment3-dir",
        type=str,
        required=True,
        help="Directory with experiment3_results.csv and experiment3_summary.json",
    )
    parser.add_argument("--experiment2-dir", type=str, default=None, help="Experiment 2 dir (arch JSON; heatmap recompute)")
    parser.add_argument(
        "--forensic-dir",
        type=str,
        default=None,
        help="run_paper3_forensic_pass output dir (forensic_ws_flex_graph_metrics.csv, forensic_pass_summary.json)",
    )
    parser.add_argument("--experiment1-dir", type=str, default=None, help="Experiment 1 dir for legacy coverage (proxy_pool.csv)")
    parser.add_argument(
        "--stratified-topology-csv",
        type=str,
        default=None,
        help="Stratified topology table for Figure 1 Panel A (preferred over G1/G2 means-only)",
    )
    parser.add_argument(
        "--stratified-pilot-root",
        type=str,
        default=None,
        help="q3_stratified_pilot dir for legacy two-panel G1 overlap dedup",
    )
    parser.add_argument(
        "--fig1-metric",
        type=str,
        choices=("te", "proxy"),
        default="te",
        help="Figure 1 Panel A x-axis: TE or proxy score 0.5*(TE+|ORC|)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="architecture_refinement/outputs/paper3/figures_chapter5",
    )
    parser.add_argument("--write-topology-csv", action="store_true", default=True)
    parser.add_argument("--no-write-topology-csv", action="store_false", dest="write_topology_csv")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated. Main: fig1,fig2,fig3[,fig4]. Legacy: legacy_te_orc_space,legacy_metrics_two_panel,... coverage",
    )
    args = parser.parse_args()
    only = _parse_only(args.only)
    run(
        experiment3_dir=Path(args.experiment3_dir),
        experiment2_dir=Path(args.experiment2_dir) if args.experiment2_dir else None,
        experiment1_dir=Path(args.experiment1_dir) if args.experiment1_dir else None,
        output_dir=Path(args.output_dir),
        write_topology_csv=args.write_topology_csv,
        only=only,
        stratified_topology_csv=Path(args.stratified_topology_csv) if args.stratified_topology_csv else None,
        stratified_pilot_root=Path(args.stratified_pilot_root) if args.stratified_pilot_root else None,
        forensic_dir=Path(args.forensic_dir) if args.forensic_dir else None,
        fig1_metric=str(args.fig1_metric),
    )
    print(f"[OK] Figures written under {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
