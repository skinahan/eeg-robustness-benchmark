"""
Paper 3 Plotting: Generate Plot 1, 2, 3 figures from experiment outputs.
All plots saved as PDF at 300 dpi.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Output format
FIG_FORMAT = "pdf"
FIG_DPI = 300

# Estimated FLOPS for Plot 1 cost comparison (order-of-magnitude)
# Proxy: graph ops (degree entropy + Ollivier-Ricci) on H=32, E~400; ~1e7 FLOP
# Full: training + robustness eval; ~1e12 FLOP per (model, seed)
EST_PROXY_FLOPS = 1e7
EST_FULL_FLOPS = 1e12


def plot1(
    experiment1_dir: Path,
    output_path: Optional[Path] = None,
    mean_full_eval_time_sec: Optional[float] = None,
    mean_proxy_flops: Optional[float] = None,
    mean_full_flops: Optional[float] = None,
) -> Path:
    """
    Plot 1: Proxy landscape scatter (Panel A) + time bars (Panel B) + FLOPS bars (Panel C).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib")

    exp1_dir = Path(experiment1_dir)
    if not exp1_dir.is_absolute():
        exp1_dir = _REPO_ROOT / exp1_dir

    pool_path = exp1_dir / "proxy_pool.csv"
    selected_path = exp1_dir / "selected_proxy.csv"
    summary_path = exp1_dir / "experiment1_summary.json"
    if not pool_path.exists():
        raise FileNotFoundError(f"Proxy pool not found: {pool_path}")

    df_pool = pd.read_csv(pool_path)
    df_sel = pd.read_csv(selected_path) if selected_path.exists() else pd.DataFrame()
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(2, 2, width_ratios=[1.2, 1], figure=fig)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    # Panel A: Scatter
    ax1.scatter(df_pool["te_hat"], df_pool["orc_hat"], alpha=0.3, s=8, c="gray", label="All")
    if not df_sel.empty:
        for i, (_, r) in enumerate(df_sel.iterrows()):
            ax1.scatter(
                r["te_hat"], r["orc_hat"],
                s=80, marker="o", facecolors="orange", edgecolors="black", linewidths=1.5, zorder=3,
            )
            ax1.annotate(f"S{i+1}", (r["te_hat"], r["orc_hat"]), fontsize=8, ha="center", va="bottom")
    ax1.set_xlabel(r"$\hat{TE}$")
    ax1.set_ylabel(r"$\widehat{|ORC|}$")
    ax1.set_title("Proxy landscape")
    ax1.legend(loc="lower right")
    # Data-driven bounds with small margin; y-axis capped at 0.05 for visibility
    te_min, te_max = df_pool["te_hat"].min(), df_pool["te_hat"].max()
    oc_min, oc_max = df_pool["orc_hat"].min(), df_pool["orc_hat"].max()
    margin = 0.005
    ax1.set_xlim(max(0, te_min - margin), min(1, te_max + margin))
    ax1.set_ylim(max(0, oc_min - margin), 0.05)

    # Panel B: Time bars
    mean_proxy = float(summary.get("mean_proxy_time_sec", 0.001))
    mean_full = mean_full_eval_time_sec or 300.0  # placeholder if not provided
    ratio = mean_full / mean_proxy if mean_proxy > 0 else 0
    ax2.bar([0], [mean_proxy], label="Proxy eval", color="steelblue")
    ax2.bar([1], [mean_full], label="Training+robustness", color="coral")
    ax2.set_yscale("log")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Proxy", "Full eval"])
    ax2.set_ylabel("Time (sec)")
    ax2.set_title(f"Time ratio ≈ {ratio:.0f}x")
    ax2.legend()

    # Panel C: Estimated FLOPS bars
    flops_proxy = mean_proxy_flops or float(summary.get("mean_proxy_flops_est", EST_PROXY_FLOPS))
    flops_full = mean_full_flops or float(summary.get("mean_full_flops_est", EST_FULL_FLOPS))
    flops_ratio = flops_full / flops_proxy if flops_proxy > 0 else 0
    ax3.bar([0], [flops_proxy], label="Proxy eval", color="steelblue")
    ax3.bar([1], [flops_full], label="Training+robustness", color="coral")
    ax3.set_yscale("log")
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["Proxy", "Full eval"])
    ax3.set_ylabel("Est. FLOPS")
    flops_ratio_str = f"{flops_ratio:.0e}".replace("e+0", "e").replace("e+", "e")
    ax3.set_title(f"FLOPS ratio ≈ {flops_ratio_str}")
    ax3.legend()

    plt.tight_layout()
    out = output_path or exp1_dir / "plot1_proxy_landscape.pdf"
    out = Path(out)
    if out.suffix != ".pdf":
        out = out.with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, format=FIG_FORMAT, dpi=FIG_DPI)
    plt.close()
    return out


def _collect_intensity_level_data(
    repo_root: Path,
    dataset: str,
    model_name: str,
    seed: int,
    noise_type: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Collect (intensities, r_t, clean_roc_auc) for one run. Returns None if not found."""
    from utils import results_paradigm_folder, short_run_id

    paradigm = results_paradigm_folder(dataset)
    base = repo_root / "results" / paradigm / dataset
    for stem in [short_run_id(model_name), model_name]:
        path = base / stem / "CrossSessionEvaluation" / str(seed)
        if not path.exists():
            continue
        for p in path.rglob("*.csv"):
            if "test_perturb" not in str(p):
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
                sub["intensity"] = pd.to_numeric(sub.get("intensity", np.nan), errors="coerce")
                sub = sub.dropna(subset=["intensity", "corrupted_roc_auc"])
                if sub.empty:
                    continue
                # Aggregate by intensity (mean across subjects/sessions if multiple rows per intensity)
                agg = sub.groupby("intensity", as_index=False).agg(
                    {"corrupted_roc_auc": "mean", "clean_roc_auc": "first"}
                )
                agg = agg.sort_values("intensity")
                clean = float(agg["clean_roc_auc"].iloc[0])
                if not np.isfinite(clean) or clean <= 0:
                    continue
                intensities = agg["intensity"].to_numpy()
                roc_vals = agg["corrupted_roc_auc"].to_numpy()
                r_t = roc_vals / clean
                return intensities, r_t, clean
            except Exception:
                pass
    return None


def _aggregate_r_t_curves(
    runs_data: List[Tuple[np.ndarray, np.ndarray]],
    n_boot: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate r_t curves across runs. Align by intensity index.
    Returns (intensities, mean_r_t, ci_lo, ci_hi).
    """
    if not runs_data:
        return np.array([]), np.array([]), np.array([]), np.array([])
    max_len = max(len(r[0]) for r in runs_data)
    r_t_matrix: List[np.ndarray] = []
    intensity_ref = np.array([])
    for intensities, r_t in runs_data:
        if len(intensities) >= len(intensity_ref):
            intensity_ref = intensities
        padded = np.full(max_len, np.nan)
        padded[: len(r_t)] = r_t
        r_t_matrix.append(padded)
    r_t_matrix = np.array(r_t_matrix)
    # Pad intensity_ref if needed
    if len(intensity_ref) < max_len:
        intensity_ref = np.pad(
            intensity_ref, (0, max_len - len(intensity_ref)), constant_values=np.nan
        )
    mean_r_t = np.nanmean(r_t_matrix, axis=0)
    rng = np.random.default_rng(42)
    boot_lo = np.full(max_len, np.nan)
    boot_hi = np.full(max_len, np.nan)
    for i in range(max_len):
        vals = r_t_matrix[:, i]
        valid = vals[np.isfinite(vals)]
        if len(valid) < 2:
            continue
        boot_means = [
            float(np.mean(rng.choice(valid, size=len(valid), replace=True)))
            for _ in range(n_boot)
        ]
        boot_lo[i] = float(np.percentile(boot_means, 2.5))
        boot_hi[i] = float(np.percentile(boot_means, 97.5))
    return intensity_ref, mean_r_t, boot_lo, boot_hi


def plot2(
    experiment2_dir: Path,
    output_path: Optional[Path] = None,
    experiment3_dir: Optional[Path] = None,
) -> Path:
    """
    Plot 2: Robustness curves r_t and RD_max distribution.
    Requires collected results - stub that writes placeholder if results not yet aggregated.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    exp2_dir = Path(experiment2_dir)
    if not exp2_dir.is_absolute():
        exp2_dir = _REPO_ROOT / exp2_dir

    # Try to load experiment3 results (which aggregates Exp2)
    exp3_results = None
    if experiment3_dir:
        exp3_results = Path(experiment3_dir) / "experiment3_results.csv"
    if not exp3_results or not exp3_results.exists():
        exp3_results = exp2_dir.parent / "experiment3" / "experiment3_results.csv"
    if not exp3_results.exists():
        exp3_results = _REPO_ROOT / "architecture_refinement" / "outputs" / "paper3" / "experiment3" / "experiment3_results.csv"
    if not exp3_results.exists():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Run Experiment 2 and 3 to generate Plot 2", ha="center", va="center")
        out = output_path or exp2_dir / "plot2_robustness_curves.pdf"
        out = Path(out)
        if out.suffix != ".pdf":
            out = out.with_suffix(".pdf")
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, format=FIG_FORMAT, dpi=FIG_DPI)
        plt.close()
        return out

    df = pd.read_csv(exp3_results)

    manifest_path = exp2_dir / "experiment2_manifest.json"
    dataset = "BNCI2014_001"
    noise_type = "ar1_drift"
    seeds: List[int] = []
    groups_config: Dict[str, List[str]] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        dataset = manifest.get("dataset", dataset)
        S = manifest.get("S", 5)
        seeds = list(range(42, 42 + S))
        groups_config = manifest.get("groups", {}) or {}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: r_t curves (intensity-level aggregation)
    if manifest_path.exists() and groups_config:
        from utils import short_run_id

        colors = {"G1": "C0", "G2": "C1"}
        for g, label in [("G1", "Proxy-selected"), ("G2", "Uniform")]:
            models = groups_config.get(g, [])
            runs_data: List[Tuple[np.ndarray, np.ndarray]] = []
            for model_name in models:
                for seed in seeds:
                    res = _collect_intensity_level_data(
                        _REPO_ROOT, dataset, model_name, seed, noise_type
                    )
                    if res is not None:
                        intensities, r_t, _ = res
                        runs_data.append((intensities, r_t))
            if runs_data:
                xs, mean_rt, lo, hi = _aggregate_r_t_curves(runs_data)
                valid = np.isfinite(xs) & np.isfinite(mean_rt)
                if valid.any():
                    xv = xs[valid]
                    yv = mean_rt[valid]
                    ax1.plot(xv, yv, label=label, color=colors.get(g, "gray"))
                    lo_v, hi_v = lo[valid], hi[valid]
                    has_ci = np.isfinite(lo_v) & np.isfinite(hi_v)
                    if has_ci.any():
                        ax1.fill_between(
                            xv[has_ci], lo_v[has_ci], hi_v[has_ci], alpha=0.2, color=colors.get(g, "gray")
                        )
        ax1.set_xlabel("Intensity")
        ax1.set_ylabel(r"$r_t$ (corrupted / clean)")
        ax1.set_title(r"$r_t$ curves")
        ax1.legend()
        ax1.set_ylim(0, 1.05)
        ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    else:
        ax1.text(0.5, 0.5, "r_t curves require experiment2_manifest.json\nand test_perturb results", ha="center", va="center")

    # Panel 2: Performance vs robustness
    for g, label in [("G1", "Proxy-selected"), ("G2", "Uniform")]:
        sub = df[df["group"] == g]
        if not sub.empty:
            ax2.scatter(sub["clean_roc_auc"], sub["RD_max"], alpha=0.5, label=label)
    ax2.set_xlabel("Clean ROC-AUC")
    ax2.set_ylabel("RD_max")
    ax2.set_title("Performance vs robustness")
    ax2.legend()

    plt.tight_layout()
    out = output_path or exp2_dir / "plot2_robustness_curves.pdf"
    out = Path(out)
    if out.suffix != ".pdf":
        out = out.with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, format=FIG_FORMAT, dpi=FIG_DPI)
    plt.close()
    return out


def plot3(
    experiment3_dir: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Plot 3: Proxy plane colored by RD_max (Panel A) + Performance vs proxy robustness score (Panel B).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    exp3_dir = Path(experiment3_dir)
    if not exp3_dir.is_absolute():
        exp3_dir = _REPO_ROOT / exp3_dir

    results_path = exp3_dir / "experiment3_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Experiment 3 results not found: {results_path}")

    df = pd.read_csv(results_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Proxy plane scatter, color=RD_max
    valid = df.dropna(subset=["te_hat", "orc_hat", "RD_max"])
    sc = None
    if not valid.empty:
        for g, label in [("G1", "Proxy-selected"), ("G2", "Uniform")]:
            sub = valid[valid["group"] == g]
            if not sub.empty:
                sc = ax1.scatter(sub["te_hat"], sub["orc_hat"], c=sub["RD_max"], s=30, alpha=0.7, label=label)
        if sc is not None:
            plt.colorbar(sc, ax=ax1, label="RD_max")
    ax1.set_xlabel(r"$\hat{TE}$")
    ax1.set_ylabel(r"$\widehat{|ORC|}$")
    ax1.set_title("Proxy plane (color=RD_max)")
    ax1.legend()

    # Panel B: Performance vs scalar proxy robustness score (te_hat + |orc_hat|)
    df["proxy_robustness_score"] = df["te_hat"].fillna(0) + df["orc_hat"].fillna(0).abs()
    valid_b = df.dropna(subset=["clean_roc_auc", "proxy_robustness_score"])
    for g, label in [("G1", "Proxy-selected"), ("G2", "Uniform")]:
        sub = valid_b[valid_b["group"] == g]
        if not sub.empty:
            ax2.scatter(
                sub["clean_roc_auc"], sub["proxy_robustness_score"], alpha=0.5, label=label
            )
    ax2.set_xlabel("Performance (Clean ROC-AUC)")
    ax2.set_ylabel(r"Proxy robustness score ($\hat{TE} + |\widehat{ORC}|$)")
    ax2.set_title("Performance vs proxy robustness score")
    ax2.legend()

    plt.tight_layout()
    out = output_path or exp3_dir / "plot3_proxy_plane.pdf"
    out = Path(out)
    if out.suffix != ".pdf":
        out = out.with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, format=FIG_FORMAT, dpi=FIG_DPI)
    plt.close()
    return out


def _bootstrap_ci_mean(values: List[float], n_boot: int = 1000) -> Tuple[float, float, float]:
    """Bootstrap 95% CI for mean. Returns (mean, ci_lo, ci_hi)."""
    arr = np.array([x for x in values if np.isfinite(x)])
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return mean, mean, mean
    rng = np.random.default_rng(42)
    boot_means = [
        float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)
    ]
    return mean, float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def plot4(
    experiment3_dir: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Plot 4: Bar chart comparing mean Performance (Clean ROC-AUC) and mean RD_max
    across all architecture groups (G1–G5), with 95% bootstrap CI error bars.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting")

    exp3_dir = Path(experiment3_dir)
    if not exp3_dir.is_absolute():
        exp3_dir = _REPO_ROOT / exp3_dir

    results_path = exp3_dir / "experiment3_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Experiment 3 results not found: {results_path}")

    df = pd.read_csv(results_path)
    group_order = ["G1", "G2", "G3", "G4", "G5"]
    group_labels = {
        "G1": "Proxy-selected",
        "G2": "Uniform",
        "G3": "Dense CfC",
        "G4": "Random sparse",
        "G5": "NCP",
    }
    labels = [group_labels.get(g, g) for g in group_order]

    perf_means, perf_lo, perf_hi = [], [], []
    rd_means, rd_lo, rd_hi = [], [], []
    for g in group_order:
        sub = df[df["group"] == g]
        if sub.empty:
            perf_means.append(np.nan)
            perf_lo.append(np.nan)
            perf_hi.append(np.nan)
            rd_means.append(np.nan)
            rd_lo.append(np.nan)
            rd_hi.append(np.nan)
            continue
        p_vals = sub["clean_roc_auc"].dropna().tolist()
        r_vals = sub["RD_max"].dropna().tolist()
        pm, pl, ph = _bootstrap_ci_mean(p_vals)
        rm, rl, rh = _bootstrap_ci_mean(r_vals)
        perf_means.append(pm)
        perf_lo.append(pl)
        perf_hi.append(ph)
        rd_means.append(rm)
        rd_lo.append(rl)
        rd_hi.append(rh)

    x = np.arange(len(group_order))
    width = 0.6
    perf_means_arr = np.array(perf_means)
    rd_means_arr = np.array(rd_means)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    perf_err_lo = np.nan_to_num(perf_means_arr - np.array(perf_lo), nan=0)
    perf_err_hi = np.nan_to_num(np.array(perf_hi) - perf_means_arr, nan=0)
    ax1.bar(x, np.nan_to_num(perf_means_arr, nan=0), width, color="steelblue")
    ax1.errorbar(
        x, np.nan_to_num(perf_means_arr, nan=0),
        yerr=[perf_err_lo, perf_err_hi], fmt="none", color="black", capsize=4,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("Performance (Clean ROC-AUC)")
    ax1.set_title("Mean performance by architecture")
    ax1.set_ylim(0, 1.05)

    rd_err_lo = np.nan_to_num(rd_means_arr - np.array(rd_lo), nan=0)
    rd_err_hi = np.nan_to_num(np.array(rd_hi) - rd_means_arr, nan=0)
    ax2.bar(x, np.nan_to_num(rd_means_arr, nan=0), width, color="coral")
    ax2.errorbar(
        x, np.nan_to_num(rd_means_arr, nan=0),
        yerr=[rd_err_lo, rd_err_hi], fmt="none", color="black", capsize=4,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("RD_max")
    ax2.set_title("Mean robustness (RD_max) by architecture")
    ax2.set_ylim(0, None)

    plt.tight_layout()
    out = output_path or exp3_dir / "plot4_architecture_comparison.pdf"
    out = Path(out)
    if out.suffix != ".pdf":
        out = out.with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, format=FIG_FORMAT, dpi=FIG_DPI)
    plt.close()
    return out
