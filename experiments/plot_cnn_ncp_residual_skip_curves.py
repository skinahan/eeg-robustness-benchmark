#!/usr/bin/env python3
"""
Plot ROC-AUC vs perturbation intensity for CNN-NCP vs CNN-NCP + residual skip.

Reads aggregated CSV from cnn_ncp_residual_skip_experiment.py by default; if the file
contains per-fold rows (fold_idx), aggregates with mean/std before plotting.

Multi-panel mode also writes a zoomed EOG figure (default x-axis 0–20, matching
the experiment EOG grid) unless --no_eog_zoom is set.

Intensity 0: if already present (clean baseline), curves use it as-is; otherwise the
same convention as analysis/analyze_results.py prepends a (0, clean_roc) point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_AGG_CSV = _REPO_ROOT / "experiments" / "outputs" / "cnn_ncp_residual_skip_perturb_aggregated.csv"
DEFAULT_OUT = _REPO_ROOT / "experiments" / "outputs" / "cnn_ncp_residual_skip_roc_vs_intensity.png"
DEFAULT_EOG_ZOOM_OUT = (
    _REPO_ROOT / "experiments" / "outputs" / "cnn_ncp_residual_skip_eog_zoom_0_20.png"
)
DEFAULT_EOG_ZOOM_XMAX = 20.0

MODEL_ORDER = ["cnn_ncp", "cnn_ncp_residual_skip"]
MODEL_LABELS = {
    "cnn_ncp": "CNN-NCP (base)",
    "cnn_ncp_residual_skip": "CNN-NCP + residual skip",
}

NOISE_PANEL_ORDER = ["gaussian", "dropout", "eog", "ar1_drift"]
NOISE_TITLES = {
    "gaussian": "Gaussian",
    "dropout": "Channel dropout",
    "eog": "EOG",
    "ar1_drift": "AR(1) drift",
}


def prepare_aggregated_df(df: pd.DataFrame) -> pd.DataFrame:
    """If input is per-fold long-form, aggregate to mean/std."""
    if "fold_idx" not in df.columns:
        return df.copy()
    gcols = ["model", "noise_type", "intensity"]
    out = (
        df.groupby(gcols, as_index=False)
        .agg(
            clean_roc_auc_mean=("clean_roc_auc", "mean"),
            clean_roc_auc_std=("clean_roc_auc", "std"),
            corrupted_roc_auc_mean=("corrupted_roc_auc", "mean"),
            corrupted_roc_auc_std=("corrupted_roc_auc", "std"),
            n_folds=("fold_idx", "count"),
        )
    )
    for c in ("clean_roc_auc_std", "corrupted_roc_auc_std"):
        out[c] = out[c].fillna(0.0)
    out["clean_roc_auc"] = out["clean_roc_auc_mean"]
    out["corrupted_roc_auc"] = out["corrupted_roc_auc_mean"]
    return out


def _curve_with_clean_at_intensity_zero(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a proper clean anchor at intensity 0 (analyze_results.py convention).
    If intensity 0 is already present, return sorted sub unchanged.
    """
    y_col = "corrupted_roc_auc"
    clean_col = "clean_roc_auc"
    if sub.empty:
        return sub
    sub = sub.sort_values("intensity").copy()
    has_zero = np.isclose(sub["intensity"].values, 0.0).any()
    if has_zero:
        return sub
    clean = float(sub[clean_col].dropna().iloc[0])
    rest = sub.loc[sub["intensity"] > 1e-12].copy()
    if rest.empty:
        return pd.DataFrame([{"intensity": 0.0, y_col: clean, clean_col: clean}])
    row0 = rest.iloc[0:1].copy()
    row0["intensity"] = 0.0
    row0[y_col] = clean
    if "corrupted_roc_auc_std" in row0.columns:
        row0["corrupted_roc_auc_std"] = sub["corrupted_roc_auc_std"].iloc[0]
    return pd.concat([row0, rest], ignore_index=True)


def _plot_one_model_on_ax(
    ax,
    curve: pd.DataFrame,
    name: str,
    colors: dict,
    use_std_band: bool,
) -> None:
    label = MODEL_LABELS.get(name, name)
    c = colors.get(name, None)
    x = curve["intensity"].values
    y = curve["corrupted_roc_auc"].values
    ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.4, label=label, color=c)
    if use_std_band and "corrupted_roc_auc_std" in curve.columns:
        std = curve["corrupted_roc_auc_std"].fillna(0.0).values
        ax.fill_between(x, np.clip(y - std, 0, 1), np.clip(y + std, 0, 1), alpha=0.2, color=c)


def plot_eog_zoom(
    df: pd.DataFrame,
    out_path: Path,
    xmax: float = DEFAULT_EOG_ZOOM_XMAX,
    xmin: float = 0.0,
    title: str | None = None,
) -> bool:
    """
    Single-panel EOG plot with x-axis limited to [xmin, xmax] (default 0–20).
    Points with intensity > xmax are omitted so the degradation is visible at low severity.

    Returns True if a figure was written, False if no EOG rows in df.
    """
    import matplotlib.pyplot as plt

    df = prepare_aggregated_df(df)
    sub_nt = df[df["noise_type"].astype(str) == "eog"]
    if sub_nt.empty:
        return False

    use_std = "corrupted_roc_auc_std" in df.columns
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=120)
    colors = {"cnn_ncp": "#1f77b4", "cnn_ncp_residual_skip": "#d62728"}

    for name in MODEL_ORDER:
        sub = sub_nt[sub_nt["model"].astype(str) == name]
        if sub.empty:
            continue
        curve = _curve_with_clean_at_intensity_zero(sub)
        curve = curve.loc[curve["intensity"] <= float(xmax) + 1e-9].copy()
        if curve.empty:
            continue
        _plot_one_model_on_ax(ax, curve, name, colors, use_std_band=use_std)

    ax.set_xlim(float(xmin), float(xmax))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("EOG intensity (0 = clean)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(
        title
        or f"EOG: ROC-AUC vs intensity (zoom {xmin:.0f}–{xmax:.0f})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_all_perturbation_types(
    df: pd.DataFrame,
    out_path: Path,
    title: str | None = None,
    eog_zoom_out: Path | None = None,
    eog_zoom_xmax: float = DEFAULT_EOG_ZOOM_XMAX,
) -> bool:
    """
    Returns True if an EOG zoom figure was written, False otherwise.
    """
    import matplotlib.pyplot as plt

    df = prepare_aggregated_df(df)
    use_std = "corrupted_roc_auc_std" in df.columns

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=120)
    axes_flat = axes.flatten()

    colors = {"cnn_ncp": "#1f77b4", "cnn_ncp_residual_skip": "#d62728"}

    for ax, nt in zip(axes_flat, NOISE_PANEL_ORDER):
        sub_nt = df[df["noise_type"].astype(str) == nt]
        if sub_nt.empty:
            ax.set_visible(False)
            continue
        for name in MODEL_ORDER:
            sub = sub_nt[sub_nt["model"].astype(str) == name]
            if sub.empty:
                continue
            curve = _curve_with_clean_at_intensity_zero(sub)
            _plot_one_model_on_ax(ax, curve, name, colors, use_std_band=use_std)

        ax.set_title(NOISE_TITLES.get(nt, nt))
        ax.set_xlabel("Intensity (units vary by perturbation)")
        ax.set_ylabel("ROC-AUC (0 = clean)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        title or "ROC-AUC vs intensity — cross-session mean ± std over folds",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    zoom_saved = False
    if eog_zoom_out is not None:
        zoom_title = None if title is None else f"{title} — EOG (0–{eog_zoom_xmax:.0f})"
        zoom_saved = plot_eog_zoom(
            df, Path(eog_zoom_out), xmax=eog_zoom_xmax, title=zoom_title
        )
    return zoom_saved


def plot_roc_vs_intensity(
    df: pd.DataFrame,
    out_path: Path,
    title: str | None = None,
    noise_type: str = "gaussian",
) -> None:
    """Single-panel plot (backward compatible)."""
    import matplotlib.pyplot as plt

    df = prepare_aggregated_df(df)
    if "noise_type" in df.columns:
        df = df[df["noise_type"].astype(str) == noise_type].copy()
    # else: legacy single-type CSV without noise_type column

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=120)
    colors = {"cnn_ncp": "#1f77b4", "cnn_ncp_residual_skip": "#d62728"}
    use_std = "corrupted_roc_auc_std" in df.columns

    for name in MODEL_ORDER:
        sub = df[df["model"].astype(str) == name]
        if sub.empty:
            continue
        curve = _curve_with_clean_at_intensity_zero(sub)
        _plot_one_model_on_ax(ax, curve, name, colors, use_std_band=use_std)

    ax.set_xlabel("Intensity")
    ax.set_ylabel("ROC-AUC (intensity 0 = clean)")
    ax.set_title(title or f"ROC-AUC vs {noise_type} — BNCI2014_001")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ROC-AUC vs intensity")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_AGG_CSV))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        metavar="NOISE_TYPE",
        help="If set, single panel for this noise type only (e.g. gaussian)",
    )
    parser.add_argument(
        "--eog_zoom_out",
        type=str,
        default=str(DEFAULT_EOG_ZOOM_OUT),
        help="Path for zoomed EOG plot (intensity 0–xmax); empty string disables",
    )
    parser.add_argument(
        "--eog_zoom_xmax",
        type=float,
        default=DEFAULT_EOG_ZOOM_XMAX,
        help="Upper x limit for EOG zoom plot (default: 20)",
    )
    parser.add_argument(
        "--no_eog_zoom",
        action="store_true",
        help="When plotting all panels, do not write the EOG zoom figure",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"model", "intensity", "corrupted_roc_auc", "clean_roc_auc"}
    if not required.issubset(df.columns):
        raise SystemExit(f"CSV missing columns {required}; got {list(df.columns)}")
    if "noise_type" not in df.columns and args.single is None:
        raise SystemExit("CSV has no noise_type; use --single gaussian for legacy CSV")

    if args.single:
        plot_roc_vs_intensity(df, Path(args.out), title=args.title, noise_type=args.single)
        print(f"Saved: {args.out}")
        if (
            args.single == "eog"
            and str(args.eog_zoom_out).strip()
            and not args.no_eog_zoom
        ):
            ok = plot_eog_zoom(
                df,
                Path(args.eog_zoom_out),
                xmax=args.eog_zoom_xmax,
                title=args.title,
            )
            if ok:
                print(f"Saved EOG zoom: {args.eog_zoom_out}")
    else:
        if "noise_type" not in df.columns:
            raise SystemExit("Need noise_type column for multi-panel plot")
        eog_zoom_path = None
        if not args.no_eog_zoom and str(args.eog_zoom_out).strip():
            eog_zoom_path = Path(args.eog_zoom_out)
        zoom_ok = plot_all_perturbation_types(
            df,
            Path(args.out),
            title=args.title,
            eog_zoom_out=eog_zoom_path,
            eog_zoom_xmax=args.eog_zoom_xmax,
        )
        print(f"Saved: {args.out}")
        if zoom_ok:
            print(f"Saved EOG zoom: {args.eog_zoom_out}")


if __name__ == "__main__":
    main()
