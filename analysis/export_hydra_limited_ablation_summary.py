#!/usr/bin/env python3
"""
Export a single plain-text summary of HYDRA limited ablation metrics for the paper.

Reads ``evaluation/results/unified_all_results.csv``, applies the same filters as
``analyze_limited_ablations.load_limited_ablation_results`` (parameterized by
dataset and eval protocol), runs ``prepare_subject_level_data``, and writes
``analysis/hydra_limited_ablation_paper_summary.txt``.

Usage (from project root, conda env ncp_robustness_proj):

    python analysis/export_hydra_limited_ablation_summary.py

Optional:

    python analysis/export_hydra_limited_ablation_summary.py --unified path/to/unified_all_results.csv
    python analysis/export_hydra_limited_ablation_summary.py --output path/to/out.txt
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import sys

_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from analysis.analyze_limited_ablations import (  # noqa: E402
    ABLATION_NAMES,
    MODEL_TO_ABLATION,
    AnalysisConfig,
    compute_subject_level_statistics,
    load_limited_ablation_results,
    prepare_subject_level_data,
)
from analysis.calculate_clean_scores import canonicalize_columns  # noqa: E402

UNIFIED_REL = Path("evaluation") / "results" / "unified_all_results.csv"

# Paper-facing names for the main table (main body = four rows)
PAPER_MAIN_ROW: Dict[str, str] = {
    "baseline": "Full HYDRA",
    "ablation1_no_carry_gate": "No Residual",
    "ablation2_no_branching": "No MTB",
    "ablation3_lstm_replacement": "No CfC",
}

DATASET_FAMILY: Dict[str, str] = {
    "BNCI2014_001": "MI",
    "Lee2019_MI": "MI",
    "Lee2019_SSVEP": "SSVEP",
    "BI2015a": "ERP",
}


def _fmt(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "n/a"
    if not np.isfinite(std) or std == 0.0:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def _mean_std(series: pd.Series) -> Tuple[float, float]:
    v = pd.to_numeric(series, errors="coerce").dropna()
    if v.empty:
        return float("nan"), float("nan")
    if len(v) == 1:
        return float(v.iloc[0]), 0.0
    return float(v.mean()), float(v.std(ddof=1))


def eval_mode_to_substr(eval_display: str) -> str:
    s = str(eval_display).strip()
    if "CrossSubject" in s:
        return "CrossSubject"
    if "CrossSession" in s:
        return "CrossSession"
    if "WithinSession" in s or "Within" in s:
        return "WithinSession"
    return s


def inventory_ablation_presence(unified_path: Path) -> pd.DataFrame:
    """
    Rows of (dataset, eval_mode_display, n_rows) for limited ablation models,
    tune=False, test_perturb, seeds 100–500.
    """
    df = pd.read_csv(unified_path, low_memory=False)
    df = canonicalize_columns(df)

    if "model" not in df.columns:
        raise KeyError("Expected 'model' in unified results")

    normalized_to_canonical: Dict[str, str] = {}
    for canonical in MODEL_TO_ABLATION:
        normalized_to_canonical[canonical.lower().replace("-", "_")] = canonical

    model_norm = df["model"].astype(str).str.strip().str.lower().str.replace("-", "_")
    df = df[model_norm.map(normalized_to_canonical).notna()].copy()

    if "tune" in df.columns:
        df = df[df["tune"] == False].copy()
    elif "mode" in df.columns:
        mode_norm = df["mode"].astype(str).str.strip()
        df = df[~mode_norm.str.contains("_tune", na=False)].copy()

    if "mode" in df.columns:
        mode_norm = df["mode"].astype(str).str.replace("_tune", "", regex=False).str.strip()
        df = df[mode_norm == "test_perturb"].copy()

    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
        df = df[df["seed"].isin([100, 200, 300, 400, 500])].copy()

    if df.empty:
        return pd.DataFrame(columns=["dataset", "eval_mode_display", "n_rows"])

    df["eval_mode_display"] = (
        df["eval_mode"].astype(str).str.replace("Evaluation", "", regex=False).str.strip()
    )
    g = df.groupby(["dataset", "eval_mode_display"], dropna=False).size().reset_index(name="n_rows")
    g = g[g["n_rows"] > 0].sort_values(["dataset", "eval_mode_display"])
    return g.reset_index(drop=True)


def attach_clean_to_collapsed(raw_df: pd.DataFrame, df_collapsed: pd.DataFrame) -> pd.DataFrame:
    if df_collapsed.empty:
        return df_collapsed
    out = df_collapsed.copy()
    if "clean_roc_auc" in raw_df.columns:
        clean_map = raw_df.groupby(["ablation", "subject"])["clean_roc_auc"].mean().reset_index()
        if "clean_roc_auc" in out.columns:
            out = out.drop(columns=["clean_roc_auc"])
        out = out.merge(clean_map, on=["ablation", "subject"], how="left")
    return out


def summarize_ablation_slice(
    raw_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Returns (df_points, df_collapsed, df_resolved, stats_df)."""
    config = AnalysisConfig(normalize_aupc=True, rd_summary="mean")
    df_points, df_collapsed, df_resolved = prepare_subject_level_data(raw_df, config)
    df_collapsed = attach_clean_to_collapsed(raw_df, df_collapsed)

    stats_df = pd.DataFrame()
    if not df_collapsed.empty:
        stats_df = compute_subject_level_statistics(
            df_collapsed,
            primary_metrics=["aupc_collapsed", "rd_collapsed", "rd_worst_collapsed"],
            secondary_metrics=["clean_roc_auc"],
        )
    return df_points, df_collapsed, df_resolved, stats_df


def collapsed_to_summary_table(df_collapsed: pd.DataFrame) -> pd.DataFrame:
    """One row per ablation: mean ± std across subjects."""
    if df_collapsed.empty:
        return pd.DataFrame()

    rows: List[dict] = []
    for ablation in sorted(df_collapsed["ablation"].unique()):
        g = df_collapsed[df_collapsed["ablation"] == ablation].copy()
        if "subject" in g.columns:
            g = g.drop_duplicates(subset=["subject"])

        m_clean, s_clean = _mean_std(g["clean_roc_auc"]) if "clean_roc_auc" in g.columns else (np.nan, np.nan)
        m_aupc, s_aupc = _mean_std(g["aupc_collapsed"]) if "aupc_collapsed" in g.columns else (np.nan, np.nan)
        m_rd, s_rd = _mean_std(g["rd_collapsed"]) if "rd_collapsed" in g.columns else (np.nan, np.nan)
        m_rw, s_rw = (
            _mean_std(g["rd_worst_collapsed"]) if "rd_worst_collapsed" in g.columns else (np.nan, np.nan)
        )

        rows.append(
            {
                "ablation": ablation,
                "ablation_name": ABLATION_NAMES.get(ablation, ablation),
                "n_subjects": int(g["subject"].nunique()) if "subject" in g.columns else len(g),
                "clean_roc_auc": _fmt(m_clean, s_clean),
                "aupc_collapsed": _fmt(m_aupc, s_aupc),
                "rd_collapsed": _fmt(m_rd, s_rd),
                "rd_worst_collapsed": _fmt(m_rw, s_rw),
            }
        )
    return pd.DataFrame(rows)


def build_report_text(
    unified_path: Path,
    inventory: pd.DataFrame,
    sections: List[Tuple[str, str, pd.DataFrame, Optional[pd.DataFrame]]],
    main_table_dataset: Optional[str],
    main_table_eval: Optional[str],
    main_summary_tbl: Optional[pd.DataFrame],
) -> str:
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("HYDRA LIMITED ABLATION STUDIES - PAPER SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Unified results source: {unified_path.resolve()}")
    lines.append("")
    lines.append("Metrics (aligned with benchmark / analyze_limited_ablations):")
    lines.append(
        "  Clean ROC-AUC: unperturbed test ROC-AUC; summarized as mean ± std across subjects."
    )
    lines.append(
        "  AUPC: area under the perturbation curve (ROC-AUC vs intensity), noise types collapsed,"
    )
    lines.append("        normalized (see AnalysisConfig.normalize_aupc in statistical_analysis).")
    lines.append(
        "  RD: rd_collapsed = mean relative drop ( (clean - corrupted)/clean ) across noise types."
    )
    lines.append(
        "  RD_worst: max relative drop over intensities (and noise types in worst collapsed)."
    )
    lines.append("")

    lines.append("-" * 80)
    lines.append("INVENTORY: (dataset × eval protocol) rows in unified file for HYDRA ablation models")
    lines.append("-" * 80)
    if inventory.empty:
        lines.append("  (no rows matched filters - check unified file and model names)")
    else:
        lines.append(inventory.to_string(index=False))
    lines.append("")

    for title, subtitle, summary_tbl, stats_df in sections:
        lines.append("-" * 80)
        lines.append(title)
        if subtitle:
            lines.append(subtitle)
        lines.append("-" * 80)
        if summary_tbl is not None and not summary_tbl.empty:
            lines.append(summary_tbl.to_string(index=False))
            if len(summary_tbl) < 5:
                if len(summary_tbl) == 1:
                    lines.append(
                        "  NOTE: Only Full HYDRA (baseline) appears in aggregated metrics for this slice. "
                        "Ablation-variant runs are likely absent from unified_all_results.csv for this "
                        "dataset/protocol."
                    )
                else:
                    lines.append(
                        "  NOTE: Fewer than five ablation variants in this slice (some variants missing "
                        "from aggregated metrics / unified results for this protocol)."
                    )
        else:
            lines.append("  (no data for this slice)")
        lines.append("")
        if stats_df is not None and not stats_df.empty:
            lines.append("  Paired baseline vs ablation tests (subject-level, abbreviated):")
            sub = stats_df[
                ["ablation_name", "metric", "metric_type", "mean_difference", "p_value", "cohens_dz"]
            ].copy()
            lines.append(sub.to_string(index=False))
            lines.append("")

    # Main paper table (4 rows)
    lines.append("-" * 80)
    lines.append("MAIN PAPER - COMPACT TABLE (four variants: Full HYDRA, No CfC, No MTB, No Residual)")
    lines.append(
        f"Context: dataset={main_table_dataset or 'n/a'}, eval={main_table_eval or 'n/a'}"
    )
    lines.append("-" * 80)

    tbl_for_main = main_summary_tbl
    if tbl_for_main is None or tbl_for_main.empty:
        for _, _, summary_tbl, _ in sections:
            if summary_tbl is not None and not summary_tbl.empty:
                tbl_for_main = summary_tbl
                break

    main_rows = []
    if tbl_for_main is not None and not tbl_for_main.empty:
        for ablation_key, label in PAPER_MAIN_ROW.items():
            hit = tbl_for_main[tbl_for_main["ablation"] == ablation_key]
            if not hit.empty:
                main_rows.append((label, hit.iloc[0]))
    if main_rows:
        lines.append(
            f"{'Model Variant':<16} | {'Clean AUC':<22} | {'AUPC':<22} | {'RD':<22}"
        )
        lines.append("-" * 90)
        for label, row in main_rows:
            lines.append(
                f"{label:<16} | {row['clean_roc_auc']:<22} | {row['aupc_collapsed']:<22} | {row['rd_collapsed']:<22}"
            )
    else:
        lines.append("  (could not build main table - missing summary rows for baseline + ablations 1-3)")
    lines.append("")

    lines.append("-" * 80)
    lines.append("APPENDIX - Ablation 4 (No SNR gate) and extra detail")
    lines.append("-" * 80)
    appendix_rows = []
    for _, _, summary_tbl, _ in sections:
        if summary_tbl is None or summary_tbl.empty:
            continue
        hit = summary_tbl[summary_tbl["ablation"] == "ablation4_no_snr_gate"]
        if not hit.empty:
            appendix_rows.append(hit.iloc[0])
            break
    if appendix_rows:
        r = appendix_rows[0]
        lines.append(f"  No SNR gate: Clean AUC {r['clean_roc_auc']}, AUPC {r['aupc_collapsed']}, ")
        lines.append(f"    RD {r['rd_collapsed']}, RD_worst {r['rd_worst_collapsed']}, n_subjects={r['n_subjects']}")
    else:
        lines.append("  (ablation 4 not present in any computed slice)")
    lines.append("")
    lines.append(
        "Full per-ablation CSVs and noise-resolved tables can be regenerated with "
        "`python analysis/analyze_limited_ablations.py`."
    )
    lines.append("=" * 80)
    return "\n".join(lines)


def run_export(unified_file: Optional[Path], output_file: Path) -> int:
    unified_path = Path(unified_file) if unified_file else _project_root / UNIFIED_REL
    if not unified_path.exists():
        txt = "\n".join(
            [
                "ERROR: unified results file not found:",
                str(unified_path.resolve()),
                "",
                "Run experiments or point to unified_all_results.csv with --unified.",
            ]
        )
        output_file.write_text(txt, encoding="utf-8")
        print(txt)
        return 1

    inventory = inventory_ablation_presence(unified_path)
    if not inventory.empty:
        _prio = {"BNCI2014_001": 0, "Lee2019_MI": 1, "Lee2019_SSVEP": 2, "BI2015a": 3}
        inventory = inventory.assign(
            _sort=inventory["dataset"].map(lambda d: _prio.get(str(d), 99))
        ).sort_values(["_sort", "dataset", "eval_mode_display"]).drop(columns="_sort")
    sections: List[Tuple[str, str, pd.DataFrame, Optional[pd.DataFrame]]] = []

    main_table_dataset: Optional[str] = None
    main_table_eval: Optional[str] = None
    main_summary_tbl: Optional[pd.DataFrame] = None

    if not inventory.empty:
        for _, inv_row in inventory.iterrows():
            ds = inv_row["dataset"]
            ev_display = inv_row["eval_mode_display"]
            substr = eval_mode_to_substr(ev_display)
            title = f"{DATASET_FAMILY.get(ds, '?')} - {ds}"
            subtitle = f"eval_mode (display): {ev_display} | filter substring: {substr}"

            try:
                raw = load_limited_ablation_results(
                    unified_path,
                    datasets=[str(ds)],
                    eval_mode_substr=substr,
                )
            except Exception as e:
                sections.append(
                    (title, f"{subtitle}\n(load failed: {e})", pd.DataFrame(), None)
                )
                continue

            if raw.empty:
                sections.append((title, subtitle + "\n(empty after load)", pd.DataFrame(), None))
                continue

            _, df_collapsed, _, stats_df = summarize_ablation_slice(raw)
            summary_tbl = collapsed_to_summary_table(df_collapsed)
            sections.append((title, subtitle, summary_tbl, stats_df))

            if (
                str(ds) == "BNCI2014_001"
                and "CrossSession" in str(ev_display)
                and main_summary_tbl is None
            ):
                main_table_dataset = str(ds)
                main_table_eval = str(ev_display)
                main_summary_tbl = summary_tbl

    if main_summary_tbl is None:
        for title, subtitle, summary_tbl, _ in sections:
            if summary_tbl is not None and not summary_tbl.empty:
                main_summary_tbl = summary_tbl
                if main_table_dataset is None:
                    main_table_dataset = title.split(" - ")[-1].strip() if " - " in title else title
                if main_table_eval is None:
                    main_table_eval = subtitle.split("\n")[0] if subtitle else None
                break

    report = build_report_text(
        unified_path,
        inventory,
        sections,
        main_table_dataset,
        main_table_eval,
        main_summary_tbl,
    )
    # Avoid Unicode minus / symbols breaking Windows cp1252 consoles
    report_ascii = (
        report.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", " - ")
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report_ascii, encoding="utf-8")
    try:
        print(report_ascii)
    except UnicodeEncodeError:
        print(report_ascii.encode("ascii", errors="replace").decode("ascii"))
    print(f"\n[OK] Wrote {output_file.resolve()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export HYDRA limited ablation paper summary text.")
    parser.add_argument(
        "--unified",
        type=Path,
        default=None,
        help="Path to unified_all_results.csv (default: evaluation/results/unified_all_results.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_current_dir / "hydra_limited_ablation_paper_summary.txt",
        help="Output .txt path",
    )
    args = parser.parse_args()
    return run_export(args.unified, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
