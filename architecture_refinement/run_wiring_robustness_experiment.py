#!/usr/bin/env python3
"""
Wiring Graph Robustness Comparison Experiment

Requires: conda environment with moabb, braindecode, etc.
  Run: activate_env.bat (Windows) or conda activate ncp_robustness_proj

Compares robustness under Gaussian perturbation of:
- First 5 wiring graphs from outputs/architectures/
- Basic NCP with 32 units (baseline)

Uses CNNWiredCfCMin as fixed base class, CrossSession evaluation on BNCI2014_001.
Intensity grid: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100].
Metrics: AUPC, RD_mean, RD_max, CSV_p.

Stage 1: Subject 1, seed 42 only (viability test).
Stage 2: All subjects, 5 seeds (if Stage 1 passes).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Constants
DATASET = "BNCI2014_001"
EVAL_MODE = "CrossSession"
INTENSITY_GRID = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
STAGE1_SEED = 42
STAGE2_SEEDS = [42, 100, 200, 300, 400]
BNCI_SUBJECTS = list(range(1, 10))  # 9 subjects
NCP_UNITS = 32
F2 = 16  # ncp I/O size for CNNWiredCfCMin (F1*D = 8*2)
VIABILITY_AUPC_THRESHOLD = 0.02
VIABILITY_RD_THRESHOLD = 0.05
VIABILITY_CSV_THRESHOLD = 0.001


def _now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_architectures_dir() -> Path:
    """Resolve path to outputs/architectures (first five wiring graphs)."""
    for base in [_REPO_ROOT, _THIS_DIR]:
        cand = base / "outputs" / "architectures"
        if cand.exists():
            return cand.resolve()
        cand = base / "architecture_refinement" / "outputs" / "architectures"
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(
        "Could not find outputs/architectures. "
        "Expected outputs/architectures or architecture_refinement/outputs/architectures"
    )


def load_wiring_from_json(arch_path: Path) -> object:
    """
    Load wiring from architecture JSON, compatible with CNNWiredCfCMin.
    Supports: (1) load_architecture_from_file format, (2) ws_flex format.
    """
    with open(arch_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ws_flex format (nas_pilot / plot2)
    if "hidden_adj_undirected" in data and "wiring_seed" in data:
        from architecture_refinement.arbitrary_wiring import WsFlexHiddenWiring
        import networkx as nx

        hidden_adj = np.asarray(data["hidden_adj_undirected"], dtype=np.int8)
        G = nx.from_numpy_array((hidden_adj != 0).astype(np.int8))
        if not nx.is_connected(G):
            raise ValueError(f"Hidden graph is disconnected: {arch_path}")
        return WsFlexHiddenWiring(
            input_size=1,
            hidden_graph=G,
            output_size=1,
            input_strategy="degree_proportional",
            output_strategy="uniform",
            hidden_edge_orientation="random_oriented",
            add_hidden_self_loops=True,
            seed=int(data["wiring_seed"]),
        )

    # load_architecture_from_file format
    from architecture_refinement.arbitrary_wiring import load_architecture_from_file
    return load_architecture_from_file(str(arch_path))


def prepare_pilot_dir(
    arch_dir: Path,
    output_dir: Path,
    ncp_units: int = NCP_UNITS,
    f2: int = F2,
) -> Path:
    """
    Create pilot directory with selected_architectures for the 5 wiring graphs + NCP baseline.
    Returns path to pilot dir.
    """
    pilot_dir = output_dir / "pilot_architectures"
    selected_dir = pilot_dir / "selected_architectures"
    selected_dir.mkdir(parents=True, exist_ok=True)

    # First 5 best_architecture_*.json, sorted by index
    arch_files = sorted(arch_dir.glob("best_architecture_*.json"))
    if len(arch_files) < 5:
        raise FileNotFoundError(
            f"Need at least 5 architecture files in {arch_dir}, found {len(arch_files)}"
        )
    arch_files = arch_files[:5]

    for i, p in enumerate(arch_files):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert to ws_flex format if needed
        if "hidden_adj_undirected" in data:
            arch_out = dict(data)
        elif "wiring_matrix" in data and "hidden_size" in data:
            wm = np.asarray(data["wiring_matrix"])
            h = int(data["hidden_size"])
            if wm.shape == (h, h):
                hidden_adj = (wm != 0).astype(np.int8).tolist()
            else:
                # Full matrix: extract hidden block
                i_sz = int(data.get("input_size", 0))
                o_sz = int(data.get("output_size", 0))
                start = i_sz
                end = i_sz + h
                hidden_block = wm[start:end, start:end]
                hidden_adj = (hidden_block != 0).astype(np.int8).tolist()
            arch_out = {
                "model_name": f"wiring_graph_{i + 1}",
                "wiring_kind": "ws_flex",
                "hidden_adj_undirected": hidden_adj,
                "wiring_seed": int(data.get("wiring_seed", 42)),
            }
        else:
            raise ValueError(f"Unknown architecture format in {p}: keys={list(data.keys())}")

        arch_out["model_name"] = f"wiring_graph_{i + 1}"
        arch_out["wiring_kind"] = arch_out.get("wiring_kind", "ws_flex")
        arch_out["wiring_seed"] = int(arch_out.get("wiring_seed", 42))

        out_path = selected_dir / f"wiring_graph_{i + 1}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(arch_out, f, indent=2)

    # NCP baseline
    ncp_arch = {
        "model_name": "ncp_baseline_32",
        "wiring_kind": "ncp_autoncp",
        "units": ncp_units,
        "output_size": f2,
        "sparsity_level": 0.5,
        "wiring_seed": 202603,
    }
    with open(selected_dir / "ncp_baseline_32.json", "w", encoding="utf-8") as f:
        json.dump(ncp_arch, f, indent=2)

    return pilot_dir


def register_models(pilot_dir: Path) -> List[str]:
    """Register models from pilot dir. Returns list of model names."""
    from architecture_refinement.nas_pilot_registry import register_nas_pilot_models
    # Clear any prior registration for our models to avoid conflicts
    import config as _cfg
    for name in list(_cfg._runtime_model_registry.keys()):
        if name.startswith("wiring_graph_") or name == "ncp_baseline_32":
            del _cfg._runtime_model_registry[name]
    return register_nas_pilot_models(pilot_dir)


def create_saturation_file_for_intensity_grid(output_dir: Path) -> str:
    """Create a saturation file so alpha_grid [0.1..1.0] yields intensities [10..100]."""
    sat_dir = output_dir / "saturation"
    sat_dir.mkdir(parents=True, exist_ok=True)
    sat_path = sat_dir / "saturation_points_summary.csv"
    # sigma_max=100 so alpha*100 = [10,20,...,100]
    df = pd.DataFrame([
        {"dataset": DATASET, "noise_type": "gaussian", "saturation_point": 100.0},
    ])
    df.to_csv(sat_path, index=False)
    return str(sat_path)


def run_experiment(
    model_name: str,
    subjects: List[int],
    seed: int,
    pilot_dir: Path,
    saturation_file: str,
    output_base: Path,
) -> Optional[pd.DataFrame]:
    """Run unified_experiment_runner for one model. Returns results DataFrame or None."""
    alpha_grid = "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"  # -> [0,10,20,...,100]
    sat_abs = str(Path(saturation_file).resolve())
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "evaluation" / "unified_experiment_runner.py"),
        "--nas_pilot_dir", str(pilot_dir),
        "--model", model_name,
        "--dataset", DATASET,
        "--subjects", *[str(s) for s in subjects],
        "--mode", "test_perturb",
        "--eval_mode", EVAL_MODE,
        "--seed", str(seed),
        "--disable_underfitting_retrain",
        "--test_perturb_gaussian_only",
        "--noise_perturbation_saturation_file", sat_abs,
        "--test_perturb_gaussian_alpha_grid", alpha_grid,
        "--overwrite",
    ]
    print(f"[RUN] {model_name} seed={seed} subjects={subjects}")
    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[ERROR] {model_name} failed: {proc.stderr[:500]}")
        return None

    # Collect results from results/ directory (CrossSession: results/.../seed/sub-XXX/session/test_perturb/)
    paradigm = "MotorImagery"
    results_dir = _REPO_ROOT / "results" / paradigm / DATASET
    from utils import short_run_id
    model_short = short_run_id(model_name)
    search_dirs = [
        results_dir / model_short / "CrossSessionEvaluation" / str(seed),
        results_dir / model_name / "CrossSessionEvaluation" / str(seed),
    ]
    dfs = []
    for d in search_dirs:
        if d.exists():
            for csv_path in d.rglob("*.csv"):
                if "test_perturb" in str(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        if "noise_type" in df.columns and ("corrupted_roc_auc" in df.columns or "corrupted_score" in df.columns):
                            df["model"] = model_name
                            df["seed"] = seed
                            if "corrupted_roc_auc" not in df.columns and "corrupted_score" in df.columns:
                                df["corrupted_roc_auc"] = df["corrupted_score"]
                            dfs.append(df)
                    except Exception as e:
                        print(f"[WARN] Could not read {csv_path}: {e}")
    # Also try non-tp_ filenames (legacy)
    if not dfs:
        for d in search_dirs:
            if d.exists():
                for csv_path in d.rglob("*.csv"):
                    if "test_perturb" in str(csv_path):
                        try:
                            df = pd.read_csv(csv_path)
                            if "noise_type" in df.columns and ("corrupted_roc_auc" in df.columns or "corrupted_score" in df.columns):
                                df["model"] = model_name
                                df["seed"] = seed
                                if "corrupted_roc_auc" not in df.columns and "corrupted_score" in df.columns:
                                    df["corrupted_roc_auc"] = df["corrupted_score"]
                                dfs.append(df)
                        except Exception as e:
                            print(f"[WARN] Could not read {csv_path}: {e}")
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def run_stage1(
    pilot_dir: Path,
    saturation_file: str,
    output_dir: Path,
) -> pd.DataFrame:
    """Stage 1: subject 1, seed 42 only."""
    models = [m for m in register_models(pilot_dir) if m.startswith("wiring_graph_") or m == "ncp_baseline_32"]
    all_dfs = []
    for model_name in models:
        df = run_experiment(
            model_name=model_name,
            subjects=[1],
            seed=STAGE1_SEED,
            pilot_dir=pilot_dir,
            saturation_file=saturation_file,
            output_base=output_dir,
        )
        if df is not None:
            all_dfs.append(df)
    if not all_dfs:
        raise RuntimeError("Stage 1 produced no results")
    combined = pd.concat(all_dfs, ignore_index=True)
    combined["subject"] = 1
    combined["dataset"] = DATASET
    combined["eval_mode"] = EVAL_MODE
    combined["tune"] = False
    return combined


def run_stage2(
    pilot_dir: Path,
    saturation_file: str,
    output_dir: Path,
) -> pd.DataFrame:
    """Stage 2: all subjects, 5 seeds."""
    models = [m for m in register_models(pilot_dir) if m.startswith("wiring_graph_") or m == "ncp_baseline_32"]
    all_dfs = []
    for model_name in models:
        for seed in STAGE2_SEEDS:
            df = run_experiment(
                model_name=model_name,
                subjects=BNCI_SUBJECTS,
                seed=seed,
                pilot_dir=pilot_dir,
                saturation_file=saturation_file,
                output_base=output_dir,
            )
            if df is not None:
                all_dfs.append(df)
    if not all_dfs:
        raise RuntimeError("Stage 2 produced no results")
    combined = pd.concat(all_dfs, ignore_index=True)
    if "subject" not in combined.columns and "session" in combined.columns:
        # Extract subject from session or metadata if needed
        combined["subject"] = combined.get("session", 1)  # placeholder
    combined["dataset"] = DATASET
    combined["eval_mode"] = EVAL_MODE
    combined["tune"] = False
    return combined


def check_viability(df: pd.DataFrame) -> bool:
    """
    Check if AUPC, RD_mean, RD_max, CSV show meaningful differences across models.
    Returns True if viability passes (differences exposed).
    """
    from analysis.robustness_metrics import (
        MetricConfig,
        add_normalized_p,
        compute_aupc,
        compute_rd_curve,
        compute_csv_p_curve,
        find_subject_col,
    )

    cfg = MetricConfig(metric_col="corrupted_roc_auc", intensity_col="intensity")
    df = add_normalized_p(df, cfg, normalize_within=["dataset", "noise_type"], clip=True)

    # AUPC per model
    aupc = compute_aupc(df, cfg, group_cols=["dataset", "model", "noise_type"])
    aupc_means = aupc.groupby("model")["aupc"].mean()
    aupc_range = float(aupc_means.max() - aupc_means.min()) if len(aupc_means) > 1 else 0.0

    # RD curve -> RD_mean, RD_max per model
    rd = compute_rd_curve(df, cfg, group_cols=["dataset", "model", "noise_type"])
    rd_mean_per_model = rd.groupby("model")["rd"].mean()
    rd_max_per_model = rd.groupby("model")["rd"].max()
    rd_mean_range = float(rd_mean_per_model.max() - rd_mean_per_model.min()) if len(rd_mean_per_model) > 1 else 0.0
    rd_max_range = float(rd_max_per_model.max() - rd_max_per_model.min()) if len(rd_max_per_model) > 1 else 0.0

    # CSV_p per model (need subject)
    subject_col = find_subject_col(df, cfg)
    if subject_col:
        csv_curve = compute_csv_p_curve(
            df, cfg, group_cols=["dataset", "model", "noise_type"], subject_col=subject_col
        )
        csv_means = csv_curve.groupby("model")["csv_p"].mean()
        csv_range = float(csv_means.max() - csv_means.min()) if len(csv_means) > 1 else 0.0
    else:
        csv_range = 0.0

    passed = (
        aupc_range >= VIABILITY_AUPC_THRESHOLD
        or rd_mean_range >= VIABILITY_RD_THRESHOLD
        or rd_max_range >= VIABILITY_RD_THRESHOLD
        or csv_range >= VIABILITY_CSV_THRESHOLD
    )
    print(f"[VIABILITY] AUPC range={aupc_range:.4f} (need>={VIABILITY_AUPC_THRESHOLD})")
    print(f"[VIABILITY] RD_mean range={rd_mean_range:.4f} (need>={VIABILITY_RD_THRESHOLD})")
    print(f"[VIABILITY] RD_max range={rd_max_range:.4f} (need>={VIABILITY_RD_THRESHOLD})")
    print(f"[VIABILITY] CSV range={csv_range:.6f} (need>={VIABILITY_CSV_THRESHOLD})")
    print(f"[VIABILITY] {'PASS' if passed else 'FAIL'}")
    return passed


def run_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Compute AUPC, RD_mean, RD_max, CSV per subject; run statistical tests; write report."""
    from analysis.robustness_metrics import (
        MetricConfig,
        add_normalized_p,
        compute_aupc,
        compute_rd_curve,
        compute_csv_p_curve,
        find_subject_col,
    )
    from analysis.statistical_analysis import (
        prepare_wide_format,
        run_omnibus_test,
        run_pairwise_tests,
        AnalysisConfig,
    )

    cfg = MetricConfig(metric_col="corrupted_roc_auc", intensity_col="intensity")
    if "dataset" not in df.columns:
        df["dataset"] = DATASET
    if "eval_mode" not in df.columns:
        df["eval_mode"] = EVAL_MODE
    if "tune" not in df.columns:
        df["tune"] = False
    if "noise_type" not in df.columns:
        df["noise_type"] = "gaussian"
    df = add_normalized_p(df, cfg, normalize_within=["dataset", "noise_type"], clip=True)

    # Ensure subject column
    subject_col = find_subject_col(df, cfg)
    if subject_col is None:
        df["subject"] = 1
        subject_col = "subject"

    group_cols = ["dataset", "eval_mode", "tune", "subject", "model", "noise_type"]

    # AUPC per (subject, model, seed) - group_cols must exist in df
    aupc_group = [c for c in group_cols + ["seed"] if c in df.columns]
    aupc = compute_aupc(df, cfg, group_cols=aupc_group)
    aupc_subject = aupc.groupby(["dataset", "eval_mode", "tune", "subject", "model"]).agg(
        aupc=("aupc", "mean"),
        aupc_std=("aupc", "std"),
    ).reset_index()

    # RD curve -> RD_mean, RD_max per (subject, model, seed)
    rd_group = [c for c in group_cols + ["seed"] if c in df.columns]
    rd = compute_rd_curve(df, cfg, group_cols=rd_group)
    rd_summary = rd.groupby(["dataset", "eval_mode", "tune", "subject", "model"]).agg(
        rd_mean=("rd", "mean"),
        rd_max=("rd", "max"),
    ).reset_index()

    # CSV_p (population variance across subjects - not per-subject; report at model level)
    csv_curve = compute_csv_p_curve(df, cfg, group_cols=["dataset", "eval_mode", "tune", "model", "noise_type"], subject_col=subject_col)
    csv_summary = csv_curve.groupby(["dataset", "eval_mode", "tune", "model"]).agg(
        csv_p_mean=("csv_p", "mean"),
    ).reset_index()

    # Merge into subject-level table (AUPC, RD are per-subject; CSV is model-level)
    subject_metrics = aupc_subject.merge(
        rd_summary,
        on=["dataset", "eval_mode", "tune", "subject", "model"],
        how="outer",
    )
    subject_metrics = subject_metrics.merge(
        csv_summary,
        on=["dataset", "eval_mode", "tune", "model"],
        how="left",
    )

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    subject_metrics.to_csv(analysis_dir / "subject_level_metrics.csv", index=False)

    # Empirical SNR by intensity (if available in data)
    if "empirical_snr_db" in df.columns:
        snr_by_int = df.groupby(["model", "intensity"], as_index=False)["empirical_snr_db"].mean()
        snr_by_int.to_csv(output_dir / "empirical_snr_by_intensity.csv", index=False)

    # Statistical tests (subject as unit)
    config = AnalysisConfig(alpha=0.05, parametric="auto")
    report_lines = [
        "=" * 80,
        "WIRING ROBUSTNESS EXPERIMENT - ANALYSIS REPORT",
        "=" * 80,
        f"Generated: {datetime.now().isoformat()}",
        f"Dataset: {DATASET}, Eval: {EVAL_MODE}",
        "",
    ]

    for metric in ["aupc", "rd_mean", "rd_max"]:
        if metric not in subject_metrics.columns:
            continue
        pivot = prepare_wide_format(
            subject_metrics,
            metric,
            group_cols=["dataset", "eval_mode", "tune"],
        )
        if pivot.empty or len(pivot) < 3:
            report_lines.append(f"{metric}: Insufficient data for tests")
            continue
        model_cols = [c for c in pivot.columns if c not in ["dataset", "eval_mode", "tune", "subject"]]
        if len(model_cols) < 2:
            continue
        res = run_omnibus_test(pivot, model_cols, parametric=False)
        report_lines.append(f"{metric}: {res['test_type']} stat={res['statistic']:.4f} p={res['p_value']:.4f}")
        if res["p_value"] < config.alpha:
            for i, m1 in enumerate(model_cols):
                for m2 in model_cols[i + 1 :]:
                    pw = run_pairwise_tests(pivot, m1, m2, False, config)
                    report_lines.append(f"  {m1} vs {m2}: p={pw['p_value']:.4f} dz={pw['cohens_dz']:.4f}")

    report_path = analysis_dir / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"[OK] Report: {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Wiring Graph Robustness Experiment")
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--skip_stage2", action="store_true", help="Skip Stage 2 even if viability passes")
    args = parser.parse_args()

    output_root = Path(args.output_root or str(_THIS_DIR / "outputs" / "wiring_robustness"))
    run_id = _now_run_id()
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("WIRING ROBUSTNESS EXPERIMENT")
    print("=" * 80)
    print(f"Output: {output_dir}")

    arch_dir = resolve_architectures_dir()
    pilot_dir = prepare_pilot_dir(arch_dir, output_dir)
    saturation_file = create_saturation_file_for_intensity_grid(output_dir)

    # Stage 1
    print("\n[STAGE 1] Viability test (subject 1, seed 42)")
    stage1_df = run_stage1(pilot_dir, saturation_file, output_dir)
    stage1_df.to_csv(output_dir / "stage1_results.csv", index=False)
    print(f"[OK] Stage 1: {len(stage1_df)} rows")

    if not check_viability(stage1_df):
        print("\n[ABORT] Viability failed. Differences in AUPC/RD/CSV not exposed. Stopping.")
        return 0

    if args.skip_stage2:
        print("\n[SKIP] Stage 2 skipped (--skip_stage2)")
        return 0

    # Stage 2
    print("\n[STAGE 2] Full scale (all subjects, 5 seeds)")
    stage2_df = run_stage2(pilot_dir, saturation_file, output_dir)
    stage2_df.to_csv(output_dir / "stage2_results.csv", index=False)
    print(f"[OK] Stage 2: {len(stage2_df)} rows")

    # Analysis (use stage2 results; they include all subjects and seeds)
    print("\n[ANALYSIS] Computing metrics and statistical tests")
    run_analysis(stage2_df, output_dir)

    print("\n[OK] Experiment complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
