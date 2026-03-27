"""
Hail Mary (Chapter 5): dispatch training + reduced test_perturb runs per topology × seed.

Uses evaluation.unified_experiment_runner (same pattern as run_paper3_experiment2).
Default: Gaussian noise, alpha grid clean / low / moderate (§13.4).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_DATASET = "BNCI2014_001"
DEFAULT_EVAL_MODE = "CrossSession"
DEFAULT_SATURATION = "saturation_results/saturation_points_summary.csv"
# Clean + low + moderate relative to sigma_max from saturation file
DEFAULT_GAUSSIAN_ALPHA_GRID = [0.0, 0.25, 0.5]
DEFAULT_SEEDS_START = 42
DEFAULT_NUM_SEEDS = 3


def _run_unified_job(
    *,
    repo_root: Path,
    python_exe: str,
    pilot_dir: Path,
    model_name: str,
    dataset: str,
    eval_mode: str,
    subjects: List[int],
    seed: int,
    saturation_file: str,
    alpha_grid: List[float],
    perturbation_types: List[str],
    target_snr_db: float,
    ar1_rho: float,
    overwrite: bool,
    hail_mary_stability: bool = True,
) -> int:
    cmd = [
        python_exe,
        str((repo_root / "evaluation" / "unified_experiment_runner.py").as_posix()),
        "--nas_pilot_dir",
        str(pilot_dir.as_posix()),
        "--model",
        model_name,
        "--dataset",
        dataset,
        "--subjects",
        *[str(s) for s in subjects],
        "--mode",
        "test_perturb",
        "--eval_mode",
        eval_mode,
        "--seed",
        str(seed),
        "--overwrite" if overwrite else "",
        "--disable_underfitting_retrain",
        "--noise_perturbation_saturation_file",
        saturation_file,
        "--noise_perturbation_num_steps",
        "20",
        "--test_perturb_gaussian_alpha_grid",
        ",".join(str(a) for a in alpha_grid),
        "--test_perturb_target_snr_db",
        str(target_snr_db),
        "--test_perturb_ar1_rho",
        str(ar1_rho),
        "--test_perturb_noise_types",
        ",".join(perturbation_types),
    ]
    if hail_mary_stability:
        cmd.append("--hail_mary_stability")
    cmd = [c for c in cmd if c]
    proc = subprocess.run(cmd, cwd=str(repo_root), check=False)
    return int(proc.returncode)


def run_hail_mary_experiment(
    topology_panel_dir: Path,
    output_dir: Path,
    *,
    dataset: str = DEFAULT_DATASET,
    eval_mode: str = DEFAULT_EVAL_MODE,
    subjects: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    num_seeds: int = DEFAULT_NUM_SEEDS,
    seed_start: int = DEFAULT_SEEDS_START,
    saturation_file: str = DEFAULT_SATURATION,
    alpha_grid: Optional[List[float]] = None,
    perturbation_types: Optional[List[str]] = None,
    target_snr_db: float = -5.0,
    ar1_rho: float = 0.97,
    python_exe: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    topology_ids_filter: Optional[List[str]] = None,
    hail_mary_stability: bool = True,
) -> Dict[str, Any]:
    """
    topology_panel_dir: directory containing topology_manifest.json (or .csv) and hail_mary_pilot/.
    """
    subjects = subjects if subjects is not None else list(range(1, 10))
    python_exe = python_exe or sys.executable
    alpha_grid = list(alpha_grid) if alpha_grid is not None else list(DEFAULT_GAUSSIAN_ALPHA_GRID)
    perturbation_types = perturbation_types or ["gaussian"]

    panel_dir = Path(topology_panel_dir).resolve()
    manifest_json = panel_dir / "topology_manifest.json"
    manifest_csv = panel_dir / "topology_manifest.csv"
    pilot_dir = panel_dir / "hail_mary_pilot"

    if not pilot_dir.is_dir():
        raise FileNotFoundError(f"Expected pilot dir {pilot_dir} — run run_hail_mary_build_panel first.")

    models: List[str] = []
    if manifest_json.exists():
        data = json.loads(manifest_json.read_text(encoding="utf-8"))
        for row in data.get("manifest", []):
            models.append(str(row["model_name"]))
    elif manifest_csv.exists():
        import csv as _csv

        with open(manifest_csv, newline="", encoding="utf-8") as f:
            r = _csv.DictReader(f)
            for row in r:
                models.append(str(row["model_name"]))
    else:
        models = sorted(p.stem for p in (pilot_dir / "selected_architectures").glob("*.json"))

    if topology_ids_filter:
        allowed = set(topology_ids_filter)
        filtered: List[str] = []
        if manifest_json.exists():
            data = json.loads(manifest_json.read_text(encoding="utf-8"))
            for row in data.get("manifest", []):
                if str(row.get("topology_id", "")) in allowed:
                    filtered.append(str(row["model_name"]))
        models = filtered

    if not models:
        raise RuntimeError("No models found for Hail Mary experiment.")

    if seeds is None:
        seeds = [seed_start + i for i in range(num_seeds)]

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sat_path = str(_REPO_ROOT / saturation_file) if not Path(saturation_file).is_absolute() else saturation_file

    run_manifest: Dict[str, Any] = {
        "study": "hail_mary_chapter5",
        "dataset": dataset,
        "eval_mode": eval_mode,
        "pilot_dir": str(pilot_dir),
        "saturation_file": sat_path,
        "perturbation_types": perturbation_types,
        "test_perturb_gaussian_alpha_grid": alpha_grid,
        "target_snr_db": target_snr_db,
        "seeds": seeds,
        "models": models,
        "hail_mary_stability": hail_mary_stability,
        "jobs": [],
    }

    jobs = [(m, s) for m in models for s in seeds]
    failed: List[Dict[str, Any]] = []

    for model_name, seed in jobs:
        job = {
            "model_name": model_name,
            "seed": seed,
            "result_glob": f"results/MotorImagery/{dataset}/{model_name}/CrossSessionEvaluation/{seed}/**/*.csv",
        }
        run_manifest["jobs"].append(job)
        if dry_run:
            continue
        rc = _run_unified_job(
            repo_root=_REPO_ROOT,
            python_exe=python_exe,
            pilot_dir=pilot_dir,
            model_name=model_name,
            dataset=dataset,
            eval_mode=eval_mode,
            subjects=subjects,
            seed=seed,
            saturation_file=sat_path,
            alpha_grid=alpha_grid,
            perturbation_types=perturbation_types,
            target_snr_db=target_snr_db,
            ar1_rho=ar1_rho,
            overwrite=overwrite,
            hail_mary_stability=hail_mary_stability,
        )
        if rc != 0:
            failed.append({"model_name": model_name, "seed": seed, "returncode": rc})

    manifest_path = output_dir / "hail_mary_run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    if failed:
        (output_dir / "hail_mary_failed_jobs.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")

    return {"run_manifest": str(manifest_path), "n_jobs": len(jobs), "failed": failed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Hail Mary: run topology panel × seeds via unified_experiment_runner")
    parser.add_argument(
        "--topology-panel-dir",
        type=str,
        default="architecture_refinement/outputs/hail_mary/topology_panel",
        help="Directory with topology_manifest.json and hail_mary_pilot/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="architecture_refinement/outputs/hail_mary/runs",
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--eval-mode", type=str, default=DEFAULT_EVAL_MODE)
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEEDS_START)
    parser.add_argument("--saturation-file", type=str, default=DEFAULT_SATURATION)
    parser.add_argument(
        "--alpha-grid",
        type=str,
        default="0,0.25,0.5",
        help="Comma-separated alphas for Gaussian (clean, low, moderate)",
    )
    parser.add_argument(
        "--perturbation-types",
        type=str,
        default="gaussian",
        help="Comma-separated noise types for test_perturb (default: gaussian)",
    )
    parser.add_argument("--target-snr-db", type=float, default=-5.0)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-hail-mary-stability",
        action="store_true",
        help="Disable --hail_mary_stability in unified runner (no batch loss variance callback).",
    )
    args = parser.parse_args()

    panel = Path(args.topology_panel_dir)
    if not panel.is_absolute():
        panel = _REPO_ROOT / panel
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = _REPO_ROOT / out

    alphas = [float(x.strip()) for x in args.alpha_grid.split(",") if x.strip()]
    ptypes = [x.strip() for x in args.perturbation_types.split(",") if x.strip()]

    res = run_hail_mary_experiment(
        topology_panel_dir=panel,
        output_dir=out,
        dataset=args.dataset,
        eval_mode=args.eval_mode,
        subjects=args.subjects,
        num_seeds=args.num_seeds,
        seed_start=args.seed_start,
        saturation_file=args.saturation_file,
        alpha_grid=alphas,
        perturbation_types=ptypes,
        target_snr_db=args.target_snr_db,
        python_exe=args.python,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        hail_mary_stability=not args.no_hail_mary_stability,
    )
    print(json.dumps({"run_manifest": res["run_manifest"], "n_jobs": res["n_jobs"], "n_failed": len(res["failed"])}, indent=2))
    return 0 if not res["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
