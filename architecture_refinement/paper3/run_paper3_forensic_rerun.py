"""
Paper 3 targeted rerun (Forensic Option A): train a small set of topologies under multiple seeds
to estimate topology vs training-stochasticity variance (ICC, variance decomposition).

1) train: copy listed architecture JSONs into pilot_dir/selected_architectures, then run unified jobs.
2) analyze: read long-form CSV (model, seed, RD_max) and write variance_decomposition.json.

Manifest format: see forensic_rerun_seeds_manifest.example.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architecture_refinement.paper3.run_paper3_experiment2 import ALPHA_GRID, DEFAULT_SATURATION, _run_unified_job
from architecture_refinement.paper3.run_paper3_forensic_pass import analyze_variance_decomposition_longform


def _prepare_pilot_architectures(
    *,
    model_names: List[str],
    source_arch_dir: Path,
    dest_arch_dir: Path,
) -> List[str]:
    """Copy architecture JSONs into dest; return list of models that were found."""
    dest_arch_dir.mkdir(parents=True, exist_ok=True)
    from utils import short_run_id

    found: List[str] = []
    for model in model_names:
        copied = False
        for stem in [model, short_run_id(model)]:
            src = source_arch_dir / f"{stem}.json"
            if src.exists():
                shutil.copy2(src, dest_arch_dir / f"{model}.json")
                found.append(model)
                copied = True
                break
        if not copied:
            print(f"[forensic-rerun] WARNING: missing architecture JSON for {model} in {source_arch_dir}")
    return found


def run_train_from_manifest(
    manifest_path: Path,
    *,
    repo_root: Path,
    python_exe: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pilot_dir = Path(manifest["pilot_dir"])
    if not pilot_dir.is_absolute():
        pilot_dir = repo_root / pilot_dir

    src_dir = Path(manifest.get("source_architectures_dir", ""))
    if not src_dir.is_absolute():
        src_dir = repo_root / src_dir

    model_names = list(manifest["model_names"])
    seeds = [int(s) for s in manifest["seeds"]]
    dataset = str(manifest.get("dataset", "BNCI2014_001"))
    eval_mode = str(manifest.get("eval_mode", "CrossSession"))
    subjects = [int(s) for s in manifest.get("subjects", list(range(1, 10)))]
    sat = str(manifest.get("saturation_file", DEFAULT_SATURATION))
    target_snr_db = float(manifest.get("target_snr_db", -5.0))
    python_exe = python_exe or sys.executable

    arch_dest = pilot_dir / "selected_architectures"
    found = _prepare_pilot_architectures(
        model_names=model_names,
        source_arch_dir=src_dir,
        dest_arch_dir=arch_dest,
    )

    pert_params = {"ar1_drift": {"rho": float(manifest.get("ar1_rho", 0.97))}}
    perturbation_types = list(manifest.get("perturbation_types", ["ar1_drift"]))

    failed: List[Dict[str, Any]] = []
    for m in found:
        for seed in seeds:
            rc = _run_unified_job(
                repo_root=repo_root,
                python_exe=python_exe,
                pilot_dir=pilot_dir,
                model_name=m,
                dataset=dataset,
                eval_mode=eval_mode,
                subjects=subjects,
                seed=seed,
                saturation_file=sat,
                alpha_grid=ALPHA_GRID,
                perturbation_types=perturbation_types,
                target_snr_db=target_snr_db,
                perturbation_params=pert_params,
                overwrite=overwrite,
            )
            if rc != 0:
                failed.append({"model": m, "seed": seed, "rc": rc})

    pilot_dir.mkdir(parents=True, exist_ok=True)
    (pilot_dir / "forensic_rerun_manifest_resolved.json").write_text(
        json.dumps(
            {
                **manifest,
                "resolved_pilot_dir": str(pilot_dir),
                "models_found": found,
                "failed_jobs": failed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "pilot_dir": str(pilot_dir),
        "models_scheduled": found,
        "n_jobs": len(found) * len(seeds),
        "failed_jobs": failed,
    }


def run_analyze_csv(
    csv_path: Path,
    output_json: Path,
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    for col in ("model", "seed", "RD_max"):
        if col not in df.columns:
            raise ValueError(f"CSV must contain column {col}")
    out = analyze_variance_decomposition_longform(df)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper 3 forensic targeted rerun (Option A)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Copy arch JSONs and run unified jobs from manifest")
    p_train.add_argument("--manifest", type=str, required=True)
    p_train.add_argument("--repo-root", type=str, default=None, help="Repo root (default: project root)")
    p_train.add_argument("--python", type=str, default=None)
    p_train.add_argument("--overwrite", action="store_true")

    p_an = sub.add_parser("analyze", help="Variance decomposition + ICC from long-form CSV")
    p_an.add_argument("--csv", type=str, required=True, help="Columns: model, seed, RD_max")
    p_an.add_argument("--output-json", type=str, required=True)
    p_an.add_argument("--repo-root", type=str, default=None, help="Root for relative CSV/output paths")

    args = parser.parse_args()
    repo = Path(args.repo_root) if getattr(args, "repo_root", None) else _REPO_ROOT

    if args.cmd == "train":
        mp = Path(args.manifest)
        if not mp.is_absolute():
            mp = repo / mp
        summary = run_train_from_manifest(
            mp,
            repo_root=repo,
            python_exe=getattr(args, "python", None),
            overwrite=args.overwrite,
        )
        print(json.dumps(summary, indent=2))
        return 0 if not summary.get("failed_jobs") else 1

    if args.cmd == "analyze":
        outp = Path(args.output_json)
        if not outp.is_absolute():
            outp = repo / outp
        cp = Path(args.csv)
        if not cp.is_absolute():
            cp = repo / cp
        run_analyze_csv(cp, outp)
        print(f"Wrote {outp}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
