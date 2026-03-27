"""
Orchestrate the Q3 stratified follow-up using paths from the latest complete Paper 3 run.

Steps (see plan / advisor spec):
  1) Optionally re-run Experiment 1 (regenerates proxy_pool.csv + G1 JSONs; use when pool must match fixed WS seed).
  2) Stratified sample + pilot JSONs + unified training (run_paper3_q3_stratified_train): by default
     regenerates the pilot with the 2D sampler (setup-only), then runs training jobs from that manifest
     (train-only) so sampling is not repeated. Use --combined-stratified for a single-phase sample+train.
  3) Collect perturb results from the stratified pilot, then build figures (run_paper3_q3_stratified_analysis).

Discovery: scans architecture_refinement/outputs (by default) for experiment3_results.csv, resolves
experiment1/ experiment2 folders via standard layouts, and picks the newest file by mtime among
runs that look complete (non-empty Exp3 CSV + Exp2 manifest + proxy_pool).

Stratified sampling uses a 2D proxy plane grid: --n-bins-te × --n-bins-orc must equal --n-target
(default 10×10=100; use 8×15=120 or 10×15=150 for larger pilots if the pool is big enough).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class DiscoveredRun:
    """Resolved Paper 3 directories for one Exp3 results file."""

    experiment3_csv: Path
    experiment3_dir: Path
    experiment1_dir: Path
    experiment2_dir: Path
    mtime_ns: int
    complete: bool
    notes: List[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [
            f"experiment3_results: {self.experiment3_csv} (mtime_ns={self.mtime_ns})",
            f"  experiment1_dir: {self.experiment1_dir}",
            f"  experiment2_dir: {self.experiment2_dir}",
            f"  complete={self.complete}",
        ]
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def _resolve_exp1_exp2(experiment3_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Map .../experiment3 (or paper3_experiment3) to experiment1 and experiment2 dirs.
    """
    parent = experiment3_dir.parent
    name = experiment3_dir.name

    if name == "experiment3":
        e1 = parent / "experiment1"
        e2 = parent / "experiment2"
        if e1.is_dir() and e2.is_dir():
            return e1, e2

    if name == "paper3_experiment3":
        out = parent
        e1 = out / "paper3_experiment1"
        e2 = out / "paper3_experiment2"
        if e1.is_dir() and e2.is_dir():
            return e1, e2

    # Loose fallback: siblings with conventional names
    for e1_name, e2_name in (
        ("experiment1", "experiment2"),
        ("paper3_experiment1", "paper3_experiment2"),
    ):
        e1, e2 = parent / e1_name, parent / e2_name
        if e1.is_dir() and e2.is_dir():
            return e1, e2

    return None, None


def _is_complete_run(bundle: DiscoveredRun) -> None:
    """Mutate bundle.complete and bundle.notes from filesystem checks."""
    csv_path = bundle.experiment3_csv
    if not csv_path.is_file():
        bundle.complete = False
        bundle.notes.append("experiment3_results.csv missing")
        return
    try:
        sz = csv_path.stat().st_size
    except OSError as e:
        bundle.complete = False
        bundle.notes.append(f"stat failed: {e}")
        return
    if sz < 50:
        bundle.notes.append(f"small experiment3 CSV ({sz} bytes)")
        bundle.complete = False
        return

    man = bundle.experiment2_dir / "experiment2_manifest.json"
    if not man.is_file():
        bundle.notes.append("no experiment2_manifest.json")
        bundle.complete = False
        return

    pool = bundle.experiment1_dir / "proxy_pool.csv"
    if not pool.is_file():
        bundle.notes.append("no proxy_pool.csv (needed for stratified sampling)")
        bundle.complete = False
        return

    try:
        mdata = json.loads(man.read_text(encoding="utf-8"))
        groups = mdata.get("groups") or {}
        if not groups.get("G1") or not groups.get("G2"):
            bundle.notes.append("manifest missing G1/G2 lists")
            bundle.complete = False
            return
    except (json.JSONDecodeError, OSError) as e:
        bundle.notes.append(f"manifest unreadable: {e}")
        bundle.complete = False
        return

    arch = bundle.experiment2_dir / "experiment2_pilot" / "selected_architectures"
    if not arch.is_dir() or not any(arch.glob("*.json")):
        bundle.notes.append("no architecture JSONs under experiment2_pilot/selected_architectures")
        bundle.complete = False
        return

    bundle.complete = True


def discover_paper3_runs(search_root: Path) -> List[DiscoveredRun]:
    """Find candidate experiment3_results.csv paths and resolve sibling Exp1/Exp2."""
    if not search_root.is_dir():
        return []

    skip_parts = {"q3_stratified", "__pycache__", ".git"}

    bundles: List[DiscoveredRun] = []
    for p in search_root.rglob("experiment3_results.csv"):
        if any(part in skip_parts for part in p.parts):
            continue
        exp3_dir = p.parent
        e1, e2 = _resolve_exp1_exp2(exp3_dir)
        if e1 is None or e2 is None:
            continue
        try:
            mtime_ns = p.stat().st_mtime_ns
        except OSError:
            continue
        b = DiscoveredRun(
            experiment3_csv=p.resolve(),
            experiment3_dir=exp3_dir.resolve(),
            experiment1_dir=e1.resolve(),
            experiment2_dir=e2.resolve(),
            mtime_ns=mtime_ns,
            complete=False,
        )
        _is_complete_run(b)
        bundles.append(b)

    bundles.sort(key=lambda x: x.mtime_ns, reverse=True)
    return bundles


def pick_best_run(bundles: List[DiscoveredRun], *, require_complete: bool) -> Optional[DiscoveredRun]:
    if not bundles:
        return None
    if require_complete:
        for b in bundles:
            if b.complete:
                return b
        return None
    return bundles[0]


def run_pipeline(
    *,
    experiment1_dir: Optional[Path],
    experiment2_dir: Optional[Path],
    main_experiment3_csv: Path,
    pilot_root: Path,
    analysis_output_dir: Path,
    rerun_experiment1: bool,
    stratified_setup_only: bool,
    combined_stratified: bool,
    skip_stratified: bool,
    analyze_only: bool,
    dataset: str,
    results_root: Path,
    H: int,
    n_target: int,
    n_bins_te: int,
    n_bins_orc: int,
    strat_seed: int,
    S_strat: int,
    subjects: List[int],
    saturation_file: str,
    target_snr_db: float,
    python_exe: str,
    overwrite: bool,
    omp_workaround: bool,
) -> int:
    if omp_workaround:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    if analyze_only:
        stratified_csv = pilot_root / "q3_stratified_experiment3_results.csv"
        print("\n=== Analyze only: collect + figures ===")
        from architecture_refinement.paper3.run_paper3_q3_stratified_collect import collect_stratified_results
        from architecture_refinement.paper3.run_paper3_q3_stratified_analysis import run_analysis

        manifest = pilot_root / "q3_stratified_manifest.json"
        if manifest.exists():
            man = json.loads(manifest.read_text(encoding="utf-8"))
            if man.get("groups", {}).get("G_strat"):
                collect_stratified_results(
                    pilot_root,
                    stratified_csv,
                    results_root,
                    dataset=dataset,
                )
            else:
                print("[pipeline] No G_strat models; skipping stratified CSV collection.")
        else:
            print("[pipeline] No manifest; skipping collect.")

        run_analysis(
            pilot_root=pilot_root,
            stratified_csv=stratified_csv,
            main_exp3_csv=main_experiment3_csv,
            output_dir=analysis_output_dir,
        )
        print(f"\n[pipeline] Analysis outputs under: {analysis_output_dir}")
        return 0

    assert experiment1_dir is not None and experiment2_dir is not None

    if rerun_experiment1:
        from architecture_refinement.paper3.run_paper3_experiment1 import run_experiment1

        print("\n=== Step 1: Re-run Experiment 1 (proxy pool + G1 selected_architectures) ===")
        run_experiment1(output_dir=experiment1_dir, H=H)

    if not skip_stratified:
        from architecture_refinement.paper3.run_paper3_q3_stratified_train import (
            run_stratified_setup_and_train,
        )

        if stratified_setup_only:
            print("\n=== Step 2: Q3 stratified pilot (2D sampler: JSONs + manifest only) ===")
            summary = run_stratified_setup_and_train(
                experiment1_dir=experiment1_dir,
                experiment2_dir=experiment2_dir,
                output_pilot_root=pilot_root,
                H=H,
                n_target=n_target,
                n_bins_te=n_bins_te,
                n_bins_orc=n_bins_orc,
                strat_seed=strat_seed,
                S=S_strat,
                dataset=dataset,
                eval_mode="CrossSession",
                subjects=subjects,
                saturation_file=saturation_file,
                target_snr_db=target_snr_db,
                python_exe=python_exe,
                overwrite=overwrite,
                setup_only=True,
                dry_run=False,
            )
            print(json.dumps(summary, indent=2))
        elif combined_stratified:
            print("\n=== Step 2: Q3 stratified pilot (single phase: sample + JSONs + training) ===")
            summary = run_stratified_setup_and_train(
                experiment1_dir=experiment1_dir,
                experiment2_dir=experiment2_dir,
                output_pilot_root=pilot_root,
                H=H,
                n_target=n_target,
                n_bins_te=n_bins_te,
                n_bins_orc=n_bins_orc,
                strat_seed=strat_seed,
                S=S_strat,
                dataset=dataset,
                eval_mode="CrossSession",
                subjects=subjects,
                saturation_file=saturation_file,
                target_snr_db=target_snr_db,
                python_exe=python_exe,
                overwrite=overwrite,
                setup_only=False,
                dry_run=False,
            )
            print(json.dumps(summary, indent=2))
        else:
            print("\n=== Step 2a: Regenerate stratified pilot (2D sampler: JSONs + manifest) ===")
            summary_setup = run_stratified_setup_and_train(
                experiment1_dir=experiment1_dir,
                experiment2_dir=experiment2_dir,
                output_pilot_root=pilot_root,
                H=H,
                n_target=n_target,
                n_bins_te=n_bins_te,
                n_bins_orc=n_bins_orc,
                strat_seed=strat_seed,
                S=S_strat,
                dataset=dataset,
                eval_mode="CrossSession",
                subjects=subjects,
                saturation_file=saturation_file,
                target_snr_db=target_snr_db,
                python_exe=python_exe,
                overwrite=overwrite,
                setup_only=True,
                dry_run=False,
            )
            print(json.dumps(summary_setup, indent=2))

            print("\n=== Step 2b: Stratified unified training (from manifest; no re-sampling) ===")
            summary_train = run_stratified_setup_and_train(
                experiment1_dir=experiment1_dir,
                experiment2_dir=experiment2_dir,
                output_pilot_root=pilot_root,
                H=H,
                n_target=n_target,
                n_bins_te=n_bins_te,
                n_bins_orc=n_bins_orc,
                strat_seed=strat_seed,
                S=S_strat,
                dataset=dataset,
                eval_mode="CrossSession",
                subjects=subjects,
                saturation_file=saturation_file,
                target_snr_db=target_snr_db,
                python_exe=python_exe,
                overwrite=overwrite,
                train_only=True,
                dry_run=False,
            )
            print(json.dumps(summary_train, indent=2))

    stratified_csv = pilot_root / "q3_stratified_experiment3_results.csv"

    print("\n=== Step 3: Collect (if any G_strat models) + analysis figures ===")
    from architecture_refinement.paper3.run_paper3_q3_stratified_collect import collect_stratified_results
    from architecture_refinement.paper3.run_paper3_q3_stratified_analysis import run_analysis

    manifest = pilot_root / "q3_stratified_manifest.json"
    if manifest.exists():
        man = json.loads(manifest.read_text(encoding="utf-8"))
        if man.get("groups", {}).get("G_strat"):
            collect_stratified_results(
                pilot_root,
                stratified_csv,
                results_root,
                dataset=dataset,
            )
        else:
            print("[pipeline] No G_strat models in manifest; skipping stratified CSV collection.")
    else:
        print("[pipeline] No q3_stratified_manifest.json; skipping collect (run stratified train first).")

    run_analysis(
        pilot_root=pilot_root,
        stratified_csv=stratified_csv,
        main_exp3_csv=main_experiment3_csv,
        output_dir=analysis_output_dir,
    )
    print(f"\n[pipeline] Analysis outputs under: {analysis_output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover latest Paper 3 run and execute Q3 stratified pipeline (Exp1 optional, stratified train, collect, plots).",
        epilog=(
            "Examples:\n"
            "  %(prog)s --discover-only\n"
            "  %(prog)s --require-complete --setup-only\n"
            "  %(prog)s --rerun-experiment1 --omp-workaround\n"
            "  %(prog)s --analyze-only --omp-workaround\n"
            "  %(prog)s --combined-stratified\n"
            "\n"
            "Default full run: regenerates the stratified pilot (2D sampler), then runs training jobs, "
            "then collect + analysis. After jobs finish, use --analyze-only if you only need figures."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--search-root",
        type=str,
        default="architecture_refinement/outputs",
        help="Root to scan for experiment3_results.csv (default: architecture_refinement/outputs)",
    )
    parser.add_argument("--repo-root", type=str, default=None, help="Repository root (default: infer from this file)")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Only accept runs with Exp3 CSV + manifest + proxy_pool + pilot JSONs (default: pick newest resolved run even if incomplete)",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="List discovered runs sorted by mtime and exit",
    )
    parser.add_argument("--experiment1-dir", type=str, default=None)
    parser.add_argument("--experiment2-dir", type=str, default=None)
    parser.add_argument("--main-experiment3-csv", type=str, default=None)
    parser.add_argument(
        "--pilot-root",
        type=str,
        default="architecture_refinement/outputs/paper3/q3_stratified_pilot",
    )
    parser.add_argument(
        "--analysis-output-dir",
        type=str,
        default="architecture_refinement/outputs/paper3/analysis_followups",
    )
    parser.add_argument(
        "--rerun-experiment1",
        action="store_true",
        help="Regenerate proxy_pool.csv and Exp1 selected_architectures before stratified sampling",
    )
    parser.add_argument(
        "--skip-exp1",
        action="store_true",
        help="Do not run Experiment 1 (overrides --rerun-experiment1)",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Stratified train: write pilot JSONs/manifest only (no unified training jobs)",
    )
    parser.add_argument(
        "--combined-stratified",
        action="store_true",
        help=(
            "Use one stratified step (sample + train in-process). Default is two phases: regenerate pilot "
            "(2D sampler) then train from manifest so sampling is not run twice."
        ),
    )
    parser.add_argument(
        "--skip-stratified",
        action="store_true",
        help="Skip stratified train/setup entirely",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only collect (if manifest has G_strat) + analysis; skip Exp1 and stratified train",
    )
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--n-target", type=int, default=100)
    parser.add_argument("--n-bins-te", type=int, default=10, help="Proxy plane bins on te_hat (2D stratification)")
    parser.add_argument("--n-bins-orc", type=int, default=10, help="Proxy plane bins on orc_hat (2D stratification)")
    parser.add_argument("--strat-seed", type=int, default=20260324)
    parser.add_argument("--S-strat", type=int, default=3, help="Seeds for stratified pilot (default 3)")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument(
        "--saturation-file",
        type=str,
        default="saturation_results/saturation_points_summary.csv",
    )
    parser.add_argument("--target-snr-db", type=float, default=-5.0)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--omp-workaround",
        action="store_true",
        help="Set KMP_DUPLICATE_LIB_OK=TRUE (helps some Windows OpenMP duplicate-DLL crashes)",
    )
    args = parser.parse_args()

    if args.setup_only and args.combined_stratified:
        print(
            "[pipeline] Error: use either --setup-only or --combined-stratified, not both.",
            file=sys.stderr,
        )
        return 1

    if not args.analyze_only and args.n_bins_te * args.n_bins_orc != args.n_target:
        print(
            f"[pipeline] Error: --n-target ({args.n_target}) must equal "
            f"--n-bins-te × --n-bins-orc ({args.n_bins_te}×{args.n_bins_orc}={args.n_bins_te * args.n_bins_orc}).",
            file=sys.stderr,
        )
        return 1

    repo = Path(args.repo_root) if args.repo_root else _REPO_ROOT
    if not repo.is_absolute():
        repo = _REPO_ROOT / repo
    search = Path(args.search_root)
    if not search.is_absolute():
        search = repo / search

    bundles = discover_paper3_runs(search)

    if args.discover_only:
        print(f"Scanned: {search}")
        print(f"Found {len(bundles)} candidate run(s) with resolvable Exp1/Exp2:\n")
        for i, b in enumerate(bundles):
            print(f"--- [{i}] ---")
            print(b.describe())
            print()
        if bundles:
            best = pick_best_run(bundles, require_complete=args.require_complete)
            tag = "best complete" if args.require_complete else "newest"
            print(f"Selected ({tag}): {best.experiment3_csv if best else 'none'}")
        return 0

    exp1: Optional[Path] = Path(args.experiment1_dir) if args.experiment1_dir else None
    exp2: Optional[Path] = Path(args.experiment2_dir) if args.experiment2_dir else None
    exp3_csv: Optional[Path] = Path(args.main_experiment3_csv) if args.main_experiment3_csv else None

    if exp1 and not exp1.is_absolute():
        exp1 = repo / exp1
    if exp2 and not exp2.is_absolute():
        exp2 = repo / exp2
    if exp3_csv and not exp3_csv.is_absolute():
        exp3_csv = repo / exp3_csv

    pilot = Path(args.pilot_root)
    analysis_out = Path(args.analysis_output_dir)
    if not pilot.is_absolute():
        pilot = repo / pilot
    if not analysis_out.is_absolute():
        analysis_out = repo / analysis_out

    results_root = Path(args.results_root) if args.results_root else repo
    if not results_root.is_absolute():
        results_root = repo / results_root

    if args.analyze_only:
        if exp3_csv is None:
            best_any = pick_best_run(bundles, require_complete=False)
            if best_any is not None:
                exp3_csv = best_any.experiment3_csv
        if exp3_csv is None:
            print(
                "[pipeline] --analyze-only requires --main-experiment3-csv or a discoverable "
                "experiment3_results.csv under --search-root.",
                file=sys.stderr,
            )
            return 1
        print(f"=== Analyze only; main Exp3 CSV: {exp3_csv} ===\n")
        return run_pipeline(
            experiment1_dir=None,
            experiment2_dir=None,
            main_experiment3_csv=exp3_csv,
            pilot_root=pilot,
            analysis_output_dir=analysis_out,
            rerun_experiment1=False,
            stratified_setup_only=False,
            combined_stratified=False,
            skip_stratified=True,
            analyze_only=True,
            dataset=args.dataset,
            results_root=results_root,
            H=args.H,
            n_target=args.n_target,
            n_bins_te=args.n_bins_te,
            n_bins_orc=args.n_bins_orc,
            strat_seed=args.strat_seed,
            S_strat=args.S_strat,
            subjects=list(args.subjects),
            saturation_file=args.saturation_file,
            target_snr_db=args.target_snr_db,
            python_exe=args.python,
            overwrite=args.overwrite,
            omp_workaround=args.omp_workaround,
        )

    if exp3_csv is None or exp1 is None or exp2 is None:
        best = pick_best_run(bundles, require_complete=args.require_complete)
        if best is None:
            if args.require_complete and bundles:
                print(
                    "No *complete* run found. Re-run with --discover-only to inspect, "
                    "or drop --require-complete, or pass explicit --experiment*-dir paths.",
                    file=sys.stderr,
                )
            else:
                print(
                    "No Paper 3 run found under search root. Run Experiment 2+3 first or pass explicit paths.",
                    file=sys.stderr,
                )
            return 1
        if exp3_csv is None:
            exp3_csv = best.experiment3_csv
        if exp1 is None:
            exp1 = best.experiment1_dir
        if exp2 is None:
            exp2 = best.experiment2_dir
        print("=== Using discovered run ===")
        print(best.describe())
        print()

    assert exp1 is not None and exp2 is not None and exp3_csv is not None

    rerun = args.rerun_experiment1 and not args.skip_exp1 and not args.analyze_only

    return run_pipeline(
        experiment1_dir=exp1,
        experiment2_dir=exp2,
        main_experiment3_csv=exp3_csv,
        pilot_root=pilot,
        analysis_output_dir=analysis_out,
        rerun_experiment1=rerun,
        stratified_setup_only=args.setup_only,
        combined_stratified=args.combined_stratified,
        skip_stratified=args.skip_stratified,
        analyze_only=args.analyze_only,
        dataset=args.dataset,
        results_root=results_root,
        H=args.H,
        n_target=args.n_target,
        n_bins_te=args.n_bins_te,
        n_bins_orc=args.n_bins_orc,
        strat_seed=args.strat_seed,
        S_strat=args.S_strat,
        subjects=list(args.subjects),
        saturation_file=args.saturation_file,
        target_snr_db=args.target_snr_db,
        python_exe=args.python,
        overwrite=args.overwrite,
        omp_workaround=args.omp_workaround,
    )


if __name__ == "__main__":
    sys.exit(main())
