"""
Paper 3 Master Script: Run all experiments in fail-safe order.

Fail-safe sequence (spec §10):
1. Experiment 1 (proxy collection) -> verify TE/ORC, Pareto nontrivial
2. Sanity Check (best vs worst)
3. Experiment 2 (train G1..G5)
4. Experiment 3 (proxy plane, hit-rate)
5. Generate Plot 1, 2, 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper 3: Run all experiments and generate plots")
    parser.add_argument("--output-root", type=str, default="architecture_refinement/outputs/paper3")
    parser.add_argument("--skip-exp1", action="store_true", help="Skip Experiment 1 (use existing)")
    parser.add_argument("--skip-sanity", action="store_true", help="Skip sanity check")
    parser.add_argument("--skip-exp2", action="store_true", help="Skip Experiment 2 (training)")
    parser.add_argument("--skip-exp3", action="store_true", help="Skip Experiment 3 (analysis)")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--dry-run", action="store_true", help="Dry-run Exp2 (no training)")
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--N-proxy", type=int, default=500)
    parser.add_argument("--K", type=int, default=12)
    parser.add_argument("--S", type=int, default=5)
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    args = parser.parse_args()

    root = _REPO_ROOT / args.output_root
    exp1_dir = root / "experiment1"
    sanity_dir = root / "sanity_check"
    exp2_dir = root / "experiment2"
    exp3_dir = root / "experiment3"
    root.mkdir(parents=True, exist_ok=True)

    # Step 1: Experiment 1
    if not args.skip_exp1:
        print("\n=== Step 1: Experiment 1 (Proxy landscape) ===")
        from architecture_refinement.paper3.run_paper3_experiment1 import run_experiment1

        run_experiment1(
            output_dir=exp1_dir,
            H=args.H,
            N_proxy=args.N_proxy,
            K=args.K,
            seed=202602,
        )
        summary_path = exp1_dir / "experiment1_summary.json"
        if summary_path.exists():
            import json
            s = json.loads(summary_path.read_text())
            if s.get("pareto_size", 0) < 10:
                print("[WARNING] Pareto set small (<10). Consider increasing N_proxy.")
    else:
        print("\n=== Step 1: Skipped (--skip-exp1) ===")
        if not (exp1_dir / "proxy_pool.csv").exists():
            print("[ERROR] Experiment 1 output not found. Run without --skip-exp1 first.")
            return 1

    # Step 2: Sanity Check
    if not args.skip_sanity:
        print("\n=== Step 2: Sanity Check (best vs worst) ===")
        from architecture_refinement.paper3.run_paper3_sanity_check import run_sanity_check

        run_sanity_check(
            experiment1_dir=exp1_dir,
            output_dir=sanity_dir,
            S=args.S,
            subjects=args.subjects,
            dry_run=False,
        )
    else:
        print("\n=== Step 2: Skipped (--skip-sanity) ===")

    # Step 3: Experiment 2
    if not args.skip_exp2:
        print("\n=== Step 3: Experiment 2 (Train G1..G5) ===")
        from architecture_refinement.paper3.run_paper3_experiment2 import run_experiment2

        run_experiment2(
            experiment1_dir=exp1_dir,
            output_dir=exp2_dir,
            H=args.H,
            K=args.K,
            S=args.S,
            subjects=args.subjects,
            dry_run=args.dry_run,
        )
    else:
        print("\n=== Step 3: Skipped (--skip-exp2) ===")
        if not (exp2_dir / "experiment2_manifest.json").exists():
            print("[WARNING] Experiment 2 output not found. Run without --skip-exp2 for full pipeline.")

    # Step 4: Experiment 3
    if not args.skip_exp3:
        print("\n=== Step 4: Experiment 3 (Proxy plane, hit-rate) ===")
        from architecture_refinement.paper3.run_paper3_experiment3 import run_experiment3

        run_experiment3(
            experiment2_dir=exp2_dir,
            experiment1_dir=exp1_dir,
            output_dir=exp3_dir,
        )
    else:
        print("\n=== Step 4: Skipped (--skip-exp3) ===")

    # Step 5: Plots
    if not args.skip_plots:
        print("\n=== Step 5: Generate plots ===")
        from architecture_refinement.paper3.plotting import plot1, plot2, plot3, plot4

        p1 = plot1(exp1_dir, output_path=root / "plot1_proxy_landscape.pdf")
        print(f"  Plot 1: {p1}")
        try:
            p2 = plot2(exp2_dir, output_path=root / "plot2_robustness_curves.pdf", experiment3_dir=exp3_dir)
            print(f"  Plot 2: {p2}")
        except Exception as e:
            print(f"  Plot 2: {e}")
        if (exp3_dir / "experiment3_results.csv").exists():
            try:
                p3 = plot3(exp3_dir, output_path=root / "plot3_proxy_plane.pdf")
                print(f"  Plot 3: {p3}")
            except Exception as e:
                print(f"  Plot 3: {e}")
            try:
                p4 = plot4(exp3_dir, output_path=root / "plot4_architecture_comparison.pdf")
                print(f"  Plot 4: {p4}")
            except Exception as e:
                print(f"  Plot 4: {e}")
        else:
            print("  Plot 3 & 4: Skipped (Experiment 3 results not found)")
    else:
        print("\n=== Step 5: Skipped (--skip-plots) ===")

    print(f"\nPaper 3 outputs: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
