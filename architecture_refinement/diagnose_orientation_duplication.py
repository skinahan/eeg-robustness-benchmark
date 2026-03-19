#!/usr/bin/env python3
"""
Diagnose why orient_ro and orient_sym produce identical results.
Traces: path generation, result file lookup, model registration.
"""
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

def main():
    from utils import short_run_id, create_output_path

    print("=== 1. Path generation (create_output_path) ===")
    for name in ["orient_ro", "orient_sym"]:
        h = short_run_id(name)
        path = create_output_path(name, 202618, 1, "0train", "test_perturb", session_type="CrossSessionEvaluation")
        print(f"  {name}: short_id={h}")
        print(f"    path={path}")
    print()

    print("=== 2. Result file lookup (_find_result_files) ===")
    results_root = _REPO / "results"
    if not results_root.exists():
        print(f"  results/ does not exist")
    else:
        from architecture_refinement.analyze_plot2_results import _find_result_files
        for name in ["orient_ro", "orient_sym"]:
            files = _find_result_files(_REPO, name)
            print(f"  {name}: found {len(files)} files")
            for f in files[:3]:
                print(f"    {f}")
            if len(files) > 3:
                print(f"    ... and {len(files)-3} more")
    print()

    print("=== 3. Check if both models find the SAME files ===")
    if results_root.exists():
        from architecture_refinement.analyze_plot2_results import _find_result_files
        ro_files = set(str(p) for p in _find_result_files(_REPO, "orient_ro"))
        sym_files = set(str(p) for p in _find_result_files(_REPO, "orient_sym"))
        overlap = ro_files & sym_files
        print(f"  orient_ro files: {len(ro_files)}")
        print(f"  orient_sym files: {len(sym_files)}")
        print(f"  OVERLAP (same files): {len(overlap)}")
        if overlap:
            print("  *** BUG: Both models are reading the same result files! ***")
            for p in overlap:
                print(f"    {p}")
    print()

    print("=== 4. Actual result directories on disk ===")
    if results_root.exists():
        for name in ["orient_ro", "orient_sym"]:
            short_id = short_run_id(name)
            found = []
            for p in results_root.rglob("*.csv"):
                path_str = str(p).replace("\\", "/")
                parts = path_str.split("/")
                if "test_perturb" in parts or "test_perturb" in path_str:
                    if short_id in parts:
                        found.append(p)
            found = list(set(found))
            print(f"  {name}: {len(found)} CSV files in paths with short_id")
            for fp in found[:2]:
                print(f"    {fp}")
    else:
        print("  results/ does not exist")
    print()

    print("=== 5. Model registry factory closure ===")
    # Simulate what happens when we register from pilot dir
    pilot_dir = _REPO / "architecture_refinement" / "outputs" / "orientation_sensitivity"
    if (pilot_dir / "selected_architectures").exists():
        import json
        for name in ["orient_ro", "orient_sym"]:
            p = pilot_dir / "selected_architectures" / f"{name}.json"
            if p.exists():
                arch = json.load(p.open())
                orient = arch.get("hidden_edge_orientation", "?")
                print(f"  {name}.json: hidden_edge_orientation={orient}")
    else:
        print("  Pilot dir selected_architectures not found (run experiment first)")

if __name__ == "__main__":
    main()
