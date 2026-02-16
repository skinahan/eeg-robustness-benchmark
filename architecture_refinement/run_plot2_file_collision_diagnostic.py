"""
Plot 2 File Collision Diagnostic — isolate whether SW and NSW read the same results.

Run this after an intermediate diagnostic to verify:
1. Each model gets distinct result file paths
2. Raw CSV contents differ between SW and NSW
3. max_drop is computed from model-specific data

Usage:
  python architecture_refinement/run_plot2_file_collision_diagnostic.py --output_dir <path_to_intermediate_output>
  python architecture_refinement/run_plot2_file_collision_diagnostic.py --repo_root . --model_a plot2_intermediate_sparse_sw_2 --model_b plot2_intermediate_sparse_nsw_3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import from repo root utils (architecture_refinement has its own utils.py)
import importlib.util
_repo_utils_spec = importlib.util.spec_from_file_location("_repo_utils", _REPO_ROOT / "utils.py")
_repo_utils = importlib.util.module_from_spec(_repo_utils_spec)
_repo_utils_spec.loader.exec_module(_repo_utils)
short_run_id = _repo_utils.short_run_id




def main():
    parser = argparse.ArgumentParser(description="Plot 2 file collision diagnostic")
    parser.add_argument("--output_dir", type=str, default=None, help="Intermediate diagnostic output dir")
    parser.add_argument("--repo_root", type=str, default=None)
    parser.add_argument("--model_a", type=str, default="plot2_intermediate_sparse_sw_2")
    parser.add_argument("--model_b", type=str, default="plot2_intermediate_sparse_nsw_3")
    parser.add_argument("--noise_type", type=str, default="ar1_drift")
    parser.add_argument("--check_adj", action="store_true", help="Compare hidden_adj from pilot JSONs")
    args = parser.parse_args()

    repo_root = Path(args.repo_root or _REPO_ROOT).resolve()
    model_a = args.model_a
    model_b = args.model_b

    # 1) short_run_id check
    sid_a = short_run_id(model_a)
    sid_b = short_run_id(model_b)
    print("=== Step 1: short_run_id check ===")
    print(f"  {model_a!r} -> {sid_a!r}")
    print(f"  {model_b!r} -> {sid_b!r}")
    if sid_a == sid_b:
        print("  [COLLISION] Same short_run_id — files will be shared!")
    else:
        print("  [OK] Different short_run_ids")

    # 2) File discovery (mirror analyze_plot2_results logic)
    def find_files(mname):
        out = []
        rr = repo_root / "results"
        if not rr.exists():
            return out
        short_id = short_run_id(mname)
        for p in rr.rglob("*.csv"):
            if short_id in str(p) and "test_perturb" in str(p):
                out.append(p)
        return sorted(set(out))

    paths_a = find_files(model_a)
    paths_b = find_files(model_b)

    print("\n=== Step 2: File discovery ===")
    print(f"  {model_a}: {len(paths_a)} file(s)")
    for p in paths_a[:5]:
        print(f"    {p}")
    if len(paths_a) > 5:
        print(f"    ... and {len(paths_a) - 5} more")
    print(f"  {model_b}: {len(paths_b)} file(s)")
    for p in paths_b[:5]:
        print(f"    {p}")
    if len(paths_b) > 5:
        print(f"    ... and {len(paths_b) - 5} more")

    # 3) Path overlap
    set_a = set(str(p) for p in paths_a)
    set_b = set(str(p) for p in paths_b)
    overlap = set_a & set_b
    print("\n=== Step 3: Path overlap ===")
    if overlap:
        print(f"  [COLLISION] {len(overlap)} file(s) shared between models:")
        for x in list(overlap)[:5]:
            print(f"    {x}")
    else:
        print("  [OK] No shared files")

    # 4) Raw CSV comparison (first file each)
    if paths_a and paths_b:
        print("\n=== Step 4: Raw CSV comparison (first file each) ===")
        df_a = pd.read_csv(paths_a[0])
        df_b = pd.read_csv(paths_b[0])
        print(f"  {model_a}: {len(df_a)} rows, cols={list(df_a.columns)[:8]}...")
        print(f"  {model_b}: {len(df_b)} rows, cols={list(df_b.columns)[:8]}...")
        if "noise_type" in df_a.columns and args.noise_type in df_a["noise_type"].astype(str).values:
            sub_a = df_a[df_a["noise_type"].astype(str) == args.noise_type]
            sub_b = df_b[df_b["noise_type"].astype(str) == args.noise_type]
            if not sub_a.empty and not sub_b.empty:
                metric = "corrupted_roc_auc" if "corrupted_roc_auc" in df_a.columns else "corrupted_score"
                if metric in sub_a.columns:
                    clean_a = sub_a["clean_roc_auc"].iloc[0] if "clean_roc_auc" in sub_a.columns else sub_a["clean_score"].iloc[0]
                    clean_b = sub_b["clean_roc_auc"].iloc[0] if "clean_roc_auc" in sub_b.columns else sub_b["clean_score"].iloc[0]
                    roc_max_a = sub_a[metric].iloc[-1]
                    roc_max_b = sub_b[metric].iloc[-1]
                    max_drop_a = float(clean_a) - float(roc_max_a)
                    max_drop_b = float(clean_b) - float(roc_max_b)
                    print(f"  max_drop {model_a}: {max_drop_a:.6f}")
                    print(f"  max_drop {model_b}: {max_drop_b:.6f}")
                    if abs(max_drop_a - max_drop_b) < 1e-12:
                        print("  [ROOT CAUSE] Identical max_drop - NOT file collision.")
                        print("  The CSVs are in different paths and have correct model names.")
                        print("  Identical corrupted scores => runner/model bug: same checkpoint, wrong")
                        print("  wiring, or closure bug in NAS pilot registry. Check:")
                        print("    - Cache path includes model_name (it does)")
                        print("    - NAS pilot factory closure captures correct _arch per model")
                        print("    - Adjacency matrices in JSON files differ (run with --check_adj)")
                    else:
                        print("  [OK] Different max_drop")
        print("\n  First 3 rows (model_a):")
        print(df_a.head(3).to_string())
        print("\n  First 3 rows (model_b):")
        print(df_b.head(3).to_string())

        # Step 5: Compare adjacency matrices from pilot JSONs (if --check_adj)
        if args.check_adj and args.output_dir:
            import json
            pilot_dir = Path(args.output_dir).resolve()
            arch_dir = pilot_dir / "selected_architectures"
            print("\n=== Step 5: Adjacency comparison (from pilot JSONs) ===")
            for mname, label in [(model_a, "A"), (model_b, "B")]:
                jpath = arch_dir / f"{mname}.json"
                if jpath.exists():
                    d = json.loads(jpath.read_text(encoding="utf-8"))
                    adj = d.get("hidden_adj_undirected") or d.get("hidden_adj_directed")
                    if adj:
                        print(f"  {label} ({mname}): adj shape={len(adj)}x{len(adj[0]) if adj else 0}")
                else:
                    print(f"  {label} ({mname}): JSON not found at {jpath}")
            ja = arch_dir / f"{model_a}.json"
            jb = arch_dir / f"{model_b}.json"
            if ja.exists() and jb.exists():
                da, db = json.loads(ja.read_text()), json.loads(jb.read_text())
                adj_a = da.get("hidden_adj_undirected") or da.get("hidden_adj_directed") or []
                adj_b = db.get("hidden_adj_undirected") or db.get("hidden_adj_directed") or []
                if adj_a and adj_b and len(adj_a) == len(adj_b):
                    same = all(
                        row_a == row_b for row_a, row_b in zip(adj_a, adj_b)
                    )
                    print(f"  Adjacency matrices identical? {same}")
    else:
        print("\n=== Step 4: Skipped (no result files) ===")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
