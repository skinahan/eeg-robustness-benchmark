"""
Collect experiment3-style rows for Q3 stratified pilot (newly trained G_strat models only).

Writes q3_stratified_experiment3_results.csv (mergeable with experiment3_results.csv schema).
Reuse rows (G1) are merged in analysis from the main experiment3 CSV using seeds from manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architecture_refinement.paper3.run_paper3_experiment3 import _collect_perturb_results


def collect_stratified_results(
    pilot_root: Path,
    output_csv: Path,
    repo_root: Path,
    dataset: str = "BNCI2014_001",
    noise_type: str = "ar1_drift",
) -> pd.DataFrame:
    manifest_path = pilot_root / "q3_stratified_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = manifest.get("groups", {})
    models: List[str] = list(groups.get("G_strat", []))
    seeds = list(manifest.get("seeds", []))
    if not models:
        return pd.DataFrame()

    plan_path = pilot_root / "q3_stratified_sample_plan.json"
    proxy_by_model: Dict[str, tuple] = {}
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        for it in plan.get("items", []):
            if it.get("source") == "train":
                proxy_by_model[it["model_name"]] = (
                    float(it["te_hat"]),
                    float(it["orc_hat"]),
                )

    arch_dir = pilot_root / "selected_architectures"
    rows: List[Dict[str, Any]] = []
    for model_name in models:
        te_hat, orc_hat = float("nan"), float("nan")
        if model_name in proxy_by_model:
            te_hat, orc_hat = proxy_by_model[model_name]
        else:
            ap = arch_dir / f"{model_name}.json"
            if ap.exists():
                arch = json.loads(ap.read_text(encoding="utf-8"))
                te_hat = float(arch.get("te_hat", float("nan")))
                orc_hat = float(arch.get("orc_hat", float("nan")))
        for seed in seeds:
            res = _collect_perturb_results(repo_root, dataset, model_name, seed, noise_type)
            if res:
                rows.append({
                    "model": model_name,
                    "group": "G_strat",
                    "seed": seed,
                    "te_hat": te_hat,
                    "orc_hat": orc_hat,
                    "clean_roc_auc": res["clean_roc_auc"],
                    "RD_max": res["RD_max"],
                })

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_csv(output_csv, index=False)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Q3 stratified perturb results")
    parser.add_argument(
        "--pilot-root",
        type=str,
        default="architecture_refinement/outputs/paper3/q3_stratified_pilot",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="architecture_refinement/outputs/paper3/q3_stratified_pilot/q3_stratified_experiment3_results.csv",
    )
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument("--results-root", type=str, default=None)
    args = parser.parse_args()

    pilot = Path(args.pilot_root)
    out = Path(args.output_csv)
    if not pilot.is_absolute():
        pilot = _REPO_ROOT / pilot
    if not out.is_absolute():
        out = _REPO_ROOT / out
    root = Path(args.results_root) if args.results_root else _REPO_ROOT

    df = collect_stratified_results(pilot, out, root, dataset=args.dataset)
    print(f"[collect] Wrote {len(df)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
