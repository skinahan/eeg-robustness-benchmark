#!/usr/bin/env python3
"""Merge manifests and metric CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_NPF_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT, _NPF_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.analysis.merge_results import merge_all_results
from src.config_util import add_config_args, load_merged_configs, parse_config_paths, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    args = parser.parse_args()
    cfg = load_merged_configs(parse_config_paths(args))

    topo = resolve_path(cfg, "topology_panel_path", "experiments/nas_proxy_followup/manifests/topology_panel.csv")
    real = resolve_path(cfg, "realization_manifest", "experiments/nas_proxy_followup/manifests/topology_panel_realization.csv")
    probe = resolve_path(
        cfg,
        "probe_metrics_aggregated",
        "experiments/nas_proxy_followup/outputs/probe_runs/probe_metrics_aggregated.csv",
    )
    down = resolve_path(
        cfg,
        "downstream_metrics_csv",
        "experiments/nas_proxy_followup/outputs/downstream_runs/downstream_metrics.csv",
    )
    pert = resolve_path(
        cfg,
        "perturbation_output",
        "experiments/nas_proxy_followup/outputs/evaluation/local_perturbation_results.csv",
    )

    merged = merge_all_results(str(topo), str(real), str(probe), str(down), str(pert))
    out_dir = resolve_path(cfg, "analysis_output_dir", "experiments/nas_proxy_followup/outputs/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "merged_run_table.csv", index=False)
    if "topology_id" in merged.columns:
        merged.groupby("topology_id", dropna=False).mean(numeric_only=True).to_csv(out_dir / "topology_level_summary.csv")
    print(f"Wrote {out_dir / 'merged_run_table.csv'}")


if __name__ == "__main__":
    main()
