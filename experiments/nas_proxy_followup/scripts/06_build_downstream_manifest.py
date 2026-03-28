#!/usr/bin/env python3
"""Build downstream run manifest from topology + schemes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_NPF_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT, _NPF_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.config_util import add_config_args, load_merged_configs, parse_config_paths, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    args = parser.parse_args()
    cfg = load_merged_configs(parse_config_paths(args))

    topo_path = resolve_path(
        cfg,
        "topology_panel_path",
        "experiments/nas_proxy_followup/manifests/topology_panel.csv",
    )
    agg_path = resolve_path(
        cfg,
        "probe_metrics_aggregated",
        "experiments/nas_proxy_followup/outputs/probe_runs/probe_metrics_aggregated.csv",
    )
    tdf = pd.read_csv(topo_path)
    pdf = pd.read_csv(agg_path) if agg_path.exists() else pd.DataFrame()

    schemes = list(cfg.get("mapping_schemes", ["deterministic_baseline"]))
    seeds = list(cfg.get("training_seeds", [101, 202, 303]))
    rows = []
    for _, tr in tdf.iterrows():
        for scheme in schemes:
            for ts in seeds:
                pm = ""
                if not pdf.empty:
                    m = pdf[(pdf["topology_id"] == tr["topology_id"]) & (pdf["mapping_scheme"] == scheme)]
                    if len(m) == 1:
                        pm = str(agg_path)
                rows.append(
                    {
                        "topology_id": tr["topology_id"],
                        "mapping_scheme": scheme,
                        "training_seed": ts,
                        "probe_metrics_path": pm,
                        "graph_path": tr.get("graph_path", ""),
                        "realization_path": "",
                        "checkpoint_path": "",
                        "log_path": "",
                        "status": "pending",
                    }
                )
    out = resolve_path(
        cfg,
        "downstream_run_manifest",
        "experiments/nas_proxy_followup/manifests/downstream_run_manifest.csv",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
