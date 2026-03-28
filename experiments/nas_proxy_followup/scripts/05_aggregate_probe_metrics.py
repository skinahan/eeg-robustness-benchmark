#!/usr/bin/env python3
"""Aggregate probe JSON metrics into topology-level CSV."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
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

    manifest_path = resolve_path(
        cfg,
        "probe_run_manifest",
        "experiments/nas_proxy_followup/manifests/probe_run_manifest.csv",
    )
    out_dir = resolve_path(cfg, "probe_output_dir", "experiments/nas_proxy_followup/outputs/probe_runs")
    out_csv = out_dir / "probe_metrics_aggregated.csv"

    df = pd.read_csv(manifest_path)
    groups = defaultdict(list)
    for _, row in df.iterrows():
        if row.get("status") != "ok":
            continue
        tid = row["topology_id"]
        scheme = row["mapping_scheme"]
        pseed = int(row["probe_seed"])
        pat = out_dir / f"probe_metrics_{tid}_{scheme}_s{pseed}.json"
        if not pat.exists():
            pat = _NPF_ROOT / f"probe_metrics_{tid}_{scheme}_s{pseed}.json"
        if not pat.exists():
            continue
        with open(pat, encoding="utf-8") as f:
            m = json.load(f)
        groups[(tid, scheme)].append(m)

    rows = []
    for (tid, scheme), metrics_list in groups.items():
        if not metrics_list:
            continue
        agg = {"topology_id": tid, "mapping_scheme": scheme, "probe_seed_count": len(metrics_list)}
        keys = set()
        for m in metrics_list:
            keys.update(m.keys())
        for k in keys:
            vals = [float(m[k]) for m in metrics_list if k in m and isinstance(m[k], (int, float))]
            if vals:
                agg[f"{k}_mean"] = float(np.mean(vals))
                agg[f"{k}_std"] = float(np.std(vals))
        rows.append(agg)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
