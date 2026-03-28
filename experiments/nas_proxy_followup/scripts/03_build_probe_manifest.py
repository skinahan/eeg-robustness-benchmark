#!/usr/bin/env python3
"""Build probe subset indices and probe run manifest skeleton."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_NPF_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT, _NPF_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.config_util import add_config_args, load_merged_configs, parse_config_paths, resolve_path, repo_root_from_config
from src.probe.probe_dataset import build_probe_subset_indices, save_probe_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    args = parser.parse_args()
    cfg = load_merged_configs(parse_config_paths(args))
    repo_root = repo_root_from_config(cfg)

    dcfg = cfg.get("dataset")
    if not isinstance(dcfg, dict):
        dcfg = {}
    ds = cfg.get("dataset_name") or dcfg.get("dataset_name") or "BNCI2014_001"
    protocol = cfg.get("protocol") or dcfg.get("protocol", "cross_session")
    subject = int(cfg.get("subject", dcfg.get("subject", 1)))

    ps = cfg.get("probe_subset", {})
    max_examples = int(ps.get("max_examples", 128))
    stratified = bool(ps.get("stratified", True))
    seed = int(ps.get("probe_subset_seed", 123))
    rel_idx = ps.get(
        "save_indices_path",
        "experiments/nas_proxy_followup/manifests/probe_subset_indices.json",
    )
    indices_path = (repo_root / rel_idx).resolve() if not Path(rel_idx).is_absolute() else Path(rel_idx).resolve()

    indices = build_probe_subset_indices(
        str(ds),
        str(protocol),
        max_examples=max_examples,
        stratified=stratified,
        seed=seed,
        subject=subject,
    )
    save_probe_indices(
        indices_path,
        indices,
        {"dataset": ds, "protocol": protocol, "subject": subject},
    )
    print(f"Saved {len(indices)} indices to {indices_path}")

    topo_path = resolve_path(
        cfg,
        "topology_panel_path",
        "experiments/nas_proxy_followup/manifests/topology_panel.csv",
    )
    real_path = resolve_path(
        cfg,
        "realization_manifest",
        "experiments/nas_proxy_followup/manifests/topology_panel_realization.csv",
    )
    if not topo_path.exists():
        print(f"Warning: {topo_path} missing — skip probe run manifest")
        return

    tdf = pd.read_csv(topo_path)
    schemes = list(cfg.get("mapping_schemes", ["deterministic_baseline"]))
    if real_path.exists():
        rdf = pd.read_csv(real_path)
        schemes = sorted(rdf["mapping_scheme"].unique().tolist())

    probe_seeds = list(cfg.get("probe_seeds", [11, 22, 33]))
    rows = []
    for _, tr in tdf.iterrows():
        for scheme in schemes:
            for pseed in probe_seeds:
                rows.append(
                    {
                        "topology_id": tr["topology_id"],
                        "mapping_scheme": scheme,
                        "probe_seed": pseed,
                        "graph_path": tr.get("graph_path", ""),
                        "realization_path": "",
                        "probe_checkpoint_path": "",
                        "probe_log_path": "",
                        "status": "pending",
                    }
                )
    out_manifest = resolve_path(
        cfg,
        "probe_run_manifest",
        "experiments/nas_proxy_followup/manifests/probe_run_manifest.csv",
    )
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_manifest, index=False)
    print(f"Wrote probe manifest skeleton: {out_manifest} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
