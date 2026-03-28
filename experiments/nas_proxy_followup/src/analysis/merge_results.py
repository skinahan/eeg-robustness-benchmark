"""Merge topology, realization, probe, downstream, perturbation tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def merge_all_results(
    topology_manifest_path: str,
    realization_manifest_path: str,
    probe_metrics_path: str,
    downstream_metrics_path: str,
    perturbation_metrics_path: str,
) -> pd.DataFrame:
    tdf = pd.read_csv(topology_manifest_path)

    if Path(probe_metrics_path).exists():
        out = pd.read_csv(probe_metrics_path)
        out = out.merge(tdf, on="topology_id", how="left")
    else:
        out = tdf.copy()

    if Path(realization_manifest_path).exists():
        rdf = pd.read_csv(realization_manifest_path)
        agg_kw = {}
        for c in ("density_hidden_block", "avg_shortest_path_input_to_output"):
            if c in rdf.columns:
                agg_kw[c] = "mean"
        if agg_kw and "mapping_scheme" in out.columns:
            r_small = rdf.groupby(["topology_id", "mapping_scheme"], as_index=False).agg(agg_kw)
            out = out.merge(r_small, on=["topology_id", "mapping_scheme"], how="left")

    if Path(downstream_metrics_path).exists():
        ddf = pd.read_csv(downstream_metrics_path)
        keys = [k for k in ("topology_id", "mapping_scheme") if k in ddf.columns]
        if keys:
            out = out.merge(ddf, on=keys, how="left", suffixes=("", "_ds"))

    if Path(perturbation_metrics_path).exists():
        per = pd.read_csv(perturbation_metrics_path)
        keys = [k for k in ("topology_id", "mapping_scheme", "training_seed") if k in per.columns]
        if keys and all(k in out.columns for k in keys):
            out = out.merge(per, on=keys, how="left", suffixes=("", "_pe"))
    return out
