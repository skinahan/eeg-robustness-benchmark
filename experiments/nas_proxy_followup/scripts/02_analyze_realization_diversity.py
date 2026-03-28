#!/usr/bin/env python3
"""WS-Flex → CfC realization diversity audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

_NPF_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT, _NPF_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.config_util import add_config_args, load_merged_configs, parse_config_paths, resolve_path
from src.wiring.cfc_realizer import realize_cfc_wiring
from src.wiring.realization_metrics import (
    compute_realization_descriptors,
    compute_realization_diversity_summary,
    pairwise_mask_hamming_distance,
    raw_graph_distance_distribution,
)


def _load_graph(graph_path: Path) -> nx.Graph:
    data = np.load(graph_path, allow_pickle=True)
    A = np.asarray(data["adjacency"])
    G = nx.from_numpy_array(A)
    return G


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    args = parser.parse_args()
    cfg = load_merged_configs(parse_config_paths(args))

    topo_csv = resolve_path(
        cfg,
        "topology_panel_path",
        "experiments/nas_proxy_followup/manifests/topology_panel.csv",
    )
    if not topo_csv.exists():
        raise FileNotFoundError(f"Run 01 first: missing {topo_csv}")

    schemes = list(
        cfg.get(
            "mapping_schemes",
            ["deterministic_baseline", "random_io_anchors", "degree_weighted_io_anchors"],
        )
    )
    n_rep = int(cfg.get("num_realizations_per_graph", 5))
    input_size = int(cfg.get("input_size", 16))
    output_size = int(cfg.get("output_size", 32))
    save_masks = bool(cfg.get("save_realized_masks", True))

    out_manifest = resolve_path(
        cfg,
        "realization_manifest",
        "experiments/nas_proxy_followup/manifests/topology_panel_realization.csv",
    )
    out_dir = resolve_path(
        cfg,
        "realization_output_dir",
        "experiments/nas_proxy_followup/outputs/realization_analysis",
    )
    mask_dir = out_dir / "masks"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(topo_csv)
    rows_out = []
    graphs = []
    for _, row in df.iterrows():
        gp = row.get("graph_path")
        if not gp or pd.isna(gp):
            continue
        gpath = _NPF_ROOT / str(gp) if not Path(gp).is_absolute() else Path(gp)
        if not gpath.exists():
            gpath = _REPO_ROOT / str(gp)
        G = _load_graph(gpath)
        graphs.append(G)
        tid = row["topology_id"]
        for scheme in schemes:
            for rep in range(n_rep):
                rseed = int(row["graph_seed"]) + rep * 100000 + hash(scheme) % 10000
                real = realize_cfc_wiring(
                    G,
                    scheme=scheme,
                    input_size=input_size,
                    output_size=output_size,
                    realization_seed=rseed,
                )
                desc = compute_realization_descriptors(real)
                mask_path = ""
                if save_masks:
                    mp = mask_dir / f"{tid}_{scheme}_rep{rep}.npz"
                    np.savez_compressed(
                        mp,
                        hidden_mask=real["hidden_mask"],
                        topology_id=tid,
                        scheme=scheme,
                        rep=rep,
                    )
                    try:
                        mask_path = str(mp.relative_to(_NPF_ROOT))
                    except ValueError:
                        mask_path = str(mp)

                rows_out.append(
                    {
                        "topology_id": tid,
                        "mapping_scheme": scheme,
                        "realization_rep": rep,
                        "realization_seed": rseed,
                        "graph_path": row["graph_path"],
                        "realized_hidden_mask_path": mask_path,
                        "density_hidden_block": desc["density_hidden_block"],
                        "n_input_anchors": desc["n_input_anchors"],
                        "n_output_anchors": desc["n_output_anchors"],
                        "avg_shortest_path_input_to_output": desc["avg_shortest_path_input_to_output"],
                    }
                )

    rdf = pd.DataFrame(rows_out)
    rdf.to_csv(out_manifest, index=False)

    raw_dist = raw_graph_distance_distribution(graphs)
    summary = {
        "raw_graph_dist_mean": float(np.mean(raw_dist)) if raw_dist.size else None,
        "raw_graph_dist_std": float(np.std(raw_dist)) if raw_dist.size else None,
        "by_scheme": compute_realization_diversity_summary(
            df, rdf, scheme_col="mapping_scheme", mask_root=_NPF_ROOT
        ),
    }
    summary_path = out_dir / "realization_diversity_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # pairwise distances sample: realized hidden masks within each scheme
    dist_rows = []
    for scheme in schemes:
        sub = rdf[rdf["mapping_scheme"] == scheme]
        if len(sub) < 2:
            continue
        masks = []
        for p in sub["realized_hidden_mask_path"]:
            if not p or pd.isna(p):
                continue
            pp = Path(str(p))
            if not pp.is_absolute():
                pp = _NPF_ROOT / pp
            if not pp.exists():
                pp = _REPO_ROOT / str(p)
            if pp.exists():
                masks.append(np.load(pp)["hidden_mask"])
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                dist_rows.append(
                    {
                        "mapping_scheme": scheme,
                        "pair_hamming": pairwise_mask_hamming_distance(masks[i], masks[j]),
                    }
                )
    pd.DataFrame(dist_rows).to_csv(out_dir / "raw_vs_realized_distance.csv", index=False)

    print(f"Wrote {len(rdf)} realization rows to {out_manifest}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
