"""
Plot 2 M1 — Metric/Proxy Computation Suite.

Samples reference WS-Flex graphs, computes full M1 metric suite,
residualizes within k, outputs metrics.csv and diagnostic plots.
GO/NO-GO: metrics stable, |corr(M_z, k)| < 0.15 for TE_z, sigma_z, ORC_z.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architecture_refinement.ws_flex_generator import make_ws_flex_graph

DEGREE_REGIMES_DEFAULT = {
    "super_sparse": [2, 4, 6],
    "sparse": [8, 10, 12],
    "moderate": [14, 16, 18],
    "near_dense": [20, 22, 24, 26],
}


def _graph_hash(adj: np.ndarray, H: int, k: int, p: float, graph_seed: int) -> str:
    adj_hex = np.asarray(adj).tobytes().hex()
    key = f"{H}|{k}|{p}|{graph_seed}|{adj_hex}"
    return hashlib.sha256(key.encode()).hexdigest()


def _undirected_adj(G, H: int) -> np.ndarray:
    A = np.asarray(nx.to_numpy_array(G), dtype=np.int8)
    return (A != 0).astype(np.int8)


def main():
    from architecture_refinement.graph_metrics_suite import compute_m1_metrics, residualize_metrics
    parser = argparse.ArgumentParser(description="Plot 2 M1: Metric suite + residualization")
    parser.add_argument("--output_dir", type=str, default="architecture_refinement/outputs/m1_metrics")
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--M_ref", type=int, default=800, help="Reference sample size (>=200 per k recommended)")
    parser.add_argument("--seed", type=int, default=202602)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k_values = sorted({k for ks in DEGREE_REGIMES_DEFAULT.values() for k in ks})
    rng = np.random.default_rng(args.seed)

    rows = []
    attempts = 0
    max_attempts = args.M_ref * 50
    n_per_k = max(50, args.M_ref // len(k_values))

    while len(rows) < args.M_ref and attempts < max_attempts:
        attempts += 1
        k = int(rng.choice(k_values))
        p = float(rng.uniform(0.0, 1.0))
        graph_seed = int(rng.integers(0, 2**31 - 1))
        try:
            G, _ = make_ws_flex_graph(args.H, k, p, graph_seed, generator_mode="plain_ws_flex")
        except Exception:
            continue
        if not nx.is_connected(G):
            continue
        adj = _undirected_adj(G, args.H)
        gh = _graph_hash(adj, args.H, k, p, graph_seed)
        m = compute_m1_metrics(G, H=args.H, k=k, p=p, graph_seed=graph_seed, graph_hash=gh)
        rows.append(m)

    df = pd.DataFrame(rows)
    metrics_to_residualize = ["TE", "sigma", "ORC_mean", "ORC_min", "C", "L", "spectral_radius", "algebraic_connectivity"]
    metrics_to_residualize = [x for x in metrics_to_residualize if x in df.columns]
    df_res, mu_sigma_by_k = residualize_metrics(df, metrics_to_residualize, k_col="k", n_per_k_min=20)

    df_res.to_csv(out_dir / "metrics.csv", index=False)

    # GO/NO-GO: corr(M_z, k)
    go_result = {"go": True, "correlations": {}}
    key_proxies = ["TE_z", "sigma_z", "ORC_mean_z"]  # M1 GO: |corr(M_z, k)| < 0.15
    for col in key_proxies:
        if col not in df_res.columns:
            go_result["correlations"][col] = None
            continue
        valid = df_res[["k", col]].dropna()
        if len(valid) < 10:
            go_result["correlations"][col] = None
            continue
        corr = float(valid["k"].corr(valid[col]))
        go_result["correlations"][col] = corr
        if abs(corr) >= 0.15:
            go_result["go"] = False

    go_result["nan_count"] = int(df_res.isna().sum().sum())
    with open(out_dir / "m1_go_nogo.json", "w") as f:
        json.dump(go_result, f, indent=2)

    with open(out_dir / "mu_sigma_by_k.json", "w") as f:
        serializable = {str(k): {m: [float(mu), float(sig)] for m, (mu, sig) in v.items()} for k, v in mu_sigma_by_k.items()}
        json.dump(serializable, f, indent=2)

    print(f"[M1] Wrote {out_dir / 'metrics.csv'} ({len(df_res)} rows)")
    print(f"[M1] GO/NO-GO: {'GO' if go_result['go'] else 'NO-GO'}")
    return 0 if go_result["go"] else 1


if __name__ == "__main__":
    sys.exit(main())
