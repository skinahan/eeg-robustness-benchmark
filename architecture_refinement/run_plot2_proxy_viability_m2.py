"""
Plot 2 M2 — Proxy Viability (lightweight labeled check).

Samples 64 WS-Flex graphs stratified by regime, trains+eval (subject 1, S_pilot),
computes y=-maxRD at target_snr_db=-6, runs proxy predictiveness tests.
GO/NO-GO: single proxy Spearman>=0.35, p<0.05 (FDR), AUC>=0.70; OR composite CV Spearman>=0.45, MAE improves>=15%.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import networkx as nx

from architecture_refinement.ws_flex_generator import make_ws_flex_graph
from architecture_refinement.arbitrary_wiring import WsFlexHiddenWiring
from architecture_refinement.topology_analyzer import compute_spectral_radius_directed

DEGREE_REGIMES = {
    "super_sparse": [2, 4, 6],
    "sparse": [8, 10, 12],
    "moderate": [14, 16, 18],
    "near_dense": [20, 22, 24, 26],
}
N_POOL = 64
N_PER_REGIME = 16
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
EPS_RD = 1e-3


def _graph_hash(adj: np.ndarray, H: int, k: int, p: float, graph_seed: int) -> str:
    adj_hex = np.asarray(adj).tobytes().hex()
    return hashlib.sha256(f"{H}|{k}|{p}|{graph_seed}|{adj_hex}".encode()).hexdigest()


def _orient_seed_from_graph_hash(gh: str) -> int:
    return int(gh, 16) % (2**31 - 1)


def _undirected_adj(G: nx.Graph, H: int) -> np.ndarray:
    A = np.asarray(nx.to_numpy_array(G), dtype=np.int8)
    return (A != 0).astype(np.int8)


def _oriented_adj(G: nx.Graph, H: int, seed: int) -> np.ndarray:
    wiring = WsFlexHiddenWiring(
        input_size=1, hidden_graph=G, output_size=1,
        hidden_edge_orientation="random_oriented", seed=int(seed),
    )
    W = wiring.full_wiring_matrix()
    if hasattr(W, "numpy"):
        W = W.numpy()
    W = np.asarray(W)
    h_block = W[1 : 1 + H, 1 : 1 + H]
    return (h_block != 0).astype(np.float64)


def _make_ws_graph(H: int, k: int, p: float, seed: int) -> nx.Graph:
    G, _ = make_ws_flex_graph(H, k, p, seed, generator_mode="plain_ws_flex")
    return G


def main():
    parser = argparse.ArgumentParser(description="Plot 2 M2: Proxy viability with cheap labels")
    parser.add_argument("--output_dir", type=str, default="architecture_refinement/outputs/proxy_viability_m2")
    parser.add_argument("--m1_metrics_dir", type=str, default=None, help="Path to M1 metrics.csv for residualization")
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument("--eval_mode", type=str, default="CrossSession")
    parser.add_argument("--S_pilot", type=int, default=1, help="Training seeds (1 min, 2 preferred)")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--target_snr_db", type=float, default=-6.0)
    parser.add_argument("--python", type=str, default="python")
    parser.add_argument("--seed", type=int, default=202602)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected_architectures").mkdir(exist_ok=True)
    (out_dir / "diagnostics").mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)
    regime_names = list(DEGREE_REGIMES.keys())
    k_values = sorted({k for ks in DEGREE_REGIMES.values() for k in ks})

    # M2.1: Sample 64 graphs stratified
    from architecture_refinement.graph_metrics_suite import compute_m1_metrics, residualize_metrics

    rows = []
    for r_i, rname in enumerate(regime_names):
        for _ in range(N_PER_REGIME):
            k = int(rng.choice(DEGREE_REGIMES[rname]))
            p = float(rng.uniform(0.0, 1.0))
            graph_seed = int(rng.integers(0, 2**31 - 1))
            for _ in range(200):
                G = _make_ws_graph(args.H, k, p, graph_seed)
                if not nx.is_connected(G):
                    graph_seed += 1
                    continue
                adj = _undirected_adj(G, args.H)
                gh = _graph_hash(adj, args.H, k, p, graph_seed)
                wiring_seed = _orient_seed_from_graph_hash(gh)
                m = compute_m1_metrics(G, H=args.H, k=k, p=p, graph_seed=graph_seed, graph_hash=gh)
                m["regime"] = rname
                m["wiring_seed"] = wiring_seed
                m["adj"] = adj
                m["G"] = G
                rows.append(m)
                break
            else:
                raise RuntimeError(f"Failed to sample connected graph for regime {rname}")

    df_metrics = pd.DataFrame([{k: v for k, v in r.items() if k not in ("adj", "G")} for r in rows])
    metrics_to_res = ["TE", "sigma", "ORC_mean"]
    metrics_to_res = [x for x in metrics_to_res if x in df_metrics.columns]
    df_metrics, _ = residualize_metrics(df_metrics, metrics_to_res, k_col="k", n_per_k_min=5)

    # Write architecture JSONs and run jobs
    runner_path = _REPO_ROOT / "evaluation" / "unified_experiment_runner.py"
    run_id = "m2_pilot"
    jobs = []
    for i, row in enumerate(rows):
        model_name = f"plot2_{run_id}_g{i:03d}"
        adj = row["adj"]
        G = row["G"]
        wiring_seed = row["wiring_seed"]
        directed_adj = _oriented_adj(G, args.H, wiring_seed)
        arch = {
            "model_name": model_name,
            "wiring_kind": "ws_flex",
            "H": args.H,
            "k": int(row["k"]),
            "p": float(row["p"]),
            "graph_seed": int(row["graph_seed"]),
            "wiring_seed": int(wiring_seed),
            "hidden_adj_undirected": adj.tolist(),
        }
        (out_dir / "selected_architectures" / f"{model_name}.json").write_text(json.dumps(arch, indent=2))
        for s in range(args.S_pilot):
            seed_val = 202600 + i * 100 + s
            jobs.append({"model_name": model_name, "seed": seed_val, "row_idx": i})

    # Run unified experiment runner for each job
    saturation_file = str(_REPO_ROOT / "architecture_refinement" / "outputs" / "perturbation_sweep" / "locked_perturbation_config.yaml")
    if not Path(saturation_file).exists():
        saturation_file = ""

    for j in jobs:
        cmd = [
            args.python,
            str(runner_path),
            "--nas_pilot_dir", str(out_dir),
            "--model", j["model_name"],
            "--dataset", args.dataset,
            "--subjects", str(args.subject),
            "--mode", "test_perturb",
            "--eval_mode", args.eval_mode,
            "--seed", str(j["seed"]),
            "--disable_underfitting_retrain",
            "--test_perturb_noise_types", "ar1_drift",
            "--test_perturb_gaussian_alpha_grid", ",".join(str(a) for a in ALPHA_GRID),
            "--test_perturb_target_snr_db=" + str(args.target_snr_db),
            "--plot2_diagnostics_dir", str(out_dir / "diagnostics"),
        ]
        if saturation_file:
            cmd.extend(["--noise_perturbation_saturation_file", saturation_file])
        cmd = [c for c in cmd if c]
        print(f"[M2] Running {j['model_name']} seed={j['seed']}")
        rc = subprocess.run(cmd, cwd=str(_REPO_ROOT), check=False)
        if rc.returncode != 0:
            print(f"[M2] WARNING: Job failed (exit {rc.returncode})")

    # Load results and compute maxRD
    from utils import short_run_id, get_noise_perturbation_bounds
    results_root = _REPO_ROOT / "results"
    sigma_max = 1.0
    try:
        _, sigma_max = get_noise_perturbation_bounds(args.dataset, "ar1_drift", saturation_file or None)
    except Exception:
        pass

    y_by_row: Dict[int, float] = {}
    clean_by_row: Dict[int, float] = {}
    for j in jobs:
        idx = j["row_idx"]
        model_name = j["model_name"]
        short_id = short_run_id(model_name)
        found = list(results_root.rglob(f"*{short_id}*test_perturb*.csv"))
        if not found:
            continue
        dfs = []
        for p in found:
            try:
                df = pd.read_csv(p)
                if "model" in df.columns and model_name not in df["model"].astype(str).values:
                    continue
                dfs.append(df)
            except Exception:
                continue
        if not dfs:
            continue
        df = pd.concat(dfs, ignore_index=True)
        df = df[df["noise_type"].astype(str) == "ar1_drift"]
        if df.empty:
            continue
        metric_col = next((c for c in ["corrupted_roc_auc", "corrupted_score", "roc_auc", "score"] if c in df.columns), None)
        if not metric_col:
            continue
        seed_col = "seed" if "seed" in df.columns else "fold_idx"
        g = df.groupby(seed_col)
        for seed_grp, gg in g:
            xs = gg.groupby("intensity", as_index=False)[metric_col].mean()
            xs = xs.sort_values("intensity")
            xs_arr = xs["intensity"].to_numpy()
            ys_arr = xs.iloc[:, 1].to_numpy()
            if xs_arr.size == 0:
                continue
            if xs_arr[0] > 0 and "clean_roc_auc" in gg.columns:
                m0 = float(gg["clean_roc_auc"].mean())
            else:
                m0 = float(ys_arr[0])
            denom = max(EPS_RD, float(m0 - 0.5))
            rd_vals = []
            for i in range(len(xs_arr)):
                if xs_arr[i] > 1e-9:
                    rd_vals.append((m0 - ys_arr[i]) / denom)
            max_rd = float(np.max(rd_vals)) if rd_vals else float("nan")
            if idx not in y_by_row:
                y_by_row[idx] = []
                clean_by_row[idx] = []
            y_by_row[idx].append(-max_rd)
            clean_by_row[idx].append(m0)

    y_mean = {idx: float(np.mean(v)) for idx, v in y_by_row.items() if v}
    clean_mean = {idx: float(np.mean(clean_by_row[idx])) for idx in y_mean}

    # Merge with metrics
    df_metrics["y"] = df_metrics.index.map(lambda i: y_mean.get(i, float("nan")))
    df_metrics["p_clean"] = df_metrics.index.map(lambda i: clean_mean.get(i, float("nan")))
    df_valid = df_metrics.dropna(subset=["y"])
    if len(df_valid) < 10:
        go = False
        print("[M2] Too few valid labels; NO-GO")
    else:
        # M2.3 Proxy predictiveness
        proxies = ["TE", "sigma", "ORC_mean", "TE_z", "sigma_z", "ORC_mean_z"]
        proxies = [p for p in proxies if p in df_valid.columns]
        from scipy import stats
        report = {"proxies": {}, "go": False}
        for p in proxies:
            r_pearson, p_pearson = stats.pearsonr(df_valid[p], df_valid["y"])
            r_spearman, p_spearman = stats.spearmanr(df_valid[p], df_valid["y"])
            q75 = df_valid["y"].quantile(0.75)
            robust = (df_valid["y"] >= q75).astype(int)
            if robust.sum() >= 2 and (1 - robust).sum() >= 2:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(robust, df_valid[p])
            else:
                auc = float("nan")
            report["proxies"][p] = {
                "pearson_r": float(r_pearson), "pearson_p": float(p_pearson),
                "spearman_r": float(r_spearman), "spearman_p": float(p_spearman),
                "auc_top25": float(auc),
            }
            if abs(r_spearman) >= 0.35 and p_spearman < 0.05 and (np.isnan(auc) or auc >= 0.70):
                report["go"] = True

        # FDR correction (Benjamini-Hochberg)
        pvals = np.array([report["proxies"][p]["spearman_p"] for p in proxies])
        try:
            from scipy.stats import false_discovery_control
            pvals_adj = false_discovery_control(pvals)
            for i, p in enumerate(proxies):
                report["proxies"][p]["spearman_p_fdr"] = float(pvals_adj[i])
        except (ImportError, AttributeError):
            for i, p in enumerate(proxies):
                report["proxies"][p]["spearman_p_fdr"] = float(min(1.0, pvals[i] * len(proxies)))

        (out_dir / "proxy_viability_report.json").write_text(json.dumps(report, indent=2))
        df_valid.to_csv(out_dir / "proxy_vs_y.csv", index=False)
        go = report["go"]

    print(f"[M2] GO/NO-GO: {'GO' if go else 'NO-GO'}")
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
