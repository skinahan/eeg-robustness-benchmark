"""
Paper 3 Q3 follow-up: stratified WS-Flex sample from proxy_pool.csv for dense proxy–robustness plot.

Stratification uses a 2D grid on the proxy plane (te_hat × orc_hat): equal-frequency marginal
bins on each axis, one sample per grid cell (n_bins_te * n_bins_orc == n_target; default 10×10=100).

Writes pilot JSONs under q3_stratified_pilot/, manifest with S=3 seeds, optional training via
the same unified runner as Experiment 2. Skips training for pool rows that match existing G1
architectures (same k, p, graph_seed as in experiment2 pilot).

Run after regenerating proxy_pool.csv with fixed Experiment 1 (graph_seed-aligned WS build).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architecture_refinement.metrics_te_orc import compute_paper3_proxies
from architecture_refinement.paper3.run_paper3_experiment2 import (
    ALPHA_GRID,
    DEFAULT_SATURATION,
    _run_unified_job,
)
from architecture_refinement.ws_flex_generator import build_plain_ws_flex

DEFAULT_N_TARGET = 100
DEFAULT_N_BINS_TE = 10
DEFAULT_N_BINS_ORC = 10
DEFAULT_STRAT_SEED = 20260324
DEFAULT_S = 3
BASE_SEED = 42


def _topology_key_from_arch(arch: Dict[str, Any]) -> Optional[Tuple[int, float, int]]:
    if arch.get("wiring_kind") != "ws_flex":
        return None
    k = arch.get("k")
    p = arch.get("p")
    gs = arch.get("graph_seed")
    if k is None or p is None or gs is None:
        return None
    if int(k) < 0 or float(p) < 0:
        return None
    return (int(k), float(p), int(gs))


def _topology_key_from_row(row: pd.Series) -> Tuple[int, float, int]:
    return (int(row["k"]), float(row["p"]), int(row["graph_seed"]))


def _load_g1_topology_map(exp2_arch_dir: Path) -> Dict[Tuple[int, float, int], str]:
    """Map (k, p, graph_seed) -> G1 model_name from proxy-selected JSONs."""
    out: Dict[Tuple[int, float, int], str] = {}
    if not exp2_arch_dir.exists():
        return out
    for p in sorted(exp2_arch_dir.glob("paper3_exp1_proxy_*.json")):
        arch = json.loads(p.read_text(encoding="utf-8"))
        key = _topology_key_from_arch(arch)
        if key is not None:
            out[key] = str(arch.get("model_name", p.stem))
    return out


def _g1_model_for_pool_row(
    row: pd.Series,
    g1_map: Dict[Tuple[int, float, int], str],
) -> Optional[str]:
    k, p, gs = int(row["k"]), float(row["p"]), int(row["graph_seed"])
    key = (k, p, gs)
    if key in g1_map:
        return g1_map[key]
    for (k2, p2, gs2), name in g1_map.items():
        if k == k2 and gs == gs2 and np.isclose(p, p2, rtol=0.0, atol=1e-12):
            return name
    return None


def stratified_sample_pool_2d(
    pool_df: pd.DataFrame,
    n_target: int,
    n_bins_te: int,
    n_bins_orc: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Stratify on the Paper 3 proxy plane: equal-frequency marginal bins on te_hat and orc_hat,
    then take (up to) one graph per grid cell (te_bin, orc_bin). Fills shortfall if some cells
    are empty. Requires n_bins_te * n_bins_orc == n_target.
    """
    df = pool_df.copy()
    df["te_hat"] = pd.to_numeric(df["te_hat"], errors="coerce")
    df["orc_hat"] = pd.to_numeric(df["orc_hat"], errors="coerce")
    df = df.dropna(subset=["te_hat", "orc_hat", "k", "p", "graph_seed"])
    df["proxy_score"] = 0.5 * (df["te_hat"] + np.abs(df["orc_hat"]))
    n = len(df)
    if n < n_target:
        raise ValueError(f"Pool has only {n} valid rows; need at least {n_target}")
    if n_bins_te * n_bins_orc != n_target:
        raise ValueError(
            f"n_bins_te * n_bins_orc ({n_bins_te * n_bins_orc}) must equal n_target ({n_target})"
        )

    df = df.reset_index(drop=True)
    te_order = df["te_hat"].to_numpy().argsort().argsort()
    orc_order = df["orc_hat"].to_numpy().argsort().argsort()
    df["te_bin"] = np.minimum(te_order * n_bins_te // max(n, 1), n_bins_te - 1).astype(np.int64)
    df["orc_bin"] = np.minimum(orc_order * n_bins_orc // max(n, 1), n_bins_orc - 1).astype(np.int64)

    picked_idx: List[int] = []
    seen: set[int] = set()
    for i in range(n_bins_te):
        for j in range(n_bins_orc):
            mask = (df["te_bin"] == i) & (df["orc_bin"] == j)
            idxs = [idx for idx in df.index[mask].tolist() if idx not in seen]
            if idxs:
                pick = int(rng.choice(idxs))
                picked_idx.append(pick)
                seen.add(pick)

    while len(picked_idx) < n_target:
        avail = [i for i in range(n) if i not in seen]
        if not avail:
            raise RuntimeError("Could not fill stratified sample to n_target (pool exhausted)")
        pick = int(rng.choice(avail))
        picked_idx.append(pick)
        seen.add(pick)

    return df.iloc[sorted(picked_idx)].reset_index(drop=True)


def _write_ws_flex_arch(
    *,
    H: int,
    model_name: str,
    k: int,
    p: float,
    graph_seed: int,
    te_hat: float,
    orc_hat: float,
    path: Path,
) -> None:
    G = build_plain_ws_flex(H, k, p, graph_seed)
    adj = nx.to_numpy_array(G, dtype=np.int8)
    adj = (adj != 0).astype(np.int8)
    n_edges = G.number_of_edges()
    arch = {
        "schema_version": 2,
        "model_name": model_name,
        "H": H,
        "wiring_kind": "ws_flex",
        "hidden_edge_orientation": "symmetric",
        "k": k,
        "p": p,
        "graph_seed": graph_seed,
        "wiring_seed": graph_seed,
        "te_hat": float(te_hat),
        "orc_hat": float(orc_hat),
        "n_edges": n_edges,
        "E_active": 2 * n_edges,
        "hidden_adj_undirected": adj.tolist(),
        "group": "G_stratified",
    }
    path.write_text(json.dumps(arch, indent=2), encoding="utf-8")


def _dispatch_stratified_training_jobs(
    *,
    output_pilot_root: Path,
    train_models: List[str],
    seeds: List[int],
    dataset: str,
    eval_mode: str,
    subjects: List[int],
    saturation_file: str,
    target_snr_db: float,
    python_exe: str,
    overwrite: bool,
) -> List[Dict[str, Any]]:
    """Run unified_experiment_runner jobs for each (model, seed) in G_strat."""
    failed: List[Dict[str, Any]] = []
    if not train_models:
        return failed
    sat_path = str(_REPO_ROOT / saturation_file) if not Path(saturation_file).is_absolute() else saturation_file
    pert_params = {"ar1_drift": {"rho": 0.97}}
    pilot_dir = output_pilot_root
    jobs = [(m, s) for m in train_models for s in seeds]
    for model_name, seed in jobs:
        rc = _run_unified_job(
            repo_root=_REPO_ROOT,
            python_exe=python_exe,
            pilot_dir=pilot_dir,
            model_name=model_name,
            dataset=dataset,
            eval_mode=eval_mode,
            subjects=subjects,
            seed=seed,
            saturation_file=sat_path,
            alpha_grid=ALPHA_GRID,
            perturbation_types=["ar1_drift"],
            target_snr_db=target_snr_db,
            perturbation_params=pert_params,
            overwrite=overwrite,
        )
        if rc != 0:
            failed.append({"model": model_name, "seed": seed})
    if failed:
        (output_pilot_root / "q3_stratified_failed_jobs.json").write_text(
            json.dumps(failed, indent=2), encoding="utf-8"
        )
    return failed


def run_stratified_setup_and_train(
    *,
    experiment1_dir: Path,
    experiment2_dir: Path,
    output_pilot_root: Path,
    H: int = 32,
    n_target: int = DEFAULT_N_TARGET,
    n_bins_te: int = DEFAULT_N_BINS_TE,
    n_bins_orc: int = DEFAULT_N_BINS_ORC,
    strat_seed: int = DEFAULT_STRAT_SEED,
    S: int = DEFAULT_S,
    dataset: str = "BNCI2014_001",
    eval_mode: str = "CrossSession",
    subjects: Optional[List[int]] = None,
    saturation_file: str = DEFAULT_SATURATION,
    target_snr_db: float = -5.0,
    python_exe: Optional[str] = None,
    overwrite: bool = False,
    setup_only: bool = False,
    train_only: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if setup_only and train_only:
        raise ValueError("Cannot use setup_only=True and train_only=True together")

    subjects = subjects if subjects is not None else list(range(1, 10))
    python_exe = python_exe or sys.executable

    if train_only:
        man_path = output_pilot_root / "q3_stratified_manifest.json"
        if not man_path.exists():
            raise FileNotFoundError(
                f"train_only requires an existing pilot manifest: {man_path} "
                "(run without --train-only first, or use setup_only to regenerate the pilot)."
            )
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        train_models = list(manifest.get("groups", {}).get("G_strat", []))
        seeds = [int(s) for s in manifest.get("seeds", [])]
        dataset = str(manifest.get("dataset", dataset))
        eval_mode = str(manifest.get("eval_mode", eval_mode))
        if not train_models:
            return {
                "pilot_root": str(output_pilot_root),
                "train_models": [],
                "n_reuse_g1": 0,
                "n_train_new": 0,
                "failed_jobs": 0,
                "train_only": True,
                "note": "No G_strat models in manifest; nothing to train.",
            }
        failed = _dispatch_stratified_training_jobs(
            output_pilot_root=output_pilot_root,
            train_models=train_models,
            seeds=seeds,
            dataset=dataset,
            eval_mode=eval_mode,
            subjects=subjects,
            saturation_file=saturation_file,
            target_snr_db=target_snr_db,
            python_exe=python_exe,
            overwrite=overwrite,
        )
        plan_path = output_pilot_root / "q3_stratified_sample_plan.json"
        n_reuse = 0
        if plan_path.exists():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            n_reuse = sum(1 for x in plan.get("items", []) if x.get("source") == "reuse_g1")
        return {
            "pilot_root": str(output_pilot_root),
            "train_models": train_models,
            "n_reuse_g1": n_reuse,
            "n_train_new": len(train_models),
            "failed_jobs": len(failed),
            "train_only": True,
        }

    pool_path = experiment1_dir / "proxy_pool.csv"
    if not pool_path.exists():
        raise FileNotFoundError(f"proxy_pool.csv not found: {pool_path}")

    pool_df = pd.read_csv(pool_path)
    rng = np.random.default_rng(strat_seed)
    sampled = stratified_sample_pool_2d(
        pool_df,
        n_target=n_target,
        n_bins_te=n_bins_te,
        n_bins_orc=n_bins_orc,
        rng=rng,
    )

    exp2_arch = experiment2_dir / "experiment2_pilot" / "selected_architectures"
    g1_map = _load_g1_topology_map(exp2_arch)

    output_pilot_root.mkdir(parents=True, exist_ok=True)
    selected_dir = output_pilot_root / "selected_architectures"
    selected_dir.mkdir(parents=True, exist_ok=True)

    train_models: List[str] = []
    items: List[Dict[str, Any]] = []
    strat_counter = 0

    for _, row in sampled.iterrows():
        key = _topology_key_from_row(row)
        te_hat = float(row["te_hat"])
        orc_hat = float(row["orc_hat"])
        proxy_score = float(row["proxy_score"])
        te_bin = int(row["te_bin"])
        orc_bin = int(row["orc_bin"])
        k, p, gs = key

        reuse_name = _g1_model_for_pool_row(row, g1_map)
        if reuse_name is not None:
            items.append({
                "source": "reuse_g1",
                "model_name": reuse_name,
                "k": k,
                "p": p,
                "graph_seed": gs,
                "te_hat": te_hat,
                "orc_hat": orc_hat,
                "proxy_score": proxy_score,
                "te_bin": te_bin,
                "orc_bin": orc_bin,
            })
            continue

        strat_counter += 1
        model_name = f"paper3_q3_strat_{strat_counter:02d}"
        te_r, oc_r = compute_paper3_proxies(build_plain_ws_flex(H, k, p, gs))
        _write_ws_flex_arch(
            H=H,
            model_name=model_name,
            k=k,
            p=p,
            graph_seed=gs,
            te_hat=te_r,
            orc_hat=oc_r,
            path=selected_dir / f"{model_name}.json",
        )
        train_models.append(model_name)
        items.append({
            "source": "train",
            "model_name": model_name,
            "k": k,
            "p": p,
            "graph_seed": gs,
            "te_hat": float(te_r),
            "orc_hat": float(oc_r),
            "proxy_score": proxy_score,
            "te_bin": te_bin,
            "orc_bin": orc_bin,
        })

    seeds = [BASE_SEED + i for i in range(S)]
    manifest = {
        "H": H,
        "S": S,
        "seeds": seeds,
        "dataset": dataset,
        "eval_mode": eval_mode,
        "groups": {"G_strat": train_models},
        "perturbation_types": ["ar1_drift"],
        "target_snr_db": target_snr_db,
        "experiment2_dir": str(experiment2_dir.resolve()),
        "experiment1_dir": str(experiment1_dir.resolve()),
    }
    (output_pilot_root / "q3_stratified_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    plan = {
        "stratification": "2d_proxy_plane",
        "n_target": n_target,
        "n_bins_te": n_bins_te,
        "n_bins_orc": n_bins_orc,
        "stratify_seed": strat_seed,
        "items": items,
        "train_model_count": len(train_models),
        "reuse_g1_count": sum(1 for x in items if x["source"] == "reuse_g1"),
    }
    (output_pilot_root / "q3_stratified_sample_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )

    failed: List[Dict[str, Any]] = []
    if not setup_only and not dry_run and train_models:
        failed = _dispatch_stratified_training_jobs(
            output_pilot_root=output_pilot_root,
            train_models=train_models,
            seeds=seeds,
            dataset=dataset,
            eval_mode=eval_mode,
            subjects=subjects,
            saturation_file=saturation_file,
            target_snr_db=target_snr_db,
            python_exe=python_exe,
            overwrite=overwrite,
        )

    return {
        "pilot_root": str(output_pilot_root),
        "train_models": train_models,
        "n_reuse_g1": plan["reuse_g1_count"],
        "n_train_new": len(train_models),
        "failed_jobs": len(failed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper 3 Q3 stratified training from proxy pool")
    parser.add_argument("--experiment1-dir", type=str, required=True, help="Dir with proxy_pool.csv")
    parser.add_argument("--experiment2-dir", type=str, required=True, help="Experiment 2 dir (for G1 reuse map)")
    parser.add_argument(
        "--output-pilot-root",
        type=str,
        default="architecture_refinement/outputs/paper3/q3_stratified_pilot",
    )
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--n-target", type=int, default=DEFAULT_N_TARGET)
    parser.add_argument(
        "--n-bins-te",
        type=int,
        default=DEFAULT_N_BINS_TE,
        help="Equal-frequency bins on te_hat (proxy plane); default 10 (10×10=100 with default --n-target)",
    )
    parser.add_argument(
        "--n-bins-orc",
        type=int,
        default=DEFAULT_N_BINS_ORC,
        help="Equal-frequency bins on orc_hat (proxy plane); default 10 (10×10=100 with default --n-target)",
    )
    parser.add_argument("--strat-seed", type=int, default=DEFAULT_STRAT_SEED)
    parser.add_argument("--S", type=int, default=DEFAULT_S, help="Number of training seeds (default 3)")
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument("--eval-mode", type=str, default="CrossSession")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--saturation-file", type=str, default=DEFAULT_SATURATION)
    parser.add_argument("--target-snr-db", type=float, default=-5.0)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--setup-only", action="store_true", help="Write JSONs/manifest only; do not train")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run unified jobs from existing q3_stratified_manifest.json only (no re-sampling; use after setup-only)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan only (no files except skip if combined?)")
    args = parser.parse_args()

    if args.setup_only and args.train_only:
        print("Error: use either --setup-only or --train-only, not both.", file=sys.stderr)
        return 1

    if not args.train_only and args.n_bins_te * args.n_bins_orc != args.n_target:
        print(
            f"Error: --n-target ({args.n_target}) must equal "
            f"--n-bins-te × --n-bins-orc ({args.n_bins_te}×{args.n_bins_orc}={args.n_bins_te * args.n_bins_orc}).",
            file=sys.stderr,
        )
        return 1

    exp1 = Path(args.experiment1_dir)
    exp2 = Path(args.experiment2_dir)
    out_root = Path(args.output_pilot_root)
    if not exp1.is_absolute():
        exp1 = _REPO_ROOT / exp1
    if not exp2.is_absolute():
        exp2 = _REPO_ROOT / exp2
    if not out_root.is_absolute():
        out_root = _REPO_ROOT / out_root

    if args.dry_run:
        pool_df = pd.read_csv(exp1 / "proxy_pool.csv")
        rng = np.random.default_rng(args.strat_seed)
        sampled = stratified_sample_pool_2d(
            pool_df,
            n_target=args.n_target,
            n_bins_te=args.n_bins_te,
            n_bins_orc=args.n_bins_orc,
            rng=rng,
        )
        print(
            f"[dry-run] 2D plane stratification: {args.n_bins_te}×{args.n_bins_orc} grid, "
            f"{len(sampled)} rows sampled."
        )
        print(
            f"  te_hat: [{sampled['te_hat'].min():.4f}, {sampled['te_hat'].max():.4f}], "
            f"orc_hat: [{sampled['orc_hat'].min():.4f}, {sampled['orc_hat'].max():.4f}]"
        )
        return 0

    summary = run_stratified_setup_and_train(
        experiment1_dir=exp1,
        experiment2_dir=exp2,
        output_pilot_root=out_root,
        H=args.H,
        n_target=args.n_target,
        n_bins_te=args.n_bins_te,
        n_bins_orc=args.n_bins_orc,
        strat_seed=args.strat_seed,
        S=args.S,
        dataset=args.dataset,
        eval_mode=args.eval_mode,
        subjects=list(args.subjects),
        saturation_file=args.saturation_file,
        target_snr_db=args.target_snr_db,
        python_exe=args.python,
        overwrite=args.overwrite,
        setup_only=args.setup_only,
        train_only=args.train_only,
        dry_run=False,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
