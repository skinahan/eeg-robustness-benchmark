"""
Plot 2 Intermediate Diagnostic (GO/NO-GO before full training).

Selects 8 graphs: 1 per regime × 2 cells (small-world: C high L low; non-small-world: C low L high).
Train/eval with subjects=3, seeds=1, ar1_drift. Applies success criteria including effect-size gate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import short_run_id

H = 32
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
DEGREE_REGIMES: Dict[str, List[int]] = {
    "super_sparse": [2, 4, 6],
    "sparse": [8, 10, 12],
    "moderate": [14, 16, 18],
    "near_dense": [20, 22, 24, 26],
}


def _make_ws_graph(H: int, k: int, p: float, seed: int) -> nx.Graph:
    return nx.watts_strogatz_graph(int(H), int(k), float(p), seed=int(seed))


def _make_graph(
    H: int,
    k: int,
    p: float,
    seed: int,
    generator_mode: str,
    *,
    seed_mod_params: int = 202607,
    sample_idx: int = 0,
) -> nx.Graph:
    """Make graph by mode."""
    if generator_mode == "modular_ws_flex":
        from architecture_refinement.ws_flex_generator import (
            make_ws_flex_graph,
            sample_modular_params,
        )
        M, p_out, r_out = sample_modular_params(H, seed_mod_params + sample_idx)
        G, _ = make_ws_flex_graph(
            H, k, p, seed,
            generator_mode="modular_ws_flex",
            M=M, p_out=p_out, r_out=r_out,
        )
        return G
    return _make_ws_graph(H, k, p, seed)


def _oriented_hidden_adj(G: nx.Graph, H: int, seed: int) -> np.ndarray:
    from architecture_refinement.arbitrary_wiring import WsFlexHiddenWiring
    wiring = WsFlexHiddenWiring(
        input_size=1, hidden_graph=G, output_size=1,
        hidden_edge_orientation="random_oriented", seed=int(seed),
    )
    A = wiring._hidden_block_oriented()
    return (np.asarray(A) != 0).astype(np.int8)


def _sample_graphs_for_cells(
    H: int,
    regime: str,
    n_per_cell: int,
    rng: np.random.Generator,
    max_attempts: int = 2000,
    generator_mode: str = "ws_flex",
    seed_mod_params: int = 202607,
    sample_idx_offset: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Sample small-world (C high, L low) and non-small-world (C low, L high) graphs per regime.
    Returns (small_world_graphs, non_small_world_graphs).
    """
    ks = DEGREE_REGIMES.get(regime, [])
    if not ks:
        return [], []

    def _make_cand(k: int, p: float, gs: int, ws: int, sidx: int, use_analyzer: bool = True) -> Optional[Dict[str, Any]]:
        G = _make_graph(H, k, p, gs, generator_mode, seed_mod_params=seed_mod_params, sample_idx=sidx)
        if not nx.is_connected(G):
            return None
        try:
            if use_analyzer:
                from architecture_refinement.topology_analyzer import TopologyAnalyzer
                from architecture_refinement.config import default_config
                topo = TopologyAnalyzer(default_config).analyze_graph(G)
                C = float(topo.get("clustering_coefficient", float("nan")))
                L = float(topo.get("avg_path_length", float("nan")))
            else:
                C = float(nx.average_clustering(G))
                L = float(nx.average_shortest_path_length(G))
        except Exception:
            C = float(nx.average_clustering(G))
            L = float(nx.average_shortest_path_length(G))
        if not (np.isfinite(C) and np.isfinite(L)):
            return None
        return {"k": k, "p": p, "graph_seed": gs, "wiring_seed": ws, "C": C, "L": L, "G": G}

    # p-range separation: SW-like = small p (0-0.15), NSW-like = large p.
    # For super_sparse (low k), use [0.5, 0.85] for NSW to avoid disconnection.
    P_SW = (0.0, 0.15)
    P_NSW = (0.5, 0.85) if regime == "super_sparse" else (0.6, 1.0)

    def _sample_p_range(p_lo: float, p_hi: float, n: int, k_force: Optional[int] = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set = set()
        for _ in range(max(500, n * 50)):
            k = int(k_force) if k_force is not None else int(rng.choice(ks))
            p = float(rng.uniform(p_lo, p_hi))
            gs = int(rng.integers(0, 2**31 - 1))
            ws = int(rng.integers(0, 2**31 - 1))
            c = _make_cand(k, p, gs, ws, sample_idx_offset + len(out), use_analyzer=(regime != "super_sparse"))
            if c is None:
                continue
            adj_hash = (nx.to_numpy_array(c["G"]) != 0).astype(np.int8).tobytes()
            if adj_hash in seen:
                continue
            seen.add(adj_hash)
            out.append(c)
            if len(out) >= n:
                break
        return out

    # p-range separation first. For super_sparse, also force k_lo/k_hi => different adjacency.
    k_lo, k_hi = (min(ks), max(ks)) if len(ks) >= 2 else (None, None)
    sw_pool = _sample_p_range(P_SW[0], P_SW[1], n_per_cell, k_force=k_lo if regime == "super_sparse" else None)
    nsw_pool = _sample_p_range(P_NSW[0], P_NSW[1], n_per_cell, k_force=k_hi if regime == "super_sparse" else None)
    if sw_pool and nsw_pool:
        adj_sw = (nx.to_numpy_array(sw_pool[0]["G"]) != 0).astype(np.int8)
        adj_nsw = (nx.to_numpy_array(nsw_pool[0]["G"]) != 0).astype(np.int8)
        if not np.array_equal(adj_sw, adj_nsw):
            return sw_pool[:n_per_cell], nsw_pool[:n_per_cell]

    all_cands: List[Dict[str, Any]] = []
    seen_adj_hashes: set = set()

    for attempt in range(max_attempts):
        k = int(rng.choice(ks))
        p = float(rng.uniform(0.0, 1.0))
        gs = int(rng.integers(0, 2**31 - 1))
        ws = int(rng.integers(0, 2**31 - 1))
        c = _make_cand(k, p, gs, ws, sample_idx_offset + len(all_cands))
        if c is None:
            continue
        adj = (nx.to_numpy_array(c["G"]) != 0).astype(np.int8)
        adj_hash = adj.tobytes()
        if adj_hash in seen_adj_hashes:
            continue
        seen_adj_hashes.add(adj_hash)
        all_cands.append(c)
        if len(all_cands) >= max(2, n_per_cell * 2):
            break

    # Fallback for constrained regimes (e.g. super_sparse): force k-diversity.
    # Different k => different degree => different adjacency. Guarantees distinct topologies.
    if len(all_cands) < 2 and len(ks) >= 2:
        k_lo, k_hi = min(ks), max(ks)
        for k_force in [k_lo, k_hi]:
            for attempt in range(300):
                p = float(rng.uniform(0.0, 1.0))
                gs = int(rng.integers(0, 2**31 - 1))
                ws = int(rng.integers(0, 2**31 - 1))
                c = _make_cand(k_force, p, gs, ws, sample_idx_offset + len(all_cands))
                if c is None:
                    continue
                adj_hash = (nx.to_numpy_array(c["G"]) != 0).astype(np.int8).tobytes()
                if adj_hash not in seen_adj_hashes:
                    seen_adj_hashes.add(adj_hash)
                    all_cands.append(c)
                    break

    # Last resort: sample one graph per k (k_lo and k_hi) - different k => different adjacency
    if len(all_cands) < 2 and len(ks) >= 2:
        k_lo, k_hi = min(ks), max(ks)
        for k_force in [k_lo, k_hi]:
            for attempt in range(500):
                p_val = 0.1 if k_force == k_lo else 0.5  # low p for regular, mid p for variety
                gs = int(rng.integers(0, 2**31 - 1))
                ws = int(rng.integers(0, 2**31 - 1))
                c = _make_cand(k_force, p_val, gs, ws, sample_idx_offset + len(all_cands))
                if c is not None:
                    adj_hash = (nx.to_numpy_array(c["G"]) != 0).astype(np.int8).tobytes()
                    if adj_hash not in seen_adj_hashes:
                        seen_adj_hashes.add(adj_hash)
                        all_cands.append(c)
                        break
            if len(all_cands) >= 2:
                break

    if len(all_cands) < 2:
        return [], []

    all_cands.sort(key=lambda r: -(r["C"] - r["L"]))
    sw_cands = all_cands[:n_per_cell]
    all_cands.sort(key=lambda r: r["L"] - r["C"])
    nsw_cands = all_cands[:n_per_cell]

    adj_sw = (nx.to_numpy_array(sw_cands[0]["G"]) != 0).astype(np.int8)
    adj_nsw = (nx.to_numpy_array(nsw_cands[0]["G"]) != 0).astype(np.int8)
    if np.array_equal(adj_sw, adj_nsw):
        return [], []
    return sw_cands, nsw_cands


def create_diagnostic_architectures(
    output_dir: Path,
    base_id: str = "plot2_intermediate",
    seed: int = 4242,
    generator_mode: str = "ws_flex",
    seed_mod_params: int = 202607,
) -> List[Dict[str, Any]]:
    """Create 8 architectures (4 regimes × 2 cells) and write JSON. Writes to output_dir/selected_architectures."""
    rng = np.random.default_rng(seed)
    arch_dir = output_dir / "selected_architectures"
    arch_dir.mkdir(parents=True, exist_ok=True)
    archs: List[Dict[str, Any]] = []
    idx = 0
    sample_idx_offset = 0
    for regime_idx, regime in enumerate(DEGREE_REGIMES):
        for retry in range(20):
            rng_retry = np.random.default_rng(seed + regime_idx * 100 + retry * 1000)
            sw, nsw = _sample_graphs_for_cells(
                H, regime, 1, rng_retry,
                generator_mode=generator_mode,
                seed_mod_params=seed_mod_params,
                sample_idx_offset=sample_idx_offset + retry * 100,
            )
            if sw and nsw:
                break
        if not sw or not nsw:
            import warnings
            warnings.warn(
                f"SW/NSW sampling failed for regime {regime} after 20 retries. "
                f"p-range separation (SW p∈[0,0.15], NSW p∈[0.6,1.0]) should have been used. "
                "Check connectivity (high p can disconnect low-k graphs) or increase max_attempts."
            )
            raise RuntimeError(
                f"Could not sample distinct SW and NSW graphs for regime {regime} after retries. "
                "SW uses p∈[0,0.15], NSW uses p∈[0.6,1.0]. For super_sparse, SW uses k_min, NSW uses k_max. "
                "If NSW pool is empty, high p may disconnect low-k graphs; try relaxing P_NSW to [0.5,0.9]."
            )
        sample_idx_offset += len(sw) + len(nsw)
        for cell_type, cands in [("sw", sw), ("nsw", nsw)]:
            if not cands:
                continue
            c = cands[0]
            G = c["G"]
            model_id = f"{base_id}_{regime}_{cell_type}_{idx}"
            undirected_adj = (nx.to_numpy_array(G) != 0).astype(np.int8)
            directed_adj = _oriented_hidden_adj(G, H, c["wiring_seed"])
            arch = {
                "schema_version": 2,
                "model_name": model_id,
                "wiring_kind": "ws_flex",
                "H": H,
                "k": c["k"],
                "p": c["p"],
                "graph_seed": c["graph_seed"],
                "wiring_seed": c["wiring_seed"],
                "hidden_adj_undirected": undirected_adj.tolist(),
                "hidden_adj_directed": directed_adj.tolist(),
                "regime": regime,
                "cell_type": cell_type,
                "clustering": c["C"],
                "path_length": c["L"],
            }
            (arch_dir / f"{model_id}.json").write_text(json.dumps(arch, indent=2), encoding="utf-8")
            archs.append({"model_id": model_id, "regime": regime, "cell_type": cell_type, **arch})
            idx += 1
    return archs


def run_eval(
    repo_root: Path,
    pilot_dir: Path,
    model_id: str,
    dataset: str,
    subjects: List[int],
    seed: int,
    noise_types: List[str],
    target_snr_db: float,
    python_exe: str,
    *,
    overwrite: bool = False,
) -> int:
    """Run unified runner for one model."""
    cmd = [
        python_exe,
        str((repo_root / "evaluation" / "unified_experiment_runner.py").resolve()),
        "--nas_pilot_dir", str(pilot_dir.resolve()),
        "--model", model_id,
        "--dataset", dataset,
        "--subjects", *[str(s) for s in subjects],
        "--mode", "test_perturb",
        "--eval_mode", "CrossSession",
        "--seed", str(seed),
        "--disable_underfitting_retrain",
        "--test_perturb_gaussian_alpha_grid", ",".join(str(a) for a in ALPHA_GRID),
        "--test_perturb_noise_types", ",".join(noise_types),
        "--test_perturb_target_snr_db", str(target_snr_db),
    ]
    if overwrite:
        cmd.append("--overwrite")
    proc = subprocess.run(cmd, cwd=str(repo_root), check=False)
    return int(proc.returncode)


def collect_max_drop(repo_root: Path, dataset: str, model_id: str, seed: int, noise_type: str) -> Optional[float]:
    """Collect max_drop for one model + noise type."""
    paradigm = "MotorImagery" if "BNCI" in dataset or "BI2015" in dataset else "SSVEP"
    base = Path(repo_root) / "results" / paradigm / dataset
    for stem in [short_run_id(model_id), model_id]:
        path = base / stem / "CrossSessionEvaluation" / str(seed)
        if not path.exists():
            continue
        for p in path.rglob("*.csv"):
            if "test_perturb" not in str(p):
                continue
            try:
                df = pd.read_csv(p)
                if "noise_type" not in df.columns or noise_type not in df["noise_type"].astype(str).values:
                    continue
                sub = df[df["noise_type"].astype(str) == noise_type]
                if sub.empty:
                    continue
                sub["corrupted_roc_auc"] = pd.to_numeric(sub["corrupted_roc_auc"], errors="coerce")
                clean_col = sub.get("clean_roc_auc", sub.get("clean_score", pd.Series(dtype=float)))
                clean = float(clean_col.iloc[0]) if len(clean_col) and pd.notna(clean_col.iloc[0]) else float("nan")
                roc_at_max = float(sub["corrupted_roc_auc"].iloc[-1]) if len(sub) else float("nan")
                return float(clean - roc_at_max) if np.isfinite(clean) and np.isfinite(roc_at_max) else None
            except Exception:
                pass
    return None


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Pooled Cohen's d."""
    if x.size < 2 or y.size < 2:
        return float("nan")
    vx = float(np.var(x, ddof=1))
    vy = float(np.var(y, ddof=1))
    pooled = np.sqrt(((x.size - 1) * vx + (y.size - 1) * vy) / (x.size + y.size - 2))
    if pooled <= 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / pooled)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot 2 Intermediate Diagnostic")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument("--subjects", type=int, nargs="*", default=[1, 3, 4], help="Subject IDs (default excludes subject 2 due to known dataset issues)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--target_snr_db", type=float, default=-6.0)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--skip_run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-run evals and overwrite existing results (passes --overwrite to unified runner).",
    )
    parser.add_argument(
        "--generator_mode",
        type=str,
        default="ws_flex",
        choices=["ws_flex", "modular_ws_flex"],
        help="Generator mode: ws_flex (default) or modular_ws_flex.",
    )
    parser.add_argument("--seed_mod_params", type=int, default=202607)
    args = parser.parse_args()

    repo_root = _REPO_ROOT
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pilot_dir = output_dir / "pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)

    noise_type = "ar1_drift"
    archs = create_diagnostic_architectures(
        output_dir,
        base_id="plot2_intermediate",
        seed=args.seed * 100,
        generator_mode=getattr(args, "generator_mode", "ws_flex"),
        seed_mod_params=getattr(args, "seed_mod_params", 202607),
    )

    if not args.skip_run:
        for arch in archs:
            rc = run_eval(
                repo_root, output_dir, arch["model_id"],
                args.dataset, args.subjects, args.seed,
                [noise_type], args.target_snr_db, args.python,
                overwrite=getattr(args, "overwrite", False),
            )
            if rc != 0:
                print(f"[INTERMEDIATE] Eval failed for {arch['model_id']}")

    # Collect max_drop per (model, subject) - simplified: we use seed as fold
    rows: List[Dict[str, Any]] = []
    for arch in archs:
        md = collect_max_drop(repo_root, args.dataset, arch["model_id"], args.seed, noise_type)
        rows.append({
            "model_id": arch["model_id"],
            "regime": arch["regime"],
            "cell_type": arch["cell_type"],
            "max_drop": md if md is not None else float("nan"),
        })
    df = pd.DataFrame(rows)

    max_drops = df["max_drop"].dropna().values
    max_pairwise_delta = float(np.max(max_drops) - np.min(max_drops)) if len(max_drops) >= 2 else 0.0

    # Sign-agnostic, n=1-safe sensitivity gate
    EPS_REGIME = 0.03              # minimum meaningful per-regime separation
    MIN_SEPARATING_REGIMES = 2     # require separation in >=2 regimes

    effect_size_pass = False
    regime_separates: Dict[str, bool] = {}
    regime_directional: Dict[str, bool] = {}  # retained for reporting only

    for regime in DEGREE_REGIMES:
        sub = df[df["regime"] == regime]
        sw = sub[sub["cell_type"] == "sw"]["max_drop"].dropna().values
        nsw = sub[sub["cell_type"] == "nsw"]["max_drop"].dropna().values

        if len(sw) >= 1 and len(nsw) >= 1:
            sw_m = float(np.mean(sw))
            nsw_m = float(np.mean(nsw))
            delta = nsw_m - sw_m  # positive => SW more robust (lower max_drop)

            # Per-regime separation regardless of sign
            regime_separates[regime] = bool(abs(delta) >= EPS_REGIME)

            # Optional reporting only (not used in pass/fail)
            regime_directional[regime] = bool(delta > 0)

            # Effect-size logic:
            if len(sw) >= 2 and len(nsw) >= 2:
                d = float(cohens_d(sw, nsw))
                if np.isfinite(d) and abs(d) >= 0.5:
                    effect_size_pass = True
            else:
                # n=1 case: treat sufficient separation as effect evidence
                if abs(delta) >= EPS_REGIME:
                    effect_size_pass = True
        else:
            regime_separates[regime] = False
            regime_directional[regime] = False

    primary_pass = bool(max_pairwise_delta >= 0.03)
    n_separating = sum(1 for v in regime_separates.values() if v)

    secondary_pass = bool(n_separating >= MIN_SEPARATING_REGIMES)
    all_pass = bool(primary_pass and effect_size_pass and secondary_pass)

    report = {
        "schema_version": 1,
        "max_pairwise_delta_max_drop": float(max_pairwise_delta),
        "primary_pass": primary_pass,
        "effect_size_pass": effect_size_pass,
        "regime_separates": {k: bool(v) for k, v in regime_separates.items()},
        "regime_directional": {k: bool(v) for k, v in regime_directional.items()},
        "n_separating_regimes": n_separating,
        "n_directional_regimes": n_separating,  # backward compat (now = n_separating)
        "all_pass": all_pass,
        "per_graph": rows,
    }
    (output_dir / "intermediate_diagnostic_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    df.to_csv(output_dir / "intermediate_diagnostic_max_drop.csv", index=False)

    print(f"[INTERMEDIATE] max_pairwise_delta_max_drop={max_pairwise_delta:.4f} (need >= 0.03)")
    print(f"[INTERMEDIATE] effect_size_pass={effect_size_pass}, regime_separates={regime_separates}, n_separating={n_separating}/{MIN_SEPARATING_REGIMES}")
    print(f"[INTERMEDIATE] All pass: {all_pass}")
    if not all_pass:
        print("[INTERMEDIATE] FAIL: Do not proceed to full training. Escalate perturbation or change type.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
