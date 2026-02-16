from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

# Ensure repo root is on sys.path when running as a script
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architecture_refinement.config import default_config
from architecture_refinement.topology_analyzer import TopologyAnalyzer
from architecture_refinement.arbitrary_wiring import WsFlexHiddenWiring


ALPHA_GRID_DEFAULT = [0.0, 0.25, 0.5, 0.75, 1.0]


@dataclass(frozen=True)
class Candidate:
    method: str  # "random" | "tpe"
    rep: int
    idx: int
    k: int
    p: float
    graph_seed: int
    wiring_seed: int
    te: float
    orc: float
    orc_raw: float
    connected: bool


def _now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_unique_run_id(*, repo_root: Path, output_root: str, run_id: str) -> str:
    """
    Ensure the pilot run_id will not overwrite existing pilot artifacts or registry files.

    If a collision is detected, append a suffix: "<run_id>_v2", "_v3", ...
    """
    base = str(run_id)
    reg_dir = repo_root / ".model_registry"

    def _collides(rid: str) -> bool:
        pilot_dir = (repo_root / output_root / rid)
        reg_file = reg_dir / f"nas_pilot_{rid}.py"
        return pilot_dir.exists() or reg_file.exists()

    if not _collides(base):
        return base

    for i in range(2, 10_000):
        candidate = f"{base}_v{i}"
        if not _collides(candidate):
            print(f"[NAS PILOT] Detected existing run_id '{base}', using '{candidate}' to avoid overwriting.")
            return candidate

    raise RuntimeError(f"Could not find a unique run_id based on '{base}' (too many collisions).")


def _normalize_0_1(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    lo = float(np.nanmin(v))
    hi = float(np.nanmax(v))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(v, dtype=float)
    return (v - lo) / (hi - lo)


def _make_ws_graph(H: int, k: int, p: float, seed: int) -> nx.Graph:
    # Must match spec: undirected connected WS; reject disconnected candidates.
    G = nx.watts_strogatz_graph(H, k, p, seed=int(seed))
    return G


def _compute_te_orc(analyzer: TopologyAnalyzer, G: nx.Graph) -> Tuple[float, float, float]:
    # Canonical definitions (see Patch_Note_ORC_Signed.md):
    # - TE: exact degree entropy normalized by log(N) (in [0,1])
    # - ORC: signed mean Ollivier–Ricci curvature (no abs)
    from architecture_refinement.metrics_te_orc import compute_te_orc

    te, orc, _dbg = compute_te_orc(G, orc_alpha=0.5)
    orc_raw = float(orc)
    return float(te), float(orc), float(orc_raw)


def _oriented_hidden_adj(G: nx.Graph, H: int, seed: int) -> np.ndarray:
    """
    Deterministic orientation rule: delegate to WsFlexHiddenWiring(..., hidden_edge_orientation="random_oriented", seed=seed)
    and extract the oriented HxH hidden block.
    """
    wiring = WsFlexHiddenWiring(
        input_size=1,
        hidden_graph=G,
        output_size=1,
        hidden_edge_orientation="random_oriented",
        seed=int(seed),
    )
    A = wiring._hidden_block_oriented()
    A = (np.asarray(A) != 0).astype(np.int8)
    if A.shape != (H, H):
        raise ValueError(f"Unexpected oriented hidden adjacency shape: {A.shape} (expected {(H, H)})")
    return A


def _undirected_hidden_adj(G: nx.Graph, H: int) -> np.ndarray:
    A = nx.to_numpy_array(G, dtype=np.int8)
    A = (A != 0).astype(np.int8)
    if A.shape != (H, H):
        raise ValueError(f"Unexpected undirected adjacency shape: {A.shape} (expected {(H, H)})")
    return A


def _sample_random_candidates(
    *,
    analyzer: TopologyAnalyzer,
    H: int,
    k_values: List[int],
    M: int,
    rep: int,
    base_seed: int,
    max_attempts: int,
) -> List[Candidate]:
    rng = np.random.default_rng(int(base_seed))
    out: List[Candidate] = []
    attempts = 0

    while len(out) < M and attempts < max_attempts:
        attempts += 1
        k = int(rng.choice(k_values))
        p = float(rng.uniform(0.0, 1.0))
        graph_seed = int(rng.integers(0, 2**31 - 1))
        wiring_seed = int(rng.integers(0, 2**31 - 1))
        G = _make_ws_graph(H, k, p, seed=graph_seed)
        connected = bool(nx.is_connected(G))
        if not connected:
            continue
        te, orc, orc_raw = _compute_te_orc(analyzer, G)
        out.append(
            Candidate(
                method="random",
                rep=rep,
                idx=len(out),
                k=k,
                p=p,
                graph_seed=graph_seed,
                wiring_seed=wiring_seed,
                te=te,
                orc=orc,
                orc_raw=orc_raw,
                connected=True,
            )
        )

    if len(out) < M:
        raise RuntimeError(f"Random sampling failed to produce M={M} connected candidates (got {len(out)} after {attempts} attempts).")
    return out


def _sample_tpe_candidates(
    *,
    analyzer: TopologyAnalyzer,
    H: int,
    k_values: List[int],
    M: int,
    rep: int,
    base_seed: int,
    max_trials: int,
) -> List[Candidate]:
    try:
        import optuna
    except Exception as e:
        raise ImportError(
            f"Optuna is required for TPE sampling but could not be imported: {e}\n"
            "Install it in your environment (e.g., pip install optuna)."
        )

    rng = np.random.default_rng(int(base_seed))
    collected: List[Candidate] = []

    sampler = optuna.samplers.TPESampler(seed=int(base_seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: "optuna.Trial") -> float:
        nonlocal collected
        k = int(trial.suggest_categorical("k", k_values))
        p = float(trial.suggest_float("p", 0.0, 1.0))
        graph_seed = int(rng.integers(0, 2**31 - 1))
        wiring_seed = int(rng.integers(0, 2**31 - 1))
        G = _make_ws_graph(H, k, p, seed=graph_seed)
        if not nx.is_connected(G):
            raise optuna.TrialPruned()

        te, orc, orc_raw = _compute_te_orc(analyzer, G)
        collected.append(
            Candidate(
                method="tpe",
                rep=rep,
                idx=len(collected),
                k=k,
                p=p,
                graph_seed=graph_seed,
                wiring_seed=wiring_seed,
                te=te,
                orc=orc,
                orc_raw=orc_raw,
                connected=True,
            )
        )
        # Proxy objective during sampling (selection still uses normalized TE/ORC sum)
        return float(te + orc)

    def stop_when_enough(study: "optuna.Study", trial: "optuna.Trial") -> None:
        if len(collected) >= M:
            study.stop()

    study.optimize(objective, n_trials=int(max_trials), callbacks=[stop_when_enough], show_progress_bar=False)

    if len(collected) < M:
        raise RuntimeError(f"TPE sampling failed to produce M={M} connected candidates (got {len(collected)} within max_trials={max_trials}).")
    return collected[:M]


def _candidates_to_rows(cands: List[Candidate]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for c in cands:
        rows.append(
            {
                "method": c.method,
                "rep": c.rep,
                "idx": c.idx,
                "k": c.k,
                "p": c.p,
                "graph_seed": c.graph_seed,
                "wiring_seed": c.wiring_seed,
                "te": c.te,
                "orc": c.orc,
                "orc_raw": c.orc_raw,
                "connected": c.connected,
            }
        )
    return rows


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _select_top_b(cands: List[Candidate], B: int) -> Tuple[List[Dict[str, Any]], List[Candidate]]:
    te = np.array([c.te for c in cands], dtype=float)
    orc = np.array([c.orc for c in cands], dtype=float)
    te_n = _normalize_0_1(te)
    orc_n = _normalize_0_1(orc)
    score = te_n + orc_n

    rows: List[Dict[str, Any]] = []
    for i, c in enumerate(cands):
        rows.append(
            {
                **_candidates_to_rows([c])[0],
                "te_norm": float(te_n[i]),
                "orc_norm": float(orc_n[i]),
                "score": float(score[i]),
            }
        )

    order = np.argsort(-score)  # descending
    selected = [cands[int(i)] for i in order[:B]]
    return rows, selected


def _generate_model_registry_file(
    *,
    repo_root: Path,
    run_id: str,
    pilot_dir: Path,
    alpha_grid: List[float],
) -> Path:
    reg_dir = repo_root / ".model_registry"
    reg_dir.mkdir(parents=True, exist_ok=True)
    reg_file = reg_dir / f"nas_pilot_{run_id}.py"

    # The registration file is loaded by evaluation/unified_experiment_runner.py at startup.
    # It registers runtime-only models that reconstruct wiring from JSON.
    content = f"""\
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import networkx as nx

import config as _cfg
from architecture_refinement.arbitrary_wiring import WsFlexHiddenWiring
from models.cnn_wiredcfc_min import create_cnnwiredcfc_min_classifier


_RUN_ID = {run_id!r}
_PILOT_DIR = Path({str(pilot_dir.as_posix())!r})


def _load_arch_jsons() -> list[dict]:
    paths = sorted((_PILOT_DIR / "selected_architectures").glob("*.json"))
    out = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def _make_wiring(hidden_adj_undirected: np.ndarray, wiring_seed: int):
    H = int(hidden_adj_undirected.shape[0])
    G = nx.from_numpy_array((hidden_adj_undirected != 0).astype(np.int8))
    if not nx.is_connected(G):
        raise ValueError("Hidden graph is disconnected (pilot constraint).")
    return WsFlexHiddenWiring(
        input_size=1,
        hidden_graph=G,
        output_size=1,
        input_strategy="degree_proportional",
        output_strategy="uniform",
        hidden_edge_orientation="random_oriented",
        add_hidden_self_loops=True,
        seed=int(wiring_seed),
    )


def _register_one(model_name: str, arch: dict) -> None:
    hidden_adj = np.asarray(arch["hidden_adj_undirected"], dtype=np.int8)
    wiring_seed = int(arch["wiring_seed"])

    def _factory(n_chans: int, n_times: int, n_outputs: int, **kwargs):
        wiring = _make_wiring(hidden_adj, wiring_seed=wiring_seed)
        return create_cnnwiredcfc_min_classifier(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            wiring=wiring,
            **kwargs,
        )

    # Safety: never overwrite an existing model key (pilot must not clobber prior optimizations).
    if model_name in _cfg.get_model_registry():
        raise ValueError(f"Refusing to overwrite existing model registration: {{model_name}}")
    _cfg._runtime_model_registry[model_name] = _factory


for arch in _load_arch_jsons():
    _register_one(arch["model_name"], arch)
"""
    # Exclusive create: never overwrite an existing registry file.
    with reg_file.open("x", encoding="utf-8") as f:
        f.write(content)
    return reg_file


def _run_unified_job(
    *,
    repo_root: Path,
    python_exe: str,
    model_name: str,
    dataset: str,
    eval_mode: str,
    subjects: List[int],
    seed: int,
    saturation_file: str,
    alpha_grid: List[float],
    gaussian_only: bool,
    overwrite: bool,
    pilot_dir: Path,
) -> int:
    cmd = [
        python_exe,
        str((repo_root / "evaluation" / "unified_experiment_runner.py").as_posix()),
        "--nas_pilot_dir",
        str(pilot_dir.as_posix()),
        "--model",
        model_name,
        "--dataset",
        dataset,
        "--subjects",
        *[str(s) for s in subjects],
        "--mode",
        "test_perturb",
        "--eval_mode",
        eval_mode,
        "--seed",
        str(seed),
        "--overwrite" if overwrite else "",
        "--disable_underfitting_retrain",
        "--noise_perturbation_saturation_file",
        saturation_file,
        "--noise_perturbation_num_steps",
        "20",
        "--test_perturb_gaussian_alpha_grid",
        ",".join(str(a) for a in alpha_grid),
    ]
    if gaussian_only:
        cmd.append("--test_perturb_gaussian_only")

    # Remove empty elements (from conditional flags)
    cmd = [c for c in cmd if c]
    print("[NAS PILOT] Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(repo_root), check=False)
    return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="NAS Pilot Study runner (WS-Flex: TE+ORC vs Random)")
    parser.add_argument("--run_id", type=str, default=None, help="Optional run id (default: timestamp).")
    parser.add_argument("--output_root", type=str, default="architecture_refinement/outputs/nas_pilot")
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument("--eval_mode", type=str, default="CrossSession")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--k_values", type=str, default="2,4,6,8")
    parser.add_argument("--M", type=int, default=80)
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--R", type=int, default=3)
    parser.add_argument("--alpha_grid", type=str, default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--saturation_file", type=str, default="saturation_results/saturation_points_summary.csv")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Generate candidates/selection/registry, but do not run training.")
    args = parser.parse_args()

    # Spec-aligned dry-run defaults (small, fast sanity-check run)
    if args.dry_run:
        args.M = 6
        args.B = 2
        args.R = 1
        args.subjects = [int(args.subjects[0])] if args.subjects else [1]

    repo_root = Path(__file__).resolve().parent.parent
    run_id = args.run_id or _now_run_id()
    run_id = _resolve_unique_run_id(repo_root=repo_root, output_root=args.output_root, run_id=run_id)
    pilot_dir = (repo_root / args.output_root / run_id).resolve()
    pilot_dir.mkdir(parents=True, exist_ok=False)

    k_values = [int(x.strip()) for x in str(args.k_values).split(",") if x.strip()]
    alpha_grid = [float(x.strip()) for x in str(args.alpha_grid).split(",") if x.strip()]
    if not alpha_grid:
        alpha_grid = list(ALPHA_GRID_DEFAULT)

    analyzer = TopologyAnalyzer(default_config, logger=None)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "dataset": args.dataset,
        "eval_mode": args.eval_mode,
        "subjects": list(args.subjects),
        "H": int(args.H),
        "k_values": k_values,
        "M": int(args.M),
        "B": int(args.B),
        "R": int(args.R),
        "alpha_grid": alpha_grid,
        "saturation_file": args.saturation_file,
        "model_key": "cnn_wiredcfc_min",
    }
    _write_json(pilot_dir / "pilot_manifest.json", manifest)

    all_selected_arch_rows: List[Dict[str, Any]] = []
    selected_arch_dir = pilot_dir / "selected_architectures"
    selected_arch_dir.mkdir(parents=True, exist_ok=False)

    for rep in range(1, int(args.R) + 1):
        # Seed layout: ensure independent repetitions
        base_seed_random = 10_000 * rep + 123
        base_seed_tpe = 10_000 * rep + 987

        # ---- Method A: Random ----
        random_cands = _sample_random_candidates(
            analyzer=analyzer,
            H=int(args.H),
            k_values=k_values,
            M=int(args.M),
            rep=rep,
            base_seed=base_seed_random,
            max_attempts=int(args.M) * 50,
        )
        random_rows, random_sel = _select_top_b(random_cands, int(args.B))
        _write_csv(pilot_dir / "candidates" / f"candidates_random_rep{rep}.csv", random_rows)

        # ---- Method B: TPE ----
        tpe_cands = _sample_tpe_candidates(
            analyzer=analyzer,
            H=int(args.H),
            k_values=k_values,
            M=int(args.M),
            rep=rep,
            base_seed=base_seed_tpe,
            max_trials=int(args.M) * 50,
        )
        tpe_rows, tpe_sel = _select_top_b(tpe_cands, int(args.B))
        _write_csv(pilot_dir / "candidates" / f"candidates_tpe_rep{rep}.csv", tpe_rows)

        for method, selected in [("random", random_sel), ("tpe", tpe_sel)]:
            for rank, cand in enumerate(selected, start=1):
                model_name = f"nas_pilot_{run_id}_{method}_r{rep}_b{rank}"
                G = _make_ws_graph(int(args.H), cand.k, cand.p, seed=cand.graph_seed)
                if not nx.is_connected(G):
                    raise RuntimeError("Selected graph unexpectedly disconnected (should not happen).")

                undirected_adj = _undirected_hidden_adj(G, int(args.H))
                directed_adj = _oriented_hidden_adj(G, int(args.H), seed=cand.wiring_seed)

                arch_row = {
                    "schema_version": 1,
                    "run_id": run_id,
                    "method": method,
                    "rep": rep,
                    "rank": rank,
                    "model_name": model_name,
                    "H": int(args.H),
                    "k": cand.k,
                    "p": cand.p,
                    "graph_seed": cand.graph_seed,
                    "wiring_seed": cand.wiring_seed,
                    "te": cand.te,
                    "orc": cand.orc,
                    "orc_raw": cand.orc_raw,
                    "hidden_adj_undirected": undirected_adj.tolist(),
                    "hidden_adj_directed": directed_adj.tolist(),
                }
                all_selected_arch_rows.append(arch_row)
                _write_json(selected_arch_dir / f"{model_name}.json", arch_row)

    _write_csv(pilot_dir / "selected_architectures.csv", all_selected_arch_rows)

    print(f"[NAS PILOT] Pilot directory: {pilot_dir}")

    if args.dry_run:
        print("[NAS PILOT] Dry-run complete (no training dispatched).")
        return

    # Dispatch training/evaluation runs (CrossSession only per spec)
    # Note: overwrite is recommended because intensity grids can differ from default benchmarks.
    for arch in all_selected_arch_rows:
        model_name = arch["model_name"]
        # Training seed per architecture (independent within repetition)
        train_seed = int(100_000 * int(arch["rep"]) + 1_000 * (1 if arch["method"] == "tpe" else 0) + int(arch["rank"]))
        rc = _run_unified_job(
            repo_root=repo_root,
            python_exe=args.python,
            model_name=model_name,
            dataset=args.dataset,
            eval_mode=args.eval_mode,
            subjects=list(args.subjects),
            seed=train_seed,
            saturation_file=args.saturation_file,
            alpha_grid=alpha_grid,
            gaussian_only=True,
            overwrite=bool(args.overwrite),
            pilot_dir=pilot_dir,
        )
        if rc != 0:
            raise RuntimeError(f"Unified runner failed for {model_name} (exit_code={rc}).")


if __name__ == "__main__":
    main()

