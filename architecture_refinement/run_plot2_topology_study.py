from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx

# Ensure repo root is on sys.path when running as a script
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architecture_refinement.config import default_config
from architecture_refinement.topology_analyzer import TopologyAnalyzer, compute_spectral_radius_directed
from architecture_refinement.arbitrary_wiring import WsFlexHiddenWiring
from architecture_refinement.ws_flex_generator import (
    make_ws_flex_graph,
    sample_modular_params,
    DEFAULT_M_VALUES,
    DEFAULT_P_OUT_LO,
    DEFAULT_P_OUT_HI,
    DEFAULT_R_OUT_LO,
    DEFAULT_R_OUT_HI,
)
from architecture_refinement.pareto_hv import hypervolume_2d, pareto_front_2d
from architecture_refinement.plot2_schema import validate_manifest
from architecture_refinement.capacity_utils import capacity_filter, DEFAULT_E_ACTIVE_BANDS_H32


ALPHA_GRID_DEFAULT = [0.0, 0.25, 0.5, 0.75, 1.0]


@dataclass(frozen=True)
class Candidate:
    method: str  # "random_stratified" | "tpe"
    batch: int
    idx: int
    k: int
    p: float
    graph_seed: int
    wiring_seed: int
    te: float
    orc: float
    orc_raw: float
    connected: bool
    clustering: float = float("nan")  # for coverage-aware selection (plot2_revision)
    path_length: float = float("nan")
    spectral_radius: float = float("nan")  # Plot2_revision3: third proxy
    # Plot 2 Overhaul: primary objectives (TE_res, sigma)
    sigma: float = float("nan")
    te_res: float = float("nan")
    orc_res: float = float("nan")
    # modular_ws_flex integration
    generator_mode: str = "ws_flex"
    M: Optional[int] = None
    p_out: Optional[float] = None
    r_out: Optional[float] = None


@dataclass(frozen=True)
class NormBounds:
    """
    Fixed normalization bounds for primary objectives (TE_res, sigma) and legacy (TE, ORC).
    """

    te_lo: float
    te_hi: float
    orc_lo: float
    orc_hi: float
    q_lo: float
    q_hi: float
    M_ref: int
    # Plot 2 Overhaul: primary objective bounds
    te_res_lo: float = 0.0
    te_res_hi: float = 1.0
    sigma_lo: float = 0.0
    sigma_hi: float = 1.0
    use_te_res_sigma: bool = False  # When True, use (TE_res, sigma) for Pareto/HV


def _now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_unique_run_id(*, repo_root: Path, output_root: str, run_id: str) -> str:
    """
    Avoid overwriting an existing Plot 2 run directory.
    """
    base = str(run_id)

    def _collides(rid: str) -> bool:
        return (repo_root / output_root / rid).exists()

    if not _collides(base):
        return base

    for i in range(2, 10_000):
        candidate = f"{base}_v{i}"
        if not _collides(candidate):
            print(f"[PLOT2] Detected existing run_id '{base}', using '{candidate}' to avoid overwriting.")
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


def _normalize_fixed(values: np.ndarray, *, lo: float, hi: float) -> np.ndarray:
    """
    Normalize values to [0,1] using fixed bounds (with clipping).
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    lo_f = float(lo)
    hi_f = float(hi)
    if not np.isfinite(lo_f) or not np.isfinite(hi_f) or hi_f <= lo_f:
        return np.zeros_like(v, dtype=float)
    out = (v - lo_f) / (hi_f - lo_f)
    return np.clip(out, 0.0, 1.0).astype(float)


def _compute_reference_bounds(
    *,
    analyzer: TopologyAnalyzer,
    H: int,
    k_values: Sequence[int],
    M_ref: int,
    q_lo: float,
    q_hi: float,
    rng: np.random.Generator,
    max_attempts_per_candidate: int = 200,
    degree_regimes: Optional[Dict[str, List[int]]] = None,
) -> Tuple[NormBounds, Dict[str, Any]]:
    """
    Sample a reference set of random feasible WS-Flex graphs and compute fixed TE/ORC bounds
    using the (q_lo, q_hi) quantiles. F4: If degree_regimes provided, also compute C/L tertile
    edges per regime for deterministic binning.
    """
    if int(M_ref) <= 0:
        raise ValueError("M_ref must be positive")
    if not k_values:
        raise ValueError("k_values must be non-empty")
    ql = float(q_lo)
    qh = float(q_hi)
    if not (0.0 <= ql < qh <= 1.0):
        raise ValueError(f"Invalid quantile range: q_lo={q_lo}, q_hi={q_hi} (require 0<=q_lo<q_hi<=1)")

    te_vals: List[float] = []
    orc_vals: List[float] = []
    cl_by_regime: Dict[str, List[Tuple[float, float]]] = {}

    need = int(M_ref)
    attempts = 0
    max_attempts = int(max_attempts_per_candidate) * max(1, need)
    while len(te_vals) < need:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Failed to sample M_ref={need} connected reference graphs (got {len(te_vals)}).")
        k = int(rng.choice(list(k_values)))
        p = float(rng.uniform(0.0, 1.0))
        graph_seed = int(rng.integers(0, 2**31 - 1))
        G = _make_ws_graph(H, k, p, seed=graph_seed)
        if not nx.is_connected(G):
            continue
        te, orc, _orc_raw = _compute_te_orc(analyzer, G)
        if not (np.isfinite(te) and np.isfinite(orc)):
            continue
        te_vals.append(float(te))
        orc_vals.append(float(orc))
        if degree_regimes:
            topo = analyzer.analyze_graph(G)
            C = float(topo.get("clustering_coefficient", float("nan")))
            L = float(topo.get("avg_path_length", float("nan")))
            if np.isfinite(C) and np.isfinite(L):
                regime = _k_to_regime(k, degree_regimes=degree_regimes)
                if regime is not None:
                    cl_by_regime.setdefault(str(regime), []).append((C, L))

    te_arr = np.asarray(te_vals, dtype=float)
    orc_arr = np.asarray(orc_vals, dtype=float)
    te_lo = float(np.nanquantile(te_arr, ql))
    te_hi = float(np.nanquantile(te_arr, qh))
    orc_lo = float(np.nanquantile(orc_arr, ql))
    orc_hi = float(np.nanquantile(orc_arr, qh))

    # Fallbacks for degenerate cases.
    if not (np.isfinite(te_lo) and np.isfinite(te_hi)) or te_hi <= te_lo:
        te_lo = float(np.nanmin(te_arr))
        te_hi = float(np.nanmax(te_arr))
    if not (np.isfinite(orc_lo) and np.isfinite(orc_hi)) or orc_hi <= orc_lo:
        orc_lo = float(np.nanmin(orc_arr))
        orc_hi = float(np.nanmax(orc_arr))

    bounds = NormBounds(te_lo=te_lo, te_hi=te_hi, orc_lo=orc_lo, orc_hi=orc_hi, q_lo=ql, q_hi=qh, M_ref=need)
    meta = {
        "M_ref": int(need),
        "q_lo": float(ql),
        "q_hi": float(qh),
        "te_lo": float(te_lo),
        "te_hi": float(te_hi),
        "orc_lo": float(orc_lo),
        "orc_hi": float(orc_hi),
        "te_min_ref": float(np.nanmin(te_arr)),
        "te_max_ref": float(np.nanmax(te_arr)),
        "orc_min_ref": float(np.nanmin(orc_arr)),
        "orc_max_ref": float(np.nanmax(orc_arr)),
    }
    # F4: Compute C/L tertile edges from reference for deterministic binning
    cl_bin_edges: Dict[str, Dict[str, Tuple[float, float]]] = {}
    if degree_regimes and cl_by_regime:
        for r in degree_regimes:
            pts = cl_by_regime.get(str(r), [])
            if not pts:
                cl_bin_edges[str(r)] = {"C": (float("nan"), float("nan")), "L": (float("nan"), float("nan"))}
                continue
            Cs = [p[0] for p in pts]
            Ls = [p[1] for p in pts]
            c_lo = float(np.nanpercentile(Cs, 33.33)) if len(Cs) >= 3 else float(np.nanmin(Cs))
            c_hi = float(np.nanpercentile(Cs, 66.67)) if len(Cs) >= 3 else float(np.nanmax(Cs))
            l_lo = float(np.nanpercentile(Ls, 33.33)) if len(Ls) >= 3 else float(np.nanmin(Ls))
            l_hi = float(np.nanpercentile(Ls, 66.67)) if len(Ls) >= 3 else float(np.nanmax(Ls))
            if c_hi <= c_lo:
                c_hi = c_lo + 1e-9
            if l_hi <= l_lo:
                l_hi = l_lo + 1e-9
            cl_bin_edges[str(r)] = {"C": (c_lo, c_hi), "L": (l_lo, l_hi)}
        meta["cl_bin_edges"] = cl_bin_edges
    return bounds, meta


def _make_ws_graph(H: int, k: int, p: float, seed: int) -> nx.Graph:
    return nx.watts_strogatz_graph(int(H), int(k), float(p), seed=int(seed))


def _make_graph_for_mode(
    H: int,
    k: int,
    p: float,
    graph_seed: int,
    generator_mode: str,
    seed_mod_params: int,
    rng: Optional[np.random.Generator] = None,
    sample_idx: int = 0,
) -> Tuple[nx.Graph, Optional[Dict[str, Any]]]:
    """
    Make graph by generator_mode. Returns (G, mod_params) with mod_params=None for ws_flex.
    """
    if generator_mode == "modular_ws_flex":
        sm_seed = seed_mod_params + sample_idx
        M, p_out, r_out = sample_modular_params(
            H, sm_seed, rng=rng,
            M_values=DEFAULT_M_VALUES,
            p_out_lo=DEFAULT_P_OUT_LO, p_out_hi=DEFAULT_P_OUT_HI,
            r_out_lo=DEFAULT_R_OUT_LO, r_out_hi=DEFAULT_R_OUT_HI,
        )
        G, params = make_ws_flex_graph(
            H, k, p, graph_seed,
            generator_mode="modular_ws_flex",
            M=M, p_out=p_out, r_out=r_out,
        )
        return G, {"M": M, "p_out": p_out, "r_out": r_out}
    G = _make_ws_graph(H, k, p, graph_seed)
    return G, None


def _compute_te_orc(analyzer: TopologyAnalyzer, G: nx.Graph) -> Tuple[float, float, float]:
    # Canonical definitions (see Patch_Note_ORC_Signed.md):
    # - TE: exact degree entropy normalized by log(N) (in [0,1])
    # - ORC: signed mean Ollivier–Ricci curvature (no abs)
    from architecture_refinement.metrics_te_orc import compute_te_orc

    te, orc, _dbg = compute_te_orc(G, orc_alpha=0.5)
    orc_raw = float(orc)
    return float(te), float(orc), float(orc_raw)


def _graph_hash_from_adj(
    adj: np.ndarray,
    H: int,
    k: int,
    p: float,
    graph_seed: int,
    mod_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Portable hash: hash(undirected_adj, H, k, p, graph_seed[, M, p_out, r_out]) for de-duplication."""
    adj_hex = np.asarray(adj).tobytes().hex()
    if mod_params and any(
        mod_params.get(x) is not None for x in ("M", "p_out", "r_out")
    ):
        M = mod_params.get("M", "")
        p_out = mod_params.get("p_out", "")
        r_out = mod_params.get("r_out", "")
        key = f"{H}|{k}|{p}|{graph_seed}|{M}|{p_out}|{r_out}|{adj_hex}"
    else:
        key = f"{H}|{k}|{p}|{graph_seed}|{adj_hex}"
    return hashlib.sha256(key.encode()).hexdigest()


def _orient_seed_from_graph_hash(graph_hash: str) -> int:
    """Spec G2: s_orient(g) = hash(graph_hash) mod 2^31-1 for deterministic orientation."""
    return int(graph_hash, 16) % (2**31 - 1)


def _compute_full_metrics(
    analyzer: TopologyAnalyzer,
    G: nx.Graph,
    k: int,
    H: int,
    graph_seed: int,
    p: float,
    mu_te_by_k: Optional[Dict[int, float]] = None,
    mu_orc_by_k: Optional[Dict[int, float]] = None,
    mod_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute TE, ORC, TE_res, ORC_res, sigma, C, L for a graph.
    Plot 2 Overhaul: primary objectives are (TE_res, sigma).
    """
    from architecture_refinement.metrics_te_orc import (
        compute_te_orc,
        compute_te_residual,
        compute_orc_residual,
    )
    from architecture_refinement.small_world_metrics import compute_small_worldness

    te, orc, _ = compute_te_orc(G, orc_alpha=0.5)
    topo = analyzer.analyze_graph(G)
    C = float(topo.get("clustering_coefficient", float("nan")))
    L = float(topo.get("avg_path_length", float("nan")))
    spectral_radius = float(topo.get("spectral_radius", float("nan")))

    adj = _undirected_hidden_adj(G, H)
    gh = _graph_hash_from_adj(adj, H, k, p, graph_seed, mod_params=mod_params)
    sigma, _, _, _, _, _ = compute_small_worldness(G, graph_id=gh, use_analytic_er=False)

    te_res = compute_te_residual(te, k, mu_te_by_k or {}) if mu_te_by_k else float(te)
    orc_res = compute_orc_residual(orc, k, mu_orc_by_k or {}) if mu_orc_by_k else float(orc)

    return {
        "te": float(te),
        "orc": float(orc),
        "orc_raw": float(orc),
        "te_res": float(te_res),
        "orc_res": float(orc_res),
        "sigma": float(sigma) if np.isfinite(sigma) else float("nan"),
        "C": C,
        "L": L,
        "clustering": C,
        "path_length": L,
        "spectral_radius": spectral_radius,
        "graph_hash": gh,
    }


def _undirected_hidden_adj(G: nx.Graph, H: int) -> np.ndarray:
    A = nx.to_numpy_array(G, dtype=np.int8)
    A = (A != 0).astype(np.int8)
    if A.shape != (H, H):
        raise ValueError(f"Unexpected undirected adjacency shape: {A.shape} (expected {(H, H)})")
    return A


def _oriented_hidden_adj(G: nx.Graph, H: int, seed: int) -> np.ndarray:
    """
    Deterministic orientation rule: delegate to WsFlexHiddenWiring(..., hidden_edge_orientation="random_oriented").
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


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text), encoding="utf-8")


def _load_proxy_viability_output(
    proxy_viability_dir: Path,
) -> Tuple[Dict[str, Dict[str, Tuple[float, float]]], Dict[int, float], Dict[int, float], NormBounds]:
    """
    Load frozen bin edges, mu_orc_by_k, mu_te_by_k, and normalization bounds from
    run_plot2_proxy_viability.py output.
    """
    frozen_path = proxy_viability_dir / "frozen_bin_edges.json"
    mu_orc_path = proxy_viability_dir / "mu_orc_by_k.json"
    mu_te_path = proxy_viability_dir / "mu_te_by_k.json"
    if not frozen_path.exists():
        raise FileNotFoundError(
            f"Proxy viability output not found: {frozen_path}. "
            "Run run_plot2_proxy_viability.py first with --output_dir."
        )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    bin_edges = frozen.get("frozen_bin_edges", frozen)
    # Convert to regime -> {C: (lo,hi), L: (lo,hi)}
    fixed_tertile_edges: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for r, edges in bin_edges.items():
        c = edges.get("C")
        l = edges.get("L")
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            c_lo, c_hi = float(c[0]), float(c[1])
        elif isinstance(c, dict):
            c_lo, c_hi = float(c.get("lo", 0)), float(c.get("hi", 1))
        else:
            c_lo, c_hi = float("nan"), float("nan")
        if isinstance(l, (list, tuple)) and len(l) >= 2:
            l_lo, l_hi = float(l[0]), float(l[1])
        elif isinstance(l, dict):
            l_lo, l_hi = float(l.get("lo", 0)), float(l.get("hi", 1))
        else:
            l_lo, l_hi = float("nan"), float("nan")
        fixed_tertile_edges[str(r)] = {"C": (c_lo, c_hi), "L": (l_lo, l_hi)}

    def _load_mu(path: Path) -> Dict[int, float]:
        if not path.exists():
            return {}
        d = json.loads(path.read_text(encoding="utf-8"))
        return {int(k): float(v) for k, v in d.items()}

    mu_orc_by_k = _load_mu(mu_orc_path)
    mu_te_by_k = _load_mu(mu_te_path)

    te_res_lo = float(frozen.get("te_res_lo", 0.0))
    te_res_hi = float(frozen.get("te_res_hi", 1.0))
    sigma_lo = float(frozen.get("sigma_lo", 0.0))
    sigma_hi = float(frozen.get("sigma_hi", 1.0))
    if te_res_hi <= te_res_lo:
        te_res_hi = te_res_lo + 1e-9
    if sigma_hi <= sigma_lo:
        sigma_hi = sigma_lo + 1e-9

    bounds = NormBounds(
        te_lo=te_res_lo,
        te_hi=te_res_hi,
        orc_lo=0.0,
        orc_hi=1.0,
        q_lo=0.05,
        q_hi=0.95,
        M_ref=int(frozen.get("M_ref", 5000)),
        te_res_lo=te_res_lo,
        te_res_hi=te_res_hi,
        sigma_lo=sigma_lo,
        sigma_hi=sigma_hi,
        use_te_res_sigma=True,
    )
    return fixed_tertile_edges, mu_orc_by_k, mu_te_by_k, bounds


def _get_git_commit(repo_root: Path) -> str:
    """Return git HEAD commit hash or 'unknown' (Spec 3 PATCH 4)."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(repo_root),
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _capture_environment_info(repo_root: Path, plot2_dir: Path, command_line: str) -> None:
    """
    I1: Write environment capture to plot2_dir for reproducibility.
    Writes: conda_env.yml or pip_freeze.txt, git commit, command line, hostname, CUDA info.
    """
    env_dir = plot2_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    info: Dict[str, Any] = {
        "git_commit": _get_git_commit(repo_root),
        "command_line": command_line,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "captured_at": datetime.now().isoformat(),
    }
    # Try conda env export first
    try:
        r = subprocess.run(
            ["conda", "env", "export", "--no-builds"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode == 0 and r.stdout:
            (env_dir / "conda_env.yml").write_text(r.stdout, encoding="utf-8")
            info["env_source"] = "conda_env.yml"
    except Exception:
        pass
    # Fallback: pip freeze
    if not (env_dir / "conda_env.yml").exists():
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if r.returncode == 0 and r.stdout:
                (env_dir / "pip_freeze.txt").write_text(r.stdout, encoding="utf-8")
                info["env_source"] = "pip_freeze.txt"
        except Exception:
            pass
    # CUDA info
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_count"] = int(torch.cuda.device_count())
            info["cuda_device_name"] = str(torch.cuda.get_device_name(0)) if torch.cuda.device_count() else None
    except ImportError:
        info["cuda_available"] = None
        info["cuda_note"] = "torch not imported"
    _write_json(env_dir / "environment_info.json", info)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_capacity_manifest(selected_rows: List[Dict[str, Any]], H: int) -> Dict[str, Any]:
    """Plot2_revision3 Step C: capacity manifest with H, E_active, mask_density per selected graph."""
    entries: List[Dict[str, Any]] = []
    for r in selected_rows:
        model_name = str(r.get("model_name", "unknown"))
        if "hidden_adj_undirected" in r:
            adj = np.asarray(r["hidden_adj_undirected"])
            E_active = int(np.count_nonzero(adj))
        elif "mask_density" in r:
            E_active = int(round(float(r["mask_density"]) * H * H))
        else:
            E_active = int(r.get("density", 0) * H * (H - 1) / 2) if r.get("density") else 0
        mask_density = E_active / (H * H) if H > 0 else 0.0
        entry: Dict[str, Any] = {
            "model_name": model_name,
            "method": str(r.get("method", "unknown")),
            "H": H,
            "E_active": E_active,
            "mask_density": float(mask_density),
        }
        if "spectral_radius" in r and np.isfinite(r.get("spectral_radius", float("nan"))):
            entry["spectral_radius"] = float(r["spectral_radius"])
        if "spectral_radius_directed" in r and np.isfinite(r.get("spectral_radius_directed", float("nan"))):
            entry["spectral_radius_directed"] = float(r["spectral_radius_directed"])
        entries.append(entry)
    return {"schema_version": 1, "H": H, "entries": entries}


def _graph_id_hash(row: Dict[str, Any]) -> str:
    """Stable portable hash of graph identity (Plot2_revision2; do not use Python hash())."""
    if row.get("wiring_kind") == "ws_flex" and "k" in row and "p" in row and "graph_seed" in row:
        key = f"{row['k']}_{row['p']}_{row['graph_seed']}"
    else:
        key = f"{row.get('method', '')}_{row.get('model_name', '')}_{row.get('wiring_seed', '')}_{row.get('sparsity_level', '')}"
    return hashlib.sha256(key.encode()).hexdigest()


def _pareto_membership_2d(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Fast 2D max-max Pareto membership for arrays. Returns boolean mask.
    """
    n = int(xs.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=bool)
    order = np.lexsort((-ys, -xs))  # sort by x desc, then y desc
    best_y = -np.inf
    on_pf = np.zeros((n,), dtype=bool)
    for idx in order:
        y = float(ys[idx])
        if y > best_y:
            on_pf[idx] = True
            best_y = y
    return on_pf


def _compute_hv(
    cands: Sequence[Candidate],
    *,
    bounds: NormBounds,
    ref: Tuple[float, float] = (-0.05, -0.05),
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute hypervolume. Uses (TE_res, sigma) when bounds.use_te_res_sigma else (TE, ORC)."""
    if getattr(bounds, "use_te_res_sigma", False):
        xs = np.asarray([c.te_res for c in cands], dtype=float)
        ys = np.asarray([c.sigma for c in cands], dtype=float)
        lo_x, hi_x = bounds.te_res_lo, bounds.te_res_hi
        lo_y, hi_y = bounds.sigma_lo, bounds.sigma_hi
    else:
        xs = np.asarray([c.te for c in cands], dtype=float)
        ys = np.asarray([c.orc for c in cands], dtype=float)
        lo_x, hi_x = bounds.te_lo, bounds.te_hi
        lo_y, hi_y = bounds.orc_lo, bounds.orc_hi
    x_n = _normalize_fixed(xs, lo=lo_x, hi=hi_x)
    y_n = _normalize_fixed(ys, lo=lo_y, hi=hi_y)
    pf = pareto_front_2d(list(zip(x_n.tolist(), y_n.tolist())))
    hv = hypervolume_2d(pf, ref=ref)
    return float(hv), x_n, y_n


def _degree_regimes_to_bins(degree_regimes: Dict[str, List[int]]) -> Dict[str, Tuple[int, int]]:
    """Convert degree_regimes to (k_lo, k_hi) regime_bins for capacity filter."""
    return {name: (min(ks), max(ks)) for name, ks in degree_regimes.items() if ks}


def _sample_random_stratified_batch(
    *,
    analyzer: TopologyAnalyzer,
    H: int,
    degree_regimes: Dict[str, List[int]],
    batch_size: int,
    rng: np.random.Generator,
    max_attempts_per_candidate: int = 200,
    capacity_filter_on: bool = False,
    diagnostics_out: Optional[Dict[str, Any]] = None,
    mu_te_by_k: Optional[Dict[int, float]] = None,
    mu_orc_by_k: Optional[Dict[int, float]] = None,
    generator_mode: str = "ws_flex",
    seed_mod_params: int = 202607,
    sample_idx_offset: int = 0,
) -> List[Candidate]:
    regime_names = list(degree_regimes.keys())
    n_reg = len(degree_regimes)
    if n_reg <= 0:
        raise ValueError("degree_regimes must be non-empty")

    # Equal allocation across regimes each batch.
    base = batch_size // n_reg
    rem = batch_size % n_reg
    counts = [base + (1 if i < rem else 0) for i in range(n_reg)]

    out: List[Candidate] = []
    batch = 0  # caller overwrites

    if diagnostics_out is not None:
        if "attempts_by_regime" not in diagnostics_out:
            diagnostics_out["attempts_by_regime"] = {str(r): 0 for r in regime_names}
        if "successes_by_regime" not in diagnostics_out:
            diagnostics_out["successes_by_regime"] = {str(r): 0 for r in regime_names}
        if "rejections_disconnected" not in diagnostics_out:
            diagnostics_out["rejections_disconnected"] = 0
        if "rejections_capacity" not in diagnostics_out:
            diagnostics_out["rejections_capacity"] = 0

    for r_i, rname in enumerate(regime_names):
        ks = degree_regimes[rname]
        need = int(counts[r_i])
        got = 0
        attempts = 0
        while got < need:
            attempts += 1
            if attempts > max_attempts_per_candidate * max(1, need):
                raise RuntimeError(f"Failed to sample {need} connected graphs for regime {rname} (got {got}).")
            k = int(rng.choice(ks))
            p = float(rng.uniform(0.0, 1.0))
            graph_seed = int(rng.integers(0, 2**31 - 1))
            G, mod_params = _make_graph_for_mode(
                H, k, p, graph_seed, generator_mode, seed_mod_params,
                rng=rng, sample_idx=sample_idx_offset + len(out),
            )
            if diagnostics_out is not None:
                diagnostics_out["attempts_by_regime"][str(rname)] = (
                    diagnostics_out["attempts_by_regime"].get(str(rname), 0) + 1
                )
            if not nx.is_connected(G):
                if diagnostics_out is not None:
                    diagnostics_out["rejections_disconnected"] += 1
                continue
            # Spec G2: wiring_seed = s_orient(g) = hash(graph_hash) mod 2^31-1
            adj = _undirected_hidden_adj(G, H)
            gh = _graph_hash_from_adj(adj, H, k, p, graph_seed, mod_params=mod_params)
            wiring_seed = _orient_seed_from_graph_hash(gh)
            if capacity_filter_on:
                cap_ok, _, _ = capacity_filter(G, k, wiring_seed, _degree_regimes_to_bins(degree_regimes), H)
                if not cap_ok:
                    if diagnostics_out is not None:
                        diagnostics_out["rejections_capacity"] += 1
                    continue
            if mu_te_by_k is not None and mu_orc_by_k is not None:
                m = _compute_full_metrics(
                    analyzer, G, k, H, graph_seed, p, mu_te_by_k, mu_orc_by_k,
                    mod_params=mod_params,
                )
                out.append(
                    Candidate(
                        method="random_stratified",
                        batch=batch,
                        idx=len(out),
                        k=k,
                        p=p,
                        graph_seed=graph_seed,
                        wiring_seed=wiring_seed,
                        te=m["te"],
                        orc=m["orc"],
                        orc_raw=m["orc_raw"],
                        connected=True,
                        clustering=m["clustering"],
                        path_length=m["path_length"],
                        spectral_radius=m["spectral_radius"],
                        sigma=m["sigma"],
                        te_res=m["te_res"],
                        orc_res=m["orc_res"],
                        generator_mode=generator_mode,
                        M=mod_params.get("M") if mod_params else None,
                        p_out=mod_params.get("p_out") if mod_params else None,
                        r_out=mod_params.get("r_out") if mod_params else None,
                    )
                )
            else:
                te, orc, orc_raw = _compute_te_orc(analyzer, G)
                topo = analyzer.analyze_graph(G)
                clustering = float(topo.get("clustering_coefficient", float("nan")))
                path_length = float(topo.get("avg_path_length", float("nan")))
                spectral_radius = float(topo.get("spectral_radius", float("nan")))
                out.append(
                    Candidate(
                        method="random_stratified",
                        batch=batch,
                        idx=len(out),
                        k=k,
                        p=p,
                        graph_seed=graph_seed,
                        wiring_seed=wiring_seed,
                        te=te,
                        orc=orc,
                        orc_raw=orc_raw,
                        connected=True,
                        clustering=clustering,
                        path_length=path_length,
                        spectral_radius=spectral_radius,
                        generator_mode=generator_mode,
                        M=mod_params.get("M") if mod_params else None,
                        p_out=mod_params.get("p_out") if mod_params else None,
                        r_out=mod_params.get("r_out") if mod_params else None,
                    )
                )
            got += 1
            if diagnostics_out is not None:
                diagnostics_out["successes_by_regime"][str(rname)] = (
                    diagnostics_out["successes_by_regime"].get(str(rname), 0) + 1
                )

    return out


def _sample_tpe_feasible(
    *,
    analyzer: TopologyAnalyzer,
    H: int,
    k_values: List[int],
    rng: np.random.Generator,
    n_feasible_needed: int,
    seed: int,
    max_trials: int,
    batch_idx: int,
) -> List[Candidate]:
    raise RuntimeError("_sample_tpe_feasible is deprecated; TPE is run inside _run_training_free_search to preserve study history.")


def _build_random_pool(
    *,
    analyzer: TopologyAnalyzer,
    H: int,
    degree_regimes: Dict[str, List[int]],
    M0: int,
    dM: int,
    M_max: int,
    base_seed: int,
    max_attempts_per_candidate: int = 200,
    capacity_filter_on: bool = False,
    mu_te_by_k: Optional[Dict[int, float]] = None,
    mu_orc_by_k: Optional[Dict[int, float]] = None,
    generator_mode: str = "ws_flex",
    seed_mod_params: int = 202607,
) -> Tuple[List[Candidate], Dict[str, Any]]:
    """
    Build a single shared random WS-Flex pool of size M_max (plot2_revision Baseline A/B).
    Same regime-stratified sampling as random_stratified path, no TPE, no HV early-stop.
    Returns (candidates, minimal_log).
    """
    rng = np.random.default_rng(int(base_seed))
    all_cands: List[Candidate] = []
    batch_idx = 0
    rejection_diagnostics: Dict[str, Any] = {}
    while len(all_cands) < int(M_max):
        batch_idx += 1
        target_new = int(M0) if len(all_cands) == 0 else int(dM)
        remaining = int(M_max) - len(all_cands)
        target_new = min(target_new, remaining)
        batch = _sample_random_stratified_batch(
            analyzer=analyzer,
            H=H,
            degree_regimes=degree_regimes,
            batch_size=target_new,
            rng=rng,
            max_attempts_per_candidate=max_attempts_per_candidate,
            capacity_filter_on=capacity_filter_on,
            diagnostics_out=rejection_diagnostics,
            mu_te_by_k=mu_te_by_k,
            mu_orc_by_k=mu_orc_by_k,
            generator_mode=generator_mode,
            seed_mod_params=seed_mod_params,
            sample_idx_offset=len(all_cands),
        )
        fixed = []
        for c in batch:
            cand_kw = dict(
                method="shared_random",
                batch=int(batch_idx),
                idx=int(len(all_cands) + len(fixed)),
                k=c.k,
                p=c.p,
                graph_seed=c.graph_seed,
                wiring_seed=c.wiring_seed,
                te=c.te,
                orc=c.orc,
                orc_raw=c.orc_raw,
                connected=c.connected,
                clustering=c.clustering,
                path_length=c.path_length,
                spectral_radius=c.spectral_radius,
                generator_mode=getattr(c, "generator_mode", "ws_flex"),
                M=getattr(c, "M", None),
                p_out=getattr(c, "p_out", None),
                r_out=getattr(c, "r_out", None),
            )
            if hasattr(c, "sigma") and np.isfinite(getattr(c, "sigma", float("nan"))):
                cand_kw["sigma"] = c.sigma
                cand_kw["te_res"] = getattr(c, "te_res", float("nan"))
                cand_kw["orc_res"] = getattr(c, "orc_res", float("nan"))
            fixed.append(Candidate(**cand_kw))
        all_cands.extend(fixed)
    log = {
        "schema_version": 2,
        "method": "shared_random",
        "M0": int(M0),
        "dM": int(dM),
        "M_max": int(M_max),
        "stop_reason": "max_budget",
        "n_candidates": len(all_cands),
    }
    if rejection_diagnostics:
        total_attempts = sum(rejection_diagnostics.get("attempts_by_regime", {}).values())
        total_successes = sum(rejection_diagnostics.get("successes_by_regime", {}).values())
        log["rejection_rate_by_regime"] = {
            r: 1.0 - (rejection_diagnostics.get("successes_by_regime", {}).get(r, 0) / max(1, rejection_diagnostics.get("attempts_by_regime", {}).get(r, 0)))
            for r in rejection_diagnostics.get("attempts_by_regime", {}).keys()
        }
        log["rejection_rate_overall"] = 1.0 - (total_successes / max(1, total_attempts))
        log["rejections_disconnected"] = rejection_diagnostics.get("rejections_disconnected", 0)
        log["rejections_capacity"] = rejection_diagnostics.get("rejections_capacity", 0)
    return all_cands, log


def _run_training_free_search(
    *,
    analyzer: TopologyAnalyzer,
    method: str,
    H: int,
    degree_regimes: Dict[str, List[int]],
    bounds: NormBounds,
    M0: int,
    dM: int,
    M_max: int,
    hv_window_W: int,
    hv_window_eps: float,
    hv_window_patience: int,
    pareto_new_m: int,
    pareto_patience_batches: int,
    base_seed: int,
    optuna_storage_url: Optional[str] = None,
    optuna_study_name: Optional[str] = None,
    capacity_filter_on: bool = False,
    mu_te_by_k: Optional[Dict[int, float]] = None,
    mu_orc_by_k: Optional[Dict[int, float]] = None,
    generator_mode: str = "ws_flex",
    seed_mod_params: int = 202607,
) -> Tuple[List[Candidate], Dict[str, Any]]:
    """
    Returns (all_candidates, search_log_dict).
    """
    rng = np.random.default_rng(int(base_seed))
    k_values = sorted({k for ks in degree_regimes.values() for k in ks})

    all_cands: List[Candidate] = []
    hv_history: List[Dict[str, Any]] = []
    hv_window_no_improve_streak = 0
    pareto_low_growth_streak = 0
    batch_idx = 0
    stop_reason = "max_budget"

    # For TPE, create ONE multi-objective study and keep it across batches so the sampler can learn.
    tpe_study = None
    tpe_current_batch = {"batch": 0}
    if method == "tpe":
        try:
            import optuna
        except Exception as e:
            raise ImportError(
                f"Optuna is required for Plot 2 TPE sampling but could not be imported: {e}\n"
                "Install it in your environment (e.g., pip install optuna)."
            )
        sampler = optuna.samplers.TPESampler(seed=int(base_seed), multivariate=True)
        # Persist study to disk so we can reload later for analysis.
        # If storage is None, Optuna keeps an in-memory study only (not acceptable for Plot 2).
        tpe_study = optuna.create_study(
            directions=["maximize", "maximize"],
            sampler=sampler,
            storage=str(optuna_storage_url) if optuna_storage_url else None,
            study_name=str(optuna_study_name) if optuna_study_name else None,
            load_if_exists=True,
        )

        use_primary = getattr(bounds, "use_te_res_sigma", False)

        def _tpe_objective(trial: "optuna.Trial") -> Tuple[float, float]:
            k = int(trial.suggest_categorical("k", k_values))
            p = float(trial.suggest_float("p", 0.0, 1.0))
            graph_seed = int(rng.integers(0, 2**31 - 1))
            G, mod_params = _make_graph_for_mode(
                H, k, p, graph_seed, generator_mode, seed_mod_params,
                rng=rng, sample_idx=len(all_cands),
            )
            if not nx.is_connected(G):
                raise optuna.TrialPruned()
            # Spec G2: wiring_seed = s_orient(g) = hash(graph_hash) mod 2^31-1
            adj = _undirected_hidden_adj(G, H)
            gh = _graph_hash_from_adj(adj, H, k, p, graph_seed, mod_params=mod_params)
            wiring_seed = _orient_seed_from_graph_hash(gh)
            if capacity_filter_on:
                cap_ok, _, _ = capacity_filter(G, k, wiring_seed, _degree_regimes_to_bins(degree_regimes), H)
                if not cap_ok:
                    raise optuna.TrialPruned()
            if use_primary and mu_te_by_k is not None and mu_orc_by_k is not None:
                m = _compute_full_metrics(
                    analyzer, G, k, H, graph_seed, p, mu_te_by_k, mu_orc_by_k,
                    mod_params=mod_params,
                )
                cand_idx = int(len(all_cands))
                try:
                    trial.set_user_attr("graph_seed", int(graph_seed))
                    trial.set_user_attr("wiring_seed", int(wiring_seed))
                    trial.set_user_attr("batch", int(tpe_current_batch["batch"]))
                    trial.set_user_attr("cand_idx", int(cand_idx))
                    trial.set_user_attr("connected", True)
                    trial.set_user_attr("orc_raw", float(m["orc_raw"]))
                except Exception:
                    pass
                all_cands.append(
                    Candidate(
                        method="tpe",
                        batch=int(tpe_current_batch["batch"]),
                        idx=int(cand_idx),
                        k=k,
                        p=p,
                        graph_seed=graph_seed,
                        wiring_seed=wiring_seed,
                        te=m["te"],
                        orc=m["orc"],
                        orc_raw=m["orc_raw"],
                        connected=True,
                        clustering=m["clustering"],
                        path_length=m["path_length"],
                        spectral_radius=m["spectral_radius"],
                        sigma=m["sigma"],
                        te_res=m["te_res"],
                        orc_res=m["orc_res"],
                        generator_mode=generator_mode,
                        M=mod_params.get("M") if mod_params else None,
                        p_out=mod_params.get("p_out") if mod_params else None,
                        r_out=mod_params.get("r_out") if mod_params else None,
                    )
                )
                return float(m["te_res"]), float(m["sigma"])
            else:
                te, orc, orc_raw = _compute_te_orc(analyzer, G)
                topo = analyzer.analyze_graph(G)
                clustering = float(topo.get("clustering_coefficient", float("nan")))
                path_length = float(topo.get("avg_path_length", float("nan")))
                spectral_radius = float(topo.get("spectral_radius", float("nan")))
                cand_idx = int(len(all_cands))
                try:
                    trial.set_user_attr("graph_seed", int(graph_seed))
                    trial.set_user_attr("wiring_seed", int(wiring_seed))
                    trial.set_user_attr("batch", int(tpe_current_batch["batch"]))
                    trial.set_user_attr("cand_idx", int(cand_idx))
                    trial.set_user_attr("connected", True)
                    trial.set_user_attr("orc_raw", float(orc_raw))
                except Exception:
                    pass
                all_cands.append(
                    Candidate(
                        method="tpe",
                        batch=int(tpe_current_batch["batch"]),
                        idx=int(cand_idx),
                        k=k,
                        p=p,
                        graph_seed=graph_seed,
                        wiring_seed=wiring_seed,
                        te=te,
                        orc=orc,
                        orc_raw=orc_raw,
                        connected=True,
                        clustering=clustering,
                        path_length=path_length,
                        spectral_radius=spectral_radius,
                        generator_mode=generator_mode,
                        M=mod_params.get("M") if mod_params else None,
                        p_out=mod_params.get("p_out") if mod_params else None,
                        r_out=mod_params.get("r_out") if mod_params else None,
                    )
                )
                return float(te), float(orc)

    while len(all_cands) < int(M_max):
        batch_idx += 1
        target_new = int(M0) if len(all_cands) == 0 else int(dM)
        remaining = int(M_max) - len(all_cands)
        target_new = min(target_new, remaining)

        if method == "random_stratified":
            batch = _sample_random_stratified_batch(
                analyzer=analyzer,
                H=H,
                degree_regimes=degree_regimes,
                batch_size=target_new,
                rng=rng,
                capacity_filter_on=capacity_filter_on,
                mu_te_by_k=mu_te_by_k,
                mu_orc_by_k=mu_orc_by_k,
                generator_mode=generator_mode,
                seed_mod_params=seed_mod_params,
                sample_idx_offset=len(all_cands),
            )
            # fix batch index, global idx
            fixed = []
            for c in batch:
                fixed.append(
                    Candidate(
                        **{**c.__dict__, "batch": int(batch_idx), "idx": int(len(all_cands) + len(fixed))}
                    )
                )
            batch = fixed
            all_cands.extend(batch)
        elif method == "tpe":
            if tpe_study is None:
                raise RuntimeError("Internal error: tpe_study was not initialized.")
            # TPE needs more trial attempts due to connectivity pruning.
            max_trials = int(target_new) * 50
            target_total = int(len(all_cands) + target_new)
            tpe_current_batch["batch"] = int(batch_idx)

            def _stop_when_enough(study_: "optuna.Study", trial: "optuna.FrozenTrial") -> None:
                if len(all_cands) >= target_total:
                    study_.stop()

            # Note: objective appends directly into all_cands with correct batch+idx.
            tpe_study.optimize(_tpe_objective, n_trials=int(max_trials), callbacks=[_stop_when_enough], show_progress_bar=False)
            if len(all_cands) < target_total:
                raise RuntimeError(
                    f"TPE sampling failed to produce target_new={target_new} connected candidates "
                    f"(got {len(all_cands)} / target_total={target_total} within max_trials={max_trials})."
                )
        else:
            raise ValueError(f"Unknown method: {method}")

        hv, te_n, orc_n = _compute_hv(all_cands, bounds=bounds)
        pareto_mask = _pareto_membership_2d(te_n, orc_n)
        batches = np.asarray([int(c.batch) for c in all_cands], dtype=int)
        n_pareto_total = int(np.sum(pareto_mask))
        n_pareto_in_batch = int(np.sum(pareto_mask & (batches == int(batch_idx))))

        # Windowed HV improvement: compare HV_t vs HV_{t-W}
        rel_improve_vs_tminusW: Optional[float] = None
        if len(hv_history) >= int(hv_window_W):
            # hv_history currently contains entries for batches [1..t-1]
            # Compare current HV (t) to HV at (t-W).
            idx_tminusW = int(len(hv_history) - int(hv_window_W))
            prevW = float(hv_history[idx_tminusW]["hv"])
            cur = float(hv)
            if prevW <= 0.0:
                rel_improve_vs_tminusW = float("inf") if cur > prevW else 0.0
            else:
                rel_improve_vs_tminusW = float((cur - prevW) / prevW)
            if rel_improve_vs_tminusW < float(hv_window_eps):
                hv_window_no_improve_streak += 1
            else:
                hv_window_no_improve_streak = 0

        # Pareto growth stability: count Pareto-optimal points coming from the most recent batch.
        if n_pareto_in_batch < int(pareto_new_m):
            pareto_low_growth_streak += 1
        else:
            pareto_low_growth_streak = 0

        hv_entry = {
            "batch": int(batch_idx),
            "M": int(len(all_cands)),
            "hv": float(hv),
            "n_pareto_total": int(n_pareto_total),
            "n_pareto_in_batch": int(n_pareto_in_batch),
            "te_min": float(np.min([c.te for c in all_cands])) if all_cands else float("nan"),
            "te_max": float(np.max([c.te for c in all_cands])) if all_cands else float("nan"),
            "orc_min": float(np.min([c.orc for c in all_cands])) if all_cands else float("nan"),
            "orc_max": float(np.max([c.orc for c in all_cands])) if all_cands else float("nan"),
        }
        if rel_improve_vs_tminusW is not None:
            hv_entry["rel_improve_vs_tminusW"] = float(rel_improve_vs_tminusW)
        hv_entry["hv_window_no_improve_streak"] = int(hv_window_no_improve_streak)
        hv_entry["pareto_low_growth_streak"] = int(pareto_low_growth_streak)
        hv_history.append(hv_entry)

        # Stop when ANY locked early-stopping criterion fires (after at least the initial budget).
        if len(all_cands) >= int(M0):
            if hv_window_no_improve_streak >= int(hv_window_patience) and len(hv_history) > int(hv_window_W):
                stop_reason = f"hv_window_saturation_{hv_window_no_improve_streak}x(<{hv_window_eps})"
                break
            if pareto_low_growth_streak >= int(pareto_patience_batches):
                stop_reason = f"pareto_growth_saturation_{pareto_low_growth_streak}x(<{pareto_new_m})"
                break

    log = {
        "schema_version": 2,
        "method": method,
        "M0": int(M0),
        "dM": int(dM),
        "M_max": int(M_max),
        "hv_ref": [-0.05, -0.05],
        "optuna": (
            {
                "storage": str(optuna_storage_url) if optuna_storage_url else None,
                "study_name": str(optuna_study_name) if optuna_study_name else None,
                "n_pruned": (
                    sum(1 for t in tpe_study.get_trials() if t.state == optuna.trial.TrialState.PRUNED)
                    if tpe_study is not None
                    else None
                ),
            }
            if method == "tpe"
            else None
        ),
        "normalization": {
            "kind": "reference_set_quantiles",
            "M_ref": int(bounds.M_ref),
            "q_lo": float(bounds.q_lo),
            "q_hi": float(bounds.q_hi),
            "te_lo": float(bounds.te_lo),
            "te_hi": float(bounds.te_hi),
            "orc_lo": float(bounds.orc_lo),
            "orc_hi": float(bounds.orc_hi),
        },
        "stopping": {
            "hv_window_W": int(hv_window_W),
            "hv_window_eps": float(hv_window_eps),
            "hv_window_patience": int(hv_window_patience),
            "pareto_new_m": int(pareto_new_m),
            "pareto_patience_batches": int(pareto_patience_batches),
        },
        "stop_reason": stop_reason,
        "batches": hv_history,
    }
    return all_cands, log


def _cands_to_rows(
    cands: Sequence[Candidate],
    *,
    te_norm: np.ndarray,
    orc_norm: np.ndarray,
    pareto_mask: np.ndarray,
    score: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, c in enumerate(cands):
        row = {
            "method": c.method,
            "batch": int(c.batch),
            "idx": int(c.idx),
            "k": int(c.k),
            "p": float(c.p),
            "graph_seed": int(c.graph_seed),
            "wiring_seed": int(c.wiring_seed),
            "te": float(c.te),
            "orc": float(c.orc),
            "orc_raw": float(c.orc_raw),
            "connected": bool(c.connected),
            "clustering": float(c.clustering),
            "path_length": float(c.path_length),
            "spectral_radius": float(c.spectral_radius),
            "te_norm": float(te_norm[i]),
            "orc_norm": float(orc_norm[i]),
            "score": float(score[i]),
            "pareto": bool(pareto_mask[i]),
        }
        if hasattr(c, "sigma") and np.isfinite(getattr(c, "sigma", float("nan"))):
            row["sigma"] = float(c.sigma)
            row["te_res"] = float(getattr(c, "te_res", float("nan")))
            row["orc_res"] = float(getattr(c, "orc_res", float("nan")))
        rows.append(row)
    return rows


def _k_to_regime(k: int, *, degree_regimes: Dict[str, List[int]]) -> Optional[str]:
    kk = int(k)
    for name, ks in degree_regimes.items():
        if kk in {int(x) for x in ks}:
            return str(name)
    return None


def _summarize_k_distribution(*, k_values: Sequence[int], degree_regimes: Dict[str, List[int]]) -> Dict[str, Any]:
    ks = [int(k) for k in k_values]
    counts_by_k: Dict[str, int] = {}
    counts_by_regime: Dict[str, int] = {str(name): 0 for name in degree_regimes.keys()}
    unknown_regime = 0

    for k in ks:
        key = str(int(k))
        counts_by_k[key] = int(counts_by_k.get(key, 0) + 1)
        r = _k_to_regime(int(k), degree_regimes=degree_regimes)
        if r is None:
            unknown_regime += 1
        else:
            counts_by_regime[str(r)] = int(counts_by_regime.get(str(r), 0) + 1)

    uniq = sorted({int(k) for k in ks})
    out = {
        "n": int(len(ks)),
        "unique_k": uniq,
        "n_unique_k": int(len(uniq)),
        "k_min": int(min(uniq)) if uniq else None,
        "k_max": int(max(uniq)) if uniq else None,
        "counts_by_k": {k: int(v) for k, v in sorted(counts_by_k.items(), key=lambda kv: int(kv[0]))},
        "counts_by_regime": counts_by_regime,
        "unknown_regime": int(unknown_regime),
    }
    return out


# --- Coverage-aware selection (plot2_revision): (C, L) tertile binning and within-bin z-scores ---

_CL_BIN_LABELS = ("low", "medium", "high")


def _compute_cl_bins(
    cands: Sequence[Candidate],
    degree_regimes: Dict[str, List[int]],
    fixed_tertile_edges: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
) -> Tuple[
    List[Optional[str]],
    List[str],
    List[str],
    Dict[str, Any],
]:
    """
    Assign each candidate to (regime, C_bin, L_bin). Bins are tertiles (low/medium/high)
    computed per regime. F4: If fixed_tertile_edges provided, use those; else compute from pool.
    Candidates with nan C or L get "unknown" bin.

    Returns:
        regime_per_idx: regime name or None for each candidate index
        c_bin_per_idx: "low" | "medium" | "high" | "unknown"
        l_bin_per_idx: same
        diagnostics: dict with tertile_edges_per_regime, (C,L) distribution by regime, etc.
    """
    n = len(cands)
    regime_per_idx: List[Optional[str]] = [None] * n
    c_bin_per_idx: List[str] = ["unknown"] * n
    l_bin_per_idx: List[str] = ["unknown"] * n

    C_vals = np.array([float(c.clustering) for c in cands], dtype=float)
    L_vals = np.array([float(c.path_length) for c in cands], dtype=float)
    valid_c = np.isfinite(C_vals)
    valid_l = np.isfinite(L_vals)

    regime_names = list(degree_regimes.keys())
    # Build by_regime for tertile computation and diagnostics (always needed)
    by_regime: Dict[str, List[Tuple[float, float]]] = {str(r): [] for r in regime_names}
    for i, c in enumerate(cands):
        r = _k_to_regime(int(c.k), degree_regimes=degree_regimes)
        regime_per_idx[i] = r
        if r is None:
            continue
        if valid_c[i] and valid_l[i]:
            by_regime[str(r)].append((float(C_vals[i]), float(L_vals[i])))

    # Tertile edges: use fixed (F4) if provided, else compute from pool
    tertile_edges: Dict[str, Dict[str, Tuple[float, float]]] = {}
    if fixed_tertile_edges and set(fixed_tertile_edges.keys()) == set(regime_names):
        tertile_edges = dict(fixed_tertile_edges)
    else:
        for r in regime_names:
            pts = by_regime.get(r, [])
            if not pts:
                tertile_edges[r] = {"C": (float("nan"), float("nan")), "L": (float("nan"), float("nan"))}
                continue
            Cs = [p[0] for p in pts]
            Ls = [p[1] for p in pts]
            if len(Cs) >= 3:
                c_lo, c_hi = float(np.nanpercentile(Cs, 33.33)), float(np.nanpercentile(Cs, 66.67))
                if c_hi <= c_lo:
                    c_hi = c_lo + 1e-9
            else:
                c_lo, c_hi = float(np.nanmin(Cs)), float(np.nanmax(Cs))
                if c_hi <= c_lo:
                    c_hi = c_lo + 1e-9
            if len(Ls) >= 3:
                l_lo, l_hi = float(np.nanpercentile(Ls, 33.33)), float(np.nanpercentile(Ls, 66.67))
                if l_hi <= l_lo:
                    l_hi = l_lo + 1e-9
            else:
                l_lo, l_hi = float(np.nanmin(Ls)), float(np.nanmax(Ls))
                if l_hi <= l_lo:
                    l_hi = l_lo + 1e-9
            tertile_edges[r] = {"C": (c_lo, c_hi), "L": (l_lo, l_hi)}

    for i, c in enumerate(cands):
        r = _k_to_regime(int(c.k), degree_regimes=degree_regimes)
        regime_per_idx[i] = r

    # Assign bins
    for i, c in enumerate(cands):
        r = regime_per_idx[i]
        if r is None:
            continue
        if not (valid_c[i] and valid_l[i]):
            continue
        c_lo, c_hi = tertile_edges[r]["C"]
        l_lo, l_hi = tertile_edges[r]["L"]
        if np.isnan(c_lo) or np.isnan(c_hi):
            continue
        C = float(C_vals[i])
        L = float(L_vals[i])
        if C <= c_lo:
            c_bin_per_idx[i] = "low"
        elif C <= c_hi:
            c_bin_per_idx[i] = "medium"
        else:
            c_bin_per_idx[i] = "high"
        if np.isnan(l_lo) or np.isnan(l_hi):
            continue
        if L <= l_lo:
            l_bin_per_idx[i] = "low"
        elif L <= l_hi:
            l_bin_per_idx[i] = "medium"
        else:
            l_bin_per_idx[i] = "high"

    # Diagnostics: distribution of (C, L) by regime, occupied bins
    dist_by_regime: Dict[str, Dict[str, Any]] = {}
    for r in regime_names:
        pts = by_regime.get(r, [])
        if not pts:
            dist_by_regime[r] = {"n": 0, "C_min": None, "C_max": None, "L_min": None, "L_max": None}
            continue
        Cs = [p[0] for p in pts]
        Ls = [p[1] for p in pts]
        dist_by_regime[r] = {
            "n": len(pts),
            "C_min": float(np.nanmin(Cs)),
            "C_max": float(np.nanmax(Cs)),
            "L_min": float(np.nanmin(Ls)),
            "L_max": float(np.nanmax(Ls)),
        }
    occupied_cells: Dict[str, int] = {}
    for i in range(n):
        r = regime_per_idx[i]
        if r is None or c_bin_per_idx[i] == "unknown" or l_bin_per_idx[i] == "unknown":
            continue
        key = (r, c_bin_per_idx[i], l_bin_per_idx[i])
        occupied_cells[str(key)] = occupied_cells.get(str(key), 0) + 1

    diagnostics = {
        "tertile_edges_per_regime": tertile_edges,
        "distribution_by_regime": dist_by_regime,
        "occupied_bin_keys": list(occupied_cells.keys()),
        "n_occupied_cells": len(occupied_cells),
    }
    return regime_per_idx, c_bin_per_idx, l_bin_per_idx, diagnostics


def _zscore_within_bins(
    te: np.ndarray,
    orc: np.ndarray,
    regime_per_idx: List[Optional[str]],
    c_bin_per_idx: List[str],
    l_bin_per_idx: List[str],
) -> np.ndarray:
    """
    Z-score TE and ORC within each (regime, C_bin, L_bin) cell; return score = z_te + z_orc per candidate.
    Single-element or empty bins: use 0 for z (or epsilon for std to avoid div by zero).
    """
    n = te.size
    te = np.asarray(te, dtype=float).ravel()
    orc = np.asarray(orc, dtype=float).ravel()
    if te.size != n or orc.size != n:
        raise ValueError("te, orc length must match regime/c_bin/l_bin")
    score = np.zeros(n, dtype=float)
    eps = 1e-12

    # Group indices by (regime, c_bin, l_bin)
    from collections import defaultdict
    bin_to_indices: Dict[Tuple[Optional[str], str, str], List[int]] = defaultdict(list)
    for i in range(n):
        key = (regime_per_idx[i], c_bin_per_idx[i], l_bin_per_idx[i])
        bin_to_indices[key].append(i)

    for key, indices in bin_to_indices.items():
        if not indices:
            continue
        idx_arr = np.asarray(indices, dtype=int)
        te_bin = te[idx_arr]
        orc_bin = orc[idx_arr]
        te_mean = float(np.nanmean(te_bin))
        te_std = float(np.nanstd(te_bin))
        orc_mean = float(np.nanmean(orc_bin))
        orc_std = float(np.nanstd(orc_bin))
        if te_std < eps:
            te_std = eps
        if orc_std < eps:
            orc_std = eps
        z_te = (te_bin - te_mean) / te_std
        z_orc = (orc_bin - orc_mean) / orc_std
        for j, pos in enumerate(idx_arr):
            score[pos] = float(z_te[j] + z_orc[j])
    return score


def _farthest_point_greedy(
    pool: Sequence[int],
    *,
    n_select: int,
    te_norm: np.ndarray,
    orc_norm: np.ndarray,
    score: np.ndarray,
    already_selected: Sequence[int] = (),
) -> List[int]:
    """
    Deterministic farthest-point sampling in (te_norm, orc_norm) space.

    Selection objective (greedy):
    - First point: highest score (tie: lower idx)
    - Next points: maximize min-distance to already selected points
      (tie: higher score, then lower idx)
    """
    pool_list = sorted({int(i) for i in pool})
    if int(n_select) <= 0 or not pool_list:
        return []

    selected_set = {int(i) for i in already_selected}
    pool_list = [int(i) for i in pool_list if int(i) not in selected_set]
    if not pool_list:
        return []

    # Keep a working list; remove chosen items as we go.
    chosen: List[int] = []

    def _coord(i: int) -> Tuple[float, float]:
        return float(te_norm[int(i)]), float(orc_norm[int(i)])

    def _min_dist2_to_selected(i: int, sel: Sequence[int]) -> float:
        xi, yi = _coord(int(i))
        if not sel:
            return float("inf")
        best = float("inf")
        for j in sel:
            xj, yj = _coord(int(j))
            d2 = (xi - xj) ** 2 + (yi - yj) ** 2
            if d2 < best:
                best = d2
        return float(best)

    # Seed selection:
    # - if there are already-selected points, pick the point farthest from them (tie: score desc, idx asc)
    # - else pick highest score (tie: idx asc)
    if selected_set:
        base_sel = list(selected_set)
        first = max(
            pool_list,
            key=lambda i: (
                float(_min_dist2_to_selected(int(i), base_sel)),
                float(score[int(i)]),
                -int(i),
            ),
        )
    else:
        first = max(pool_list, key=lambda i: (float(score[int(i)]), -int(i)))
    chosen.append(int(first))
    pool_list.remove(int(first))

    # Iteratively pick max-min-distance
    while pool_list and len(chosen) < int(n_select):
        base_sel = list(selected_set) + chosen
        best_i = None
        best_key = None
        for i in pool_list:
            md2 = _min_dist2_to_selected(int(i), base_sel)
            key = (float(md2), float(score[int(i)]), -int(i))
            if best_key is None or key > best_key:
                best_key = key
                best_i = int(i)
        if best_i is None:
            break
        chosen.append(int(best_i))
        pool_list.remove(int(best_i))

    return [int(i) for i in chosen]


def _regime_neighbor_order(regime_names: Sequence[str], target: str) -> List[str]:
    """
    Deterministic neighbor ordering by index distance (1, 2, ...), preferring lower-index ties.
    """
    names = [str(r) for r in regime_names]
    tgt = str(target)
    if tgt not in names:
        return []
    i = int(names.index(tgt))
    out: List[str] = []
    for dist in range(1, len(names)):
        j_left = i - dist
        j_right = i + dist
        if 0 <= j_left < len(names):
            out.append(str(names[j_left]))
        if 0 <= j_right < len(names):
            out.append(str(names[j_right]))
    return out


def _select_top_b_coverage_aware(
    cands: Sequence[Candidate],
    B: int,
    *,
    degree_regimes: Dict[str, List[int]],
    selection_allow_missing_regimes: bool = False,
    rank_by_proxy: bool = True,
    uniform_seed: Optional[int] = None,
    fixed_tertile_edges: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Coverage-aware selection (plot2_revision): regime -> (C,L) tertile bins -> within-bin
    ranking. If rank_by_proxy=True, rank by z-scored TE+ORC; if False (Baseline A), rank
    uniformly at random within each cell (using uniform_seed). Allocate budget uniformly
    across occupied bins; enforce no single (C_bin, L_bin) > 50% of selected.
    Returns (selected_indices, selection_meta).
    """
    from collections import defaultdict

    n = len(cands)
    regime_names = list(degree_regimes.keys())
    n_reg = len(regime_names)
    if n_reg <= 0:
        raise ValueError("degree_regimes must be non-empty for coverage-aware selection")
    B = int(B)
    max_per_cell = max(1, int(B * 0.50))  # collapse constraint: no bin > 50%

    regime_per_idx, c_bin_per_idx, l_bin_per_idx, bin_diagnostics = _compute_cl_bins(
        cands, degree_regimes, fixed_tertile_edges=fixed_tertile_edges
    )
    te = np.asarray([c.te for c in cands], dtype=float)
    orc = np.asarray([c.orc for c in cands], dtype=float)
    within_bin_score = _zscore_within_bins(te, orc, regime_per_idx, c_bin_per_idx, l_bin_per_idx)

    # Only consider candidates with valid (regime, c_bin, l_bin)
    valid = [
        i
        for i in range(n)
        if regime_per_idx[i] is not None
        and c_bin_per_idx[i] != "unknown"
        and l_bin_per_idx[i] != "unknown"
    ]
    if len(valid) < B and not selection_allow_missing_regimes:
        raise AssertionError(
            f"Coverage-aware selection: only {len(valid)} candidates with valid (C,L) bins, need >= B={B}."
        )

    # Per-regime budget
    base_r = B // n_reg
    rem_r = B % n_reg
    B_per_regime = {regime_names[i]: base_r + (1 if i < rem_r else 0) for i in range(n_reg)}

    # Group valid candidates by (regime, c_bin, l_bin)
    cell_to_indices: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for i in valid:
        r = regime_per_idx[i]
        if r is None:
            continue
        key = (str(r), c_bin_per_idx[i], l_bin_per_idx[i])
        cell_to_indices[key].append(i)

    # Within each cell: proxy ranking (desc) or uniform random (Baseline A)
    if rank_by_proxy:
        for key in cell_to_indices:
            indices = cell_to_indices[key]
            indices.sort(key=lambda i: (-float(within_bin_score[i]), i))
    else:
        seed_val = int(uniform_seed) if uniform_seed is not None else 202602
        for idx, (key, indices) in enumerate(cell_to_indices.items()):
            rng = np.random.default_rng(seed_val + idx)
            order = np.arange(len(indices), dtype=int)
            rng.shuffle(order)
            cell_to_indices[key] = [indices[int(j)] for j in order]

    # Allocate quota per cell: B_r per regime, uniform across occupied cells, cap at max_per_cell
    selected_idxs: List[int] = []
    cell_quotas: Dict[Tuple[str, str, str], int] = {}
    for r in regime_names:
        B_r = B_per_regime[r]
        cells_in_r = [c for c in cell_to_indices if c[0] == r]
        if not cells_in_r:
            continue
        n_cells = len(cells_in_r)
        base_q = B_r // n_cells
        rem_q = B_r % n_cells
        for j, cell in enumerate(cells_in_r):
            q = base_q + (1 if j < rem_q else 0)
            cell_quotas[cell] = min(max_per_cell, max(0, q))

    # Select from each cell up to quota (by within_bin_score)
    for cell, indices in cell_to_indices.items():
        quota = cell_quotas.get(cell, 0)
        take = indices[:quota]
        selected_idxs.extend(take)

    # If we have fewer than B (e.g. due to cap or empty cells), fill from remaining valid
    selected_set = set(selected_idxs)
    if len(selected_idxs) < B:
        remaining = [i for i in valid if i not in selected_set]
        if rank_by_proxy:
            remaining.sort(key=lambda i: (-float(within_bin_score[i]), i))
        else:
            seed_val = int(uniform_seed) if uniform_seed is not None else 202602
            rng = np.random.default_rng(seed_val + 99999)
            order = np.arange(len(remaining), dtype=int)
            rng.shuffle(order)
            remaining = [remaining[int(j)] for j in order]
        for i in remaining:
            if len(selected_set) >= B:
                break
            selected_set.add(i)
            selected_idxs.append(i)
    selected_idxs = selected_idxs[:B]

    # Selected set (C_bin, L_bin) counts and collapse/coverage scores
    cell_counts: Dict[str, int] = defaultdict(int)
    for i in selected_idxs:
        r = regime_per_idx[i]
        if r is None:
            continue
        key = (str(r), c_bin_per_idx[i], l_bin_per_idx[i])
        cell_counts[str(key)] += 1
    total_sel = len(selected_idxs) or 1
    collapse_score = float(max(cell_counts.values(), default=0) / total_sel)
    n_bins_total = 9 * n_reg  # 3x3 per regime
    n_occupied_selected = len([k for k, v in cell_counts.items() if v > 0])
    coverage_score = float(n_occupied_selected) / float(n_bins_total) if n_bins_total > 0 else 0.0

    selection_meta = {
        "selection_coverage_level": "regime_cl_bins",
        "selection_allow_missing_regimes": bool(selection_allow_missing_regimes),
        "selection_strategy": "coverage_aware_uniform" if not rank_by_proxy else "coverage_aware",
        "coverage_aware_candidate_diagnostics": bin_diagnostics,
        "selected_cell_counts": dict(cell_counts),
        "regime_counts": _summarize_k_distribution(
            k_values=[int(cands[i].k) for i in selected_idxs],
            degree_regimes=degree_regimes,
        ).get("counts_by_regime", {}),
        "collapse_score": collapse_score,
        "coverage_score": coverage_score,
        "n_occupied_bins_selected": n_occupied_selected,
        "n_bins_total": n_bins_total,
    }
    selection_meta["selected_k_distribution"] = _summarize_k_distribution(
        k_values=[int(cands[i].k) for i in selected_idxs],
        degree_regimes=degree_regimes,
    )
    return selected_idxs, selection_meta


def _select_top_b(
    cands: Sequence[Candidate],
    B: int,
    *,
    bounds: NormBounds,
    degree_regimes: Dict[str, List[int]],
    selection_coverage_level: str = "regime_cl_bins",
    selection_allow_missing_regimes: bool = False,
    selection_strategy: str = "pareto_farthest_regime",
    cl_bin_edges: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
) -> Tuple[List[Candidate], Dict[str, Any]]:
    te = np.asarray([c.te for c in cands], dtype=float)
    orc = np.asarray([c.orc for c in cands], dtype=float)
    te_n = _normalize_fixed(te, lo=bounds.te_lo, hi=bounds.te_hi)
    orc_n = _normalize_fixed(orc, lo=bounds.orc_lo, hi=bounds.orc_hi)
    score = te_n + orc_n
    # NOTE:
    # - Global Pareto is used for search/HV diagnostics and for "global" selection strategies.
    # - For regime-coverage selection, we typically want *within-regime* Pareto (so each regime can contribute).
    pareto_mask_global = _pareto_membership_2d(te_n, orc_n)
    pareto_mask = pareto_mask_global

    n = int(len(cands))
    idxs = np.arange(n, dtype=int)

    # Deterministic total order: score desc, then idx asc
    global_order = np.lexsort((idxs, -score))

    selection_coverage_level = str(selection_coverage_level).strip().lower()
    if selection_coverage_level not in {"none", "regime", "regime_cl_bins", "regime_cl_bins_fixed"}:
        raise ValueError(
            f"Unknown selection_coverage_level={selection_coverage_level!r} "
            "(expected 'none'|'regime'|'regime_cl_bins'|'regime_cl_bins_fixed')."
        )

    selected_idxs: List[int] = []
    selection_meta: Dict[str, Any] = {
        "selection_coverage_level": selection_coverage_level,
        "selection_allow_missing_regimes": bool(selection_allow_missing_regimes),
        "selection_strategy": str(selection_strategy),
    }

    # Coverage-aware selection (plot2_revision): (C,L) bins, within-bin z-scored proxy, collapse <= 50%
    # regime_cl_bins_fixed: same as regime_cl_bins but uses frozen tertile edges from proxy viability
    if selection_coverage_level in {"regime_cl_bins", "regime_cl_bins_fixed"}:
        selected_idxs, cov_meta = _select_top_b_coverage_aware(
            cands,
            B,
            degree_regimes=degree_regimes,
            selection_allow_missing_regimes=selection_allow_missing_regimes,
            fixed_tertile_edges=cl_bin_edges,
        )
        selection_meta.update(cov_meta)
        if selection_meta.get("collapse_score") is not None and selection_meta["collapse_score"] > 0.50:
            raise AssertionError(
                f"Coverage-aware selection produced collapse_score={selection_meta['collapse_score']:.4f} > 0.50. "
                "This should not happen (cap enforced); check allocation logic."
            )
    else:
        selection_strategy = str(selection_strategy).strip().lower()
        allowed_strategies = {
            "score",
            "pareto_farthest_global",
            "pareto_farthest_regime",
            "pareto_score_global",
            "pareto_score_regime",
        }
        if selection_strategy not in allowed_strategies:
            raise ValueError(f"Unknown selection_strategy={selection_strategy!r} (expected one of {sorted(allowed_strategies)}).")

        if selection_strategy == "score":
            # Legacy behavior: ignore Pareto mask and select by score (optionally with regime quotas).
            if selection_coverage_level == "none":
                selected_idxs = [int(i) for i in global_order[: int(B)]]
            else:
                regime_names = list(degree_regimes.keys())
                n_reg = int(len(regime_names))
                if n_reg <= 0:
                    raise ValueError("degree_regimes must be non-empty when selection_coverage_level='regime'")

                # Equal allocation across regimes (deterministic remainder allocation in regime_names order).
                base = int(B) // n_reg
                rem = int(B) % n_reg
                quotas = {str(r): int(base + (1 if i < rem else 0)) for i, r in enumerate(regime_names)}
                shortfalls: Dict[str, int] = {str(r): 0 for r in regime_names}

                # Pre-bucket candidate indices by regime
                by_regime: Dict[str, List[int]] = {str(r): [] for r in regime_names}
                unknown: List[int] = []
                for i, c in enumerate(cands):
                    r = _k_to_regime(int(c.k), degree_regimes=degree_regimes)
                    if r is None or str(r) not in by_regime:
                        unknown.append(int(i))
                    else:
                        by_regime[str(r)].append(int(i))

                # Select top within each regime by score (score desc, idx asc)
                for r in regime_names:
                    r = str(r)
                    need = int(quotas.get(r, 0))
                    if need <= 0:
                        continue
                    inds = by_regime.get(r, [])
                    if not inds:
                        shortfalls[r] = int(need)
                        continue
                    inds_arr = np.asarray(inds, dtype=int)
                    order_r = np.lexsort((inds_arr, -score[inds_arr]))
                    take = [int(inds_arr[int(j)]) for j in order_r[:need]]
                    if len(take) < need:
                        shortfalls[r] = int(need - len(take))
                    selected_idxs.extend(take)

                # Backfill to reach B using global order (excluding already selected).
                selected_set = set(int(i) for i in selected_idxs)
                for i in global_order.tolist():
                    if len(selected_set) >= int(B):
                        break
                    ii = int(i)
                    if ii in selected_set:
                        continue
                    selected_set.add(ii)
                    selected_idxs.append(ii)

                selection_meta["regime_quotas"] = quotas
                selection_meta["regime_shortfalls"] = shortfalls
                selection_meta["n_unknown_regime_candidates"] = int(len(unknown))

        elif selection_strategy in {"pareto_score_global", "pareto_farthest_global"} or selection_coverage_level == "none":
            # Global selection from Pareto pool; ignore regime quotas (or explicitly requested none-coverage).
            pareto_idxs = [int(i) for i in np.where(pareto_mask_global)[0].tolist()]
            selection_meta["n_pareto_total"] = int(len(pareto_idxs))
            selection_meta["pareto_kind"] = "global"
            if selection_strategy == "pareto_score_global":
                # Select by score among Pareto; tie-break by idx asc.
                pareto_arr = np.asarray(pareto_idxs, dtype=int)
                order_p = np.lexsort((pareto_arr, -score[pareto_arr]))
                selected_idxs = [int(pareto_arr[int(j)]) for j in order_p[: int(B)]]
            else:
                selected_idxs = _farthest_point_greedy(
                    pareto_idxs, n_select=int(B), te_norm=te_n, orc_norm=orc_n, score=score, already_selected=()
                )

            # If Pareto pool is too small, fill remainder using global score order.
            if len(selected_idxs) < int(B):
                selected_set = set(int(i) for i in selected_idxs)
                for i in global_order.tolist():
                    if len(selected_set) >= int(B):
                        break
                    ii = int(i)
                    if ii in selected_set:
                        continue
                    selected_set.add(ii)
                    selected_idxs.append(ii)

        else:
            regime_names = list(degree_regimes.keys())
            n_reg = int(len(regime_names))
            if n_reg <= 0:
                raise ValueError("degree_regimes must be non-empty when selection_coverage_level='regime'")

            # Hard requirement: we need at least 2 topologies per regime (Plot 2 diversity constraint).
            min_per_regime = 2
            if int(B) < int(min_per_regime) * int(n_reg) and not bool(selection_allow_missing_regimes):
                raise ValueError(
                    f"Selection requires at least {min_per_regime} architectures per regime, but B={int(B)} "
                    f"and n_regimes={int(n_reg)} (need B >= {int(min_per_regime) * int(n_reg)})."
                )

            # Guardrails for search-space collapse (candidate-level), unless explicitly overridden.
            if not bool(selection_allow_missing_regimes):
                cand_ks = [int(c.k) for c in cands]
                cand_dist = _summarize_k_distribution(k_values=cand_ks, degree_regimes=degree_regimes)
                selection_meta["candidate_k_distribution"] = cand_dist
                cand_reg_counts = cand_dist.get("counts_by_regime", {})
                n_reg_present_cands = int(sum(1 for _r, ct in cand_reg_counts.items() if int(ct) > 0))
                if n_reg_present_cands < int(n_reg):
                    raise AssertionError(
                        f"Candidate sampling collapsed: only {n_reg_present_cands}/{n_reg} regimes present in candidates. "
                        f"counts_by_regime={cand_reg_counts}"
                    )
                if int(cand_dist.get("n_unique_k", 0)) <= 1 and int(len(cands)) >= 2:
                    raise AssertionError(
                        f"Candidate sampling collapsed: only one unique k in candidates (k={cand_dist.get('unique_k')})."
                    )

            # Bucket all candidate indices by regime.
            by_regime_all: Dict[str, List[int]] = {str(r): [] for r in regime_names}
            unknown: List[int] = []
            for i, c in enumerate(cands):
                r = _k_to_regime(int(c.k), degree_regimes=degree_regimes)
                if r is None or str(r) not in by_regime_all:
                    unknown.append(int(i))
                else:
                    by_regime_all[str(r)].append(int(i))

            candidates_available_by_regime = {str(r): int(len(by_regime_all.get(str(r), []))) for r in regime_names}
            selection_meta["candidates_available_by_regime"] = candidates_available_by_regime
            selection_meta["n_unknown_regime_candidates"] = int(len(unknown))

            # Enforce that each regime has enough candidates to pick two distinct architectures.
            if not bool(selection_allow_missing_regimes):
                missing = {r: ct for r, ct in candidates_available_by_regime.items() if int(ct) < int(min_per_regime)}
                if missing:
                    raise AssertionError(
                        f"Insufficient candidates per regime for Plot 2: need >= {min_per_regime} per regime. "
                        f"candidates_available_by_regime={candidates_available_by_regime}"
                    )

            # Compute *within-regime* Pareto sets (max-max) using normalized TE/ORC.
            pareto_mask_within = np.zeros((n,), dtype=bool)
            by_regime_pareto_within: Dict[str, List[int]] = {str(r): [] for r in regime_names}
            for r in regime_names:
                r = str(r)
                inds = by_regime_all.get(r, [])
                if not inds:
                    continue
                inds_arr = np.asarray([int(i) for i in inds], dtype=int)
                sub_mask = _pareto_membership_2d(te_n[inds_arr], orc_n[inds_arr])
                sub_idx = np.where(sub_mask)[0].tolist()
                pareto_inds = [int(inds_arr[int(j)]) for j in sub_idx]
                by_regime_pareto_within[r] = pareto_inds
                pareto_mask_within[pareto_inds] = True

            pareto_mask = pareto_mask_within
            selection_meta["pareto_kind"] = "within_regime"
            selection_meta["n_pareto_total"] = int(np.sum(pareto_mask_within))
            selection_meta["pareto_available_by_regime"] = {
                str(r): int(len(by_regime_pareto_within.get(str(r), []))) for r in regime_names
            }

            # Step 1: pick top-2 per regime, prioritizing within-regime Pareto points.
            selected_from_pareto_by_regime: Dict[str, int] = {str(r): 0 for r in regime_names}
            shortfalls: Dict[str, int] = {str(r): 0 for r in regime_names}
            for r in regime_names:
                r = str(r)
                pareto_inds = [int(i) for i in by_regime_pareto_within.get(r, [])]
                all_inds = [int(i) for i in by_regime_all.get(r, [])]
                if not all_inds:
                    shortfalls[r] = int(min_per_regime)
                    continue

                # Order Pareto candidates by score desc, idx asc.
                take: List[int] = []
                if pareto_inds:
                    p_arr = np.asarray(pareto_inds, dtype=int)
                    order_p = np.lexsort((p_arr, -score[p_arr]))
                    ordered_p = [int(p_arr[int(j)]) for j in order_p.tolist()]
                    take.extend(ordered_p[: int(min_per_regime)])

                selected_from_pareto_by_regime[r] = int(len(take))

                # If Pareto set has <2, backfill within the same regime by score (may include dominated points).
                if len(take) < int(min_per_regime):
                    remaining = [int(i) for i in all_inds if int(i) not in set(take)]
                    if remaining:
                        rem_arr = np.asarray(remaining, dtype=int)
                        order_r = np.lexsort((rem_arr, -score[rem_arr]))
                        need = int(min_per_regime) - int(len(take))
                        take.extend([int(rem_arr[int(j)]) for j in order_r[:need]])

                if len(take) < int(min_per_regime):
                    shortfalls[r] = int(min_per_regime) - int(len(take))
                    if not bool(selection_allow_missing_regimes):
                        raise AssertionError(
                            f"Regime {r} could not provide {min_per_regime} architectures after backfill. "
                            f"candidates_available_by_regime={candidates_available_by_regime}, "
                            f"pareto_available_by_regime={selection_meta.get('pareto_available_by_regime')}"
                        )

                selected_idxs.extend([int(i) for i in take])

            selection_meta["min_per_regime"] = int(min_per_regime)
            selection_meta["selected_from_pareto_by_regime"] = selected_from_pareto_by_regime
            selection_meta["regime_shortfalls"] = shortfalls

            # Step 2: fill remaining slots up to B from within-regime Pareto pool (then fall back to global score).
            if len(selected_idxs) < int(B):
                remaining_pareto = sorted({int(i) for i in np.where(pareto_mask_within)[0].tolist()} - {int(i) for i in selected_idxs})
                if remaining_pareto:
                    if selection_strategy == "pareto_score_regime":
                        rem_arr = np.asarray(remaining_pareto, dtype=int)
                        order_p = np.lexsort((rem_arr, -score[rem_arr]))
                        extra = [int(rem_arr[int(j)]) for j in order_p[: int(B) - len(selected_idxs)]]
                    else:
                        extra = _farthest_point_greedy(
                            remaining_pareto,
                            n_select=int(B) - len(selected_idxs),
                            te_norm=te_n,
                            orc_norm=orc_n,
                            score=score,
                            already_selected=selected_idxs,
                        )
                    selected_idxs.extend([int(i) for i in extra])

            if len(selected_idxs) < int(B):
                selected_set = set(int(i) for i in selected_idxs)
                for i in global_order.tolist():
                    if len(selected_set) >= int(B):
                        break
                    ii = int(i)
                    if ii in selected_set:
                        continue
                    selected_set.add(ii)
                    selected_idxs.append(ii)

            # Selection-level distribution is recorded for diagnostics.
            selected_ks = [int(cands[i].k) for i in selected_idxs]
            selection_meta["selected_k_distribution"] = _summarize_k_distribution(k_values=selected_ks, degree_regimes=degree_regimes)

    # Preserve order in output list, but also record deterministic selection indices.
    selected_idxs = [int(i) for i in selected_idxs[: int(B)]]
    selected = [cands[int(i)] for i in selected_idxs]

    # Collapse score per spec §3.4, §7: regime = max regime fraction; regime_cl_bins* = already in selection_meta
    if selection_coverage_level in {"regime_cl_bins", "regime_cl_bins_fixed"}:
        # collapse_score and coverage_score already set by _select_top_b_coverage_aware
        pass
    elif selection_coverage_level == "regime" and degree_regimes and selected_idxs:
        sel_ks = [int(cands[i].k) for i in selected_idxs]
        sel_dist = _summarize_k_distribution(k_values=sel_ks, degree_regimes=degree_regimes)
        if "selected_k_distribution" not in selection_meta:
            selection_meta["selected_k_distribution"] = sel_dist
        counts = sel_dist.get("counts_by_regime", {})
        total = sum(counts.values()) or 1
        selection_meta["collapse_score"] = float(max(counts.values(), default=0) / total)
    else:
        selection_meta["collapse_score"] = None

    meta = {
        "te_norm": te_n,
        "orc_norm": orc_n,
        "score": score,
        "pareto_mask": pareto_mask,
        "pareto_mask_global": pareto_mask_global,
        "order": global_order,
        "selected_idxs": selected_idxs,
        **selection_meta,
    }
    return selected, meta


def _generate_and_select_external_random(
    *,
    H: int,
    B_ext: int,
    output_size: int,
    degree_regimes: Dict[str, List[int]],
    pool_size: int,
    base_seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Plot 2 G3: Generate a pool of ncps Random wirings, compute mask_density for each,
    and select B_ext with density bins aligned to WS-Flex regimes (stratified).
    F2: Recurrent size must equal H for capacity matching with WS-Flex (not output_size).
    Returns (selected arch rows for CSV/JSON, selection summary dict).
    """
    from ncps.wirings import Random

    rng = np.random.default_rng(int(base_seed) % (2**31 - 1))
    # F2: Recurrent units = H for capacity matching with WS-Flex baselines
    units = int(H)
    # Four density bins aligned to WS-Flex regimes: k/H for k in [2,6], [7,12], [13,18], [19,26]
    regime_names = list(degree_regimes.keys())
    bin_edges: List[float] = [0.0]
    for r in regime_names:
        ks = degree_regimes[r]
        if ks:
            k_hi = max(ks)
            bin_edges.append(float(k_hi) / float(H))
    bin_edges.append(1.0)
    bin_edges = sorted(set(bin_edges))
    n_bins = max(1, min(4, len(bin_edges) - 1))
    # Use first n_bins+1 edges for n_bins bins
    bin_edges = bin_edges[: n_bins + 1]

    # Build pool: Random(units=H, output_dim=output_size) for capacity matching (F2)
    pool: List[Dict[str, Any]] = []
    for _ in range(pool_size):
        sp = float(rng.uniform(0.05, 0.95))
        seed = int(rng.integers(0, 2**31 - 1))
        try:
            w = Random(units=units, output_dim=int(output_size), sparsity_level=sp, random_seed=seed)
            A = np.asarray(w.adjacency_matrix)
            density = float(np.sum(A != 0)) / float(units * units) if units > 0 else 0.0
        except Exception:
            continue
        # Assign to bin
        bin_idx = 0
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= density < bin_edges[i + 1]:
                bin_idx = min(i, n_bins - 1)
                break
            if density >= bin_edges[i + 1]:
                bin_idx = min(i + 1, n_bins - 1)
        pool.append({
            "sparsity_level": sp,
            "wiring_seed": seed,
            "mask_density": density,
            "bin_idx": bin_idx,
        })

    if not pool:
        raise RuntimeError("External random pool is empty (all Random(...) failed).")
    # Stratified selection: equal allocation per bin
    quotas: List[int] = []
    rem = int(B_ext) % n_bins
    base_q = int(B_ext) // n_bins
    for i in range(n_bins):
        quotas.append(base_q + (1 if i < rem else 0))
    by_bin: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(n_bins)}
    for c in pool:
        by_bin[c["bin_idx"]].append(c)
    # F6: Use seeded random selection within each bin instead of deterministic sort
    selected: List[Dict[str, Any]] = []
    for b in range(n_bins):
        need = quotas[b]
        cands = by_bin[b]
        if not cands:
            continue
        bin_rng = np.random.default_rng(int(base_seed + b) % (2**31 - 1))
        take = min(need, len(cands))
        if take >= len(cands):
            selected.extend(cands)
        else:
            idxs = bin_rng.choice(len(cands), size=int(take), replace=False)
            for i in idxs:
                selected.append(cands[int(i)])
    # If we got fewer than B_ext (e.g. empty bins), backfill from remaining pool using seeded random
    selected_set = {(x["sparsity_level"], x["wiring_seed"]) for x in selected}
    remaining = [c for c in pool if (c["sparsity_level"], c["wiring_seed"]) not in selected_set]
    if len(selected) < int(B_ext) and remaining:
        backfill_rng = np.random.default_rng(int(base_seed + n_bins) % (2**31 - 1))
        need_backfill = int(B_ext) - len(selected)
        take = min(need_backfill, len(remaining))
        idxs = backfill_rng.choice(len(remaining), size=int(take), replace=False)
        for i in idxs:
            selected.append(remaining[int(i)])
    selected = selected[: int(B_ext)]

    summary = {
        "n_bins": n_bins,
        "bin_edges": bin_edges,
        "pool_size": len(pool),
        "n_selected": len(selected),
        "selection_method": "seeded_random_per_bin",
        "base_seed": int(base_seed),
        "density_range": [min(c["mask_density"] for c in selected), max(c["mask_density"] for c in selected)] if selected else [float("nan"), float("nan")],
    }
    return selected, summary


def _derive_training_seeds_from_base_seed(base_seed: int, S: int) -> List[int]:
    """
    Deterministic *run-level* training seed list.

    Rationale:
    - For fair/reproducible comparisons, all competing models/topologies in the same Plot 2 run
      should be trained/evaluated with the exact same set of experimental seeds.
    - Previously this was derived from (run_id, model_name), which produced different seed sets per model.
    - Resume mode reads jobs from jobs.csv, so changing this only affects new runs.
    """
    b = int(base_seed) % (2**31 - 1)
    rng = np.random.default_rng(int(b if b != 0 else 1))
    seeds: List[int] = []
    while len(seeds) < int(S):
        s = int(rng.integers(1, 2**31 - 1))
        if s not in seeds:
            seeds.append(s)
    return seeds


def _should_skip_plot2_job(
    *,
    model_name: str,
    dataset: str,
    eval_mode: str,
    subjects: List[int],
    seed: int,
    saturation_file: str,
    alpha_grid: List[float],
    perturbation_types: Optional[List[str]] = None,
) -> bool:
    """Return True if check_skip_eval says existing results already cover this job (noise types + alpha grid).
    Used to avoid spinning up the unified runner when the inner check would skip anyway.
    """
    from evaluation.experiment_utils import check_skip_eval
    from evaluation.unified_experiment_runner import _get_test_perturb_expected_scope
    from config import get_paradigm
    from moabb.datasets import BNCI2014_001, Lee2019_SSVEP, BI2015a

    if perturbation_types is None:
        perturbation_types = ["gaussian"]

    eval_mode_eval = f"{eval_mode}Evaluation" if not str(eval_mode).endswith("Evaluation") else str(eval_mode)
    if dataset == "Lee2019_SSVEP":
        paradigm_name = "SSVEP"
    elif dataset == "BI2015a":
        paradigm_name = "ERP"
    else:
        paradigm_name = "MotorImagery"

    temp_dataset_obj = None
    try:
        if dataset == "BNCI2014_001":
            temp_dataset_obj = BNCI2014_001()
            temp_dataset_obj.subject_list = subjects
        elif dataset == "Lee2019_SSVEP":
            temp_dataset_obj = Lee2019_SSVEP()
            temp_dataset_obj.subject_list = subjects
        elif dataset == "BI2015a":
            temp_dataset_obj = BI2015a()
            temp_dataset_obj.subject_list = subjects
        temp_paradigm = get_paradigm(resample=None, dataset=dataset)
    except Exception as e:
        print(f"[PLOT2] Warning: could not create paradigm/dataset for skip check: {e}")
        return False

    gaussian_only = perturbation_types == ["gaussian"]
    exp_types, exp_by_noise = _get_test_perturb_expected_scope(
        dataset,
        test_perturb_noise_types=None if gaussian_only else perturbation_types,
        test_perturb_gaussian_only=gaussian_only,
        test_perturb_gaussian_alpha_grid=alpha_grid,
        test_perturb_num_steps=20,
        saturation_file=saturation_file,
    )
    return check_skip_eval(
        model_name,
        seed,
        subjects,
        "test_perturb",
        None,
        None,
        eval_mode=eval_mode_eval,
        paradigm=paradigm_name,
        dataset=dataset,
        paradigm_obj=temp_paradigm,
        dataset_obj=temp_dataset_obj,
        tuned=False,
        expected_noise_types=exp_types,
        expected_intensities_by_noise=exp_by_noise,
        test_perturb_num_steps=20,
        test_perturb_saturation_file=saturation_file,
    )


def _run_unified_job(
    *,
    repo_root: Path,
    python_exe: str,
    plot2_dir: Path,
    model_name: str,
    dataset: str,
    eval_mode: str,
    subjects: List[int],
    seed: int,
    saturation_file: str,
    alpha_grid: List[float],
    perturbation_types: Optional[List[str]] = None,
    target_snr_db: float = -5.0,
    perturbation_params: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> int:
    if perturbation_types is None:
        perturbation_types = ["gaussian"]
    gaussian_only = perturbation_types == ["gaussian"]
    params = perturbation_params or {}

    cmd = [
        python_exe,
        str((repo_root / "evaluation" / "unified_experiment_runner.py").as_posix()),
        "--nas_pilot_dir",
        str(plot2_dir.as_posix()),
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
        "--test_perturb_target_snr_db="
        + (str(target_snr_db) if isinstance(target_snr_db, (int, float)) else ",".join(str(t) for t in target_snr_db)),
        "--test_perturb_ar1_rho",
        str(params.get("ar1_drift", {}).get("rho", 0.97)),
        "--test_perturb_spatial_ell_multiplier",
        str(params.get("spatial_gaussian", {}).get("ell_multiplier", 1.0)),
        "--test_perturb_emg_f_low",
        str(params.get("emg_band", {}).get("f_low", 20.0)),
        "--test_perturb_emg_f_high",
        str(params.get("emg_band", {}).get("f_high", 80.0)),
        "--plot2_diagnostics_dir",
        str((plot2_dir / "diagnostics").as_posix()),
    ]
    if gaussian_only:
        cmd.append("--test_perturb_gaussian_only")
    else:
        cmd.extend(["--test_perturb_noise_types", ",".join(perturbation_types)])
    if params.get("emg_band", {}).get("envelope_on", False):
        cmd.append("--test_perturb_emg_use_envelope")
    cmd = [c for c in cmd if c]
    print("[PLOT2] Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(repo_root), check=False)
    return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 2 topology study runner (WS-Flex: multiobjective TE+ORC)")
    parser.add_argument("--run_id", type=str, default=None, help="Optional run id (default: timestamp).")
    parser.add_argument("--output_root", type=str, default="architecture_refinement/outputs/plot2_topology_study")
    parser.add_argument(
        "--plot2_dir",
        type=str,
        default=None,
        help="Optional explicit Plot2 run directory. If it exists, the runner can resume from it.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing Plot2 directory (skip search/selection if artifacts exist).",
    )
    parser.add_argument(
        "--force_search",
        action="store_true",
        help="Force rerunning the topology search/selection even if resume artifacts exist in --plot2_dir.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="If a training job fails, record it and continue with remaining jobs.",
    )
    parser.add_argument("--dataset", type=str, default="BNCI2014_001")
    parser.add_argument("--eval_mode", type=str, default="CrossSession")
    parser.add_argument(
        "--scale",
        type=str,
        default="full",
        choices=["mini", "full"],
        help="Plot 2 scale: 'mini' = 3 subjects, 1 seed, B=8, B_ext=8 (fast validation); 'full' = full-scale run.",
    )
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument(
        "--B",
        type=int,
        default=12,
        help=(
            "Requested number of WS-Flex topologies selected per method. "
            "Under the locked Plot 2 policy (within-regime Pareto, top-2 per regime), "
            "this will be coerced to B = 2 * (#degree_regimes)."
        ),
    )
    parser.add_argument(
        "--B_ext",
        type=int,
        default=12,
        help="Number of external random wiring (G3) architectures to select (density-matched stratified).",
    )
    parser.add_argument("--S", type=int, default=5, help="Training seeds per topology.")
    parser.add_argument(
        "--training_seed_base",
        type=int,
        default=None,
        help=(
            "Optional base seed used to derive the shared Plot 2 training seed set. "
            "If omitted, the base is derived deterministically from run_id."
        ),
    )
    parser.add_argument(
        "--training_seeds",
        type=str,
        default="",
        help=(
            "Optional explicit comma-separated list of training seeds to use for all models/topologies "
            "(e.g. '1,2,3,4,5'). If provided, overrides --S and --training_seed_base."
        ),
    )
    parser.add_argument(
        "--proxy_viability_dir",
        type=str,
        default="",
        help=(
            "Required for Plot 2 Overhaul: path to run_plot2_proxy_viability.py output. "
            "Loads frozen bin edges, mu_orc_by_k, mu_te_by_k, and normalization bounds."
        ),
    )
    parser.add_argument(
        "--selection_coverage_level",
        type=str,
        default="regime_cl_bins_fixed",
        choices=["none", "regime", "regime_cl_bins", "regime_cl_bins_fixed"],
        help=(
            "Coverage when selecting topologies. 'regime_cl_bins_fixed' (default, Plot 2 Overhaul): "
            "manifest-locked C/L bins from G0 viability. 'regime_cl_bins': dynamic tertiles. "
            "'regime': per-degree-regime quotas only. 'none': global only."
        ),
    )
    parser.add_argument(
        "--selection_strategy",
        type=str,
        default="pareto_farthest_regime",
        choices=[
            "pareto_farthest_regime",
            "pareto_farthest_global",
            "pareto_score_regime",
            "pareto_score_global",
            "score",
        ],
        help=(
            "How to select B trained topologies from training-free candidates. "
            "'pareto_farthest_regime' computes Pareto sets *within each degree regime* and selects at least "
            "two architectures per regime (preferring within-regime Pareto points), then fills remaining slots "
            "with farthest-point diversity."
        ),
    )
    parser.add_argument(
        "--selection_allow_missing_regimes",
        action="store_true",
        help="If set, do not assert regime coverage / multi-k selection (log diagnostics only).",
    )
    parser.add_argument(
        "--allow_regime_collapse",
        action="store_true",
        help="If set, allow selection where >50%% of models come from one regime (otherwise hard fail).",
    )
    parser.add_argument("--alpha_grid", type=str, default="0,0.25,0.5,0.75,1.0")
    parser.add_argument(
        "--perturbation_types",
        type=str,
        default="ar1_drift",
        help=(
            "Comma-separated perturbation types for Plot 2: ar1_drift (default primary), gaussian, spatial_gaussian, emg_band, iid_gaussian. "
            "AR(1) drift is the primary topology-sensitive stress test (Spec 2). Gaussian i.i.d. is a documented negative control."
        ),
    )
    parser.add_argument(
        "--target_snr_db",
        type=float,
        default=-6.0,
        help="Target SNR in dB at alpha_max for correlated perturbations (Spec 3; default -6.0 for Plot 2).",
    )
    parser.add_argument(
        "--target_snr_dbs",
        type=str,
        default=None,
        help="Comma-separated target SNRs for dual-SNR eval (Plot 2 Overhaul, e.g. '-12,-6'). Overrides --target_snr_db.",
    )
    parser.add_argument("--ar1_rho", type=float, default=0.97, help="AR(1) drift coefficient for ar1_drift (default 0.97).")
    parser.add_argument("--spatial_ell_multiplier", type=float, default=1.0, help="Multiplier for spatial correlation length (default 1.0).")
    parser.add_argument("--emg_f_low", type=float, default=20.0, help="EMG band low cutoff in Hz (default 20).")
    parser.add_argument("--emg_f_high", type=float, default=80.0, help="EMG band high cutoff in Hz (default 80).")
    parser.add_argument("--emg_envelope_on", action="store_true", help="Use slow amplitude envelope for EMG noise (bursty EMG).")
    parser.add_argument("--saturation_file", type=str, default="saturation_results/saturation_points_summary.csv")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Generate candidates/selection/jobs but do not run training.")
    parser.add_argument("--capacity_filter", action="store_true", help="Reject graphs whose E_active falls outside regime band (Plot2_revision3 Step C).")
    parser.add_argument(
        "--generator_mode",
        type=str,
        default="ws_flex",
        choices=["ws_flex", "modular_ws_flex"],
        help="Generator mode: ws_flex (default) or modular_ws_flex.",
    )
    parser.add_argument("--seed_mod_params", type=int, default=202607, help="RNG seed for modular param sampling.")
    parser.add_argument("--write_jobs_only", action="store_true", help="Write jobs.csv and exit (implies dry_run).")

    # Search budget + early stopping (spec defaults)
    parser.add_argument("--M0", type=int, default=200)
    parser.add_argument("--dM", type=int, default=100)
    parser.add_argument("--M_max", type=int, default=1000)
    parser.add_argument("--M_ref", type=int, default=200, help="Reference set size for fixed TE/ORC normalization.")
    parser.add_argument(
        "--require_phase2_diagnostics",
        action="store_true",
        help="If set, require M0 >= 500 for Phase 2 diagnostics; exit with error otherwise.",
    )
    parser.add_argument("--ref_q_lo", type=float, default=0.01, help="Lower quantile for reference-set normalization bounds.")
    parser.add_argument("--ref_q_hi", type=float, default=0.99, help="Upper quantile for reference-set normalization bounds.")
    parser.add_argument("--hv_window_W", type=int, default=3, help="Hypervolume improvement window (batches).")
    parser.add_argument("--hv_window_eps", type=float, default=0.02, help="Minimum relative HV improvement over window.")
    parser.add_argument("--hv_window_patience", type=int, default=2, help="Consecutive windows below eps required to stop.")
    parser.add_argument("--pareto_new_m", type=int, default=5, help="Minimum Pareto-optimal points from newest batch.")
    parser.add_argument("--pareto_patience_batches", type=int, default=3, help="Consecutive low-growth batches required to stop.")

    # Optuna persistence (TPE only): we must keep studies for later analysis.
    parser.add_argument(
        "--tpe_study_db",
        type=str,
        default="",
        help=(
            "Path to SQLite DB file for persisting the Optuna multi-objective study (TPE only). "
            "If empty, defaults to <plot2_dir>/optuna/tpe_study.sqlite3."
        ),
    )
    parser.add_argument(
        "--tpe_study_name",
        type=str,
        default="",
        help="Optuna study name (TPE only). If empty, defaults to 'plot2_<run_id>_tpe'.",
    )

    # Deprecated (kept for backwards compatibility with older runs/scripts; no longer used by locked policy).
    parser.add_argument("--hv_rel_tol", type=float, default=0.01, help="(deprecated) Old stepwise HV tolerance; ignored.")
    parser.add_argument("--hv_patience_batches", type=int, default=3, help="(deprecated) Old stepwise HV patience; ignored.")

    # Degree regimes (Plot 2 spec §3.3 locked: super_sparse [2,6], sparse [7,12], moderate [13,18], near_dense [19,26])
    # Watts-Strogatz requires even k; use even k only in each range.
    parser.add_argument("--regime_super_sparse", type=str, default="2,4,6")
    parser.add_argument("--regime_sparse", type=str, default="8,10,12")
    parser.add_argument("--regime_moderate", type=str, default="14,16,18")
    parser.add_argument("--regime_near_dense", type=str, default="20,22,24,26")

    # NCP baseline (AutoNCP)
    parser.add_argument("--ncp_sparsity_level", type=float, default=0.5)
    parser.add_argument(
        "--ncp_io_size",
        type=int,
        default=16,
        help="CfC chamber I/O size (should match F2=F1*D for CNNWiredCfCMin). Used to size baseline wiring fairly.",
    )

    args = parser.parse_args()
    if args.write_jobs_only:
        args.dry_run = True

    # Mini-scale config (Plot 2 spec §4): S_small=3 subjects, 1 seed, B_small=8, B_small_ext=8
    if getattr(args, "scale", "full") == "mini":
        args.subjects = list(args.subjects)[:3] if len(args.subjects) >= 3 else list(args.subjects)[:2]
        if not args.subjects:
            args.subjects = [1, 2]
        args.S = 1
        args.B = 8
        args.B_ext = 8
        if args.training_seeds:
            # Leave explicit seeds as-is if provided; else we already set S=1
            pass
        print(f"[PLOT2] Mini-scale: subjects={args.subjects}, S={args.S}, B={args.B}, B_ext={args.B_ext}")

    # Small, fast sanity-check defaults when dry-running
    if args.dry_run:
        args.M0 = min(int(args.M0), 40)
        args.dM = min(int(args.dM), 20)
        args.M_max = min(int(args.M_max), 80)
        args.M_ref = min(int(args.M_ref), 80)
        # Keep selection meaningful under the locked Plot 2 policy: need >=2 architectures per regime.
        # For a fast dry-run, we set B to the minimum satisfying this constraint.
        _preview_regimes = {
            "super_sparse": [int(x) for x in str(args.regime_super_sparse).split(",") if x.strip()],
            "sparse": [int(x) for x in str(args.regime_sparse).split(",") if x.strip()],
            "moderate": [int(x) for x in str(args.regime_moderate).split(",") if x.strip()],
            "near_dense": [int(x) for x in str(args.regime_near_dense).split(",") if x.strip()],
        }
        _n_reg_nonempty = int(sum(1 for _ks in _preview_regimes.values() if list(_ks)))
        _min_per_regime = 2
        _min_B = int(max(_min_per_regime, _min_per_regime * _n_reg_nonempty))
        args.B = int(_min_B)
        args.S = min(int(args.S), 2)
        args.subjects = [int(args.subjects[0])] if args.subjects else [1]

    # FIX 4: Optional requirement that Phase 2 diagnostics run at N >= 500
    if getattr(args, "require_phase2_diagnostics", False) and int(args.M0) < 500:
        print("[PLOT2] --require_phase2_diagnostics set but M0 < 500. Phase 2 diagnostics require N >= 500.")
        sys.exit(1)

    repo_root = Path(__file__).resolve().parent.parent

    # Determine plot2_dir (new run vs resume)
    if args.plot2_dir:
        plot2_dir = Path(args.plot2_dir).resolve()
        plot2_dir.mkdir(parents=True, exist_ok=True)
        run_id = plot2_dir.name
    else:
        run_id = args.run_id or _now_run_id()
        plot2_dir = (repo_root / args.output_root / run_id).resolve()
        if plot2_dir.exists():
            if args.resume:
                # resume in-place, do not generate a new run_id
                pass
            else:
                run_id = _resolve_unique_run_id(repo_root=repo_root, output_root=args.output_root, run_id=run_id)
                plot2_dir = (repo_root / args.output_root / run_id).resolve()
        plot2_dir.mkdir(parents=True, exist_ok=True)

    # I1: Environment capture (reproducibility)
    _capture_environment_info(repo_root, plot2_dir, " ".join(sys.argv))
    wall_clock_start = time.perf_counter()

    # If resuming and core artifacts exist, skip search/selection and just dispatch jobs.
    manifest_path = plot2_dir / "plot2_manifest.json"
    selected_csv_path = plot2_dir / "selected_architectures.csv"
    jobs_csv_path = plot2_dir / "jobs.csv"
    # Auto-resume when an explicit plot2_dir is provided and artifacts exist (unless --force_search).
    can_resume = (
        (args.resume or (args.plot2_dir is not None))
        and (not args.force_search)
        and manifest_path.exists()
        and selected_csv_path.exists()
        and jobs_csv_path.exists()
    )

    if can_resume:
        print(f"[PLOT2] Resuming from existing run dir (skipping search): {plot2_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        import pandas as pd

        jobs_df = pd.read_csv(jobs_csv_path)
        jobs = jobs_df.to_dict(orient="records")
        # Manifest stores alpha grid as a JSON list (preferred). Accept comma-string for backwards compat.
        alpha_obj = manifest.get("gaussian_alpha_grid", ALPHA_GRID_DEFAULT)
        if isinstance(alpha_obj, list):
            alpha_grid = [float(x) for x in alpha_obj]
        else:
            alpha_grid = [float(x) for x in str(alpha_obj).split(",") if str(x).strip()]
        if not alpha_grid:
            alpha_grid = list(ALPHA_GRID_DEFAULT)
        if args.dry_run:
            print("[PLOT2] Dry-run complete (resume mode, no training dispatched).")
            return

        failed: List[Dict[str, Any]] = []
        for job in jobs:
            job_dataset = str(job.get("dataset", manifest.get("dataset", "BNCI2014_001")))
            job_eval_mode = str(job.get("eval_mode", manifest.get("eval_mode", "CrossSession")))
            job_subjects = [int(x) for x in str(job["subjects"]).split(",") if str(x).strip()]
            job_saturation = str(job.get("saturation_file", manifest.get("saturation_file", "saturation_results/saturation_points_summary.csv")))
            job_alpha = [float(x) for x in str(job.get("alpha_grid", ",".join(str(a) for a in alpha_grid))).split(",") if str(x).strip()]
            job_perturb_raw = job.get("perturbation_types", manifest.get("perturbation_types", "gaussian"))
            job_perturb = [x.strip() for x in str(job_perturb_raw).split(",") if str(x).strip()]
            if not job_perturb:
                job_perturb = ["gaussian"]
            if not args.overwrite and _should_skip_plot2_job(
                model_name=str(job["model_name"]),
                dataset=job_dataset,
                eval_mode=job_eval_mode,
                subjects=job_subjects,
                seed=int(job["seed"]),
                saturation_file=job_saturation,
                alpha_grid=job_alpha,
                perturbation_types=job_perturb,
            ):
                print(f"[PLOT2] Skipping (results exist): model={job['model_name']} seed={job['seed']}")
                continue
            rc = _run_unified_job(
                repo_root=repo_root,
                python_exe=str(args.python),
                plot2_dir=plot2_dir,
                model_name=str(job["model_name"]),
                dataset=job_dataset,
                eval_mode=job_eval_mode,
                subjects=job_subjects,
                seed=int(job["seed"]),
                saturation_file=job_saturation,
                alpha_grid=job_alpha,
                perturbation_types=job_perturb,
                target_snr_db=job.get("target_snr_db", manifest.get("target_snr_db", -6.0)),
                perturbation_params=manifest.get("perturbation_params"),
                overwrite=bool(args.overwrite),
            )
            if rc != 0:
                failed.append({**job, "exit_code": int(rc)})
                if not args.continue_on_error:
                    raise RuntimeError(
                        f"Unified runner failed for model={job['model_name']} seed={job['seed']} (exit_code={rc}). "
                        f"Re-run with --resume --continue_on_error to keep going."
                    )

        if failed:
            # write a small failure table to make re-runs easy
            out = plot2_dir / "failed_jobs.csv"
            pd.DataFrame(failed).to_csv(out, index=False)
            raise RuntimeError(f"[PLOT2] {len(failed)} jobs failed. See: {out}")

        print("[PLOT2] Resume run complete (all jobs succeeded).")
        return

    alpha_grid = [float(x.strip()) for x in str(args.alpha_grid).split(",") if x.strip()]
    if not alpha_grid:
        alpha_grid = list(ALPHA_GRID_DEFAULT)
    perturbation_types = [x.strip() for x in str(getattr(args, "perturbation_types", "gaussian")).split(",") if x.strip()]
    if not perturbation_types:
        perturbation_types = ["gaussian"]
    # Plot2_revision2: locked Plot 2 core run uses a single perturbation type (AR(1) drift at −5 dB). Do not mix.
    if len(perturbation_types) > 1:
        import warnings
        warnings.warn(
            "Plot 2 core spec (Plot2_revision2.md) uses a single perturbation type: AR(1) drift at −5 dB. "
            f"Multiple types requested: {perturbation_types}. Use one type for the final Plot 2 run.",
            UserWarning,
            stacklevel=2,
        )

    # ---- Optuna persistence settings (TPE) ----
    # Use a run-local default so every Plot 2 run keeps its own study for later analysis.
    if str(getattr(args, "tpe_study_db", "")).strip():
        tpe_db_path = Path(str(args.tpe_study_db)).expanduser().resolve()
    else:
        tpe_db_path = (plot2_dir / "optuna" / "tpe_study.sqlite3").resolve()
    tpe_db_path.parent.mkdir(parents=True, exist_ok=True)
    # Optuna uses SQLAlchemy; for Windows paths, use forward slashes in sqlite URL.
    tpe_storage_url = f"sqlite:///{tpe_db_path.as_posix()}"
    tpe_study_name = str(args.tpe_study_name).strip() or f"plot2_{run_id}_tpe"
    _write_json(
        plot2_dir / "optuna" / "tpe_study_info.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "study_name": str(tpe_study_name),
            "storage_url": str(tpe_storage_url),
            "db_path": str(tpe_db_path),
        },
    )

    H = int(args.H)
    degree_regimes = {
        "super_sparse": [int(x) for x in str(args.regime_super_sparse).split(",") if x.strip()],
        "sparse": [int(x) for x in str(args.regime_sparse).split(",") if x.strip()],
        "moderate": [int(x) for x in str(args.regime_moderate).split(",") if x.strip()],
        "near_dense": [int(x) for x in str(args.regime_near_dense).split(",") if x.strip()],
    }
    # Validate ks
    for name, ks in degree_regimes.items():
        if not ks:
            raise ValueError(f"Empty k list for regime: {name}")
        for k in ks:
            if k % 2 != 0 or k < 2 or k > H - 2:
                raise ValueError(f"Invalid k={k} for regime {name} (H={H}); require even, 2 <= k <= H-2")

    # Locked Plot 2 selection policy: when using regime-only (no C,L bins), coerce B = 2 * (#regimes).
    _cov = str(getattr(args, "selection_coverage_level", "regime_cl_bins")).strip().lower()
    _strat = str(getattr(args, "selection_strategy", "pareto_farthest_regime")).strip().lower()
    if (
        _cov == "regime"
        and _strat in {"pareto_farthest_regime", "pareto_score_regime"}
        and (not bool(getattr(args, "selection_allow_missing_regimes", False)))
    ):
        args.B = int(2 * len(degree_regimes))

    # Plot 2 Overhaul: allow MODE_NONE (selection_coverage_level=none) for selection protocol ablation (M3).
    # MODE_REGIME = regime (2 per regime, no C,L bins); MODE_NONE = none (any B=8).
    # Legacy: regime_cl_bins_fixed remains default for non-overhaul runs.

    # Plot 2 Overhaul: regime_cl_bins_fixed requires proxy viability output
    proxy_viability_dir = getattr(args, "proxy_viability_dir", "") or ""
    if _cov == "regime_cl_bins_fixed" and not proxy_viability_dir.strip():
        raise ValueError(
            "Plot 2 Overhaul: --selection_coverage_level regime_cl_bins_fixed requires --proxy_viability_dir. "
            "Run run_plot2_proxy_viability.py first."
        )

    analyzer = TopologyAnalyzer(default_config, logger=None)

    # ---- Reference-set normalization (fixed bounds shared across methods) ----
    fixed_tertile_edges: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
    mu_te_by_k: Dict[int, float] = {}
    mu_orc_by_k: Dict[int, float] = {}
    if proxy_viability_dir.strip():
        pv_path = Path(proxy_viability_dir).resolve()
        fixed_tertile_edges, mu_orc_by_k, mu_te_by_k, norm_bounds = _load_proxy_viability_output(pv_path)
        norm_meta = {"cl_bin_edges": fixed_tertile_edges}
        print(f"[PLOT2] Loaded proxy viability from {pv_path} (TE_res/sigma primary objectives)")
    else:
        ref_rng = np.random.default_rng(202600)
        k_values = sorted({k for ks in degree_regimes.values() for k in ks})
        norm_bounds, norm_meta = _compute_reference_bounds(
            analyzer=analyzer,
            H=H,
            k_values=k_values,
            M_ref=int(args.M_ref),
            q_lo=float(args.ref_q_lo),
            q_hi=float(args.ref_q_hi),
            rng=ref_rng,
            degree_regimes=degree_regimes,
        )

    # Shared seed set used for all models/topologies in this run (fair comparisons).
    # This is intentionally *decoupled* from graph generation seeds (graph_seed/wiring_seed).
    training_seeds_str = str(getattr(args, "training_seeds", "")).strip()
    training_seeds: List[int] = []
    training_seed_strategy = "run_id_hash"
    training_seed_base = None
    if training_seeds_str:
        seen = set()
        for tok in training_seeds_str.split(","):
            if not str(tok).strip():
                continue
            s = int(str(tok).strip())
            if not (1 <= s < (2**31 - 1)):
                raise ValueError(f"Invalid training seed {s}; require 1 <= seed < 2**31-1.")
            if s in seen:
                continue
            seen.add(s)
            training_seeds.append(int(s))
        if not training_seeds:
            raise ValueError("--training_seeds provided but parsed empty list.")
        training_seed_strategy = "explicit"
    else:
        if getattr(args, "training_seed_base", None) is not None:
            training_seed_base = int(args.training_seed_base)
            training_seed_strategy = "fixed_base"
        else:
            # NOTE: Python's built-in hash() is salted per process by default; use a stable hash.
            payload = f"{run_id}::plot2_training_seed_base".encode("utf-8")
            digest = hashlib.sha256(payload).digest()
            training_seed_base = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**31 - 1)
            training_seed_strategy = "run_id_hash"
        training_seeds = _derive_training_seeds_from_base_seed(int(training_seed_base), int(args.S))

    generator_mode = str(getattr(args, "generator_mode", "ws_flex"))
    seed_mod_params = int(getattr(args, "seed_mod_params", 202607))

    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "dataset": str(args.dataset),
        "eval_mode": str(args.eval_mode),
        "subjects": list(args.subjects),
        "H": H,
        "ncp_io_size": int(args.ncp_io_size),
        "degree_regimes": degree_regimes,
        "search_budget": {
            "M0": int(args.M0),
            "dM": int(args.dM),
            "M_max": int(args.M_max),
            "hv_ref": [-0.05, -0.05],
            "M_ref": int(args.M_ref),
            "ref_q_lo": float(args.ref_q_lo),
            "ref_q_hi": float(args.ref_q_hi),
            "hv_window_W": int(args.hv_window_W),
            "hv_window_eps": float(args.hv_window_eps),
            "hv_window_patience": int(args.hv_window_patience),
            "pareto_new_m": int(args.pareto_new_m),
            "pareto_patience_batches": int(args.pareto_patience_batches),
        },
        "optuna": {
            "tpe": {
                "study_name": str(tpe_study_name),
                "storage_url": str(tpe_storage_url),
                "db_path": str(tpe_db_path),
                "sampler_seed": 202601,
            }
        },
        "normalization": {"kind": "reference_set_quantiles", "ref_seed": 202600, **norm_meta},
        "reproducibility": {
            "reference_bounds_seed": 202600,
            "cl_bin_edges_seed": 202600,
            "orientation_policy": "random_oriented",
            "orientation_seed_mapping": "wiring_seed",
            "external_random_base_seed": 202604,
        },
        "selection": {
            "B": int(args.B),
            "S": int(args.S),
            "score": "te_norm + orc_norm",
            "strategy": str(args.selection_strategy),
            "coverage_level": str(args.selection_coverage_level),
            "allow_missing_regimes": bool(args.selection_allow_missing_regimes),
            "selection_policy": "stratified_by_regime",
        },
        "training_seeds": {
            "strategy": str(training_seed_strategy),
            "S": int(len(training_seeds)),
            "base_seed": (int(training_seed_base) if training_seed_base is not None else None),
            "seeds": [int(s) for s in training_seeds],
        },
        "gaussian_alpha_grid": alpha_grid,
        "perturbation_types": perturbation_types,
        "primary_perturbation_type": perturbation_types[0] if perturbation_types else "ar1_drift",
        "target_snr_db": float(getattr(args, "target_snr_db", -5.0)),
        "perturbation_params": {
            "ar1_drift": {"rho": float(getattr(args, "ar1_rho", 0.97))},
            "spatial_gaussian": {"ell_multiplier": float(getattr(args, "spatial_ell_multiplier", 1.0))},
            "emg_band": {
                "f_low": float(getattr(args, "emg_f_low", 20.0)),
                "f_high": float(getattr(args, "emg_f_high", 80.0)),
                "envelope_on": bool(getattr(args, "emg_envelope_on", False)),
            },
        },
        "saturation_file": str(args.saturation_file),
        "model_key": "cnn_wiredcfc_min",
        "ncp_baseline": {
            "wiring_kind": "ncp_autoncp",
            # Capacity-fair sizing: NCP recurrent chamber must match WS-Flex H (32 units).
            # output_size = ncp_io_size (F2) for CNNWiredCfCMin proj_size compatibility.
            "units": int(H),
            "output_size": int(args.ncp_io_size),
            "sparsity_level": float(args.ncp_sparsity_level),
        },
        "git_commit": _get_git_commit(repo_root),
        # PATCH 0.1: required manifest fields (Plot_2_Investigation.txt)
        "perturbation_type": perturbation_types[0] if perturbation_types else "ar1_drift",
        "empirical_snr_db": None,
        "alpha_grid": alpha_grid,
        "generator_bounds": {
            "k_min": int(min(kk for ks in degree_regimes.values() for kk in ks)) if degree_regimes else 2,
            "k_max": int(max(kk for ks in degree_regimes.values() for kk in ks)) if degree_regimes else 30,
            "p_min": 0.0,
            "p_max": 1.0,
        },
        "selection_method": str(args.selection_strategy),
        "search_seeds": {"baseline_a_b": 202602, "tpe": 202601},
        "baseline_definitions": {
            "baseline_a": "True random WS-Flex; same N as B, selection uniform random within (regime, C_bin, L_bin); TE/ORC not used for selection.",
            "baseline_b": "Random WS-Flex + offline proxy filtering; same pool as A, selection by z_bin(TE)+z_bin(ORC) within bins.",
            "baseline_c": "TPE (adaptive proxy-guided); N trials, same coverage-aware proxy selection as B.",
        },
        "shared_random_pool_size": int(args.M_max),
        # Spec §6.3 NEW 3: selected_architectures path and TE/ORC formula reference
        "selected_architectures_csv_path": "selected_architectures.csv",
        "te_orc_formulas": "TE: degree entropy normalized by log(N); ORC: signed mean Ollivier-Ricci curvature, alpha=0.5",
        # D1: RNG stream separation (Plot 2 Overhaul)
        "seed_ref": None if proxy_viability_dir.strip() else 202600,
        "seed_pool_ab": 202602,
        "seed_tpe": 202601,
        "seed_orient": "wiring_seed",
        "seed_train": int(training_seed_base) if training_seed_base is not None else None,
        "generator_mode": generator_mode,
        "seed_mod_params": seed_mod_params,
        "modular_param_bounds": {
            "M_values": list(DEFAULT_M_VALUES),
            "p_out": [DEFAULT_P_OUT_LO, DEFAULT_P_OUT_HI],
            "r_out": [DEFAULT_R_OUT_LO, DEFAULT_R_OUT_HI],
        } if generator_mode == "modular_ws_flex" else None,
        # F1: Manifest-locked metric views (undirected vs directed)
        "metrics_graph_view": {
            "undirected_pre_orient": ["TE", "TE_res", "C", "L", "sigma", "ORC", "ORC_res"],
            "directed_mask": ["rho", "rho_norm"],
        },
        # F2: Capacity schema for all baselines
        "capacity_schema": {
            "H_fixed": int(H),
            "description": "All baselines use recurrent size H. WS-Flex: H hidden nodes. NCP: units=H. External: units=H.",
            "E_active_bands": dict(DEFAULT_E_ACTIVE_BANDS_H32) if H == 32 else {},
        },
    }
    # Plot 2 Overhaul: proxy viability fields when using regime_cl_bins_fixed
    if proxy_viability_dir.strip():
        pv_path = Path(proxy_viability_dir).resolve()
        report_path = pv_path / "proxy_viability_report.json"
        frozen_path = pv_path / "frozen_bin_edges.json"
        manifest["proxy_viability_run_id"] = str(pv_path.name)
        manifest["proxy_viability_dir"] = str(pv_path)
        manifest["bin_edge_source"] = "G0_neutral_reference"
        manifest["primary_objectives"] = ["TE_res", "sigma"]
        if frozen_path.exists():
            manifest["frozen_bin_edges"] = json.loads(frozen_path.read_text(encoding="utf-8")).get("frozen_bin_edges", {})
        if (pv_path / "mu_orc_by_k.json").exists():
            manifest["mu_orc_by_k"] = json.loads((pv_path / "mu_orc_by_k.json").read_text(encoding="utf-8"))
        if (pv_path / "mu_te_by_k.json").exists():
            manifest["mu_te_by_k"] = json.loads((pv_path / "mu_te_by_k.json").read_text(encoding="utf-8"))
        if report_path.exists():
            rpt = json.loads(report_path.read_text(encoding="utf-8"))
            manifest["proxy_viability_seed_ref"] = rpt.get("seed_ref")
    validate_manifest(manifest, strict=True)
    _write_json(plot2_dir / "plot2_manifest.json", manifest)
    _write_json(plot2_dir / "manifest.json", manifest)

    # ---- Diagnostics (human-readable + machine-readable) ----
    _write_text(
        plot2_dir / "diagnostics" / "README.txt",
        "\n".join(
            [
                "Plot 2 diagnostics",
                "",
                "- k_distribution_<method>.json:",
                "  Contains candidate-level and selected-level k/regime distributions, plus selection policy metadata.",
                "",
            ]
        ),
    )

    # ---- Training-free search: shared random pool (Baseline A/B) + TPE (Baseline C) ----
    search_outputs: Dict[str, Dict[str, Any]] = {}
    selected_rows: List[Dict[str, Any]] = []
    selection_collapse_scores: Dict[str, Optional[float]] = {}
    selection_coverage_scores: Dict[str, Optional[float]] = {}
    selected_arch_dir = plot2_dir / "selected_architectures"
    selected_arch_dir.mkdir(parents=True, exist_ok=False)

    # Build one shared random pool for Baseline A (uniform-in-bin) and Baseline B (proxy-in-bin)
    print("[PLOT2] Search start: shared random pool (baseline_a / baseline_b)")
    shared_random_pool, shared_hv_log = _build_random_pool(
        analyzer=analyzer,
        H=H,
        degree_regimes=degree_regimes,
        M0=int(args.M0),
        dM=int(args.dM),
        M_max=int(args.M_max),
        base_seed=202602,
        capacity_filter_on=bool(getattr(args, "capacity_filter", False)),
        mu_te_by_k=mu_te_by_k if mu_te_by_k else None,
        mu_orc_by_k=mu_orc_by_k if mu_orc_by_k else None,
        generator_mode=generator_mode,
        seed_mod_params=seed_mod_params,
    )
    # Norms and Pareto for shared pool (for CSV and diagnostics)
    if getattr(norm_bounds, "use_te_res_sigma", False):
        _x = np.asarray([c.te_res for c in shared_random_pool], dtype=float)
        _y = np.asarray([c.sigma for c in shared_random_pool], dtype=float)
        pool_te_norm = _normalize_fixed(_x, lo=norm_bounds.te_res_lo, hi=norm_bounds.te_res_hi)
        pool_orc_norm = _normalize_fixed(_y, lo=norm_bounds.sigma_lo, hi=norm_bounds.sigma_hi)
    else:
        _te = np.asarray([c.te for c in shared_random_pool], dtype=float)
        _orc = np.asarray([c.orc for c in shared_random_pool], dtype=float)
        pool_te_norm = _normalize_fixed(_te, lo=norm_bounds.te_lo, hi=norm_bounds.te_hi)
        pool_orc_norm = _normalize_fixed(_orc, lo=norm_bounds.orc_lo, hi=norm_bounds.orc_hi)
    pool_score = pool_te_norm + pool_orc_norm
    pool_pareto = _pareto_membership_2d(pool_te_norm, pool_orc_norm)
    rows_shared = _cands_to_rows(
        shared_random_pool,
        te_norm=pool_te_norm,
        orc_norm=pool_orc_norm,
        pareto_mask=pool_pareto,
        score=pool_score,
    )
    _write_csv(plot2_dir / "candidates" / "candidates_shared_random.csv", rows_shared)
    _write_json(plot2_dir / "hypervolume" / "hypervolume_shared_random.json", shared_hv_log)
    # B4: Rejection rate diagnostics
    if any(k in shared_hv_log for k in ("rejection_rate_by_regime", "rejection_rate_overall")):
        (plot2_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
        _write_json(
            plot2_dir / "diagnostics" / "rejection_rate_shared_random.json",
            {k: shared_hv_log[k] for k in ("rejection_rate_by_regime", "rejection_rate_overall", "rejections_disconnected", "rejections_capacity") if k in shared_hv_log},
        )

    # Baseline A: uniform random within (regime, C_bin, L_bin); Baseline B: proxy-ranked within bins
    # F4: Use fixed cl_bin_edges from reference for deterministic binning
    cl_bin_edges = norm_meta.get("cl_bin_edges")
    selected_idxs_a, meta_a = _select_top_b_coverage_aware(
        shared_random_pool,
        int(args.B),
        degree_regimes=degree_regimes,
        selection_allow_missing_regimes=bool(args.selection_allow_missing_regimes),
        rank_by_proxy=False,
        uniform_seed=202602,
        fixed_tertile_edges=cl_bin_edges,
    )
    selected_idxs_b, meta_b = _select_top_b_coverage_aware(
        shared_random_pool,
        int(args.B),
        degree_regimes=degree_regimes,
        selection_allow_missing_regimes=bool(args.selection_allow_missing_regimes),
        rank_by_proxy=True,
        fixed_tertile_edges=cl_bin_edges,
    )
    selected_a = [shared_random_pool[i] for i in selected_idxs_a]
    selected_b = [shared_random_pool[i] for i in selected_idxs_b]
    for meta in (meta_a, meta_b):
        meta["te_norm"] = pool_te_norm
        meta["orc_norm"] = pool_orc_norm
        meta["score"] = pool_score
        meta["pareto_mask"] = pool_pareto

    def _process_ws_flex_baseline(
        method: str,
        cands: List[Candidate],
        selected: List[Candidate],
        meta: Dict[str, Any],
        hv_log: Dict[str, Any],
        write_candidates_csv: bool = False,
        write_hypervolume: bool = False,
    ) -> None:
        collapse_score = meta.get("collapse_score")
        selection_collapse_scores[str(method)] = collapse_score
        selection_coverage_scores[str(method)] = meta.get("coverage_score")
        if collapse_score is not None and collapse_score > 0.50 and not getattr(args, "allow_regime_collapse", False):
            raise ValueError(
                f"Plot 2 selection collapsed for method {method!r}: collapse_score={collapse_score:.3f} > 0.50. "
                "Use stratified selection or pass --allow_regime_collapse with justification."
            )
        k_dist_candidates = _summarize_k_distribution(
            k_values=[int(c.k) for c in cands],
            degree_regimes=degree_regimes,
        )
        k_dist_selected = _summarize_k_distribution(
            k_values=[int(c.k) for c in selected],
            degree_regimes=degree_regimes,
        )
        _write_json(
            plot2_dir / "diagnostics" / f"k_distribution_{method}.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "method": method,
                "selection": {
                    "strategy": str(args.selection_strategy),
                    "coverage_level": str(args.selection_coverage_level),
                    "allow_missing_regimes": bool(args.selection_allow_missing_regimes),
                    "B": int(args.B),
                },
                "selection_meta": {
                    "selection_strategy": meta.get("selection_strategy"),
                    "selection_coverage_level": meta.get("selection_coverage_level"),
                    "selection_allow_missing_regimes": meta.get("selection_allow_missing_regimes"),
                    "collapse_score": meta.get("collapse_score"),
                    "coverage_score": meta.get("coverage_score"),
                    "n_occupied_bins_selected": meta.get("n_occupied_bins_selected"),
                    "n_bins_total": meta.get("n_bins_total"),
                    "selected_cell_counts": meta.get("selected_cell_counts"),
                    "regime_counts": meta.get("regime_counts"),
                    "coverage_aware_candidate_diagnostics": meta.get("coverage_aware_candidate_diagnostics"),
                    "n_pareto_total": meta.get("n_pareto_total"),
                    "pareto_available_by_regime": meta.get("pareto_available_by_regime"),
                    "regime_quotas": meta.get("regime_quotas"),
                    "regime_shortfalls": meta.get("regime_shortfalls"),
                    "regime_neighbor_backfill_taken": meta.get("regime_neighbor_backfill_taken"),
                    "n_unknown_regime_candidates": meta.get("n_unknown_regime_candidates"),
                    "candidate_k_distribution": meta.get("candidate_k_distribution"),
                    "selected_k_distribution": meta.get("selected_k_distribution"),
                },
                "candidates": k_dist_candidates,
                "selected": k_dist_selected,
            },
        )
        n_candidates_by_regime = k_dist_candidates.get("counts_by_regime", {})
        orc_raw_arr = np.array([float(c.orc_raw) for c in cands])
        k_arr = np.array([int(c.k) for c in cands])
        if orc_raw_arr.size >= 2 and np.std(k_arr) > 0:
            orc_raw_vs_k_correlation = float(np.corrcoef(orc_raw_arr, k_arr)[0, 1])
        else:
            orc_raw_vs_k_correlation = float("nan")
        pareto_mask_arr = meta.get("pareto_mask")
        pareto_width = int(np.sum(pareto_mask_arr)) if pareto_mask_arr is not None else int(meta.get("n_pareto_total", 0))
        pareto_by_regime: Dict[str, int] = {str(r): 0 for r in degree_regimes}
        if pareto_mask_arr is not None:
            for i, c in enumerate(cands):
                if i < len(pareto_mask_arr) and pareto_mask_arr[i]:
                    r = _k_to_regime(int(c.k), degree_regimes=degree_regimes)
                    if r is not None:
                        pareto_by_regime[str(r)] = pareto_by_regime.get(str(r), 0) + 1
        _write_json(
            plot2_dir / "diagnostics" / f"phase2_diagnostics_{method}.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "method": method,
                "n_candidates": len(cands),
                "n_candidates_by_regime": n_candidates_by_regime,
                "orc_raw_vs_k_correlation": orc_raw_vs_k_correlation,
                "pareto_width": pareto_width,
                "pareto_by_regime": pareto_by_regime,
                "note": "High orc_raw_vs_k_correlation implies proxy is density-dominated.",
            },
        )
        if write_candidates_csv:
            rows = _cands_to_rows(
                cands,
                te_norm=meta["te_norm"],
                orc_norm=meta["orc_norm"],
                pareto_mask=meta["pareto_mask"],
                score=meta["score"],
            )
            _write_csv(plot2_dir / "candidates" / f"candidates_{method}.csv", rows)
        if write_hypervolume:
            _write_json(plot2_dir / "hypervolume" / f"hypervolume_{method}.json", hv_log)
        for rank, cand in enumerate(selected, start=1):
            model_name = f"plot2_{run_id}_{method}_b{rank}"
            cand_generator_mode = getattr(cand, "generator_mode", "ws_flex")
            if cand_generator_mode == "modular_ws_flex" and getattr(cand, "M", None) is not None:
                G, _ = make_ws_flex_graph(
                    H, cand.k, cand.p, cand.graph_seed,
                    generator_mode="modular_ws_flex",
                    M=cand.M,
                    p_out=getattr(cand, "p_out", 0.1),
                    r_out=getattr(cand, "r_out", 0.05),
                )
            else:
                G = _make_ws_graph(H, cand.k, cand.p, seed=cand.graph_seed)
            if not nx.is_connected(G):
                raise RuntimeError("Selected graph unexpectedly disconnected (should not happen).")
            topo = analyzer.analyze_graph(G)
            undirected_adj = _undirected_hidden_adj(G, H)
            directed_adj = _oriented_hidden_adj(G, H, seed=cand.wiring_seed)
            spectral_radius_directed = float(compute_spectral_radius_directed(directed_adj))
            mean_degree_undirected = float(undirected_adj.sum() / float(H)) if H > 0 else float("nan")
            if cand_generator_mode != "modular_ws_flex" and (
                not np.isfinite(mean_degree_undirected) or abs(mean_degree_undirected - float(cand.k)) > 1e-6
            ):
                raise RuntimeError(
                    f"Effective mean degree sanity check failed for selected graph: "
                    f"mean_degree_undirected={mean_degree_undirected} vs requested k={cand.k} (H={H})."
                )
            arch_row = {
                "schema_version": 2,
                "run_id": run_id,
                "method": method,
                "rank": int(rank),
                "model_name": model_name,
                "H": H,
                "wiring_kind": "ws_flex",
                "k": int(cand.k),
                "p": float(cand.p),
                "graph_seed": int(cand.graph_seed),
                "wiring_seed": int(cand.wiring_seed),
                "te": float(cand.te),
                "orc": float(cand.orc),
                "orc_raw": float(cand.orc_raw),
                "mean_degree_undirected": float(mean_degree_undirected),
                "density": float(topo.get("density", float("nan"))),
                "clustering": float(topo.get("clustering_coefficient", float("nan"))),
                "path_length": float(topo.get("avg_path_length", float("nan"))),
                "spectral_radius": float(topo.get("spectral_radius", float("nan"))),
                "spectral_radius_directed": float(spectral_radius_directed),
                "selection_method": str(args.selection_strategy),
                "hidden_adj_undirected": undirected_adj.tolist(),
                "hidden_adj_directed": directed_adj.tolist(),
                "graph_hash": _graph_hash_from_adj(
                    undirected_adj, H, int(cand.k), float(cand.p), int(cand.graph_seed),
                    mod_params={"M": cand.M, "p_out": cand.p_out, "r_out": cand.r_out}
                    if cand_generator_mode == "modular_ws_flex" and getattr(cand, "M", None) is not None
                    else None,
                ),
                "generator_mode": cand_generator_mode,
            }
            if cand_generator_mode == "modular_ws_flex" and getattr(cand, "M", None) is not None:
                arch_row["M"] = int(cand.M)
                arch_row["p_out"] = float(cand.p_out)
                arch_row["r_out"] = float(cand.r_out) if cand.r_out is not None else None
            cl_edges = norm_meta.get("cl_bin_edges")
            if cl_edges and np.isfinite(cand.clustering) and np.isfinite(cand.path_length):
                r = _k_to_regime(int(cand.k), degree_regimes=degree_regimes)
                if r and r in cl_edges:
                    c_lo, c_hi = cl_edges[r]["C"]
                    l_lo, l_hi = cl_edges[r]["L"]
                    arch_row["C_bin"] = "low" if cand.clustering <= c_lo else ("medium" if cand.clustering <= c_hi else "high")
                    arch_row["L_bin"] = "low" if cand.path_length <= l_lo else ("medium" if cand.path_length <= l_hi else "high")
            if hasattr(cand, "sigma") and np.isfinite(getattr(cand, "sigma", float("nan"))):
                arch_row["sigma"] = float(cand.sigma)
                arch_row["te_res"] = float(getattr(cand, "te_res", float("nan")))
                arch_row["orc_res"] = float(getattr(cand, "orc_res", float("nan")))
            selected_rows.append(arch_row)
            _write_json(selected_arch_dir / f"{model_name}.json", arch_row)
        search_outputs[method] = {
            "n_candidates": int(len(cands)),
            "stop_reason": hv_log.get("stop_reason") if hv_log else None,
        }

    _process_ws_flex_baseline("baseline_a", shared_random_pool, selected_a, meta_a, shared_hv_log, write_candidates_csv=False, write_hypervolume=False)
    _process_ws_flex_baseline("baseline_b", shared_random_pool, selected_b, meta_b, shared_hv_log, write_candidates_csv=False, write_hypervolume=False)

    # Baseline C: TPE pool + proxy-ranked coverage-aware selection
    print("[PLOT2] Search start: tpe")
    tpe_cands, tpe_hv_log = _run_training_free_search(
        analyzer=analyzer,
        method="tpe",
        H=H,
        degree_regimes=degree_regimes,
        bounds=norm_bounds,
        M0=int(args.M0),
        dM=int(args.dM),
        M_max=int(args.M_max),
        hv_window_W=int(args.hv_window_W),
        hv_window_eps=float(args.hv_window_eps),
        hv_window_patience=int(args.hv_window_patience),
        pareto_new_m=int(args.pareto_new_m),
        pareto_patience_batches=int(args.pareto_patience_batches),
        base_seed=202601,
        optuna_storage_url=str(tpe_storage_url),
        optuna_study_name=str(tpe_study_name),
        capacity_filter_on=bool(getattr(args, "capacity_filter", False)),
        mu_te_by_k=mu_te_by_k if mu_te_by_k else None,
        mu_orc_by_k=mu_orc_by_k if mu_orc_by_k else None,
        generator_mode=generator_mode,
        seed_mod_params=seed_mod_params,
    )
    selected_c, meta_c = _select_top_b(
        tpe_cands,
        int(args.B),
        bounds=norm_bounds,
        degree_regimes=degree_regimes,
        selection_coverage_level=str(args.selection_coverage_level),
        selection_allow_missing_regimes=bool(args.selection_allow_missing_regimes),
        selection_strategy=str(args.selection_strategy),
        cl_bin_edges=norm_meta.get("cl_bin_edges"),
    )
    _process_ws_flex_baseline("tpe", tpe_cands, selected_c, meta_c, tpe_hv_log, write_candidates_csv=True, write_hypervolume=True)

    # ---- NCP baseline (hand-designed wiring) ----
    # Implemented as an AutoNCP wiring (units=H). This is evaluated with the same CNN–WiredCfC-Min skeleton.
    ncp_model_name = f"plot2_{run_id}_ncp"
    ncp_arch = {
        "schema_version": 2,
        "run_id": run_id,
        "method": "baseline",
        "baseline_type": "ncp_autoncp",
        "rank": 1,
        "model_name": ncp_model_name,
        "wiring_kind": "ncp_autoncp",
        "units": int(manifest["ncp_baseline"]["units"]),
        "output_size": int(manifest["ncp_baseline"]["output_size"]),
        "sparsity_level": float(manifest["ncp_baseline"]["sparsity_level"]),
        "wiring_seed": 202603,
        "selection_method": "baseline",
    }
    selected_rows.append(ncp_arch)
    _write_json(selected_arch_dir / f"{ncp_model_name}.json", ncp_arch)

    # ---- G3: External random wiring baseline (Plot 2 spec §2.5, §6.3) ----
    B_ext = int(getattr(args, "B_ext", 12))
    if B_ext > 0:
        external_candidates, external_summary = _generate_and_select_external_random(
            H=H,
            B_ext=B_ext,
            output_size=int(manifest["ncp_baseline"]["output_size"]),
            degree_regimes=degree_regimes,
            pool_size=max(4 * B_ext, 80),
            base_seed=202604,
        )
        _write_json(plot2_dir / "diagnostics" / "external_random_selection.json", external_summary)
        for rank, ext in enumerate(external_candidates, start=1):
            ext_model_name = f"plot2_{run_id}_external_random_b{rank}"
            ext_arch = {
                "schema_version": 2,
                "run_id": run_id,
                "method": "external_random",
                "rank": int(rank),
                "model_name": ext_model_name,
                "wiring_kind": "external_random",
                "units": int(H),
                "output_size": int(manifest["ncp_baseline"]["output_size"]),
                "sparsity_level": float(ext["sparsity_level"]),
                "wiring_seed": int(ext["wiring_seed"]),
                "mask_density": float(ext["mask_density"]),
                "selection_method": "density_stratified",
            }
            selected_rows.append(ext_arch)
            _write_json(selected_arch_dir / f"{ext_model_name}.json", ext_arch)

    # PATCH 0.3: selection regime histogram (prevents silent collapse to one k)
    regime_counts: Dict[str, int] = {str(r): 0 for r in degree_regimes}
    for row in selected_rows:
        if row.get("wiring_kind") != "ws_flex" or "k" not in row:
            continue
        r = _k_to_regime(int(row["k"]), degree_regimes=degree_regimes)
        if r is not None:
            regime_counts[str(r)] = regime_counts.get(str(r), 0) + 1
    total_ws_flex_selected = sum(regime_counts.values())
    _write_json(
        plot2_dir / "diagnostics" / "selection_regime_histogram.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "counts_by_regime": regime_counts,
            "total_ws_flex_selected": total_ws_flex_selected,
        },
    )
    _write_json(
        plot2_dir / "diagnostics" / "selection_diagnostics.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "selection_coverage_level": str(args.selection_coverage_level),
            "selection_strategy": str(args.selection_strategy),
            "B": int(args.B),
            "selection_collapse_scores": {k: v for k, v in selection_collapse_scores.items()},
            "selection_coverage_scores": {k: v for k, v in selection_coverage_scores.items()},
        },
    )
    # FIX 3: Regime collapse check — if >50% of selected models from one regime, hard fail unless overridden
    if total_ws_flex_selected > 0:
        max_in_regime = max(regime_counts.values())
        if max_in_regime > 0.5 * total_ws_flex_selected and not getattr(args, "allow_regime_collapse", False):
            raise ValueError(
                f"Plot 2 selection collapsed: one regime has {max_in_regime}/{total_ws_flex_selected} models (>50%%). "
                "Use stratified selection (Path A) or pass --allow_regime_collapse with justification."
            )

    # Plot2_revision2: emit selected_A/B/C.jsonl (and optional D) with stable graph IDs and graph_id hash
    def _write_manifest_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                out = {k: v for k, v in r.items() if k not in ("hidden_adj_undirected", "hidden_adj_directed")}
                out["graph_id"] = _graph_id_hash(r)
                f.write(json.dumps(out) + "\n")

    for method_key, filename in [("baseline_a", "selected_A.jsonl"), ("baseline_b", "selected_B.jsonl"), ("tpe", "selected_C.jsonl")]:
        subset = [r for r in selected_rows if r.get("method") == method_key]
        if subset:
            _write_manifest_jsonl(subset, plot2_dir / filename)
    ext_subset = [r for r in selected_rows if r.get("method") == "external_random"]
    if ext_subset:
        _write_manifest_jsonl(ext_subset, plot2_dir / "selected_D.jsonl")

    _write_csv(plot2_dir / "selected_architectures.csv", selected_rows)
    capacity_manifest = _build_capacity_manifest(selected_rows, H)
    _write_json(plot2_dir / "capacity_manifest.json", capacity_manifest)
    _write_json(plot2_dir / "plot2_search_summary.json", search_outputs)

    # Update manifest with selection_collapse_scores (spec §7) and re-write
    manifest["selection_collapse_scores"] = {k: v for k, v in selection_collapse_scores.items()}
    # I2: Compute budget
    wall_clock_sec = time.perf_counter() - wall_clock_start
    n_graphs_sampled = len(shared_random_pool) + search_outputs.get("tpe", {}).get("n_trials", 0)
    n_graphs_trained = len(selected_rows) * len(training_seeds)
    manifest["compute_budget"] = {
        "graphs_sampled_pool_ab": len(shared_random_pool),
        "graphs_sampled_tpe": search_outputs.get("tpe", {}).get("n_trials", 0),
        "graphs_sampled_total": n_graphs_sampled,
        "graphs_trained_total": n_graphs_trained,
        "wall_clock_seconds": round(wall_clock_sec, 2),
        "gpu_hours_estimate": None,  # Populated by training jobs if tracked
    }
    # Plot 2 Overhaul: dual-SNR manifest key; analysis_target_snr_db for locked analyzer (mini: -12)
    target_snr_dbs_str = getattr(args, "target_snr_dbs", None) or ""
    if target_snr_dbs_str.strip():
        target_snr_list = [float(x.strip()) for x in target_snr_dbs_str.split(",") if x.strip()]
        if target_snr_list:
            manifest["target_snr_dbs"] = target_snr_list
            # G0: analyzer uses first (harshest) for locked config; override via manifest.analysis_target_snr_db
            manifest["analysis_target_snr_db"] = float(target_snr_list[0])
    else:
        single = float(getattr(args, "target_snr_db", -6.0))
        manifest["target_snr_db"] = single
        manifest["analysis_target_snr_db"] = single
    _write_json(plot2_dir / "plot2_manifest.json", manifest)
    _write_json(plot2_dir / "manifest.json", manifest)

    # ---- Jobs table (topology × training seed) ----
    target_snr_dbs_str = getattr(args, "target_snr_dbs", None) or ""
    target_snr_val: Any = float(getattr(args, "target_snr_db", -6.0))
    if target_snr_dbs_str.strip():
        target_snr_val = [float(x.strip()) for x in target_snr_dbs_str.split(",") if x.strip()]
        if not target_snr_val:
            target_snr_val = float(getattr(args, "target_snr_db", -6.0))
    jobs: List[Dict[str, Any]] = []
    for arch in selected_rows:
        model_name = str(arch["model_name"])
        method = str(arch.get("method", "unknown"))
        for s in training_seeds:
            jobs.append(
                {
                    "run_id": run_id,
                    "model_name": model_name,
                    "method": method,
                    "seed": int(s),
                    "dataset": str(args.dataset),
                    "eval_mode": str(args.eval_mode),
                    "subjects": ",".join(str(x) for x in list(args.subjects)),
                    "mode": "test_perturb",
                    "saturation_file": str(args.saturation_file),
                    "alpha_grid": ",".join(str(a) for a in alpha_grid),
                    "perturbation_types": ",".join(perturbation_types),
                    "target_snr_db": target_snr_val,
                }
            )
    _write_csv(plot2_dir / "jobs.csv", jobs)

    print(f"[PLOT2] Plot2 directory: {plot2_dir}")
    if args.dry_run:
        print("[PLOT2] Dry-run complete (no training dispatched).")
        return

    # ---- Dispatch training/evaluation ----
    failed: List[Dict[str, Any]] = []
    for job in jobs:
        job_subjects = [int(x) for x in str(job["subjects"]).split(",") if x.strip()]
        job_alpha = [float(x) for x in str(job["alpha_grid"]).split(",") if x.strip()]
        job_perturb = [x.strip() for x in str(job.get("perturbation_types", "gaussian")).split(",") if x.strip()]
        if not job_perturb:
            job_perturb = ["gaussian"]
        if not args.overwrite and _should_skip_plot2_job(
            model_name=str(job["model_name"]),
            dataset=str(job["dataset"]),
            eval_mode=str(job["eval_mode"]),
            subjects=job_subjects,
            seed=int(job["seed"]),
            saturation_file=str(job["saturation_file"]),
            alpha_grid=job_alpha,
            perturbation_types=job_perturb,
        ):
            print(f"[PLOT2] Skipping (results exist): model={job['model_name']} seed={job['seed']}")
            continue
        rc = _run_unified_job(
            repo_root=repo_root,
            python_exe=str(args.python),
            plot2_dir=plot2_dir,
            model_name=str(job["model_name"]),
            dataset=str(job["dataset"]),
            eval_mode=str(job["eval_mode"]),
            subjects=job_subjects,
            seed=int(job["seed"]),
            saturation_file=str(job["saturation_file"]),
            alpha_grid=job_alpha,
            perturbation_types=job_perturb,
            target_snr_db=job.get("target_snr_db", manifest.get("target_snr_db", -6.0)),
            perturbation_params=manifest.get("perturbation_params"),
            overwrite=bool(args.overwrite),
        )
        if rc != 0:
            failed.append({**job, "exit_code": int(rc)})
            if not args.continue_on_error:
                raise RuntimeError(f"Unified runner failed for model={job['model_name']} seed={job['seed']} (exit_code={rc}).")

    if failed:
        import pandas as pd

        out = plot2_dir / "failed_jobs.csv"
        pd.DataFrame(failed).to_csv(out, index=False)
        raise RuntimeError(f"[PLOT2] {len(failed)} jobs failed. See: {out}")


if __name__ == "__main__":
    main()

