"""
Plot 2 Proxy Viability Diagnostic (GO/NO-GO).

G0-only reference set. Computes frozen bin edges, μ_ORC(k), μ_TE(k).
Runs gates V1–V5. De-duplication by graph_hash.
Outputs must pass before any Plot 2 training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architecture_refinement.config import default_config
from architecture_refinement.topology_analyzer import TopologyAnalyzer
from architecture_refinement.metrics_te_orc import (
    compute_te_orc,
    compute_orc_residual,
    compute_te_residual,
)
from architecture_refinement.small_world_metrics import compute_small_worldness
from architecture_refinement.ws_flex_generator import (
    make_ws_flex_graph,
    sample_modular_params,
    DEFAULT_M_VALUES,
    DEFAULT_P_OUT_LO,
    DEFAULT_P_OUT_HI,
    DEFAULT_R_OUT_LO,
    DEFAULT_R_OUT_HI,
)

# Default degree regimes (G0)
DEGREE_REGIMES_DEFAULT: Dict[str, List[int]] = {
    "super_sparse": [2, 4, 6],
    "sparse": [8, 10, 12],
    "moderate": [14, 16, 18],
    "near_dense": [20, 22, 24, 26],
}

# Gate defaults
CORR_THRESHOLD = 0.60
EPS_VAR = 1e-3
# Watts-Strogatz at high k (near_dense) converges to similar structure; sigma variance is inherently low.
EPS_VAR_NEAR_DENSE = 1e-4
N_MIN_CELL = 30
# near_dense: fewer k values + similar structure → harder to fill cells; use lower occupancy threshold.
N_MIN_CELL_NEAR_DENSE = 20
OVERLAP_THRESHOLD = 0.40
PARETO_MIN_POINTS = 10
PARETO_MIN_REGIMES = 3
PARETO_MIN_CELLS = 6
V3_MIN_CELLS_PER_REGIME = 6


def _make_ws_graph(H: int, k: int, p: float, seed: int) -> nx.Graph:
    return nx.watts_strogatz_graph(int(H), int(k), float(p), seed=int(seed))


def _undirected_adj(G: nx.Graph, H: int) -> np.ndarray:
    A = nx.to_numpy_array(G, dtype=np.int8)
    return (A != 0).astype(np.int8)


def _graph_hash(
    adj: np.ndarray,
    H: int,
    k: int,
    p: float,
    graph_seed: int,
    mod_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Portable hash: hash(undirected_adj, H, k, p, graph_seed[, M, p_out, r_out])."""
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


def _k_to_regime(k: int, degree_regimes: Dict[str, List[int]]) -> Optional[str]:
    for name, ks in degree_regimes.items():
        if int(k) in {int(x) for x in ks}:
            return str(name)
    return None


def _assign_c_bin(C: float, c_lo: float, c_hi: float) -> str:
    if C <= c_lo:
        return "low"
    if C <= c_hi:
        return "medium"
    return "high"


def _assign_l_bin(L: float, l_lo: float, l_hi: float) -> str:
    if L <= l_lo:
        return "low"
    if L <= l_hi:
        return "medium"
    return "high"


def _sample_g0_reference(
    H: int,
    degree_regimes: Dict[str, List[int]],
    M_ref: int,
    rng: np.random.Generator,
    max_attempts_factor: int = 200,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    Sample M_ref connected WS-Flex graphs using G0 (uniform k, p).
    Returns (rows, graph_hashes).
    """
    k_values = sorted({k for ks in degree_regimes.values() for k in ks})
    analyzer = TopologyAnalyzer(default_config)
    rows: List[Dict[str, Any]] = []
    seen_hashes: Set[str] = set()
    attempts = 0
    max_attempts = max_attempts_factor * M_ref

    while len(rows) < M_ref:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Failed to sample M_ref={M_ref} connected G0 graphs (got {len(rows)})."
            )
        k = int(rng.choice(k_values))
        p = float(rng.uniform(0.0, 1.0))
        graph_seed = int(rng.integers(0, 2**31 - 1))
        G = _make_ws_graph(H, k, p, seed=graph_seed)
        if not nx.is_connected(G):
            continue

        adj = _undirected_adj(G, H)
        gh = _graph_hash(adj, H, k, p, graph_seed)
        if gh in seen_hashes:
            continue
        seen_hashes.add(gh)

        te, orc, _ = compute_te_orc(G, orc_alpha=0.5)
        topo = analyzer.analyze_graph(G)
        C = float(topo.get("clustering_coefficient", float("nan")))
        L = float(topo.get("avg_path_length", float("nan")))
        sigma, _, _, C_ER, L_ER, _ = compute_small_worldness(
            G, graph_id=gh, use_analytic_er=False
        )

        regime = _k_to_regime(k, degree_regimes)
        density = float(nx.density(G))

        row = {
            "k": k,
            "p": p,
            "graph_seed": graph_seed,
            "TE": float(te) if np.isfinite(te) else float("nan"),
            "ORC": float(orc) if np.isfinite(orc) else float("nan"),
            "C": C,
            "L": L,
            "sigma": float(sigma) if np.isfinite(sigma) else float("nan"),
            "regime": regime or "unknown",
            "density": density,
            "graph_hash": gh,
        }
        rows.append(row)

    return rows, seen_hashes


def _sample_g1_targeted(
    H: int,
    degree_regimes: Dict[str, List[int]],
    M_ref: int,
    rng: np.random.Generator,
    bin_edges: Dict[str, Dict[str, Tuple[float, float]]],
    mu_te_by_k: Dict[int, float],
    mu_orc_by_k: Dict[int, float],
    max_attempts_factor: int = 200,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    G1: ws_flex targeted candidate generation. Regime-stratified (k,p), assign bins from G0 edges.
    Returns (rows, graph_hashes).
    """
    regime_names = list(degree_regimes.keys())
    analyzer = TopologyAnalyzer(default_config)
    rows: List[Dict[str, Any]] = []
    seen_hashes: Set[str] = set()
    attempts = 0
    max_attempts = max_attempts_factor * M_ref

    while len(rows) < M_ref:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Failed to sample M_ref={M_ref} G1 graphs (got {len(rows)})."
            )
        regime = regime_names[len(rows) % len(regime_names)]
        k = int(rng.choice(degree_regimes[regime]))
        p = float(rng.uniform(0.0, 1.0))
        graph_seed = int(rng.integers(0, 2**31 - 1))
        G = _make_ws_graph(H, k, p, seed=graph_seed)
        if not nx.is_connected(G):
            continue

        adj = _undirected_adj(G, H)
        gh = _graph_hash(adj, H, k, p, graph_seed)
        if gh in seen_hashes:
            continue
        seen_hashes.add(gh)

        te, orc, _ = compute_te_orc(G, orc_alpha=0.5)
        topo = analyzer.analyze_graph(G)
        C = float(topo.get("clustering_coefficient", float("nan")))
        L = float(topo.get("avg_path_length", float("nan")))
        sigma, _, _, _, _, _ = compute_small_worldness(
            G, graph_id=gh, use_analytic_er=False
        )
        te_res = compute_te_residual(te, k, mu_te_by_k)
        orc_res = compute_orc_residual(orc, k, mu_orc_by_k)
        regime = _k_to_regime(k, degree_regimes)
        density = float(nx.density(G))

        row = {
            "k": k,
            "p": p,
            "graph_seed": graph_seed,
            "TE": float(te) if np.isfinite(te) else float("nan"),
            "ORC": float(orc) if np.isfinite(orc) else float("nan"),
            "TE_res": float(te_res) if np.isfinite(te_res) else float("nan"),
            "ORC_res": float(orc_res) if np.isfinite(orc_res) else float("nan"),
            "C": C,
            "L": L,
            "sigma": float(sigma) if np.isfinite(sigma) else float("nan"),
            "regime": regime or "unknown",
            "density": density,
            "graph_hash": gh,
        }
        rows.append(row)

    _assign_bins(rows, bin_edges, degree_regimes)
    return rows, seen_hashes


def _sample_g2_targeted(
    H: int,
    degree_regimes: Dict[str, List[int]],
    M_ref: int,
    rng: np.random.Generator,
    bin_edges: Dict[str, Dict[str, Tuple[float, float]]],
    mu_te_by_k: Dict[int, float],
    mu_orc_by_k: Dict[int, float],
    seed_mod_params: int = 202607,
    M_values: Tuple[int, ...] = DEFAULT_M_VALUES,
    p_out_lo: float = DEFAULT_P_OUT_LO,
    p_out_hi: float = DEFAULT_P_OUT_HI,
    r_out_lo: float = DEFAULT_R_OUT_LO,
    r_out_hi: float = DEFAULT_R_OUT_HI,
    max_attempts_factor: int = 200,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    G2: modular_ws_flex targeted candidate generation. (k,p) + M, p_out, r_out. Use G0 bin edges.
    Returns (rows, graph_hashes).
    """
    k_values = sorted({k for ks in degree_regimes.values() for k in ks})
    analyzer = TopologyAnalyzer(default_config)
    rows: List[Dict[str, Any]] = []
    seen_hashes: Set[str] = set()
    attempts = 0
    max_attempts = max_attempts_factor * M_ref

    while len(rows) < M_ref:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Failed to sample M_ref={M_ref} G2 graphs (got {len(rows)})."
            )
        k = int(rng.choice(k_values))
        p = float(rng.uniform(0.0, 1.0))
        graph_seed = int(rng.integers(0, 2**31 - 1))
        sm_seed = seed_mod_params + len(rows)
        M, p_out, r_out = sample_modular_params(
            H, sm_seed, M_values=M_values, p_out_lo=p_out_lo, p_out_hi=p_out_hi,
            r_out_lo=r_out_lo, r_out_hi=r_out_hi,
        )
        G, _ = make_ws_flex_graph(
            H, k, p, graph_seed,
            generator_mode="modular_ws_flex",
            M=M, p_out=p_out, r_out=r_out,
        )
        if not nx.is_connected(G):
            continue

        adj = _undirected_adj(G, H)
        mod_params = {"M": M, "p_out": p_out, "r_out": r_out}
        gh = _graph_hash(adj, H, k, p, graph_seed, mod_params=mod_params)
        if gh in seen_hashes:
            continue
        seen_hashes.add(gh)

        te, orc, _ = compute_te_orc(G, orc_alpha=0.5)
        topo = analyzer.analyze_graph(G)
        C = float(topo.get("clustering_coefficient", float("nan")))
        L = float(topo.get("avg_path_length", float("nan")))
        sigma, _, _, _, _, _ = compute_small_worldness(
            G, graph_id=gh, use_analytic_er=False
        )
        te_res = compute_te_residual(te, k, mu_te_by_k)
        orc_res = compute_orc_residual(orc, k, mu_orc_by_k)
        regime = _k_to_regime(k, degree_regimes)
        density = float(nx.density(G))

        row = {
            "k": k,
            "p": p,
            "graph_seed": graph_seed,
            "M": M,
            "p_out": p_out,
            "r_out": r_out,
            "TE": float(te) if np.isfinite(te) else float("nan"),
            "ORC": float(orc) if np.isfinite(orc) else float("nan"),
            "TE_res": float(te_res) if np.isfinite(te_res) else float("nan"),
            "ORC_res": float(orc_res) if np.isfinite(orc_res) else float("nan"),
            "C": C,
            "L": L,
            "sigma": float(sigma) if np.isfinite(sigma) else float("nan"),
            "regime": regime or "unknown",
            "density": density,
            "graph_hash": gh,
        }
        rows.append(row)

    _assign_bins(rows, bin_edges, degree_regimes)
    return rows, seen_hashes


def _compute_mu_by_k(rows: List[Dict[str, Any]], key: str) -> Dict[int, float]:
    """E[metric | k] per k."""
    by_k: Dict[int, List[float]] = {}
    for r in rows:
        k = int(r["k"])
        v = r.get(key)
        if v is not None and np.isfinite(v):
            by_k.setdefault(k, []).append(float(v))
    return {k: float(np.mean(vals)) for k, vals in by_k.items() if vals}


def _compute_frozen_bin_edges(
    rows: List[Dict[str, Any]], degree_regimes: Dict[str, List[int]]
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """C and L tertile edges (q=1/3, 2/3) per regime."""
    cl_by_regime: Dict[str, List[Tuple[float, float]]] = {
        str(r): [] for r in degree_regimes
    }
    for r in rows:
        reg = r.get("regime")
        if reg not in cl_by_regime:
            continue
        C, L = r.get("C"), r.get("L")
        if np.isfinite(C) and np.isfinite(L):
            cl_by_regime[reg].append((float(C), float(L)))

    edges: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for reg in degree_regimes:
        pts = cl_by_regime.get(str(reg), [])
        if not pts:
            edges[str(reg)] = {"C": (float("nan"), float("nan")), "L": (float("nan"), float("nan"))}
            continue
        Cs = [p[0] for p in pts]
        Ls = [p[1] for p in pts]
        c_lo = float(np.nanpercentile(Cs, 33.33))
        c_hi = float(np.nanpercentile(Cs, 66.67))
        l_lo = float(np.nanpercentile(Ls, 33.33))
        l_hi = float(np.nanpercentile(Ls, 66.67))
        if c_hi <= c_lo:
            c_hi = c_lo + 1e-9
        if l_hi <= l_lo:
            l_hi = l_lo + 1e-9
        edges[str(reg)] = {"C": (c_lo, c_hi), "L": (l_lo, l_hi)}
    return edges


def _assign_bins(
    rows: List[Dict[str, Any]],
    bin_edges: Dict[str, Dict[str, Tuple[float, float]]],
    degree_regimes: Dict[str, List[int]],
) -> None:
    """Assign C_bin, L_bin to each row in place."""
    for r in rows:
        reg = r.get("regime")
        if reg not in bin_edges:
            r["C_bin"] = "unknown"
            r["L_bin"] = "unknown"
            continue
        c_lo, c_hi = bin_edges[reg]["C"]
        l_lo, l_hi = bin_edges[reg]["L"]
        C, L = r.get("C"), r.get("L")
        if np.isfinite(C) and np.isfinite(L):
            r["C_bin"] = _assign_c_bin(float(C), c_lo, c_hi)
            r["L_bin"] = _assign_l_bin(float(L), l_lo, l_hi)
        else:
            r["C_bin"] = "unknown"
            r["L_bin"] = "unknown"


def _gate_v1(
    df: pd.DataFrame,
    corr_threshold: float = CORR_THRESHOLD,
    eps_var: float = EPS_VAR,
) -> Tuple[bool, List[str]]:
    """Redundancy: |corr(P,k)|<0.60, |corr(P,density)|<0.60, Var(P|r)>=eps_var."""
    reasons: List[str] = []
    for P in ["TE_res", "sigma"]:
        if P not in df.columns:
            reasons.append(f"V1: Missing column {P}")
            continue
        vals = df[P].dropna()
        if len(vals) < 3:
            reasons.append(f"V1: Insufficient data for {P}")
            continue
        corr_k = df[[P, "k"]].corr().loc[P, "k"] if "k" in df.columns else 0.0
        corr_d = df[[P, "density"]].corr().loc[P, "density"] if "density" in df.columns else 0.0
        if abs(corr_k) >= corr_threshold:
            reasons.append(f"V1: |corr({P},k)|={abs(corr_k):.3f} >= {corr_threshold}")
        if abs(corr_d) >= corr_threshold:
            reasons.append(f"V1: |corr({P},density)|={abs(corr_d):.3f} >= {corr_threshold}")
        # Within-regime variance (relaxed for near_dense: WS at high k has inherently low sigma variance)
        for reg in df["regime"].dropna().unique():
            sub = df[df["regime"] == reg][P].dropna()
            if len(sub) >= 2:
                v = float(sub.var())
                eps = EPS_VAR_NEAR_DENSE if str(reg) == "near_dense" else eps_var
                if v < eps:
                    reasons.append(f"V1: Var({P}|{reg})={v:.2e} < {eps}")
    return len(reasons) == 0, reasons


def _pareto_membership_2d(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Boolean mask of Pareto membership (max-max)."""
    n = int(xs.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=bool)
    order = np.lexsort((-ys, -xs))
    best_y = -np.inf
    on_pf = np.zeros((n,), dtype=bool)
    for idx in order:
        y = float(ys[idx])
        if y > best_y:
            on_pf[idx] = True
            best_y = y
    return on_pf


def _gate_v2(
    df: pd.DataFrame,
    min_points: int = PARETO_MIN_POINTS,
    min_regimes: int = PARETO_MIN_REGIMES,
    min_cells: int = PARETO_MIN_CELLS,
) -> Tuple[bool, List[str]]:
    """Pareto width: >=10 points, >=3 regimes, >=6 cells."""
    reasons: List[str] = []
    te_res = df["TE_res"].values if "TE_res" in df.columns else df["TE"].values
    sigma = df["sigma"].values
    valid = np.isfinite(te_res) & np.isfinite(sigma)
    if valid.sum() < 2:
        reasons.append("V2: Insufficient valid (TE_res, sigma) points")
        return False, reasons
    xs = np.asarray(te_res, dtype=float)
    ys = np.asarray(sigma, dtype=float)
    pf_mask = _pareto_membership_2d(xs, ys) & valid
    n_pf = int(np.sum(pf_mask))
    if n_pf < min_points:
        reasons.append(f"V2: Pareto front has {n_pf} points < {min_points}")
    regimes_on_pf = set()
    cells_on_pf = set()
    for i in np.where(pf_mask)[0]:
        r = df.iloc[i]["regime"] if "regime" in df.columns else None
        cb = df.iloc[i].get("C_bin")
        lb = df.iloc[i].get("L_bin")
        if r:
            regimes_on_pf.add(str(r))
        if cb and lb and cb != "unknown" and lb != "unknown":
            cells_on_pf.add((str(cb), str(lb)))
    if len(regimes_on_pf) < min_regimes:
        reasons.append(f"V2: Pareto spans {len(regimes_on_pf)} regimes < {min_regimes}")
    if len(cells_on_pf) < min_cells:
        reasons.append(f"V2: Pareto spans {len(cells_on_pf)} cells < {min_cells}")
    return len(reasons) == 0, reasons


def _gate_v3(
    df: pd.DataFrame,
    degree_regimes: Dict[str, List[int]],
    min_cells: int = V3_MIN_CELLS_PER_REGIME,
) -> Tuple[bool, List[str]]:
    """Cell feasibility: per regime, >=6 of 9 cells non-empty."""
    reasons: List[str] = []
    for reg in degree_regimes:
        sub = df[df["regime"] == reg]
        if sub.empty:
            reasons.append(f"V3: Regime {reg} has no candidates")
            continue
        cells = set()
        for _, r in sub.iterrows():
            cb, lb = r.get("C_bin"), r.get("L_bin")
            if cb and lb and cb != "unknown" and lb != "unknown":
                cells.add((str(cb), str(lb)))
        if len(cells) < min_cells:
            reasons.append(f"V3: Regime {reg} has {len(cells)}/9 cells < {min_cells}")
    return len(reasons) == 0, reasons


def _gate_v4(
    df: pd.DataFrame,
    degree_regimes: Dict[str, List[int]],
    n_min: int = N_MIN_CELL,
    min_cells: int = V3_MIN_CELLS_PER_REGIME,
    min_cells_near_dense: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    """Cell occupancy: per regime, >=min_cells of 9 cells have >= N_min candidates."""
    if min_cells_near_dense is None:
        min_cells_near_dense = min_cells
    reasons: List[str] = []
    for reg in degree_regimes:
        sub = df[df["regime"] == reg]
        if sub.empty:
            reasons.append(f"V4: Regime {reg} has no candidates")
            continue
        n_min_reg = N_MIN_CELL_NEAR_DENSE if str(reg) == "near_dense" else n_min
        min_cells_reg = min_cells_near_dense if str(reg) == "near_dense" else min_cells
        cell_counts: Dict[Tuple[str, str], int] = {}
        for _, r in sub.iterrows():
            cb, lb = r.get("C_bin"), r.get("L_bin")
            if cb and lb and cb != "unknown" and lb != "unknown":
                key = (str(cb), str(lb))
                cell_counts[key] = cell_counts.get(key, 0) + 1
        n_ok = sum(1 for c in cell_counts.values() if c >= n_min_reg)
        if n_ok < min_cells_reg:
            reasons.append(
                f"V4: Regime {reg} has {n_ok}/9 cells with >= {n_min_reg} candidates"
            )
    return len(reasons) == 0, reasons


def _get_git_commit(repo_root: Path) -> str:
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


def _capture_environment_info(output_dir: Path, command_line: str) -> None:
    """I1: Write environment capture for reproducibility."""
    env_dir = output_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    info: Dict[str, Any] = {
        "git_commit": _get_git_commit(_REPO_ROOT),
        "command_line": command_line,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "captured_at": datetime.now().isoformat(),
    }
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
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_count"] = int(torch.cuda.device_count())
            info["cuda_device_name"] = str(torch.cuda.get_device_name(0)) if torch.cuda.device_count() else None
    except ImportError:
        info["cuda_available"] = None
        info["cuda_note"] = "torch not imported"
    (env_dir / "environment_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def _gate_v5(
    ref_hashes: Set[str],
    pool_ab_hashes: Optional[Set[str]] = None,
    pool_tpe_hashes: Optional[Set[str]] = None,
    overlap_threshold: float = OVERLAP_THRESHOLD,
) -> Tuple[bool, List[str]]:
    """
    Overlap: ref vs A/B vs TPE must not overlap.
    V5 is evaluated later when A/B/C selections exist; here we only check ref has no duplicates.
    """
    reasons: List[str] = []
    if pool_ab_hashes is not None and ref_hashes:
        overlap = len(ref_hashes & pool_ab_hashes) / max(1, len(ref_hashes))
        if overlap > overlap_threshold:
            reasons.append(f"V5: ref vs A/B overlap {overlap:.3f} > {overlap_threshold}")
    if pool_tpe_hashes is not None and ref_hashes:
        overlap = len(ref_hashes & pool_tpe_hashes) / max(1, len(ref_hashes))
        if overlap > overlap_threshold:
            reasons.append(f"V5: ref vs TPE overlap {overlap:.3f} > {overlap_threshold}")
    return len(reasons) == 0, reasons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot 2 Proxy Viability Diagnostic (GO/NO-GO). G0-only reference."
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--M_ref", type=int, default=5000)
    parser.add_argument("--seed_ref", type=int, default=None)
    parser.add_argument(
        "--relaxed_v2",
        action="store_true",
        help="Use relaxed V2 (min 2 regimes, 5 cells on Pareto). Use when near_dense rarely appears on Pareto.",
    )
    parser.add_argument(
        "--relaxed_v4",
        action="store_true",
        help="Use relaxed V4 for near_dense: allow 5/9 cells with >= 20 candidates (default 6/9). Use when near_dense is hard to fill.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _capture_environment_info(output_dir, " ".join(sys.argv))

    H = int(args.H)
    M_ref = int(args.M_ref)
    seed_ref = args.seed_ref
    if seed_ref is None:
        seed_ref = int(hashlib.sha256(b"plot2_proxy_viability_seed_ref").hexdigest()[:8], 16)
    rng = np.random.default_rng(seed_ref)

    degree_regimes = DEGREE_REGIMES_DEFAULT

    print(f"[PROXY_VIABILITY] Sampling M_ref={M_ref} G0 reference graphs (H={H})...")
    rows, ref_hashes = _sample_g0_reference(H, degree_regimes, M_ref, rng)
    df = pd.DataFrame(rows)

    # Compute μ_ORC(k), μ_TE(k)
    mu_orc_by_k = _compute_mu_by_k(rows, "ORC")
    mu_te_by_k = _compute_mu_by_k(rows, "TE")

    # Add TE_res, ORC_res
    for r in rows:
        k = int(r["k"])
        r["TE_res"] = compute_te_residual(r["TE"], k, mu_te_by_k)
        r["ORC_res"] = compute_orc_residual(r["ORC"], k, mu_orc_by_k)
    df = pd.DataFrame(rows)

    # Frozen bin edges (G0 only)
    bin_edges = _compute_frozen_bin_edges(rows, degree_regimes)
    _assign_bins(rows, bin_edges, degree_regimes)
    df = pd.DataFrame(rows)

    # Gates
    v1_ok, v1_reasons = _gate_v1(df)
    v2_min_regimes = 2 if getattr(args, "relaxed_v2", False) else PARETO_MIN_REGIMES
    v2_min_cells = 5 if getattr(args, "relaxed_v2", False) else PARETO_MIN_CELLS
    v2_ok, v2_reasons = _gate_v2(df, min_regimes=v2_min_regimes, min_cells=v2_min_cells)
    v3_ok, v3_reasons = _gate_v3(df, degree_regimes)
    v4_min_cells_near_dense = 5 if getattr(args, "relaxed_v4", False) else None
    v4_ok, v4_reasons = _gate_v4(
        df, degree_regimes, min_cells_near_dense=v4_min_cells_near_dense
    )
    v5_ok, v5_reasons = _gate_v5(ref_hashes)  # No A/B/TPE pools yet

    all_pass = v1_ok and v2_ok and v3_ok and v4_ok and v5_ok

    report = {
        "schema_version": 1,
        "bin_edge_source": "G0_neutral_reference",
        "H": H,
        "M_ref": M_ref,
        "seed_ref": seed_ref,
        "relaxed_v2": bool(getattr(args, "relaxed_v2", False)),
        "relaxed_v4": bool(getattr(args, "relaxed_v4", False)),
        "gates": {
            "V1_redundancy": {"pass": v1_ok, "reasons": v1_reasons},
            "V2_pareto_width": {"pass": v2_ok, "reasons": v2_reasons},
            "V3_cell_feasibility": {"pass": v3_ok, "reasons": v3_reasons},
            "V4_cell_occupancy": {"pass": v4_ok, "reasons": v4_reasons},
            "V5_overlap": {"pass": v5_ok, "reasons": v5_reasons},
        },
        "all_gates_pass": all_pass,
    }

    # Outputs
    df.to_csv(output_dir / "reference_metrics.csv", index=False)
    (output_dir / "proxy_viability_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    # Normalization bounds for TE_res and sigma (q 0.05, 0.95)
    te_res_arr = df["TE_res"].dropna().values
    sigma_arr = df["sigma"].dropna().values
    te_res_lo = float(np.nanquantile(te_res_arr, 0.05)) if len(te_res_arr) > 0 else 0.0
    te_res_hi = float(np.nanquantile(te_res_arr, 0.95)) if len(te_res_arr) > 0 else 1.0
    sigma_lo = float(np.nanquantile(sigma_arr, 0.05)) if len(sigma_arr) > 0 else 0.0
    sigma_hi = float(np.nanquantile(sigma_arr, 0.95)) if len(sigma_arr) > 0 else 1.0
    if te_res_hi <= te_res_lo:
        te_res_hi = te_res_lo + 1e-9
    if sigma_hi <= sigma_lo:
        sigma_hi = sigma_lo + 1e-9

    frozen = {
        "schema_version": 1,
        "bin_edge_source": "G0_neutral_reference",
        "C_edges": {r: {"lo": bin_edges[r]["C"][0], "hi": bin_edges[r]["C"][1]} for r in bin_edges},
        "L_edges": {r: {"lo": bin_edges[r]["L"][0], "hi": bin_edges[r]["L"][1]} for r in bin_edges},
        "frozen_bin_edges": bin_edges,
        "te_res_lo": te_res_lo,
        "te_res_hi": te_res_hi,
        "sigma_lo": sigma_lo,
        "sigma_hi": sigma_hi,
    }
    (output_dir / "frozen_bin_edges.json").write_text(
        json.dumps(frozen, indent=2), encoding="utf-8"
    )
    (output_dir / "mu_orc_by_k.json").write_text(
        json.dumps({str(k): v for k, v in mu_orc_by_k.items()}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "mu_te_by_k.json").write_text(
        json.dumps({str(k): v for k, v in mu_te_by_k.items()}, indent=2),
        encoding="utf-8",
    )

    print(f"[PROXY_VIABILITY] Gates: V1={v1_ok} V2={v2_ok} V3={v3_ok} V4={v4_ok} V5={v5_ok}")
    if all_pass:
        print("[PROXY_VIABILITY] All gates PASS. Proceed to perturbation sensitivity and intermediate diagnostic.")
    else:
        print("[PROXY_VIABILITY] FAIL: Do NOT proceed to training. Fix bounds/generator and rerun.")
        for g, (ok, reasons) in [
            ("V1", (v1_ok, v1_reasons)),
            ("V2", (v2_ok, v2_reasons)),
            ("V3", (v3_ok, v3_reasons)),
            ("V4", (v4_ok, v4_reasons)),
            ("V5", (v5_ok, v5_reasons)),
        ]:
            if not ok and reasons:
                print(f"  {g}: {'; '.join(reasons)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
