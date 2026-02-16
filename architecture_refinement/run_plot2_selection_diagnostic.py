"""
Phase 3: Selection policy diagnostics (Plot_2_Investigation.txt).

Given a fixed pool P of accepted graphs with metrics, compare selectors A/B/C/D and report
selected-set diversity and collapse score (max_regime_fraction). Gate: Selector C must
yield meaningful multi-regime coverage. Selector D: coverage-aware (C,L) tertile bins
per regime, within-bin z-scored TE+ORC, collapse <= 50% (plot2_revision).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architecture_refinement.pareto_hv import pareto_front_2d


def _pareto_membership_2d(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Max-max Pareto membership."""
    n = int(xs.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=bool)
    order = np.lexsort((-ys, -xs))
    best_y = -np.inf
    on_pf = np.zeros((n,), dtype=bool)
    for idx in order:
        if ys[idx] > best_y:
            on_pf[idx] = True
            best_y = float(ys[idx])
    return on_pf


def _crowding_distance(x: np.ndarray, y: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """Compute crowding distance for indices (in x,y space); higher = more diverse."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx_arr = np.asarray(indices, dtype=int)
    xs = x[idx_arr]
    ys = y[idx_arr]
    n = len(idx_arr)
    if n <= 2:
        return np.ones(n, dtype=float) * (1.0 if n > 0 else 0.0)
    dx = np.max(xs) - np.min(xs)
    dy = np.max(ys) - np.min(ys)
    dx = dx if dx > 0 else 1.0
    dy = dy if dy > 0 else 1.0
    order_x = np.argsort(xs)
    order_y = np.argsort(ys)
    dist = np.zeros(n, dtype=float)
    for i, o in enumerate(order_x):
        pos = np.where(idx_arr == idx_arr[o])[0][0]
        if i == 0 or i == n - 1:
            dist[pos] += 1.0
        else:
            dist[pos] += (xs[order_x[i + 1]] - xs[order_x[i - 1]]) / dx
    for i, o in enumerate(order_y):
        pos = np.where(idx_arr == idx_arr[o])[0][0]
        if i == 0 or i == n - 1:
            dist[pos] += 1.0
        else:
            dist[pos] += (ys[order_y[i + 1]] - ys[order_y[i - 1]]) / dy
    return dist


# Selector A: global top-B by scalar score (TE_norm + ORC_norm)
def select_a(pool: List[Dict], score: np.ndarray, B: int) -> List[int]:
    order = np.argsort(-score)[:B]
    return [int(i) for i in order]


# Selector B: Pareto front then crowding distance
def select_b(pool: List[Dict], te_norm: np.ndarray, orc_norm: np.ndarray, B: int) -> List[int]:
    pf = _pareto_membership_2d(te_norm, orc_norm)
    pareto_idx = np.where(pf)[0].tolist()
    if len(pareto_idx) <= B:
        return pareto_idx[:B]
    dist = _crowding_distance(te_norm, orc_norm, pareto_idx)
    # Sort by distance desc, take top B
    order = np.argsort(-dist)[:B]
    return [int(pareto_idx[i]) for i in order]


# Selector C: stratified by regime; floor(B/R) per regime, within regime by Pareto then crowding
def select_c(
    pool: List[Dict],
    te_norm: np.ndarray,
    orc_norm: np.ndarray,
    regime_per_idx: List[Optional[str]],
    regime_names: List[str],
    B: int,
) -> List[int]:
    R = len(regime_names)
    base = B // R
    rem = B % R
    quotas = {regime_names[i]: base + (1 if i < rem else 0) for i in range(R)}
    by_regime: Dict[str, List[int]] = {r: [] for r in regime_names}
    for i, reg in enumerate(regime_per_idx):
        if reg in by_regime:
            by_regime[reg].append(i)
    selected: List[int] = []
    for reg in regime_names:
        need = quotas[reg]
        inds = by_regime.get(reg, [])
        if need <= 0 or not inds:
            continue
        sub_te = te_norm[inds]
        sub_orc = orc_norm[inds]
        sub_pf = _pareto_membership_2d(sub_te, sub_orc)
        pareto_inds = [inds[j] for j in range(len(inds)) if sub_pf[j]]
        take: List[int] = []
        if pareto_inds:
            if len(pareto_inds) <= need:
                take.extend(pareto_inds)
            else:
                dist = _crowding_distance(te_norm, orc_norm, pareto_inds)
                order = np.argsort(-dist)[:need]
                take.extend([pareto_inds[j] for j in order])
        if len(take) < need:
            rest = [i for i in inds if i not in take]
            take.extend(rest[: need - len(take)])
        selected.extend(take)
    score = te_norm + orc_norm
    already = set(selected)
    if len(selected) < B:
        for i in np.argsort(-score):
            if len(already) >= B:
                break
            if i not in already:
                selected.append(int(i))
                already.add(int(i))
    return selected[:B]


# Selector D: coverage-aware (plot2_revision): (C,L) tertile bins per regime, within-bin z-scored TE+ORC,
# uniform bin allocation, collapse <= 50%
def _compute_cl_bins_pool(
    n: int,
    regime_per_idx: List[Optional[str]],
    c_vals: np.ndarray,
    l_vals: np.ndarray,
    regime_names: List[str],
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Tertile (low/medium/high) bins for C and L per regime; return c_bin_per_idx, l_bin_per_idx, diagnostics."""
    c_bin_per_idx: List[str] = ["unknown"] * n
    l_bin_per_idx: List[str] = ["unknown"] * n
    valid_c = np.isfinite(c_vals)
    valid_l = np.isfinite(l_vals)
    by_regime: Dict[str, List[Tuple[float, float]]] = {r: [] for r in regime_names}
    for i in range(n):
        r = regime_per_idx[i] if regime_per_idx[i] in regime_names else None
        if r is None:
            continue
        if valid_c[i] and valid_l[i]:
            by_regime[r].append((float(c_vals[i]), float(l_vals[i])))
    tertile_edges: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for r in regime_names:
        pts = by_regime.get(r, [])
        if not pts:
            tertile_edges[r] = {"C": (float("nan"), float("nan")), "L": (float("nan"), float("nan"))}
            continue
        Cs = [p[0] for p in pts]
        Ls = [p[1] for p in pts]
        c_lo = float(np.nanpercentile(Cs, 33.33)) if len(Cs) >= 3 else float(np.nanmin(Cs))
        c_hi = float(np.nanpercentile(Cs, 66.67)) if len(Cs) >= 3 else float(np.nanmax(Cs))
        if c_hi <= c_lo:
            c_hi = c_lo + 1e-9
        l_lo = float(np.nanpercentile(Ls, 33.33)) if len(Ls) >= 3 else float(np.nanmin(Ls))
        l_hi = float(np.nanpercentile(Ls, 66.67)) if len(Ls) >= 3 else float(np.nanmax(Ls))
        if l_hi <= l_lo:
            l_hi = l_lo + 1e-9
        tertile_edges[r] = {"C": (c_lo, c_hi), "L": (l_lo, l_hi)}
    for i in range(n):
        r = regime_per_idx[i] if regime_per_idx[i] in regime_names else None
        if r is None or not (valid_c[i] and valid_l[i]):
            continue
        c_lo, c_hi = tertile_edges[r]["C"]
        l_lo, l_hi = tertile_edges[r]["L"]
        if np.isnan(c_lo):
            continue
        C, L = float(c_vals[i]), float(l_vals[i])
        c_bin_per_idx[i] = "low" if C <= c_lo else ("medium" if C <= c_hi else "high")
        l_bin_per_idx[i] = "low" if L <= l_lo else ("medium" if L <= l_hi else "high")
    occupied = set()
    for i in range(n):
        r, cb, lb = regime_per_idx[i], c_bin_per_idx[i], l_bin_per_idx[i]
        if r and cb != "unknown" and lb != "unknown":
            occupied.add((r, cb, lb))
    diagnostics = {"tertile_edges_per_regime": tertile_edges, "n_occupied_cells": len(occupied)}
    return c_bin_per_idx, l_bin_per_idx, diagnostics


def _zscore_within_bins_pool(
    te: np.ndarray,
    orc: np.ndarray,
    regime_per_idx: List[Optional[str]],
    c_bin_per_idx: List[str],
    l_bin_per_idx: List[str],
) -> np.ndarray:
    """Z-score TE and ORC within each (regime, c_bin, l_bin); return score = z_te + z_orc."""
    n = te.size
    score = np.zeros(n, dtype=float)
    eps = 1e-12
    bin_to_indices: Dict[Tuple[Optional[str], str, str], List[int]] = defaultdict(list)
    for i in range(n):
        bin_to_indices[(regime_per_idx[i], c_bin_per_idx[i], l_bin_per_idx[i])].append(i)
    for indices in bin_to_indices.values():
        if not indices:
            continue
        idx_arr = np.asarray(indices, dtype=int)
        te_b = te[idx_arr]
        orc_b = orc[idx_arr]
        te_std = max(float(np.nanstd(te_b)), eps)
        orc_std = max(float(np.nanstd(orc_b)), eps)
        z_te = (te_b - float(np.nanmean(te_b))) / te_std
        z_orc = (orc_b - float(np.nanmean(orc_b))) / orc_std
        for j, pos in enumerate(idx_arr):
            score[pos] = float(z_te[j] + z_orc[j])
    return score


def select_d(
    pool: List[Dict],
    te_raw: np.ndarray,
    orc_raw: np.ndarray,
    regime_per_idx: List[Optional[str]],
    regime_names: List[str],
    B: int,
    clustering_col: str = "clustering",
    path_length_col: str = "path_length",
    rank_by_proxy: bool = True,
    uniform_seed: Optional[int] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Coverage-aware selection: (C,L) tertile bins per regime; within-bin ranking by
    z-scored TE+ORC if rank_by_proxy=True, else uniform random (Baseline A). Uniform
    bin allocation, collapse <= 50%. Returns (selected_indices, diagnostics).
    """
    n = len(pool)
    if clustering_col not in pool[0] or path_length_col not in pool[0]:
        raise ValueError(f"Pool must have columns {clustering_col!r} and {path_length_col!r} for Selector D")
    c_vals = np.array([float(p.get(clustering_col, float("nan"))) for p in pool], dtype=float)
    l_vals = np.array([float(p.get(path_length_col, float("nan"))) for p in pool], dtype=float)
    c_bin_per_idx, l_bin_per_idx, bin_diag = _compute_cl_bins_pool(
        n, regime_per_idx, c_vals, l_vals, regime_names
    )
    within_bin_score = _zscore_within_bins_pool(
        te_raw, orc_raw, regime_per_idx, c_bin_per_idx, l_bin_per_idx
    )
    valid = [
        i
        for i in range(n)
        if regime_per_idx[i] in regime_names
        and c_bin_per_idx[i] != "unknown"
        and l_bin_per_idx[i] != "unknown"
    ]
    if len(valid) < B:
        B_actual = len(valid)
    else:
        B_actual = B
    n_reg = len(regime_names)
    max_per_cell = max(1, int(B_actual * 0.50))
    base_r = B_actual // n_reg
    rem_r = B_actual % n_reg
    B_per_regime = {regime_names[i]: base_r + (1 if i < rem_r else 0) for i in range(n_reg)}
    cell_to_indices: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for i in valid:
        r = regime_per_idx[i]
        if r:
            cell_to_indices[(r, c_bin_per_idx[i], l_bin_per_idx[i])].append(i)
    if rank_by_proxy:
        for key in cell_to_indices:
            cell_to_indices[key].sort(key=lambda i: (-float(within_bin_score[i]), i))
    else:
        seed_val = int(uniform_seed) if uniform_seed is not None else 202602
        for idx, (key, indices) in enumerate(cell_to_indices.items()):
            rng = np.random.default_rng(seed_val + idx)
            order = np.arange(len(indices), dtype=int)
            rng.shuffle(order)
            cell_to_indices[key] = [indices[int(j)] for j in order]
    cell_quotas: Dict[Tuple[str, str, str], int] = {}
    for r in regime_names:
        B_r = B_per_regime[r]
        cells_in_r = [c for c in cell_to_indices if c[0] == r]
        if not cells_in_r:
            continue
        base_q = B_r // len(cells_in_r)
        rem_q = B_r % len(cells_in_r)
        for j, cell in enumerate(cells_in_r):
            q = base_q + (1 if j < rem_q else 0)
            cell_quotas[cell] = min(max_per_cell, max(0, q))
    selected_idxs: List[int] = []
    for cell, indices in cell_to_indices.items():
        quota = cell_quotas.get(cell, 0)
        selected_idxs.extend(indices[:quota])
    selected_set = set(selected_idxs)
    if len(selected_idxs) < B_actual:
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
            if len(selected_set) >= B_actual:
                break
            selected_set.add(i)
            selected_idxs.append(i)
    selected_idxs = selected_idxs[:B_actual]
    cell_counts: Dict[str, int] = defaultdict(int)
    for i in selected_idxs:
        r = regime_per_idx[i]
        if r:
            cell_counts[str((r, c_bin_per_idx[i], l_bin_per_idx[i]))] += 1
    total_sel = len(selected_idxs) or 1
    collapse_score = float(max(cell_counts.values(), default=0) / total_sel)
    n_bins_total = 9 * n_reg
    n_occupied_selected = len([k for k, v in cell_counts.items() if v > 0])
    coverage_score = float(n_occupied_selected) / float(n_bins_total) if n_bins_total > 0 else 0.0
    diagnostics = {
        "collapse_score": collapse_score,
        "coverage_score": coverage_score,
        "selected_cell_counts": dict(cell_counts),
        "n_occupied_bins_selected": n_occupied_selected,
        "n_bins_total": n_bins_total,
        "coverage_aware_candidate_diagnostics": bin_diag,
    }
    return selected_idxs, diagnostics


def _graph_signature(row: Dict[str, Any]) -> Tuple[Any, ...]:
    """Signature for overlap: (k, p, graph_seed) if available else (k, p)."""
    k = row.get("k", row.get("K"))
    p = row.get("p", row.get("P"))
    gs = row.get("graph_seed")
    if gs is not None and str(gs) != "nan":
        return (k, p, gs)
    return (k, p)


def _graph_id_hash(row: Dict[str, Any]) -> str:
    """Stable portable hash of graph identity (do not use Python hash())."""
    sig = _graph_signature(row)
    key = "_".join(str(x) for x in sig)
    return hashlib.sha256(key.encode()).hexdigest()


def _overlap_fraction(sel1: List[Dict], sel2: List[Dict]) -> float:
    """Fraction of sel1 that appears in sel2 (by graph signature)."""
    if not sel1:
        return 0.0
    sig2 = {_graph_signature(r) for r in sel2}
    return sum(1 for r in sel1 if _graph_signature(r) in sig2) / len(sel1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 2 Phase 3: selection policy diagnostic (Stage 1 when --pool_tpe set)")
    parser.add_argument("--pool_csv", type=str, required=True, help="Random pool CSV (or single pool for legacy). Columns: k, p, regime, TE_raw/TE, ORC_raw/ORC, clustering, path_length")
    parser.add_argument("--pool_tpe", type=str, default=None, help="TPE pool CSV for Stage 1 (baseline C). If set, run three-baseline validation + overlap + gate.")
    parser.add_argument("--B", type=int, default=12, help="Number to select")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--regime_bins", type=str, default=None, help="JSON regime bins if regime not in CSV")
    parser.add_argument("--exit_no_go", action="store_true", help="Exit with code 1 if Stage 1 gate is NO-GO.")
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.pool_csv)
    if "TE_raw" not in df.columns and "TE" in df.columns:
        df["TE_raw"] = df["TE"]
    if "ORC_raw" not in df.columns and "ORC" in df.columns:
        df["ORC_raw"] = df["ORC"]
    for col in ["k", "TE_raw", "ORC_raw"]:
        if col not in df.columns:
            raise ValueError(f"Pool CSV must have column {col} (or TE/ORC aliased to TE_raw/ORC_raw)")
    df = df.dropna(subset=["TE_raw", "ORC_raw"])
    n = len(df)
    if n < args.B:
        print(f"Pool size {n} < B {args.B}; reducing B to {n}")
        B = n
    else:
        B = args.B

    pool = df.to_dict(orient="records")
    te_raw = df["TE_raw"].to_numpy(dtype=float)
    orc_raw = df["ORC_raw"].to_numpy(dtype=float)
    te_lo, te_hi = float(np.nanmin(te_raw)), float(np.nanmax(te_raw))
    orc_lo, orc_hi = float(np.nanmin(orc_raw)), float(np.nanmax(orc_raw))
    if te_hi <= te_lo:
        te_hi = te_lo + 1.0
    if orc_hi <= orc_lo:
        orc_hi = orc_lo + 1.0
    te_norm = (te_raw - te_lo) / (te_hi - te_lo)
    orc_norm = (orc_raw - orc_lo) / (orc_hi - orc_lo)
    score = te_norm + orc_norm

    if "regime" in df.columns:
        regime_per_idx = df["regime"].astype(str).tolist()
        regime_names = sorted(df["regime"].dropna().unique().astype(str).tolist())
    else:
        regime_per_idx = ["unknown"] * n
        regime_names = ["unknown"]

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.pool_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1 (plot2_revision): three baselines A/B/C, overlap check, GO/NO-GO
    if args.pool_tpe:
        df_tpe = pd.read_csv(args.pool_tpe)
        if "TE_raw" not in df_tpe.columns and "TE" in df_tpe.columns:
            df_tpe["TE_raw"] = df_tpe["TE"]
        if "ORC_raw" not in df_tpe.columns and "ORC" in df_tpe.columns:
            df_tpe["ORC_raw"] = df_tpe["ORC"]
        df_tpe = df_tpe.dropna(subset=["TE_raw", "ORC_raw"])
        pool_tpe = df_tpe.to_dict(orient="records")
        te_raw_tpe = df_tpe["TE_raw"].to_numpy(dtype=float)
        orc_raw_tpe = df_tpe["ORC_raw"].to_numpy(dtype=float)
        if "regime" in df_tpe.columns:
            regime_tpe = df_tpe["regime"].astype(str).tolist()
            regime_names_tpe = sorted(df_tpe["regime"].dropna().unique().astype(str).tolist())
        else:
            regime_tpe = ["unknown"] * len(pool_tpe)
            regime_names_tpe = ["unknown"]
        B_s1 = min(B, len(pool), len(pool_tpe))
        sel_a_idx, diag_a = select_d(
            pool, te_raw, orc_raw, regime_per_idx, regime_names, B_s1,
            rank_by_proxy=False, uniform_seed=202602,
        )
        sel_b_idx, diag_b = select_d(
            pool, te_raw, orc_raw, regime_per_idx, regime_names, B_s1,
            rank_by_proxy=True,
        )
        sel_c_idx, diag_c = select_d(
            pool_tpe, te_raw_tpe, orc_raw_tpe, regime_tpe, regime_names_tpe, B_s1,
            rank_by_proxy=True,
        )
        sel_a_records = [pool[i] for i in sel_a_idx]
        sel_b_records = [pool[i] for i in sel_b_idx]
        sel_c_records = [pool_tpe[i] for i in sel_c_idx]
        overlap_ab = _overlap_fraction(sel_a_records, sel_b_records)
        overlap_ac = _overlap_fraction(sel_a_records, sel_c_records)
        overlap_bc = _overlap_fraction(sel_b_records, sel_c_records)
        # Plot2_revision2: overlap_ok uses only A-C and B-C < 50%; A-B may be high (same pool).
        overlap_ok = overlap_ac < 0.50 and overlap_bc < 0.50
        collapse_ok = (
            (diag_a["collapse_score"] or 1.0) <= 0.50
            and (diag_b["collapse_score"] or 1.0) <= 0.50
            and (diag_c["collapse_score"] or 1.0) <= 0.50
        )
        # Coverage: when B < n_bins_total, max achievable = B/n_bins_total (one graph per bin)
        n_bins_total = int(diag_a.get("n_bins_total", 36))
        coverage_threshold = min(0.5, B_s1 / n_bins_total) if n_bins_total > 0 else 0.5
        coverage_ok = (
            (diag_a["coverage_score"] or 0.0) >= coverage_threshold
            and (diag_b["coverage_score"] or 0.0) >= coverage_threshold
            and (diag_c["coverage_score"] or 0.0) >= coverage_threshold
        )
        stage1_go = collapse_ok and coverage_ok and overlap_ok
        stage1_gate = {
            "pass": stage1_go,
            "reason": "GO" if stage1_go else "NO-GO",
            "collapse_ok": collapse_ok,
            "coverage_ok": coverage_ok,
            "coverage_threshold_used": float(coverage_threshold),
            "overlap_ok": overlap_ok,
            "overlap_A_B": float(overlap_ab),
            "overlap_A_C": float(overlap_ac),
            "overlap_B_C": float(overlap_bc),
            "baseline_a": {"collapse_score": diag_a["collapse_score"], "coverage_score": diag_a["coverage_score"]},
            "baseline_b": {"collapse_score": diag_b["collapse_score"], "coverage_score": diag_b["coverage_score"]},
            "baseline_c": {"collapse_score": diag_c["collapse_score"], "coverage_score": diag_c["coverage_score"]},
        }
        stage1_report = {
            "schema_version": 1,
            "stage": "Stage 1",
            "B": B_s1,
            "pool_random_size": len(pool),
            "pool_tpe_size": len(pool_tpe),
            "stage1_gate": stage1_gate,
        }
        out_stage1 = out_dir / "phase3_stage1_report.json"
        with open(out_stage1, "w", encoding="utf-8") as f:
            json.dump(stage1_report, f, indent=2)
        print(f"Stage 1 report written to {out_stage1}")

        # Plot2_revision2: emit selected_A/B/C.jsonl with stable graph IDs and graph_id hash
        def _write_manifest_jsonl(records: List[Dict], path: Path) -> None:
            with open(path, "w", encoding="utf-8") as f:
                for r in records:
                    out = {
                        "k": r.get("k", r.get("K")),
                        "p": r.get("p", r.get("P")),
                        "graph_seed": r.get("graph_seed"),
                        "regime": r.get("regime"),
                        "TE": r.get("TE_raw", r.get("TE")),
                        "ORC": r.get("ORC_raw", r.get("ORC")),
                        "clustering": r.get("clustering"),
                        "path_length": r.get("path_length"),
                        "density": r.get("density"),
                    }
                    out["graph_id"] = _graph_id_hash(r)
                    f.write(json.dumps(out) + "\n")

        _write_manifest_jsonl(sel_a_records, out_dir / "selected_A.jsonl")
        _write_manifest_jsonl(sel_b_records, out_dir / "selected_B.jsonl")
        _write_manifest_jsonl(sel_c_records, out_dir / "selected_C.jsonl")
        print(f"Stage 1 manifests written: selected_A.jsonl, selected_B.jsonl, selected_C.jsonl")

        print(f"Stage 1 gate: {stage1_gate['reason']} (pass={stage1_go})")
        if not stage1_go:
            parts = []
            if not collapse_ok:
                parts.append("collapse > 0.50")
            if not coverage_ok:
                parts.append(f"coverage < threshold ({coverage_threshold:.2f})")
            if not overlap_ok:
                parts.append("A-C or B-C overlap >= 50% (A-C={:.0%} B-C={:.0%})".format(overlap_ac, overlap_bc))
            print("  NO-GO: " + "; ".join(parts))
        if not stage1_go and args.exit_no_go:
            sys.exit(1)
        return

    def summarize(selected_idxs: List[int], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        sel = [pool[i] for i in selected_idxs]
        ks = [s["k"] for s in sel if "k" in s]
        regimes = [regime_per_idx[i] for i in selected_idxs]
        regime_counts = {r: regimes.count(r) for r in regime_names}
        total = len(regimes)
        max_frac = max(regime_counts.values()) / total if total else 0.0
        out = {
            "unique_k": sorted(set(int(k) for k in ks)),
            "regime_counts": regime_counts,
            "collapse_score": float(max_frac),
            "n_selected": len(selected_idxs),
            "density_range": [min(s.get("density", float("nan")) for s in sel if "density" in s and s.get("density") is not None) or float("nan"),
                             max(s.get("density", float("nan")) for s in sel if "density" in s and s.get("density") is not None) or float("nan")],
            "TE_range": [min(s.get("TE_raw", float("nan")) for s in sel), max(s.get("TE_raw", float("nan")) for s in sel)],
            "ORC_range": [min(s.get("ORC_raw", float("nan")) for s in sel), max(s.get("ORC_raw", float("nan")) for s in sel)],
        }
        if extra:
            out.update(extra)
        return out

    sel_a = select_a(pool, score, B)
    sel_b = select_b(pool, te_norm, orc_norm, B)
    sel_c = select_c(pool, te_norm, orc_norm, regime_per_idx, regime_names, B)

    report = {
        "schema_version": 1,
        "B": B,
        "pool_size": n,
        "selector_a_global_top_B": summarize(sel_a),
        "selector_b_pareto_crowding": summarize(sel_b),
        "selector_c_stratified_regime": summarize(sel_c),
        "gate": {
            "selector_c_collapse_score": summarize(sel_c)["collapse_score"],
            "pass": summarize(sel_c)["collapse_score"] <= 0.5,
            "reason": "Selector C yields multi-regime coverage (collapse <= 0.5)" if summarize(sel_c)["collapse_score"] <= 0.5 else "Selector C collapse > 0.5",
        },
    }

    # Selector D: coverage-aware (requires clustering, path_length in CSV)
    has_cl_pl = "clustering" in df.columns and "path_length" in df.columns
    if has_cl_pl:
        sel_d, d_diag = select_d(
            pool, te_raw, orc_raw, regime_per_idx, regime_names, B,
            clustering_col="clustering", path_length_col="path_length",
        )
        report["selector_d_coverage_aware"] = summarize(
            sel_d,
            extra={
                "collapse_score": d_diag["collapse_score"],
                "coverage_score": d_diag["coverage_score"],
                "selected_cell_counts": d_diag["selected_cell_counts"],
                "n_occupied_bins_selected": d_diag["n_occupied_bins_selected"],
                "n_bins_total": d_diag["n_bins_total"],
            },
        )
        report["selector_d_diagnostics"] = {
            "collapse_score": d_diag["collapse_score"],
            "coverage_score": d_diag["coverage_score"],
        }
    else:
        report["selector_d_coverage_aware"] = None
        report["selector_d_note"] = "Skipped (pool CSV must have clustering and path_length columns)"

    out_json = out_dir / "phase3_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Phase 3 report written to {out_json}")
    for name, sel in [("A (global top-B)", sel_a), ("B (Pareto+crowding)", sel_b), ("C (stratified)", sel_c)]:
        s = summarize(sel)
        print(f"  {name}: collapse_score={s['collapse_score']:.3f}, regime_counts={s['regime_counts']}")
    if has_cl_pl:
        s_d = report["selector_d_coverage_aware"]
        print(f"  D (coverage-aware): collapse_score={s_d['collapse_score']:.3f}, coverage_score={s_d.get('coverage_score', float('nan')):.3f}, regime_counts={s_d['regime_counts']}")
    print(f"Gate PASS (Selector C): {report['gate']['pass']}")


if __name__ == "__main__":
    main()
