"""
Plot 2 smoke test utilities (experiment_three spec).

Helpers for mask statistics and robustness metrics used by run_plot2_smoke_test.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def summarize_wiring_mask(arch_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute recurrent mask statistics from a Plot 2 architecture dict (read-only).

    Uses hidden_adj_directed or hidden_adj_undirected and H from the arch JSON.
    No model instantiation required.

    Returns
    -------
    dict with:
        n_active_edges : int
        n_possible_edges : int  (H*H for the hidden block)
        mask_density : float   (active / possible)
        per_layer : optional list of dicts (WS-Flex: single hidden block)
    """
    H = int(arch_dict.get("H", 0))
    if H <= 0:
        return {
            "n_active_edges": 0,
            "n_possible_edges": 0,
            "mask_density": float("nan"),
        }

    adj = arch_dict.get("hidden_adj_directed") or arch_dict.get("hidden_adj_undirected")
    if adj is None:
        return {
            "n_active_edges": 0,
            "n_possible_edges": int(H * H),
            "mask_density": float("nan"),
        }

    A = np.asarray(adj, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != H or A.shape[1] != H:
        n_possible = int(H * H)
        return {
            "n_active_edges": 0,
            "n_possible_edges": n_possible,
            "mask_density": float("nan"),
        }

    n_active = int(np.count_nonzero(A != 0))
    n_possible = int(H * H)
    density = float(n_active / n_possible) if n_possible > 0 else float("nan")

    out: Dict[str, Any] = {
        "n_active_edges": n_active,
        "n_possible_edges": n_possible,
        "mask_density": density,
    }
    # Optional per-layer: WS-Flex has one hidden block
    out["per_layer"] = [
        {
            "name": "hidden",
            "n_active_edges": n_active,
            "n_possible_edges": n_possible,
            "mask_density": density,
        }
    ]
    return out


def compute_smoke_robustness_metrics(
    per_graph_df: pd.DataFrame,
    model_names: List[str],
    primary_noise_type: str = "ar1_drift",
) -> Dict[str, Any]:
    """
    Build robustness metrics dict for smoke_test_report.json from per_graph table.

    Expects per_graph_df to have columns from analyze_plot2_results (per_graph_aupc.csv):
    model_name, noise_type, aupc_alpha_mean, clean_roc_auc_mean, and optionally
    max_drop_mean, mid_drop_mean.

    Returns
    -------
    dict with:
        aupc_by_model, clean_roc_auc_by_model, max_drop_by_model, mid_drop_by_model,
        max_pairwise_delta_AUPC, max_pairwise_delta_max_drop, max_pairwise_delta_mid_drop
    """
    if "noise_type" in per_graph_df.columns:
        df = per_graph_df[
            per_graph_df["noise_type"].astype(str) == primary_noise_type
        ].copy()
    else:
        df = per_graph_df.copy()

    aupc_by_model: Dict[str, float] = {}
    clean_roc_auc_by_model: Dict[str, float] = {}
    max_drop_by_model: Dict[str, float] = {}
    mid_drop_by_model: Dict[str, float] = {}

    for mn in model_names:
        row = df[df["model_name"].astype(str) == mn]
        if row.empty:
            aupc_by_model[mn] = float("nan")
            clean_roc_auc_by_model[mn] = float("nan")
            max_drop_by_model[mn] = float("nan")
            mid_drop_by_model[mn] = float("nan")
            continue
        r = row.iloc[0]
        aupc_by_model[mn] = _safe_float(r.get("aupc_alpha_mean"))
        clean_roc_auc_by_model[mn] = _safe_float(r.get("clean_roc_auc_mean"))
        max_drop_by_model[mn] = _safe_float(r.get("max_drop_mean"))
        mid_drop_by_model[mn] = _safe_float(r.get("mid_drop_mean"))

    def max_pairwise_delta(values: Dict[str, float], names: List[str]) -> float:
        vals = [values.get(m) for m in names]
        finite = [x for x in vals if x is not None and np.isfinite(x)]
        if len(finite) < 2:
            return 0.0
        return float(max(abs(a - b) for i, a in enumerate(finite) for b in finite[i + 1 :]))

    return {
        "aupc_by_model": aupc_by_model,
        "clean_roc_auc_by_model": clean_roc_auc_by_model,
        "max_drop_by_model": max_drop_by_model,
        "mid_drop_by_model": mid_drop_by_model,
        "max_pairwise_delta_AUPC": max_pairwise_delta(aupc_by_model, model_names),
        "max_pairwise_delta_max_drop": max_pairwise_delta(max_drop_by_model, model_names),
        "max_pairwise_delta_mid_drop": max_pairwise_delta(mid_drop_by_model, model_names),
    }


def _safe_float(x: Any) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")
