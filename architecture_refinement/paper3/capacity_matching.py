"""
Capacity matching for PAPER 3: ensure all models within 5% of P_target.

P_target is defined by a reference CfC model.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from typing import Optional, Tuple

from .models import (
    CfCRecurrentModel,
    NCPRecurrentModel,
    LSTMRecurrentModel,
    CNNBaselineModel,
    count_parameters,
)
from architecture_refinement.ws_flex_generator import make_ws_flex_graph


def _make_reference_cfc(
    C: int = 1,
    D_in: int = 16,
    H: int = 32,
    k: int = 4,
    p: float = 0.3,
    seed: int = 42,
) -> CfCRecurrentModel:
    """Create reference CfC with typical WS-Flex wiring."""
    G, _ = make_ws_flex_graph(H=H, k=k, p=p, seed=seed, generator_mode="plain_ws_flex")
    if not nx.is_connected(G):
        G = nx.watts_strogatz_graph(H, max(2, k), p, seed=seed)
    return CfCRecurrentModel(C=C, D_in=D_in, H=H, n_outputs=2, hidden_graph=G, wiring_seed=seed)


def compute_p_target(
    C: int = 1,
    D_in: int = 16,
    H: int = 32,
    k: int = 4,
    p: float = 0.3,
    seed: int = 42,
) -> int:
    """
    Compute P_target from reference CfC model.

    Returns:
        Number of trainable parameters in reference model.
    """
    model = _make_reference_cfc(C=C, D_in=D_in, H=H, k=k, p=p, seed=seed)
    return count_parameters(model)


def lstm_params_formula(H: int, D_in: int, C: int, n_outputs: int = 2) -> int:
    """
    Approximate LSTM params: 4 * [H*(H + D_in) + H] for recurrent + encoder + readout.

    Encoder: C * D_in + D_in
    LSTM: 4 * (D_in * H + H * H + H) = 4 * (H*(D_in + H) + H)
    Readout: H * n_outputs + n_outputs
    """
    encoder = C * D_in + D_in
    lstm_core = 4 * (D_in * H + H * H + H)
    readout = H * n_outputs + n_outputs
    return encoder + lstm_core + readout


def find_lstm_hidden_for_target(
    P_target: int,
    D_in: int = 16,
    C: int = 1,
    n_outputs: int = 2,
    tol: float = 0.05,
) -> Optional[int]:
    """
    Find H_LSTM such that |P_model - P_target| / P_target <= tol.

    Uses binary search over H.
    """
    H_lo, H_hi = 8, 128
    best_H, best_err = None, float("inf")
    for H in range(H_lo, H_hi + 1):
        P = lstm_params_formula(H, D_in, C, n_outputs)
        err = abs(P - P_target) / max(1, P_target)
        if err <= tol and err < best_err:
            best_err = err
            best_H = H
    return best_H


def find_cnn_config_for_target(
    P_target: int,
    D_in: int = 16,
    C: int = 1,
    H_ref: int = 32,
    n_outputs: int = 2,
    tol: float = 0.05,
    kernel_size: int = 7,
) -> Optional[dict]:
    """
    Find CNN config (conv_channels) such that |P - P_target| / P_target <= tol.
    Tries (D_in, mid, H) with mid, H varied.
    """
    best = None
    best_err = float("inf")
    for mid in [16, 24, 32, 40, 48]:
        for H in [H_ref - 8, H_ref - 4, H_ref, H_ref + 4, H_ref + 8]:
            if H < 8:
                continue
            try:
                model = CNNBaselineModel(
                    C=C, D_in=D_in, H=H, n_outputs=n_outputs,
                    conv_channels=(D_in, mid, H), kernel_size=kernel_size,
                )
                P = count_parameters(model)
                err = abs(P - P_target) / max(1, P_target)
                if err <= tol and err < best_err:
                    best_err = err
                    best = {"conv_channels": (D_in, mid, H), "H": H, "params": P}
            except Exception:
                continue
    return best


def find_ncp_units_for_target(
    P_target: int,
    D_in: int = 16,
    C: int = 1,
    n_outputs: int = 2,
    tol: float = 0.05,
    sparsity_level: float = 0.85,
) -> Optional[Tuple[int, int]]:
    """
    Find (ncp_units, actual_P) such that |P - P_target| / P_target <= tol.

    Returns (units, param_count) or None if no suitable config found.
    """
    encoder = C * D_in + D_in
    readout = n_outputs * n_outputs + n_outputs
    core_budget = P_target - encoder - readout
    if core_budget <= 0:
        return None

    best = None
    best_err = float("inf")
    for units in range(10, 64):
        if units - n_outputs < 2:
            continue
        try:
            model = NCPRecurrentModel(
                C=C, D_in=D_in, H=32,
                n_outputs=n_outputs,
                ncp_units=units,
                sparsity_level=sparsity_level,
            )
            P = count_parameters(model)
            err = abs(P - P_target) / max(1, P_target)
            if err <= tol and err < best_err:
                best_err = err
                best = (units, P)
        except Exception:
            continue
    return best


def get_capacity_matched_configs(
    C: int = 1,
    D_in: int = 16,
    H_ref: int = 32,
    k: int = 4,
    p: float = 0.3,
    tol: float = 0.05,
) -> dict:
    """
    Return config dicts for CfC, NCP, LSTM all within tol of P_target.

    Returns:
        {
            "P_target": int,
            "CfC": {"H": int, "k": int, "p": float, "params": int},
            "LSTM": {"H": int, "params": int},
            "NCP": {"ncp_units": int, "params": int},
        }
    """
    P_target = compute_p_target(C=C, D_in=D_in, H=H_ref, k=k, p=p)
    G, _ = make_ws_flex_graph(H=H_ref, k=k, p=p, seed=42, generator_mode="plain_ws_flex")
    if not nx.is_connected(G):
        G = nx.watts_strogatz_graph(H_ref, max(2, k), p, seed=42)

    cfc_model = CfCRecurrentModel(C=C, D_in=D_in, H=H_ref, n_outputs=2, hidden_graph=G, wiring_seed=42)
    cfc_params = count_parameters(cfc_model)

    lstm_H = find_lstm_hidden_for_target(P_target, D_in=D_in, C=C, n_outputs=2, tol=tol)
    lstm_params = lstm_params_formula(lstm_H, D_in, C, 2) if lstm_H else None

    ncp_result = find_ncp_units_for_target(P_target, D_in=D_in, C=C, n_outputs=2, tol=tol)
    ncp_units, ncp_params = ncp_result if ncp_result else (None, None)

    cnn_result = find_cnn_config_for_target(P_target, D_in=D_in, C=C, H_ref=H_ref, n_outputs=2, tol=tol)
    cnn_config = cnn_result if cnn_result else None

    return {
        "P_target": P_target,
        "CfC": {"H": H_ref, "k": k, "p": p, "params": cfc_params},
        "LSTM": {"H": lstm_H, "params": lstm_params},
        "NCP": {"ncp_units": ncp_units, "params": ncp_params},
        "CNN": cnn_config,
    }
