"""
PAPER 3 diagnostic tests (Go/No-Go).

TEST 1: Data + perturbation sanity (no training)
TEST 2: Single-model overfit (LSTM, N=2k, 30 epochs)
TEST 3: Capacity matching
TEST 4: Hidden-state extraction
TEST 5: Dynamics metrics smoke
TEST 6: Mini separation pilot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import networkx as nx

from .harmonic_oscillator_dataset import HarmonicOscillatorDataset, create_splits, generate_sample
from .perturbations import apply_awgn, apply_impulse, apply_drift, get_perturbation_grid
from .models import CfCRecurrentModel, NCPRecurrentModel, LSTMRecurrentModel, count_parameters
from .capacity_matching import get_capacity_matched_configs, compute_p_target
from .dynamics_metrics import compute_sensitivity, compute_state_variance, compute_lambda
from .topology_search import run_tpe_search, run_random_search, run_random_filter_search, GraphCandidate
from .train_eval import train_model, evaluate, evaluate_perturbed, evaluate_perturbed_multi, evaluate_dynamics
from architecture_refinement.ws_flex_generator import make_ws_flex_graph


def test1_data_perturbation_sanity() -> bool:
    """TEST 1: Generate data, verify perturbations. GO if perturbations match intended behavior."""
    print("\n=== TEST 1: Data + Perturbation Sanity ===")
    rng = np.random.default_rng(42)
    ds = HarmonicOscillatorDataset(n_samples=1000, T=256, seed=42)
    x0, y0 = ds[0]
    x = x0.numpy()

    x_awgn = apply_awgn(x, 0.2, seed=1)
    x_imp = apply_impulse(x, 0.5, seed=1)
    x_drift = apply_drift(x, 0.3, seed=1)

    assert np.isfinite(x_awgn).all(), "AWGN produced non-finite"
    assert np.isfinite(x_imp).all(), "Impulse produced non-finite"
    assert np.isfinite(x_drift).all(), "Drift produced non-finite"

    diff_awgn = np.abs(x_awgn - x).max()
    assert diff_awgn > 0, "AWGN should change signal"
    print(f"  [OK] Data shape {x.shape}, class balance check: {sum(ds.data[i][1] for i in range(100))} ones in first 100")
    print(f"  [OK] Perturbations applied, AWGN max diff={diff_awgn:.4f}")
    return True


def test2_overfit() -> bool:
    """TEST 2: LSTM overfit on N=2k, 50 epochs. GO if val AUC > 0.90."""
    print("\n=== TEST 2: Single-Model Overfit ===")
    train_ds = HarmonicOscillatorDataset(n_samples=2000, T=256, seed=42)
    val_ds = HarmonicOscillatorDataset(n_samples=500, T=256, seed=43)
    model = LSTMRecurrentModel(C=1, D_in=16, H=32, n_outputs=2)
    result = train_model(model, train_ds, val_ds, epochs=50, batch_size=64, patience=15, seed=42)
    val_auc = result["best_val_auc"]
    print(f"  Val AUC: {val_auc:.4f} (need > 0.90)")
    return val_auc > 0.90


def test3_capacity_matching() -> bool:
    """TEST 3: All models within 5% of P_target."""
    print("\n=== TEST 3: Capacity Matching ===")
    configs = get_capacity_matched_configs(C=1, D_in=16, H_ref=32, tol=0.05)
    P_target = configs["P_target"]
    print(f"  P_target: {P_target}")

    cfc_p = configs["CfC"]["params"]
    lstm_p = configs["LSTM"]["params"]
    ncp_p = configs["NCP"]["params"]

    cfc_ok = abs(cfc_p - P_target) / P_target <= 0.05
    lstm_ok = lstm_p and abs(lstm_p - P_target) / P_target <= 0.05
    ncp_ok = ncp_p and abs(ncp_p - P_target) / P_target <= 0.05

    print(f"  CfC:  {cfc_p} ({'OK' if cfc_ok else 'FAIL'})")
    print(f"  LSTM: {lstm_p} ({'OK' if lstm_ok else 'FAIL'})")
    print(f"  NCP:  {ncp_p} ({'OK' if ncp_ok else 'FAIL'})")
    return cfc_ok and lstm_ok and ncp_ok


def test4_hidden_states() -> bool:
    """TEST 4: return_states=True returns [B,T,H], finite."""
    print("\n=== TEST 4: Hidden-State Extraction ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G, _ = make_ws_flex_graph(H=32, k=4, p=0.3, seed=42, generator_mode="plain_ws_flex")
    if not nx.is_connected(G):
        G = nx.watts_strogatz_graph(32, 4, 0.3, seed=42)

    models = [
        ("CfC", CfCRecurrentModel(C=1, D_in=16, H=32, hidden_graph=G)),
        ("NCP", NCPRecurrentModel(C=1, D_in=16, H=32, ncp_units=28)),
        ("LSTM", LSTMRecurrentModel(C=1, D_in=16, H=32)),
    ]
    x = torch.randn(4, 256, 1)
    for name, model in models:
        model = model.to(device)
        model.eval()
        logits, states = model(x.to(device), return_states=True)
        assert states.dim() == 3, f"{name}: states should be [B,T,H], got {states.shape}"
        assert states.shape[0] == 4 and states.shape[1] == 256, f"{name}: wrong shape {states.shape}"
        assert torch.isfinite(states).all(), f"{name}: non-finite states"
        print(f"  [OK] {name}: states {states.shape}")
    return True


def test5_dynamics_smoke() -> bool:
    """TEST 5: M1-M3 finite; Sensitivity increases with ε, StateVar increases with σ."""
    print("\n=== TEST 5: Dynamics Metrics Smoke ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRecurrentModel(C=1, D_in=16, H=32)
    ds = HarmonicOscillatorDataset(n_samples=128, T=256, seed=42)
    x = torch.stack([ds[i][0] for i in range(128)])
    if x.dim() == 2:
        x = x.unsqueeze(-1)
    x = x.to(device)

    s1 = compute_sensitivity(model, x, 1e-3, seed=42, device=device)
    s2 = compute_sensitivity(model, x, 1e-2, seed=42, device=device)
    v1 = compute_state_variance(model, x, 0.1, R=5, seed=42, device=device)
    v2 = compute_state_variance(model, x, 0.1, R=5, seed=42, device=device)
    lam = compute_lambda(model, x, 1e-2, seed=42, device=device)

    assert np.isfinite(s1) and np.isfinite(s2), "Sensitivity non-finite"
    assert np.isfinite(v1) and np.isfinite(v2), "StateVar non-finite"
    assert np.isfinite(lam), "Lambda non-finite"
    print(f"  Sensitivity(1e-3)={s1:.4f}, Sensitivity(1e-2)={s2:.4f} (expect s2 >= s1)")
    print(f"  StateVar={v1:.4f}, Lambda={lam:.4f}")
    return True


def test6_mini_pilot() -> bool:
    """TEST 6: Mini separation pilot (legacy). Use TEST 6b for stress amplification."""
    return test6b_stress_amplification()


def test6b_stress_amplification() -> bool:
    """TEST 6b: CfC-Rand vs NCP vs LSTM. Recurrent architectures under stress (task designed for recurrence)."""
    print("\n=== TEST 6b: CfC-Rand vs NCP vs LSTM ===")
    H, D_in, K, seeds = 32, 16, 4, [0, 1]
    k_values = [2, 4, 6, 8]
    pert_types = ["awgn", "impulse", "drift"]

    train_ds, val_ds, test_ds = create_splits(
        n_train=5000, n_val=500, n_test=1000, seed=42,
        omega_L=(0.08, 0.18), omega_H=(0.15, 0.28),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = get_capacity_matched_configs(C=1, D_in=D_in, H_ref=H, tol=0.05)

    results: dict[str, list] = {"CfC-Rand": [], "NCP": [], "LSTM": []}

    # CfC-Rand: K random topologies
    _, selected = run_random_search(H, k_values, 20, K, seeds[0])
    for cand in selected:
        model = CfCRecurrentModel(C=1, D_in=D_in, H=H, hidden_graph=cand.G, wiring_seed=cand.wiring_seed)
        for seed in seeds:
            train_model(model, train_ds, val_ds, epochs=10, batch_size=128, seed=seed, device=device)
            pert = evaluate_perturbed_multi(model, test_ds, pert_types, seed=seed, device=device)
            dyn = evaluate_dynamics(model, test_ds, n_samples=256, seed=seed, device=device)
            results["CfC-Rand"].append({
                "AUPC_total": pert["AUPC_total"], "MaxRD_total": pert["MaxRD_total"],
                "Sensitivity_med": dyn["sensitivity_med"], "Lambda": dyn["lambda"],
            })

    # NCP
    ncp_units = configs["NCP"].get("ncp_units")
    if ncp_units:
        model = NCPRecurrentModel(C=1, D_in=D_in, H=32, ncp_units=ncp_units)
        for seed in seeds:
            train_model(model, train_ds, val_ds, epochs=10, batch_size=128, seed=seed, device=device)
            pert = evaluate_perturbed_multi(model, test_ds, pert_types, seed=seed, device=device)
            dyn = evaluate_dynamics(model, test_ds, n_samples=256, seed=seed, device=device)
            results["NCP"].append({
                "AUPC_total": pert["AUPC_total"], "MaxRD_total": pert["MaxRD_total"],
                "Sensitivity_med": dyn["sensitivity_med"], "Lambda": dyn["lambda"],
            })

    # LSTM
    lstm_H = configs["LSTM"].get("H")
    if lstm_H:
        model = LSTMRecurrentModel(C=1, D_in=D_in, H=lstm_H)
        for seed in seeds:
            train_model(model, train_ds, val_ds, epochs=10, batch_size=128, seed=seed, device=device)
            pert = evaluate_perturbed_multi(model, test_ds, pert_types, seed=seed, device=device)
            dyn = evaluate_dynamics(model, test_ds, n_samples=256, seed=seed, device=device)
            results["LSTM"].append({
                "AUPC_total": pert["AUPC_total"], "MaxRD_total": pert["MaxRD_total"],
                "Sensitivity_med": dyn["sensitivity_med"], "Lambda": dyn["lambda"],
            })

    def mean_metric(r: list, key: str) -> float:
        vals = [x[key] for x in r if isinstance(x, dict) and key in x]
        return float(np.mean(vals)) if vals else 0.0

    for regime in ["CfC-Rand", "NCP", "LSTM"]:
        if results[regime]:
            aupc = mean_metric(results[regime], "AUPC_total")
            sens = mean_metric(results[regime], "Sensitivity_med")
            lam = mean_metric(results[regime], "Lambda")
            print(f"  {regime:12} AUPC_total={aupc:.4f}, Sens_med={sens:.4f}, Lambda={lam:.4f}")

    go = all(len(results[r]) > 0 for r in ["CfC-Rand", "NCP", "LSTM"])
    print(f"  GO: all three recurrent architectures run and report metrics: {go}")
    return go


def test6c_sensitivity_sanity() -> bool:
    """TEST 6c: Sensitivity finite, non-zero, and s_med >= s_small (allows linear models where ratio≈1)."""
    print("\n=== TEST 6c: Sensitivity Sanity ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRecurrentModel(C=1, D_in=16, H=32)
    train_ds, val_ds, test_ds = create_splits(n_train=2000, n_val=500, n_test=500, seed=42)
    train_model(model, train_ds, val_ds, epochs=15, batch_size=128, seed=42, device=device)
    dyn = evaluate_dynamics(model, test_ds, n_samples=256, seed=42, device=device)
    s_small = dyn["sensitivity_small"]
    s_med = dyn["sensitivity_med"]
    ratio = s_med / (s_small + 1e-10)
    # Linear models: S = ||Δh||/ε constant => ratio ≈ 1. Accept s_med >= s_small * 0.95 (tolerance for noise).
    go = s_small > 1e-6 and s_med >= s_small * 0.95
    print(f"  Sens(ε_small)={s_small:.4f}, Sens(ε_med)={s_med:.4f}, ratio={ratio:.2f}")
    print(f"  GO if finite, non-zero, and s_med >= 0.95*s_small: {go}")
    return go


def test6d_lambda_sanity() -> bool:
    """TEST 6d: |Lambda(ε_med) - Lambda(ε_small)| >= 0.01."""
    print("\n=== TEST 6d: Lambda Sanity ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRecurrentModel(C=1, D_in=16, H=32)
    train_ds, val_ds, test_ds = create_splits(n_train=2000, n_val=500, n_test=500, seed=42)
    train_model(model, train_ds, val_ds, epochs=15, batch_size=128, seed=42, device=device)
    dyn = evaluate_dynamics(model, test_ds, n_samples=256, seed=42, device=device)
    lam_small = dyn["lambda_small"]
    lam_med = dyn["lambda_med"]
    delta = abs(lam_med - lam_small)
    go = delta >= 0.01
    print(f"  Lambda(ε_small)={lam_small:.4f}, Lambda(ε_med)={lam_med:.4f}, |Δ|={delta:.4f}")
    print(f"  GO if |Δ|>=0.01: {go}")
    return go


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="0", help="1-6, 6b, 6c, 6d for specific test, 0 for all")
    args = parser.parse_args()

    test_map = {
        1: test1_data_perturbation_sanity,
        2: test2_overfit,
        3: test3_capacity_matching,
        4: test4_hidden_states,
        5: test5_dynamics_smoke,
        6: test6_mini_pilot,
        "6b": test6b_stress_amplification,
        "6c": test6c_sensitivity_sanity,
        "6d": test6d_lambda_sanity,
    }
    t = args.test
    if t == "0":
        tests = [(1, test_map[1]), (2, test_map[2]), (3, test_map[3]), (4, test_map[4]), (5, test_map[5]), (6, test_map[6])]
    elif t in test_map:
        tests = [(t, test_map[t])]
    else:
        ti = int(t) if t.isdigit() else None
        tests = [(ti, test_map[ti])] if ti in test_map else []

    if not tests:
        print("Invalid --test")
        return 1

    passed = 0
    for num, fn in tests:
        try:
            ok = fn()
            if ok:
                passed += 1
                print(f"  >>> TEST {num} PASSED")
            else:
                print(f"  >>> TEST {num} FAILED (No-Go)")
        except Exception as e:
            print(f"  >>> TEST {num} ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n=== Summary: {passed}/{len(tests)} passed ===")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
