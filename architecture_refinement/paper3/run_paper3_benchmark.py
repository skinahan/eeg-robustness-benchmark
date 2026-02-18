"""
PAPER 3 Task-Agnostic Robustness Benchmark - Main Entry Point.

Usage:
  python -m architecture_refinement.paper3.run_paper3_benchmark --stage diagnostics
  python -m architecture_refinement.paper3.run_paper3_benchmark --stage mini
  python -m architecture_refinement.paper3.run_paper3_benchmark --stage full
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import networkx as nx

from .harmonic_oscillator_dataset import HarmonicOscillatorDataset, create_splits
from .perturbations import get_perturbation_grid, PerturbationType
from .models import CfCRecurrentModel, NCPRecurrentModel, LSTMRecurrentModel, count_parameters
from .capacity_matching import get_capacity_matched_configs
from .topology_search import run_random_search, GraphCandidate
from .train_eval import train_model, evaluate_perturbed, evaluate_dynamics
from .run_diagnostics import (
    test1_data_perturbation_sanity,
    test2_overfit,
    test3_capacity_matching,
    test4_hidden_states,
    test5_dynamics_smoke,
    test6_mini_pilot,
)
from architecture_refinement.ws_flex_generator import make_ws_flex_graph


def run_diagnostics() -> int:
    """Run TEST 1-5."""
    print("=== PAPER 3 Diagnostics (TEST 1-5) ===\n")
    tests = [
        (1, test1_data_perturbation_sanity),
        (2, test2_overfit),
        (3, test3_capacity_matching),
        (4, test4_hidden_states),
        (5, test5_dynamics_smoke),
    ]
    passed = 0
    for num, fn in tests:
        try:
            ok = fn()
            if ok:
                passed += 1
                print(f"  >>> TEST {num} PASSED")
            else:
                print(f"  >>> TEST {num} FAILED")
        except Exception as e:
            print(f"  >>> TEST {num} ERROR: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


def run_mini(output_dir: Path) -> int:
    """Run TEST 6 mini pilot: B_evals=20, K=3, 2 seeds, 10 epochs, P1 only."""
    print("=== PAPER 3 Mini Pilot (TEST 6) ===\n")
    ok = test6_mini_pilot()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mini_result.txt").write_text(f"TEST 6: {'PASSED' if ok else 'FAILED'}\n")
    return 0 if ok else 1


def run_full(output_dir: Path) -> int:
    """Full benchmark: CfC-Rand vs NCP vs LSTM. Recurrent architectures, capacity-matched, P1-P3."""
    print("=== PAPER 3 Full Benchmark (CfC-Rand vs NCP vs LSTM) ===\n")
    output_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    H, D_in, K, seeds = 32, 16, 6, [0, 1, 2, 3, 4]
    k_values = [2, 4, 6, 8]
    n_train, n_val, n_test = 20_000, 2_000, 5_000

    train_ds, val_ds, test_ds = create_splits(
        n_train=n_train, n_val=n_val, n_test=n_test, seed=42,
        omega_L=(0.08, 0.18), omega_H=(0.15, 0.28),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = get_capacity_matched_configs(C=1, D_in=D_in, H_ref=H, tol=0.05)

    all_rows = []
    pert_types = ["awgn", "impulse", "drift"]

    # CfC-Rand: K random topologies
    _, selected = run_random_search(H, k_values, 20, K, seeds[0])
    for cand in selected:
        model = CfCRecurrentModel(C=1, D_in=D_in, H=H, hidden_graph=cand.G, wiring_seed=cand.wiring_seed)
        for seed in seeds:
            train_model(model, train_ds, val_ds, epochs=20, batch_size=256, seed=seed, device=device)
            for pt in pert_types:
                pert = evaluate_perturbed(model, test_ds, pt, seed=seed, device=device)
                all_rows.append({
                    "regime": "CfC-Rand", "graph_id": f"{cand.k}_{cand.p}_{cand.graph_seed}",
                    "seed": seed, "pert": pt,
                    "AUPC": pert["AUPC"], "MaxRD": pert["MaxRD"], "score_clean": pert["score_clean"],
                })
            dyn = evaluate_dynamics(model, test_ds, n_samples=512, seed=seed, device=device)
            all_rows.append({
                "regime": "CfC-Rand", "graph_id": f"{cand.k}_{cand.p}_{cand.graph_seed}",
                "seed": seed, "pert": "dynamics",
                "sensitivity": dyn["sensitivity"], "sensitivity_small": dyn["sensitivity_small"],
                "sensitivity_med": dyn["sensitivity_med"], "state_var": dyn["state_var"],
                "lambda": dyn["lambda"], "lambda_small": dyn["lambda_small"], "lambda_med": dyn["lambda_med"],
            })

    ncp_units = configs["NCP"].get("ncp_units")
    if ncp_units:
        model = NCPRecurrentModel(C=1, D_in=D_in, H=32, ncp_units=ncp_units)
        for seed in seeds:
            train_model(model, train_ds, val_ds, epochs=20, batch_size=256, seed=seed, device=device)
            for pt in pert_types:
                pert = evaluate_perturbed(model, test_ds, pt, seed=seed, device=device)
                all_rows.append({"regime": "NCP", "graph_id": "-", "seed": seed, "pert": pt, "AUPC": pert["AUPC"], "MaxRD": pert["MaxRD"], "score_clean": pert["score_clean"]})
            dyn = evaluate_dynamics(model, test_ds, n_samples=512, seed=seed, device=device)
            all_rows.append({"regime": "NCP", "graph_id": "-", "seed": seed, "pert": "dynamics",
                "sensitivity": dyn["sensitivity"], "sensitivity_small": dyn["sensitivity_small"],
                "sensitivity_med": dyn["sensitivity_med"], "state_var": dyn["state_var"],
                "lambda": dyn["lambda"], "lambda_small": dyn["lambda_small"], "lambda_med": dyn["lambda_med"]})

    lstm_H = configs["LSTM"].get("H")
    if lstm_H:
        model = LSTMRecurrentModel(C=1, D_in=D_in, H=lstm_H)
        for seed in seeds:
            train_model(model, train_ds, val_ds, epochs=20, batch_size=256, seed=seed, device=device)
            for pt in pert_types:
                pert = evaluate_perturbed(model, test_ds, pt, seed=seed, device=device)
                all_rows.append({"regime": "LSTM", "graph_id": "-", "seed": seed, "pert": pt, "AUPC": pert["AUPC"], "MaxRD": pert["MaxRD"], "score_clean": pert["score_clean"]})
            dyn = evaluate_dynamics(model, test_ds, n_samples=512, seed=seed, device=device)
            all_rows.append({"regime": "LSTM", "graph_id": "-", "seed": seed, "pert": "dynamics",
                "sensitivity": dyn["sensitivity"], "sensitivity_small": dyn["sensitivity_small"],
                "sensitivity_med": dyn["sensitivity_med"], "state_var": dyn["state_var"],
                "lambda": dyn["lambda"], "lambda_small": dyn["lambda_small"], "lambda_med": dyn["lambda_med"]})

    all_keys = set()
    for r in all_rows:
        all_keys.update(r.keys())
    with open(output_dir / "all_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(all_keys), extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    (output_dir / "config.json").write_text(json.dumps({
        "H": H, "D_in": D_in, "K": K, "seeds": seeds,
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "omega_L": [0.08, 0.18], "omega_H": [0.15, 0.28],
        "pert_types": pert_types,
        "capacity_configs": {k: str(v) for k, v in configs.items()},
    }, indent=2))

    report = [
        "PAPER 3 Full Benchmark Report (CfC-Rand vs NCP vs LSTM)",
        f"Output: {output_dir}",
        f"Regimes: CfC-Rand (K={K} random topologies), NCP (ncp_units={ncp_units}), LSTM (H={lstm_H})",
        f"Task: overlapping omega bands (0.08-0.18 vs 0.15-0.28), P1-P3 (awgn, impulse, drift)",
        f"Results: {len(all_rows)} rows in all_results.csv",
    ]
    (output_dir / "report.txt").write_text("\n".join(report))
    print(f"\nResults saved to {output_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["diagnostics", "mini", "full"], default="diagnostics")
    parser.add_argument("--output-dir", type=str, default="architecture_refinement/outputs/paper3")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.stage == "diagnostics":
        return run_diagnostics()
    elif args.stage == "mini":
        return run_mini(output_dir)
    elif args.stage == "full":
        return run_full(output_dir)
    return 1


if __name__ == "__main__":
    sys.exit(main())
