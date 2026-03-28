#!/usr/bin/env python3
"""Gaussian perturbation evaluation on downstream checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_NPF_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT, _NPF_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.config_util import add_config_args, load_merged_configs, parse_config_paths, repo_root_from_config, resolve_path
from src.evaluation.perturbation_eval import evaluate_gaussian_perturbation
from src.probe.probe_runner import build_probe_model, load_graph_for_topology_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cfg = load_merged_configs(parse_config_paths(args))
    repo_root = repo_root_from_config(cfg)
    want_cuda = str(cfg.get("device", "cpu")).lower().startswith("cuda")
    device = torch.device("cuda" if want_cuda and torch.cuda.is_available() else "cpu")

    man_path = resolve_path(
        cfg,
        "downstream_run_manifest",
        "experiments/nas_proxy_followup/manifests/downstream_run_manifest.csv",
    )
    out_csv = resolve_path(
        cfg,
        "perturbation_output",
        "experiments/nas_proxy_followup/outputs/evaluation/local_perturbation_results.csv",
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pec = cfg.get("perturbation_eval", {})
    levels = [float(x) for x in pec.get("levels", [0.0, 0.25, 0.5])]
    pseed = int(pec.get("perturbation_seed", 999))

    from src.probe.probe_dataset import encode_moabb_labels, infer_eeg_n_channels, load_bnci_cross_session_arrays

    dcfg = cfg.get("dataset")
    if not isinstance(dcfg, dict):
        dcfg = {}
    subject = int(cfg.get("subject", dcfg.get("subject", 1)))
    X, y, _meta, sess = load_bnci_cross_session_arrays(subject=subject)
    y_enc = encode_moabb_labels(y)
    test_mask = sess == "1test"
    X_test = X[test_mask]
    y_test = y_enc[test_mask]

    model_cfg = cfg.get("model", {})
    D_in = int(model_cfg.get("D_in", 16))
    H = int(model_cfg.get("H", 32))
    n_ch = int(model_cfg.get("n_channels", infer_eeg_n_channels(X_test)))
    n_out = int(model_cfg.get("n_outputs", int(y_enc.max()) + 1))

    df = pd.read_csv(man_path)
    if args.limit:
        df = df.head(args.limit)
    rows = []
    for _, row in df.iterrows():
        if row.get("status") != "ok" or not row.get("checkpoint_path"):
            continue
        ckpt = Path(row["checkpoint_path"])
        if not ckpt.is_absolute():
            ckpt = _NPF_ROOT / ckpt
        if not ckpt.exists():
            ckpt = repo_root / row["checkpoint_path"]
        G = load_graph_for_topology_row(row.to_dict(), repo_root)
        model = build_probe_model(
            G,
            n_channels=n_ch,
            D_in=D_in,
            H=H,
            n_outputs=n_out,
            mapping_scheme=str(row["mapping_scheme"]),
            wiring_seed=int(row["training_seed"]),
        )
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state"])
        model.to(device)
        ev = evaluate_gaussian_perturbation(
            model,
            X_test,
            y_test,
            levels=levels,
            perturbation_seed=pseed,
            device=device,
        )
        for e in ev:
            rows.append(
                {
                    "topology_id": row["topology_id"],
                    "mapping_scheme": row["mapping_scheme"],
                    "training_seed": row["training_seed"],
                    "perturbation_family": pec.get("family", "gaussian"),
                    "level": e["level"],
                    "test_auc": e["test_auc"],
                    "relative_degradation": e["relative_degradation"],
                }
            )
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
