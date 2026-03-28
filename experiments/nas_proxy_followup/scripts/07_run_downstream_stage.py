#!/usr/bin/env python3
"""Full downstream training (optional; long-running)."""

from __future__ import annotations

import argparse
import json
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
from src.models.downstream_runner import run_downstream_training
from src.probe.probe_dataset import encode_moabb_labels, infer_eeg_n_channels, load_bnci_cross_session_arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cfg = load_merged_configs(parse_config_paths(args))
    repo_root = repo_root_from_config(cfg)
    want_cuda = str(cfg.get("device", "cpu")).lower().startswith("cuda")
    device = torch.device("cuda" if want_cuda and torch.cuda.is_available() else "cpu")

    manifest_path = resolve_path(
        cfg,
        "downstream_run_manifest",
        "experiments/nas_proxy_followup/manifests/downstream_run_manifest.csv",
    )
    out_base = resolve_path(cfg, "downstream_output_dir", "experiments/nas_proxy_followup/outputs/downstream_runs")
    out_base.mkdir(parents=True, exist_ok=True)

    dcfg = cfg.get("dataset")
    if not isinstance(dcfg, dict):
        dcfg = {}
    subject = int(cfg.get("subject", dcfg.get("subject", 1)))
    X, y, _meta, sess = load_bnci_cross_session_arrays(subject=subject)
    y_enc = encode_moabb_labels(y)
    train_mask = sess == "0train"
    X_train = X[train_mask]
    y_train = y_enc[train_mask]

    model_cfg = dict(cfg.get("model", {}))
    model_cfg.setdefault("n_channels", infer_eeg_n_channels(X_train))
    model_cfg.setdefault("n_outputs", int(y_enc.max()) + 1)

    df = pd.read_csv(manifest_path).to_dict("records")
    if args.limit:
        df = df[: args.limit]
    for i, row in enumerate(df):
        dt = cfg.get("downstream_training", cfg)
        ckpt = out_base / f"down_{row['topology_id']}_{row['mapping_scheme']}_s{row['training_seed']}.pt"
        logp = out_base / f"down_{row['topology_id']}_{row['mapping_scheme']}_s{row['training_seed']}.log.json"
        try:
            hist = run_downstream_training(
                row,
                X_train,
                y_train,
                repo_root,
                device,
                int(row["training_seed"]),
                cfg,
                model_cfg,
                checkpoint_path=ckpt,
            )
            row["checkpoint_path"] = str(ckpt)
            row["log_path"] = str(logp)
            row["status"] = "ok"
            with open(logp, "w", encoding="utf-8") as f:
                json.dump(hist, f, indent=2)
        except Exception as e:
            row["status"] = f"error: {e!s}"
        df[i] = row

    pd.DataFrame(df).to_csv(manifest_path, index=False)
    print(f"Updated {manifest_path}")


if __name__ == "__main__":
    main()
