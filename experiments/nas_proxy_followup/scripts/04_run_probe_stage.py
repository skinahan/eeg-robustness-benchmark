#!/usr/bin/env python3
"""Run probe training and per-run proxy metrics."""

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
from src.probe.fd_sensitivity import compute_fd_sensitivity
from src.probe.jacobian_metrics import estimate_jacobian_spectral_norm
from src.probe.probe_dataset import encode_moabb_labels, infer_eeg_n_channels, load_bnci_cross_session_arrays
from src.probe.probe_runner import build_probe_model, load_graph_for_topology_row, train_probe_loop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--limit", type=int, default=0, help="Max probe rows (0=all)")
    args = parser.parse_args()
    cfg = load_merged_configs(parse_config_paths(args))
    repo_root = repo_root_from_config(cfg)
    want_cuda = str(cfg.get("device", "cpu")).lower().startswith("cuda")
    device = torch.device("cuda" if want_cuda and torch.cuda.is_available() else "cpu")

    manifest_path = resolve_path(
        cfg,
        "probe_run_manifest",
        "experiments/nas_proxy_followup/manifests/probe_run_manifest.csv",
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run 03 first: {manifest_path}")

    dcfg = cfg.get("dataset")
    if not isinstance(dcfg, dict):
        dcfg = {}
    subject = int(cfg.get("subject", dcfg.get("subject", 1)))
    X, y, _meta, sess = load_bnci_cross_session_arrays(subject=subject)
    y_enc = encode_moabb_labels(y)

    train_mask = sess == "0train"
    X_train = X[train_mask]
    y_train = y_enc[train_mask]

    ps = cfg.get("probe_subset", {})
    rel_idx = ps.get(
        "save_indices_path",
        "experiments/nas_proxy_followup/manifests/probe_subset_indices.json",
    )
    idx_path = (repo_root / rel_idx).resolve() if not Path(rel_idx).is_absolute() else Path(rel_idx)
    with open(idx_path, encoding="utf-8") as f:
        probe_js = json.load(f)
    probe_ix = probe_js["indices"]
    X_probe = X[probe_ix]
    y_probe = y_enc[probe_ix]

    pt = cfg.get("probe_training", {})
    epochs = int(pt.get("epochs", 3))
    batch_size = int(pt.get("batch_size", 64))
    lr = float(pt.get("learning_rate", 1e-3))
    model_cfg = cfg.get("model", {})
    D_in = int(model_cfg.get("D_in", 16))
    H = int(model_cfg.get("H", 32))
    n_ch = int(model_cfg.get("n_channels", infer_eeg_n_channels(X_train)))
    n_out = int(model_cfg.get("n_outputs", int(y_enc.max()) + 1))

    out_dir = resolve_path(cfg, "probe_output_dir", "experiments/nas_proxy_followup/outputs/probe_runs")
    out_dir.mkdir(parents=True, exist_ok=True)

    jcfg = cfg.get("jacobian_proxy", {})
    fcfg = cfg.get("fd_proxy", {})

    df = pd.read_csv(manifest_path)
    rows = df.to_dict("records")
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    for i, row in enumerate(rows):
        tid = row["topology_id"]
        scheme = row["mapping_scheme"]
        pseed = int(row["probe_seed"])
        try:
            G = load_graph_for_topology_row({"graph_path": row["graph_path"]}, repo_root)
        except Exception as e:
            rows[i]["status"] = f"error: {e!s}"
            continue
        model = build_probe_model(
            G,
            n_channels=n_ch,
            D_in=D_in,
            H=H,
            n_outputs=n_out,
            mapping_scheme=scheme,
            wiring_seed=pseed,
        ).to(device)
        ckpt = out_dir / f"probe_{tid}_{scheme}_s{pseed}.pt"
        logp = out_dir / f"probe_{tid}_{scheme}_s{pseed}.log.json"
        hist = train_probe_loop(
            model,
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            checkpoint_path=ckpt,
        )
        try:
            row["probe_checkpoint_path"] = str(ckpt.relative_to(_NPF_ROOT))
        except ValueError:
            row["probe_checkpoint_path"] = str(ckpt)
        try:
            row["probe_log_path"] = str(logp.relative_to(_NPF_ROOT))
        except ValueError:
            row["probe_log_path"] = str(logp)
        row["status"] = "ok"
        with open(logp, "w", encoding="utf-8") as f:
            json.dump(hist, f, indent=2)

        model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
        model.eval()
        xb = torch.from_numpy(np.ascontiguousarray(X_probe[: batch_size])).float().to(device)
        if xb.ndim == 3 and xb.shape[1] < xb.shape[2]:
            xb = xb.transpose(1, 2)
        metrics = {}
        if jcfg.get("enabled", True):
            try:
                metrics.update(estimate_jacobian_spectral_norm(model, xb, num_power_iters=int(jcfg.get("num_power_iters", 20))))
            except Exception as je:
                metrics["jacobian_error"] = str(je)
        if fcfg.get("enabled", True):
            metrics.update(
                compute_fd_sensitivity(
                    model,
                    xb,
                    float(fcfg.get("noise_std", 0.01)),
                    num_repeats=int(fcfg.get("num_repeats", 3)),
                )
            )
        mp = out_dir / f"probe_metrics_{tid}_{scheme}_s{pseed}.json"
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        rows[i] = row

    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"Updated {manifest_path}")


if __name__ == "__main__":
    main()
