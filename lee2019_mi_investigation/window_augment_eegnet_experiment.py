#!/usr/bin/env python3
"""
Lee2019_MI: compare EEGNet with (A) baseline 1s crop vs (B) training-only sliding windows (1s, 0.5s stride).

Leakage safety:
  - LeaveOneGroupOut on session is applied to *trials* first.
  - Sliding windows are computed only on outer-training trials.
  - Validation uses a single fixed crop per trial (start=0, length=win_samples) for both conditions.
  - Do not call sliding_window_tensor before the session split.

If config.py changes Lee2019_MI tmin/tmax, epoch length and window counts update automatically.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder

# Repo root (parent of lee2019_mi_investigation/)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


from evaluation.mi_sliding_window import fixed_crop_batch, sliding_window_tensor


def _roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    from evaluation.metrics import compute_classification_metrics

    m = compute_classification_metrics(y_true, y_proba, num_classes=2)
    return float(m["roc_auc"])


def _run_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    condition: str,
    win_samples: int,
    stride_samples: int,
    crop_start: int,
    seed: int,
) -> Dict:
    """Train EEGNet and return validation ROC-AUC on outer val set."""
    from globals import get_max_epochs_for_dataset, set_seeds
    from models.eegnet import create_eegnet_classifier

    set_seeds(seed)
    n_chans = X_train.shape[1]
    if condition == "baseline":
        X_tr, y_tr = fixed_crop_batch(X_train, y_train, crop_start, win_samples)
        n_train_windows = len(X_tr)
    else:
        X_tr, y_tr = sliding_window_tensor(X_train, y_train, win_samples, stride_samples)
        n_train_windows = len(X_tr)

    X_va, y_va = fixed_crop_batch(X_val, y_val, crop_start, win_samples)

    model = create_eegnet_classifier(
        n_chans, win_samples, n_outputs=2, seed=seed
    )
    model.verbose = 0
    model.max_epochs = get_max_epochs_for_dataset("Lee2019_MI", eval_mode="CrossSession")
    model.fit(X_tr, y_tr)

    model.module_.eval()
    with torch.no_grad():
        proba = model.predict_proba(X_va)
    auc = _roc_auc(y_va, proba)

    return {
        "roc_auc": auc,
        "n_train_trials": int(len(X_train)),
        "n_train_windows": int(n_train_windows),
        "n_val_trials": int(len(X_val)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Lee2019_MI window aug vs baseline (EEGNet)")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Used when --seeds is omitted (single run).",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        metavar="S",
        help="One or more experimental seeds; runs full protocol per seed. Overrides --seed when set.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: lee2019_mi_investigation/output",
    )
    args = ap.parse_args()
    seeds = list(args.seeds) if args.seeds is not None else [args.seed]
    out_dir = args.out_dir or (Path(__file__).resolve().parent / "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    from config import get_dataset_sampling_rate, get_paradigm
    from globals import set_seeds
    from moabb.datasets import Lee2019_MI

    sfreq = float(get_dataset_sampling_rate("Lee2019_MI"))
    win_samples = int(round(1.0 * sfreq))
    stride_samples = int(round(0.5 * sfreq))
    crop_start = 0

    paradigm = get_paradigm(dataset="Lee2019_MI")
    dataset = Lee2019_MI()

    rows: List[Dict] = []
    for exp_seed in seeds:
        set_seeds(exp_seed)
        print(f"=== experimental seed {exp_seed} ===")
        for subject in args.subjects:
            dataset.subject_list = [subject]
            X, y, metadata = paradigm.get_data(dataset, subjects=[subject])
            y_enc = LabelEncoder().fit_transform(y)
            n_times_epoch = int(X.shape[2])
            assert n_times_epoch >= win_samples, (
                f"Subject {subject}: n_times={n_times_epoch} < win_samples={win_samples}"
            )
            print(
                f"Subject {subject}: n_trials={len(X)}, epoch_n_times={n_times_epoch}, "
                f"win_samples={win_samples}, stride_samples={stride_samples}"
            )

            groups = metadata["session"].values
            logo = LeaveOneGroupOut()
            for fold_idx, (train_idx, valid_idx) in enumerate(
                logo.split(X, y_enc, groups=groups)
            ):
                X_tr_raw, y_tr_raw = X[train_idx], y_enc[train_idx]
                X_va_raw, y_va_raw = X[valid_idx], y_enc[valid_idx]
                holdout_session = str(metadata.iloc[valid_idx]["session"].values[0])

                fold_seed = exp_seed + fold_idx
                for condition in ("baseline", "augmented"):
                    stats = _run_fold(
                        X_tr_raw,
                        y_tr_raw,
                        X_va_raw,
                        y_va_raw,
                        condition=condition,
                        win_samples=win_samples,
                        stride_samples=stride_samples,
                        crop_start=crop_start,
                        seed=fold_seed,
                    )
                    rows.append(
                        {
                            "seed": exp_seed,
                            "subject": subject,
                            "fold": fold_idx,
                            "holdout_session": holdout_session,
                            "condition": condition,
                            "roc_auc": stats["roc_auc"],
                            "n_train_trials": stats["n_train_trials"],
                            "n_train_windows": stats["n_train_windows"],
                            "n_val_trials": stats["n_val_trials"],
                            "win_samples": win_samples,
                            "stride_samples": stride_samples,
                            "crop_start": crop_start,
                            "sfreq_hz": sfreq,
                            "epoch_n_times": int(X.shape[2]),
                        }
                    )

    config = {
        "dataset": "Lee2019_MI",
        "subjects": args.subjects,
        "seeds": seeds,
        "win_seconds": 1.0,
        "stride_seconds": 0.5,
        "win_samples": win_samples,
        "stride_samples": stride_samples,
        "val_crop": "first_1s_same_as_train_baseline",
        "note": "Sliding windows only on outer-train trials; val is single crop per trial, no leakage",
        "repo_root": str(_ROOT),
    }

    stem = "window_augment_eegnet_experiment"
    csv_path = out_dir / f"{stem}_summary.csv"
    json_path = out_dir / f"{stem}_config.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    def _json_default(o: object):
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        raise TypeError(f"Not JSON serializable: {type(o)}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"config": config, "rows": rows}, f, indent=2, default=_json_default)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
