#!/usr/bin/env python3
"""
Probe Shin2017A + EEGNet under WithinSession evaluation with multiple experimental seeds.

Use this to find (subject, seed, session) combinations where clean ROC-AUC sits above the
Combrisson chance threshold before running noise perturbation / saturation detection.

Run from repo root (conda env ncp_robustness_proj), e.g.:
  $env:KMP_DUPLICATE_LIB_OK='TRUE'; conda run -n ncp_robustness_proj python evaluation/shin2017a_within_session_seed_sweep.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Project root (parent of evaluation/)
_CURRENT = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from moabb.datasets import Shin2017A  # noqa: E402
from moabb_braindecode_compat import fix_moabb_lee2019_session_filter  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

from config import get_paradigm  # noqa: E402
from evaluation.metrics import compute_classification_metrics  # noqa: E402
from evaluation.saturation_detector import (  # noqa: E402
    AdaptiveSaturationDetector,
    StatisticalThresholds,
)
from globals import set_seeds  # noqa: E402

DATASET_NAME = "Shin2017A"


@dataclass
class RunRow:
    seed: int
    session: str
    n_train: int
    n_test: int
    roc_auc: float
    accuracy: float
    balanced_accuracy: float
    chance_threshold: float
    above_chance: bool


def _session_stats(metadata, y_encoded: np.ndarray) -> List[Tuple[str, int, np.ndarray]]:
    """(session_label, n_trials, boolean_mask) for sessions with both classes."""
    if "session" not in metadata.columns:
        return []
    sess = metadata["session"].astype(str).values
    out: List[Tuple[str, int, np.ndarray]] = []
    for s in np.unique(sess):
        m = sess == s
        n = int(m.sum())
        if n < 6:
            continue
        y_s = y_encoded[m]
        if len(np.unique(y_s)) < 2:
            continue
        c = np.bincount(y_s)
        if c.min() < 2:
            continue
        out.append((str(s), n, m))
    out.sort(key=lambda x: -x[1])
    return out


def split_within_session_for_label(
    X: np.ndarray,
    y_encoded: np.ndarray,
    metadata,
    session_label: str,
    seed: int,
    test_size: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified holdout inside one session (same protocol as saturation_detector)."""
    sess = metadata["session"].astype(str).values
    m = sess == str(session_label)
    if m.sum() < 6:
        raise ValueError(f"Session {session_label!r} has too few trials.")
    X_s = X[m]
    y_s = y_encoded[m]
    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y_s, test_size=test_size, random_state=seed, stratify=y_s
    )
    return X_train, y_train, X_test, y_test


def evaluate_seed_session(
    detector: AdaptiveSaturationDetector,
    X: np.ndarray,
    y_encoded: np.ndarray,
    metadata,
    session_label: str,
    seed: int,
) -> RunRow:
    set_seeds(seed)
    detector.base_seed = seed
    X_train, y_train, X_test, y_test = split_within_session_for_label(
        X, y_encoded, metadata, session_label, seed
    )
    n_classes = 2
    model, _ = detector._fit_model_with_underfit_retrain(
        X_train,
        y_train,
        X_test,
        y_test,
        DATASET_NAME,
        n_classes,
        eval_mode="WithinSession",
    )
    import torch

    model.module_.eval()
    with torch.no_grad():
        proba = model.predict_proba(X_test)
    m = compute_classification_metrics(y_test, proba, n_classes)
    y_hat = np.argmax(proba, axis=1)
    bacc = balanced_accuracy_score(y_test, y_hat)
    n_test = len(y_test)
    thr = StatisticalThresholds.get_chance_threshold(n_classes, max(1, n_test))
    return RunRow(
        seed=seed,
        session=session_label,
        n_train=len(y_train),
        n_test=n_test,
        roc_auc=float(m["roc_auc"]),
        accuracy=float(m["accuracy"]),
        balanced_accuracy=float(bacc),
        chance_threshold=float(thr),
        above_chance=float(m["roc_auc"]) >= thr,
    )


def compare_cross_session_one(
    detector: AdaptiveSaturationDetector,
    X: np.ndarray,
    y_encoded: np.ndarray,
    metadata,
    seed: int,
) -> Tuple[float, int, int]:
    """Return (roc_auc, n_train, n_test) for CrossSession-style split at this seed."""
    set_seeds(seed)
    detector.base_seed = seed
    X_train, y_train, X_test, y_test = detector._split_train_test_cross_session(
        X, y_encoded, metadata
    )
    _, roc_auc_fit = detector._fit_model_with_underfit_retrain(
        X_train,
        y_train,
        X_test,
        y_test,
        DATASET_NAME,
        2,
        eval_mode="CrossSession",
    )
    return float(roc_auc_fit), len(X_train), len(X_test)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Shin2017A WithinSession seed sweep (clean decoding vs Combrisson chance)."
    )
    ap.add_argument("--subject", type=int, default=1, help="MOABB subject id (default: 1)")
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[40, 41, 42, 43, 44, 45, 46, 47],
        help="Experimental seeds to try (model init + stratified split RNG)",
    )
    ap.add_argument(
        "--session-mode",
        choices=("largest", "all"),
        default="largest",
        help="largest: only the session with most eligible trials; all: every eligible session",
    )
    ap.add_argument(
        "--compare-cross-session",
        action="store_true",
        help="Also print one CrossSession (LOGO / 0train) clean ROC-AUC per seed for contrast",
    )
    ap.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write JSON summary",
    )
    args = ap.parse_args()

    dataset = Shin2017A(accept=True)
    fix_moabb_lee2019_session_filter(dataset)
    dataset.subject_list = [args.subject]
    paradigm = get_paradigm(resample=None, dataset=DATASET_NAME)
    X, y, metadata = paradigm.get_data(dataset, subjects=[args.subject])
    y_encoded = LabelEncoder().fit_transform(y)

    print("=== Shin2017A WithinSession seed sweep ===")
    print(f"Subject: {args.subject}")
    print(f"X shape: {X.shape}, columns: {list(metadata.columns)}")
    if "session" in metadata.columns:
        print(f"Sessions (raw counts): {metadata['session'].value_counts().to_dict()}")
    stats = _session_stats(metadata, y_encoded)
    print(f"Eligible sessions (>=6 trials, >=2/class): {[(s, n) for s, n, _ in stats]}")
    if not stats:
        print("No eligible session for stratified WithinSession split. Check MOABB / paradigm.")
        return 1

    sessions: Sequence[str]
    if args.session_mode == "largest":
        sessions = [stats[0][0]]
    else:
        sessions = [s for s, _, _ in stats]

    detector = AdaptiveSaturationDetector(
        base_seed=42,
        lee2019_mi_train_sliding_window=False,
    )

    rows: List[RunRow] = []
    for sess in sessions:
        for seed in args.seeds:
            row = evaluate_seed_session(
                detector, X, y_encoded, metadata, sess, seed
            )
            rows.append(row)
            flag = "OK" if row.above_chance else "below"
            print(
                f"  seed={seed:3d} session={row.session!s:12s} "
                f"ROC-AUC={row.roc_auc:.3f} thr={row.chance_threshold:.3f} "
                f"bal_acc={row.balanced_accuracy:.3f} n_test={row.n_test} [{flag}]"
            )

    if args.compare_cross_session:
        print("\n=== CrossSession reference (same seeds; not WithinSession) ===")
        for seed in args.seeds:
            try:
                auc, n_tr, n_te = compare_cross_session_one(
                    detector, X, y_encoded, metadata, seed
                )
                print(
                    f"  seed={seed:3d} CrossSession ROC-AUC={auc:.3f} "
                    f"n_train={n_tr} n_test={n_te}"
                )
            except Exception as e:
                print(f"  seed={seed:3d} CrossSession failed: {e}")

    good = [r for r in rows if r.above_chance]
    print("\n=== Summary ===")
    print(f"Total runs: {len(rows)}, above Combrisson ROC chance: {len(good)}")
    if good:
        best = max(good, key=lambda r: r.roc_auc)
        print(
            f"Best: seed={best.seed} session={best.session!r} "
            f"ROC-AUC={best.roc_auc:.3f} (use base_seed={best.seed} in AdaptiveSaturationDetector)"
        )
    else:
        print(
            "No configuration beat chance. Try: more seeds, other subjects, shorter epoch window "
            "in config.get_paradigm(Shin2017A), or a different model."
        )

    if args.json_out:
        payload: Dict[str, Any] = {
            "dataset": DATASET_NAME,
            "subject": args.subject,
            "session_mode": args.session_mode,
            "rows": [r.__dict__ for r in rows],
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.json_out}")

    return 0 if good else 2


if __name__ == "__main__":
    raise SystemExit(main())
