"""Shared helpers for Lee2019_MI diagnostic scripts."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Project root (parent of lee2019_mi_investigation/)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def repo_root() -> Path:
    return _REPO_ROOT


def get_paradigm_for_window(
    tmin: float,
    tmax: float,
    resample: Optional[float] = None,
):
    """Build MotorImagery for Lee2019_MI with explicit tmin/tmax (seconds within task)."""
    from moabb.paradigms import MotorImagery

    if resample is None:
        from config import get_dataset_sampling_rate

        resample = get_dataset_sampling_rate("Lee2019_MI")
    return MotorImagery(
        events=["left_hand", "right_hand"],
        fmin=8,
        fmax=30,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        resample=resample,
        n_classes=2,
    )


def load_lee2019_mi(
    subjects: List[int],
    paradigm_factory: Callable[[], Any],
):
    """Load X, y, metadata using paradigm.get_data."""
    from moabb.datasets import Lee2019_MI

    dataset = Lee2019_MI()
    dataset.subject_list = list(subjects)
    paradigm = paradigm_factory()
    X, y, metadata = paradigm.get_data(dataset, subjects=subjects)
    return X, y, metadata, dataset, paradigm


def cross_session_fold_indices(
    metadata: pd.DataFrame,
    y_encoded: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """MOABB-style CrossSession: LeaveOneGroupOut on session."""
    groups = metadata["session"].values
    logo = LeaveOneGroupOut()
    return list(logo.split(np.zeros(len(groups)), y_encoded, groups=groups))


def within_session_kfold_scores(
    X: np.ndarray,
    y: np.ndarray,
    session_mask: np.ndarray,
    predict_proba_fn,
    n_splits: int = 5,
    random_state: int = 42,
) -> float:
    """Mean ROC-AUC over StratifiedKFold within one session (MOABB-style)."""
    Xs = X[session_mask]
    ys = y[session_mask]
    if len(np.unique(ys)) < 2:
        return float("nan")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, test_idx in cv.split(Xs, ys):
        proba = predict_proba_fn(Xs[train_idx], ys[train_idx], Xs[test_idx])
        if proba.ndim == 2:
            proba = proba[:, 1]
        scores.append(roc_auc_score(ys[test_idx], proba))
    return float(np.mean(scores))


def write_outputs(
    out_dir: Path,
    stem: str,
    config: Dict[str, Any],
    summary_rows: List[Dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / f"{stem}_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    csv_path = out_dir / f"{stem}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"Wrote {cfg_path}")
    print(f"Wrote {csv_path}")


def encode_y(y) -> Tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    return le.fit_transform(y), le


def make_csp_lda_pipeline(n_components: int = 6):
    """Lightweight MOABB-style baseline: CSP + LDA on (trials, chans, times)."""
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("csp", CSP(n_components=n_components, reg="oas")),
            ("lda", LinearDiscriminantAnalysis()),
        ]
    )


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def within_session_per_session_kfold_aucs(
    X: np.ndarray,
    y_enc: np.ndarray,
    metadata: pd.DataFrame,
    subject: int,
    n_splits: int = 5,
    random_state: int = 42,
    n_components: int = 6,
) -> List[dict]:
    """MOABB WithinSession-style: StratifiedKFold per session."""
    rows = []
    sub_mask = metadata["subject"].values == subject
    for sess in sorted(metadata.loc[sub_mask, "session"].unique()):
        m = sub_mask & (metadata["session"].values == sess)
        Xs = X[m]
        ys = y_enc[m]
        if len(np.unique(ys)) < 2:
            rows.append(
                {
                    "subject": subject,
                    "session": str(sess),
                    "mean_roc_auc": float("nan"),
                    "n_trials": len(ys),
                    "note": "single_class",
                }
            )
            continue
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_aucs = []
        for tr, te in cv.split(Xs, ys):
            pipe = make_csp_lda_pipeline(n_components=n_components)
            pipe.fit(Xs[tr], ys[tr])
            proba = pipe.predict_proba(Xs[te])[:, 1]
            fold_aucs.append(roc_auc_binary(ys[te], proba))
        rows.append(
            {
                "subject": subject,
                "session": str(sess),
                "mean_roc_auc": float(np.mean(fold_aucs)),
                "std_roc_auc": float(np.std(fold_aucs)),
                "n_trials": len(ys),
                "n_splits": n_splits,
            }
        )
    return rows


def cross_session_csp_lda_aucs(
    X: np.ndarray,
    y_enc: np.ndarray,
    metadata: pd.DataFrame,
    n_components: int = 6,
) -> Tuple[List[float], List[str]]:
    """Leave-one-session-out ROC-AUC per fold (MOABB CrossSession-style)."""
    folds = cross_session_fold_indices(metadata, y_enc)
    aucs: List[float] = []
    holdout_sessions: List[str] = []
    for train_idx, test_idx in folds:
        pipe = make_csp_lda_pipeline(n_components=n_components)
        pipe.fit(X[train_idx], y_enc[train_idx])
        proba = pipe.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_binary(y_enc[test_idx], proba))
        sess = metadata.iloc[test_idx]["session"].values[0]
        holdout_sessions.append(str(sess))
    return aucs, holdout_sessions
