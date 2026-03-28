"""
Per-epoch validation ROC-AUC for skorch EpochScoring (Project Hail Mary §13.2).

Records history key ``valid_roc_auc`` (binary: positive-class probability; multiclass: OvR macro).
Enable via HAIL_MARY_LEARNABILITY_METRICS or HAIL_MARY_STABILITY (see cnn_wiredcfc_min._build_cfc_callbacks).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import make_scorer, roc_auc_score
from skorch.callbacks import EpochScoring


def _roc_auc_y_true_y_proba(y_true, y_proba, **kwargs) -> float:
    # sklearn's _Scorer may pass through kwargs (e.g. from make_scorer metadata); ignore.
    _ = kwargs
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if y_true.size == 0:
        return float("nan")
    if y_proba.ndim == 1:
        y_proba = np.column_stack([1.0 - y_proba, y_proba])
    elif y_proba.ndim == 2 and y_proba.shape[1] == 1:
        p = np.asarray(y_proba[:, 0], dtype=float)
        y_proba = np.column_stack([1.0 - p, p])
    elif y_proba.ndim != 2 or y_proba.shape[1] < 1:
        return float("nan")

    u = np.unique(y_true)
    if len(u) < 2:
        return float("nan")

    # Binary: two columns (standard head) — positive column is index 1 (sklearn convention).
    if len(u) == 2 and y_proba.shape[1] == 2:
        return float(roc_auc_score(y_true, y_proba[:, 1]))

    # Binary with K>2 logits (e.g. 4-class model, val split only sees two classes): subset
    # columns by class id so they match y_true, renormalize, then score vs positive class.
    if len(u) == 2 and y_proba.shape[1] > 2:
        neg, pos = int(u[0]), int(u[1])
        if neg < 0 or pos < 0 or max(neg, pos) >= y_proba.shape[1]:
            return float("nan")
        sub = np.asarray(y_proba[:, [neg, pos]], dtype=float)
        sub = np.maximum(sub, 0.0)
        rs = sub.sum(axis=1, keepdims=True)
        sub = sub / np.maximum(rs, 1e-12)
        y_bin = (y_true == pos).astype(int)
        return float(roc_auc_score(y_bin, sub[:, 1]))

    # Multiclass: validation may omit classes; sklearn requires y_score columns to match labels.
    labels = np.sort(u).astype(int)
    if labels.min() < 0 or labels.max() >= y_proba.shape[1]:
        return float("nan")
    sub = np.asarray(y_proba[:, labels], dtype=float)
    sub = np.maximum(sub, 0.0)
    sub = sub / np.maximum(sub.sum(axis=1, keepdims=True), 1e-12)
    return float(
        roc_auc_score(
            y_true, sub, multi_class="ovr", average="macro", labels=labels
        )
    )


def make_valid_roc_auc_epoch_scoring() -> EpochScoring:
    try:
        scorer = make_scorer(_roc_auc_y_true_y_proba, response_method="predict_proba")
    except TypeError:
        scorer = make_scorer(_roc_auc_y_true_y_proba, needs_proba=True)
    return EpochScoring(
        scoring=scorer,
        lower_is_better=False,
        on_train=False,
        name="valid_roc_auc",
        use_caching=True,
    )
