import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    num_classes: int,
    average_for_multiclass: str = "macro",
) -> Dict[str, float]:
    """
    Compute ROC-AUC, Accuracy, Precision, Recall, and F1 efficiently.

    - Supports binary and multiclass.
    - For binary, uses probability of the positive class (column 1) and threshold 0.5 for labels.
    - For multiclass, uses one-vs-rest ROC-AUC and argmax labels; metrics averaged with `average_for_multiclass`.
    """
    is_multiclass = num_classes > 2

    if is_multiclass:
        # y_pred_proba expected shape: [N, num_classes]
        y_pred_labels = np.argmax(y_pred_proba, axis=1)
        roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class="ovr")
        acc = accuracy_score(y_true, y_pred_labels)
        prec = precision_score(y_true, y_pred_labels, average=average_for_multiclass, zero_division=0)
        rec = recall_score(y_true, y_pred_labels, average=average_for_multiclass, zero_division=0)
        f1 = f1_score(y_true, y_pred_labels, average=average_for_multiclass, zero_division=0)
    else:
        # Binary case: use column 1 as positive class probability if available
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] >= 2:
            y_pos = y_pred_proba[:, 1]
            y_pred_labels = (y_pos >= 0.5).astype(int)
        else:
            # If only one probability/score provided, treat as positive score
            y_pos = y_pred_proba.reshape(-1)
            y_pred_labels = (y_pos >= 0.5).astype(int)
        roc_auc = roc_auc_score(y_true, y_pos)
        acc = accuracy_score(y_true, y_pred_labels)
        prec = precision_score(y_true, y_pred_labels, zero_division=0)
        rec = recall_score(y_true, y_pred_labels, zero_division=0)
        f1 = f1_score(y_true, y_pred_labels, zero_division=0)

    return {
        "roc_auc": float(roc_auc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


