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
    # Ensure y_true is a numpy array of integers
    y_true = np.asarray(y_true)
    if y_true.dtype.kind in ['U', 'S', 'O']:  # String or object dtype
        # Convert string labels to integers using LabelEncoder
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_true = le.fit_transform(y_true)
    else:
        # Convert to integers if not already
        y_true = y_true.astype(int)
    
    # Ensure labels are in valid range [0, num_classes-1]
    unique_labels = np.unique(y_true)
    if len(unique_labels) > num_classes:
        raise ValueError(f"Found {len(unique_labels)} unique labels but num_classes={num_classes}. Labels: {unique_labels}")
    if num_classes == 2 and not np.all(np.isin(unique_labels, [0, 1])):
        # For binary classification, ensure labels are 0 and 1
        # If labels are in a different range, remap them
        if len(unique_labels) == 2:
            # Remap to 0 and 1
            label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_labels))}
            y_true = np.array([label_map[label] for label in y_true])
            unique_labels = np.unique(y_true)
    
    # Ensure y_pred_proba is a numpy array
    y_pred_proba = np.asarray(y_pred_proba)
    
    # Ensure y_pred_proba has the correct shape
    if y_pred_proba.ndim == 1:
        y_pred_proba = y_pred_proba.reshape(-1, 1)
    
    is_multiclass = num_classes > 2

    if is_multiclass:
        # For multiclass, sklearn's roc_auc_score with multi_class="ovr" requires that 
        # the number of columns in y_pred_proba matches the number of unique classes in y_true
        n_unique_classes = len(unique_labels)
        n_proba_classes = y_pred_proba.shape[1]
        
        if n_proba_classes != n_unique_classes:
            if n_proba_classes > n_unique_classes:
                # y_pred_proba has more columns than unique classes in y_true
                # This can happen if validation set is missing some classes
                # We need to select only the columns for classes that actually appear
                max_label = unique_labels.max()
                
                # Check if labels are consecutive starting from 0
                if max_label < n_proba_classes and np.array_equal(unique_labels, np.arange(max_label + 1)):
                    # Labels are consecutive [0, 1, ..., max_label], select first (max_label+1) columns
                    y_pred_proba = y_pred_proba[:, :max_label + 1]
                else:
                    # Labels are non-consecutive (e.g., [0, 1, 3])
                    # Remap y_true to consecutive indices and select corresponding columns
                    label_remap = {old_label: new_idx for new_idx, old_label in enumerate(sorted(unique_labels))}
                    y_true_remapped = np.array([label_remap[label] for label in y_true])
                    
                    # Select columns corresponding to the unique labels
                    selected_columns = sorted(unique_labels)
                    if max(selected_columns) >= n_proba_classes:
                        raise ValueError(
                            f"Cannot align y_pred_proba (shape {y_pred_proba.shape}) with unique labels {unique_labels}. "
                            f"Max label {max(selected_columns)} exceeds number of columns {n_proba_classes}"
                        )
                    y_pred_proba = y_pred_proba[:, selected_columns]
                    y_true = y_true_remapped
            else:
                raise ValueError(
                    f"y_pred_proba has {n_proba_classes} columns but y_true has {n_unique_classes} unique classes. "
                    f"Unique labels: {unique_labels}. Expected at least {n_unique_classes} columns."
                )
        
        # y_pred_proba expected shape: [N, n_unique_classes] after alignment
        # Normalize probabilities to ensure they sum to 1.0 for each sample (required by sklearn)
        # First, ensure non-negative values (clip negative values to 0)
        y_pred_proba = np.maximum(y_pred_proba, 0.0)
        # Then normalize so each row sums to 1.0
        row_sums = y_pred_proba.sum(axis=1, keepdims=True)
        # Avoid division by zero - if sum is 0, set to uniform distribution
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        y_pred_proba = y_pred_proba / row_sums
        
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
            # Ensure probabilities are in [0, 1] range
            y_pos = np.clip(y_pos, 0.0, 1.0)
            y_pred_labels = (y_pos >= 0.5).astype(int)
        else:
            # If only one probability/score provided, treat as positive score
            y_pos = y_pred_proba.reshape(-1)
            # Ensure probabilities are in [0, 1] range
            y_pos = np.clip(y_pos, 0.0, 1.0)
            y_pred_labels = (y_pos >= 0.5).astype(int)
        
        # Ensure y_pred_labels and y_true have the same dtype for sklearn metrics
        y_pred_labels = y_pred_labels.astype(int)
        
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


