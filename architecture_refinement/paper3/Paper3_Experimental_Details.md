# Paper 3 Experimental Details (Reproducibility)

This document consolidates the experimental configuration and protocol used in Paper 3 experiments. It serves as the single source of truth for checklist items 4, 6, 7, and 8.

---

## 1. Hyperparameters (Locked Config)

Paper 3 uses NAS pilot models (G1–G5) registered via `architecture_refinement/nas_pilot_registry.py`, which invokes `create_cnnwiredcfc_min_classifier` with default keyword arguments. The unified experiment runner does not pass any hyperparameter overrides; the only runtime override is `max_epochs`, set via `get_max_epochs_for_dataset()`.

### Architecture parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| F1 | 8 | `models/cnn_wiredcfc_min.py` `create_cnnwiredcfc_min_classifier` |
| D | 2 | Same |
| drop_prob | 0.15 | Same |
| kernel_length | 128 | Same |
| temporal_kernel_size | 3 | Same |
| temporal_stride | 4 | Same |
| max_seq_length | 250 | Same |
| mixed_memory | True | Same |

### Training parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| lr | 1e-3 | `models/cnn_wiredcfc_min.py` |
| batch_size | 64 | Same |
| weight_decay | 1e-3 | Same |
| gradient_clip_value | 1.0 | Same |
| max_epochs (BNCI2014_001 CrossSession) | 200 | `globals.py` `get_max_epochs_for_dataset()` |

### Note on max epochs

- **Paper 1 supplement:** states max epochs 100.
- **Current codebase:** `globals.py` sets `DEFAULT_MAX_EPOCHS = 200` (comment: "Increased from 100").
- **BNCI2014_001 CrossSession:** receives 200 epochs from `get_max_epochs_for_dataset("BNCI2014_001", eval_mode="CrossSession")`.
- **Paper 3:** uses 200 epochs for BNCI2014_001 CrossSession experiments.

---

## 2. Data Split Protocol

### MOABB CrossSession

- **Evaluator:** Leave-one-session-out (LeaveOneGroupOut by session).
- **Per subject:** For each fold, one session is held out for evaluation; all other sessions are used for training.
- **Test set:** The held-out session only. Never used for training or validation.

### Internal validation split

- **API:** `ValidSplit(0.2, stratified=True, random_state=seed)`
- **Applied to:** `X_train` (training-session data only), before `model.fit()`.
- **Effect:** 80% of training-session data → actual training; 20% → validation for early stopping.
- **Stratification:** Class balance preserved in the 20% validation split.
- **Seed:** From `get_seed()` (RAND_SEED = 42 unless overridden).

### Data flow

1. MOABB provides `X`, `y`, metadata with session labels.
2. LeaveOneGroupOut produces `train_idx`, `valid_idx` per fold.
3. `X_train = X[train_idx]`, `y_train = y[train_idx]` (training sessions only).
4. `X_valid = X[valid_idx]`, `y_valid = y[valid_idx]` (held-out session).
5. `model.fit(X_train, y_train)`; internally, ValidSplit splits `X_train` into 80% train / 20% validation.
6. Held-out session (`X_valid`, `y_valid`) is used only for final evaluation.

---

## 3. Loss and Label Format

### Loss

- **Function:** `torch.nn.CrossEntropyLoss`
- **Inputs:** Raw logits (no softmax) and integer class labels (0 and 1 for binary).

### ROC-AUC for binary classification

- **Computation:** `roc_auc_score(y_true, y_pos)` where `y_pos = y_pred_proba[:, 1]`.
- **Probability source:** `predict_proba()` returns softmax over logits; column 1 is the positive-class probability.
- **Code:** `evaluation/metrics.py` `compute_classification_metrics()`.

### Label encoding

- **Binary:** Labels 0 (negative) and 1 (positive).
- **MOABB string labels:** Converted via `sklearn.preprocessing.LabelEncoder` in `compute_classification_metrics()` when needed.

---

## 4. Preprocessing (BNCI2014_001)

| Parameter | Value | Source |
|-----------|-------|--------|
| Paradigm | MotorImagery | `config.py` `get_paradigm()` |
| Events | left_hand, right_hand | Same (2-class subset) |
| Filter band | fmin=8, fmax=35 Hz | Same |
| Epoch window | tmin=0.0, tmax=None | Same (full trial) |
| Resampling | 250 Hz | `get_dataset_sampling_rate("BNCI2014_001")` |
| Baseline | None | Same |
| Channels | MOABB default (22 EEG) | Paradigm/dataset |

**Note:** `tmax=None` means the full trial length is used (end of epoch as defined by the dataset).

---

## 5. Dataset Identity

**BNCI2014_001** (MOABB identifier) corresponds to **BCI Competition IV Dataset 2a** (motor imagery). It has 9 subjects, two sessions per subject, 22 EEG channels, and 250 Hz sampling. Performance is reported as ROC-AUC. The dataset is described in:

> Tangermann, M., Müller, K. R., Aertsen, A., Birbaumer, N., Braun, C., Brunner, C., Leeb, R., Mehring, C., Miller, K. J., et al. "Review of the BCI Competition IV." *Frontiers in Neuroscience*, vol. 6, 2012, article 55. DOI: 10.3389/fnins.2012.00055

Paper 3 uses the 2-class subset (left hand vs right hand motor imagery).

---

## 6. Early Stopping

- **Monitor:** `valid_loss`
- **Patience:** 20 epochs
- **Threshold:** 1e-5 (minimum improvement to count as non-stagnant)
- **Load best:** True (restore best weights after stopping)
- **Direction:** Lower is better (implicit for loss in skorch)

**Source:** `globals.py` `get_early_stopping_callback()`

---

## 7. Compute Resources

### Hardware (fixed statement)

- **GPU:** Nvidia GeForce 3060
- **RAM:** 16 GB

### Wall-clock timing

Per-run timing is stored in CSV result files produced by `unified_experiment_runner.py`:

| Column | Description |
|--------|-------------|
| `training_time` | Training time (seconds) |
| `evaluation_time` | Evaluation time (seconds) |
| `total_time` | Total time (seconds) |

These columns are present in the per-subject CSV outputs under `results/<paradigm>/<dataset>/<model>/<eval_mode>/<seed>/sub-<id>/<session>/test_perturb/`.
