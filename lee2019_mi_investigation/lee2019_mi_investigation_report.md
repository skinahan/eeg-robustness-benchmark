# Lee2019_MI: diagnostic synthesis and resolution

This report ties together the **[benchmark reference](benchmark_reference.md)** and **[config matrix](config_matrix.md)** with the runnable scripts in this folder. Use it as a checklist when “chronic underfitting” is reported.

## 1. What “underfitting” usually means here

- **EEGNet** training logs show **`valid_acc` stuck near chance** while early stopping or underfitting-retrain runs fire.
- **Primary metric** in MOABB `MotorImagery` and in this repo’s evaluation is **ROC-AUC** (binary), not raw accuracy — compare like-for-like.

## 2. Likely cause classes

| Class | Symptom | How to verify |
|-------|---------|----------------|
| **A. Preprocessing / window** | CSP+LDA and EEGNet both poor | Run [`ablation_time_window.py`](ablation_time_window.py); large gaps between `paper_1_3p5` and `full_0_4` implicate the epoch window. |
| **B. Session transfer** | Within-session OK, CrossSession bad | Run [`split_comparison.py`](split_comparison.py); large gap → session shift / non-stationarity (expected to be hard). |
| **C. Deep model vs linear** | CSP+LDA good, EEGNet bad | Run [`linear_baseline_same_splits.py`](linear_baseline_same_splits.py) vs `unified_experiment_runner` on same subjects — tune LR / epochs / `ValidSplit`, not the dataset. |
| **D. Subject / illiteracy** | All methods poor for **one** subject | Run [`baseline_moabb_evaluator.py`](baseline_moabb_evaluator.py) across subjects; cite Lee et al. (2019) on BCI illiteracy. |
| **E. Data integrity** | Odd shapes, class imbalance | Run [`data_audit_lee2019_mi.py`](data_audit_lee2019_mi.py). |

## 3. Repo configuration (current)

- **Time window:** `Lee2019_MI` uses **`tmin=1.0`, `tmax=3.5`** in [`config.py`](../config.py) to align with the MOABB/Lee note on **paper online decoding** (see `moabb/datasets/Lee2019.py` comments).
- **CrossSession** in [`unified_experiment_runner.py`](../evaluation/unified_experiment_runner.py) matches **LeaveOneGroupOut on `metadata.session`**, consistent with MOABB’s `CrossSessionEvaluation` logic.
- **EEGNet** uses **`ValidSplit(0.2)`** and early stopping — high variance when **N ≈ 100** trials per session.

## 4. Resolution playbook

1. **If ablations show the paper window wins:** keep the **[1.0, 3.5] s** crop; document it in methods (already in `config.py` comments).
2. **If CrossSession ≪ WithinSession:** report **session transfer** as the main difficulty; do not treat it as a single “bug.”
3. **If CSP+LDA >> EEGNet:** adjust **training** (epochs, patience, learning rate, optional `ValidSplit` fraction for small N) — **after** confirming preprocessing.
4. **If all baselines are poor for a subject:** treat as **dataset variability** / illiteracy; do not force the pipeline to match high-performing subjects.

## 5. Scripts output (artifacts)

All scripts write CSV/JSON under `output/` for traceability and plotting. Re-run after any change to [`config.py`](../config.py) preprocessing.

## References

- Lee et al. (2019), *GigaScience*, DOI [10.1093/gigascience/giz002](https://doi.org/10.1093/gigascience/giz002)
- MOABB `Lee2019_MI` and `MotorImagery` documentation in the installed `moabb` package
