# Lee2019 SSVEP Investigation Report: Suspiciously High Results Analysis

## Executive Summary

After investigating the Lee2019 SSVEP evaluation pipeline, I've identified several potential issues that could explain the suspiciously high results, even under significant noise perturbation. The main concerns center around **class imbalance handling**, **evaluation methodology**, and **noise application consistency**.

## Key Findings

### 1. **CRITICAL: Potential Class Imbalance Issues**

**Problem**: The evaluation pipeline uses `LabelEncoder().fit_transform(y)` to convert SSVEP class labels to integers (0,1,2,3), but this encoding may not preserve the original class distribution or meaningful ordering.

**Evidence**:
- SSVEP classes represent different stimulation frequencies (5.45, 6.67, 8.57, 12 Hz)
- `LabelEncoder` assigns integer labels based on alphabetical order, not frequency order
- This could lead to inconsistent class mapping across different subjects/sessions

**Impact**: If classes are not properly balanced or if the encoding introduces bias, models might achieve artificially high performance by exploiting these imbalances.

### 2. **StratifiedKFold Assumption Violation**

**Problem**: The code uses `StratifiedKFold(n_splits=3, shuffle=True, random_state=self.seed)` which assumes balanced classes, but SSVEP datasets may have inherent imbalances.

**Evidence from code**:
```python
# In session_evaluator.py line 92
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.seed)
```

**Impact**: If classes are imbalanced, StratifiedKFold may create folds that don't represent the true class distribution, leading to overoptimistic results.

### 3. **ROC-AUC OvR Strategy Masking Issues**

**Problem**: The evaluation uses One-vs-Rest (OvR) ROC-AUC strategy for multi-class SSVEP (4 classes), which can mask class-specific performance issues.

**Evidence from code**:
```python
# In two_stage_hp_opt.py lines 90-91
else:
    # Multi-class classification - use all probabilities with OvR strategy
    auc = roc_auc_score(y_valid_part, y_pred_proba, multi_class='ovr')
```

**Impact**: OvR ROC-AUC can give high scores even if some classes are poorly classified, especially in imbalanced scenarios.

### 4. **Noise Application Inconsistencies**

**Problem**: In `test_perturb` mode, noise is applied only to validation sets during cross-validation, but the application may not be consistent across classes.

**Evidence from code**:
```python
# In session_evaluator.py lines 180-184
for i, (train_idx, valid_idx) in enumerate(cv.split(X_to_corrupt, y_mask)):
    X_to_corrupt[valid_idx] = noise_augmentor.transform(X_to_corrupt[valid_idx])
    # ... evaluation ...
    # Reset X_to_corrupt to original data
    X_to_corrupt[valid_idx] = X_mask[valid_idx]
```

**Impact**: If noise affects different classes differently, or if the reset mechanism doesn't work properly, it could lead to inconsistent evaluation.

### 5. **Hardcoded Channel Count Issue**

**Problem**: The code has hardcoded assumptions about channel counts that may not match Lee2019's actual configuration.

**Evidence from code**:
```python
# In session_evaluator.py line 154
self.model = wrapped_model_fn(n_chans=22, n_times=int(self.resample * 4), n_outputs=2)
```

**Note**: Lee2019 has 62 channels, but this line hardcodes 22 channels (which is correct for BNCI2014_001).

### 6. **Session-Based Evaluation Potential Issues**

**Problem**: The evaluation splits data by sessions (`0train`, `1test`), but different sessions may have different class distributions or quality.

**Evidence**: The code processes each session separately, which could lead to inconsistent evaluation if sessions have different characteristics.

## Recommendations

### Immediate Actions

1. **Verify Class Distribution**:
   ```python
   # Add this analysis to your evaluation pipeline
   for subject in subjects:
       X, y, metadata = paradigm.get_data(dataset, subjects=[subject])
       print(f"Subject {subject} class distribution:")
       unique, counts = np.unique(y, return_counts=True)
       for u, c in zip(unique, counts):
           print(f"  Class {u}: {c} samples")
   ```

2. **Check Label Encoding Consistency**:
   ```python
   # Ensure consistent encoding across subjects
   label_encoder = LabelEncoder()
   all_labels = []
   for subject in subjects:
       _, y, _ = paradigm.get_data(dataset, subjects=[subject])
       all_labels.extend(y)
   
   # Fit encoder on all labels
   label_encoder.fit(all_labels)
   print("Class mapping:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))
   ```

3. **Add Per-Class Performance Metrics**:
   ```python
   from sklearn.metrics import classification_report, confusion_matrix
   
   # After model evaluation
   y_pred = model.predict(X_test)
   print("Per-class performance:")
   print(classification_report(y_test, y_pred))
   print("Confusion matrix:")
   print(confusion_matrix(y_test, y_pred))
   ```

4. **Verify Noise Application**:
   ```python
   # Check if noise is applied consistently
   original_signal_power = np.mean(X**2)
   noisy_signal_power = np.mean(X_noisy**2)
   print(f"Signal power change: {noisy_signal_power/original_signal_power:.3f}")
   ```

### Medium-term Improvements

1. **Implement Balanced Evaluation**:
   - Use balanced accuracy instead of raw accuracy
   - Implement stratified sampling that respects class balance
   - Add class-weighted loss functions

2. **Enhanced Cross-Validation**:
   - Use `StratifiedGroupKFold` if available
   - Implement custom CV that ensures balanced folds
   - Add per-class performance tracking

3. **Robust Metrics**:
   - Add macro/micro averaged F1-scores
   - Implement per-class ROC-AUC
   - Add balanced accuracy metrics

## Expected Baseline Performance

Based on the web search results, typical SSVEP classification performance should be:
- **Clean data**: 70-90% accuracy (depending on subject and method)
- **With noise**: Performance should degrade significantly (20-40% drop with moderate noise)
- **Chance level**: 25% for 4-class SSVEP

If your results are consistently above 95% even with high noise levels, this strongly suggests methodological issues rather than superior model performance.

## Next Steps

1. Run the analysis script to verify class distributions
2. Check label encoding consistency across subjects
3. Implement per-class performance metrics
4. Verify noise application is working correctly
5. Compare results with expected SSVEP baselines
6. Consider re-running experiments with corrected methodology

The high performance under noise perturbation is likely due to one or more of these methodological issues rather than genuine model robustness.
