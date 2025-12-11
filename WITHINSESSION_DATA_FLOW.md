# WithinSession Evaluation Data Flow

## Overview
This document traces the complete data flow for WithinSession evaluation mode, showing how fold results are compiled and aggregated correctly after the bug fixes.

---

## Data Flow Steps

### 1. Subject Loop (Lines 851-936)
```python
for subject in self.subjects:
    self.current_subject = subject
    # Get data for this subject
    X, y, metadata = self.paradigm.get_data(self.dataset_obj, subjects=[subject])
```

**Output:** One iteration per subject, with `self.current_subject` correctly set

---

### 2. Session Loop (Lines 875-891)
```python
for session in metadata['session'].unique():
    session_idx = metadata['session'] == session
    first_idx = session_idx.index[0]
    X_session, y_session = X[session_idx], y_encoded[session_idx]
    session_metadata = metadata[session_idx]
```

**Output:** Data filtered to specific session

---

### 3. Fold Generation (Lines 882-891)
```python
folds = list(enumerate(cv_splitter.split(X_session, y_session)))
for fold_idx, (train_idx, valid_idx) in folds:
    # Save unadjusted indices for session_metadata
    train_idx_session = train_idx.copy()
    # Adjust indices for global dataset arrays
    train_idx_global = train_idx + first_idx
    valid_idx_global = valid_idx + first_idx
    # Store global indices for data extraction
    fold_indices.append((fold_idx, (train_idx_global, valid_idx_global), session))
    # Use session-specific indices for metadata
    fold_metadata.append(session_metadata.iloc[train_idx_session])
```

**Key Points:**
- `fold_idx` resets to 0-4 for each session (5-fold CV)
- `train_idx_global` and `valid_idx_global` index into full subject dataset
- `train_idx_session` indexes into session-specific metadata
- `fold_metadata` list grows across all sessions

**Example with 2 sessions:**
```
Session '0train':
  fold_indices[0] = (0, (indices_0_0, ...), '0train')
  fold_indices[1] = (1, (indices_0_1, ...), '0train')
  ...
  fold_indices[4] = (4, (indices_0_4, ...), '0train')
  fold_metadata[0-4] = metadata for session '0train', folds 0-4

Session '1test':
  fold_indices[5] = (0, (indices_1_0, ...), '1test')
  fold_indices[6] = (1, (indices_1_1, ...), '1test')
  ...
  fold_indices[9] = (4, (indices_1_4, ...), '1test')
  fold_metadata[5-9] = metadata for session '1test', folds 0-4
```

---

### 4. Fold Evaluation (Lines 899-906)
```python
all_results = []
for i, (fold_idx, (train_idx, valid_idx), session) in enumerate(fold_indices):
    X_train = X[train_idx]
    y_train = y_encoded[train_idx]
    X_valid = X[valid_idx]
    y_valid = y_encoded[valid_idx]
    metadata_train = fold_metadata[i]  # ✓ Use enumeration index, not fold_idx
    
    fold_results = self._evaluate_cv_fold(
        X_train, y_train, X_valid, y_valid, 
        fold_idx, cv_metadata, session, metadata_train
    )
    all_results.extend(fold_results)
```

**Key Points:**
- Use enumeration index `i` to access `fold_metadata` (not `fold_idx`)
- Each fold generates one or more result dictionaries
- Results include: fold_idx, session, subject, scores, metrics

**Example result structure:**
```python
{
    'fold_idx': 0,
    'session': '0train',
    'subject': 1,  # ✓ Correctly set to self.current_subject
    'score': 0.85,
    'validation_roc_auc': 0.85,
    'validation_accuracy': 0.80,
    ...
}
```

---

### 5. Result Compilation (Lines 395-405)
```python
for i, result in enumerate(all_results):
    all_results[i].update({
        'fold_idx': fold_idx,
        'cv_type': cv_metadata['cv_type'],
        'split_level': cv_metadata['split_level'],
        'session': session,
        'subject': self.current_subject  # ✓ Use current subject, not subjects[0]
    })
```

**Key Points:**
- Each result gets enriched with metadata
- Subject ID correctly reflects the subject being evaluated
- Session and fold_idx correctly identify which data was used

---

### 6. DataFrame Creation (Line 909)
```python
results_df = pd.DataFrame(all_results)
```

**Example DataFrame:**
```
   fold_idx  session  subject  score  validation_roc_auc  ...
0         0   0train        1   0.85                0.85
1         1   0train        1   0.87                0.87
2         2   0train        1   0.84                0.84
3         3   0train        1   0.86                0.86
4         4   0train        1   0.88                0.88
5         0    1test        1   0.82                0.82
6         1    1test        1   0.83                0.83
7         2    1test        1   0.81                0.81
8         3    1test        1   0.84                0.84
9         4    1test        1   0.85                0.85
```

---

### 7. Aggregation by Session (Lines 996-1018)
```python
agg_results = []
for session in results_df['session'].unique():
    session_df = results_df[results_df['session'] == session]
    if len(session_df) > 0:
        agg_row = {
            'subject': session_df['subject'].iloc[0],
            'session': session,
            'score': session_df['validation_score'].mean(),  # Mean across folds
            'model': self.model,
            'mode': self.mode,
            'eval_mode': self.eval_mode,
            'seed': self.seed,
            'tune': self.tune
        }
        agg_results.append(agg_row)

return pd.DataFrame(agg_results)
```

**Example Aggregated DataFrame:**
```
   session  subject  score  model    mode      eval_mode  seed  tune
0   0train        1  0.860  eegnet  baseline  WithinSession  42  False
1    1test        1  0.830  eegnet  baseline  WithinSession  42  False
```

**Key Points:**
- Results grouped by session
- Mean score calculated across 5 folds per session
- Each row represents one session's average performance
- Subject ID correctly preserved from fold results

---

## Special Case: test_perturb Mode

For `test_perturb` mode, the aggregation is more complex (lines 958-992):

```python
for intensity in results_df['intensity'].unique():
    intensity_df = results_df[results_df['intensity'] == intensity]
    for session in intensity_df['session'].unique():
        session_df = intensity_df[intensity_df['session'] == session]
        agg_row = {
            'subject': session_df['subject'].iloc[0],
            'session': session,
            'score': session_df['corrupted_score'].mean(),
            'clean_score': session_df['clean_score'].mean(),
            'corrupted_score': session_df['corrupted_score'].mean(),
            'relative_drop': session_df['relative_drop'].mean(),
            'noise_type': noise_type,
            'intensity': intensity,
            ...
        }
        agg_results.append(agg_row)
```

**Key Points:**
- Groups by both intensity AND session
- Calculates means for clean_score, corrupted_score, and relative_drop
- One row per (session, noise_type, intensity) combination

---

## Verification Checklist

✓ **Fold metadata**: Uses enumeration index `i`, not `fold_idx`  
✓ **Index adjustment**: Separate indices for global data vs session metadata  
✓ **Subject recording**: Uses `self.current_subject`, not `subjects[0]`  
✓ **Current subject**: Set correctly before each fold evaluation  
✓ **Aggregation**: Groups by session and calculates means correctly  
✓ **Data alignment**: Metadata matches actual training data  

---

## Summary

The data flow is now correct:
1. Subject loop sets `current_subject`
2. Session loop filters data and creates fold splits
3. Fold evaluation uses correct metadata (via enumeration index)
4. Results record correct subject ID (via `current_subject`)
5. Aggregation groups by session and calculates fold means

All fold results are compiled correctly, with proper metadata alignment and subject tracking.

