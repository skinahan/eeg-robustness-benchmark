# CrossSubject Memory Optimization Recommendation

## Problem Analysis

### Current Memory Bottlenecks

1. **Loading all subjects at once** (Line 1152 in `unified_experiment_runner.py`):
   - Loads ALL subjects' data: `X, y, metadata = self.paradigm.get_data(self.dataset_obj, subjects=self.subjects)`
   - For 9 subjects with shape (5760, 62, 4001): ~11.6 GB (float64) or ~5.8 GB (float32)
   - This data stays in memory for the entire experiment

2. **Creating train/valid splits** (Lines 1210-1213):
   - `X_train = X[train_idx]` creates a copy of ~7.7 GB (2/3 of data)
   - `X_valid = X[valid_idx]` creates a copy of ~3.9 GB (1/3 of data)
   - Even with SliceDataset, we still need the full dataset in memory

3. **Hyperparameter optimization**:
   - Additional CV splits within each fold create more copies
   - Multiple trials multiply memory usage

### Memory Footprint Estimate

For 9 subjects with shape (5760, 62, 4001):
- **Full dataset (float64)**: 11.6 GB
- **Full dataset (float32)**: 5.8 GB
- **Train split (float32)**: ~3.9 GB
- **Valid split (float32)**: ~1.9 GB
- **HPO internal CV**: Additional 1-2 GB per trial
- **Total peak memory**: ~10-15 GB (even with optimizations)

## Recommended Solution: Fold-by-Fold Execution

### Core Strategy

**Split each CrossSubject fold into a separate run of `unified_experiment_runner.py`**, rather than processing all folds in one run.

### Implementation Approach

#### Option 1: Two-Phase Execution (Recommended)

**Phase 1: Fold Planning Script**
- Create a new script `plan_crosssubject_folds.py` that:
  - Determines which subjects go in each fold (using `ThreeFoldSubjectSplit` logic)
  - Generates separate command-line invocations for each fold
  - Saves fold configuration to a JSON file

**Phase 2: Per-Fold Execution**
- Modify `unified_experiment_runner.py` to accept a `--fold_idx` parameter
- When `fold_idx` is specified:
  - Load ONLY the subjects needed for that fold
  - Skip fold iteration, process only the specified fold
  - Save results with fold identifier

**Phase 3: Aggregation**
- After all folds complete, run an aggregation script that:
  - Collects results from all fold runs
  - Combines them into a single results DataFrame
  - Maintains compatibility with existing analysis pipelines

#### Option 2: Single Script with Fold Selection

Modify `unified_experiment_runner.py` to:
- Accept `--fold_idx` and `--train_subjects` / `--eval_subjects` parameters
- When these are provided, load only the specified subjects
- Process only that fold
- Can be called multiple times from a wrapper script

### Memory Benefits

**Before (Current)**:
- Load all 9 subjects: 11.6 GB
- Create train split: +3.9 GB (copy)
- Create valid split: +1.9 GB (copy)
- **Peak memory**: ~15-20 GB

**After (Fold-by-Fold)**:
- Load only 6 train subjects: ~7.7 GB
- Load only 3 eval subjects: ~3.9 GB (can be loaded separately)
- **Peak memory**: ~8-10 GB (50% reduction)

### Implementation Details

#### 1. Modify `ThreeFoldSubjectSplit` to Export Fold Configuration

```python
def get_fold_subjects(self, subjects):
    """Return which subjects go in each fold."""
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    eval_group_size = n_subjects // 3
    
    fold_configs = []
    for fold_idx in range(3):
        eval_start = fold_idx * eval_group_size
        eval_end = eval_start + eval_group_size
        
        eval_subjects = unique_subjects[eval_start:eval_end].tolist()
        train_subjects = np.concatenate([
            unique_subjects[:eval_start],
            unique_subjects[eval_end:]
        ]).tolist()
        
        fold_configs.append({
            'fold_idx': fold_idx,
            'train_subjects': train_subjects,
            'eval_subjects': eval_subjects
        })
    
    return fold_configs
```

#### 2. Add Fold Parameters to `unified_experiment_runner.py`

```python
parser.add_argument("--fold_idx", type=int, default=None,
                    help="Specific fold to process (for CrossSubject mode)")
parser.add_argument("--train_subjects", type=int, nargs="+", default=None,
                    help="Training subjects for this fold (for CrossSubject mode)")
parser.add_argument("--eval_subjects", type=int, nargs="+", default=None,
                    help="Evaluation subjects for this fold (for CrossSubject mode)")
```

#### 3. Modify CrossSubject Flow

```python
if self.eval_mode == "CrossSubject":
    if self.fold_idx is not None:
        # Fold-by-fold mode: Load only needed subjects
        all_subjects = self.train_subjects + self.eval_subjects
        X, y, metadata = self.paradigm.get_data(
            self.dataset_obj, subjects=all_subjects
        )
        
        # Split into train/valid based on subject IDs
        train_mask = metadata['subject'].isin(self.train_subjects)
        valid_mask = metadata['subject'].isin(self.eval_subjects)
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        # Process single fold
        fold_results = self._evaluate_cv_fold(
            X_train, y_train, X_valid, y_valid, 
            self.fold_idx, cv_metadata, session, metadata_train
        )
    else:
        # Legacy mode: Load all subjects (for backward compatibility)
        # ... existing code ...
```

#### 4. Create Wrapper Script

```python
# run_crosssubject_folds.py
import subprocess
import json
from evaluation.unified_experiment_runner import ThreeFoldSubjectSplit

def main():
    # Determine fold configuration
    splitter = ThreeFoldSubjectSplit()
    all_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Your subjects
    fold_configs = splitter.get_fold_subjects(all_subjects)
    
    # Run each fold separately
    for fold_config in fold_configs:
        cmd = [
            "python", "evaluation/unified_experiment_runner.py",
            "--model", "cnn_ncp",
            "--dataset", "BNCI2014_001",
            "--subjects"] + [str(s) for s in all_subjects] + [
            "--mode", "tune",
            "--eval_mode", "CrossSubject",
            "--fold_idx", str(fold_config['fold_idx']),
            "--train_subjects"] + [str(s) for s in fold_config['train_subjects']] + [
            "--eval_subjects"] + [str(s) for s in fold_config['eval_subjects']],
            "--seed", "42"
        ]
        
        print(f"Running fold {fold_config['fold_idx']}...")
        subprocess.run(cmd, check=True)
    
    # Aggregate results
    aggregate_results(fold_configs)

if __name__ == "__main__":
    main()
```

### Benefits

1. **50-60% memory reduction**: Only load subjects needed for current fold
2. **Better parallelization**: Can run folds in parallel on different nodes
3. **Fault tolerance**: If one fold fails, others can still complete
4. **Scalability**: Can handle larger subject counts without OOM
5. **Backward compatibility**: Legacy mode still works for smaller datasets

### Trade-offs

1. **Additional complexity**: Need wrapper script and fold management
2. **Multiple runs**: Requires multiple invocations (but can be automated)
3. **Result aggregation**: Need to combine results from multiple runs
4. **Cache management**: Model caching needs to account for fold_idx

### Migration Path

1. **Phase 1**: Implement fold-by-fold mode alongside existing code
2. **Phase 2**: Test with small datasets to verify correctness
3. **Phase 3**: Use fold-by-fold mode for large CrossSubject experiments
4. **Phase 4**: Keep legacy mode for smaller datasets or backward compatibility

### Alternative: Hybrid Approach

For very large datasets, consider:
- **Chunked loading**: Load subjects in batches, process, then load next batch
- **Memory-mapped arrays**: Use numpy.memmap for large arrays
- **Streaming**: Process data in smaller chunks during training

However, fold-by-fold execution is the most straightforward and effective solution.

## Recommendation

**Implement Option 1 (Two-Phase Execution)** because:
1. Clean separation of concerns
2. Easy to test and debug
3. Maintains backward compatibility
4. Provides maximum memory savings
5. Enables parallel execution

This approach will reduce memory requirements from ~15-20 GB to ~8-10 GB, making CrossSubject experiments feasible on systems with limited memory.
