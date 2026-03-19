# CrossSubject Fold-by-Fold Execution Guide

## Overview

The fold-by-fold execution system splits CrossSubject experiments into separate runs, one per fold, to reduce memory requirements by 50-60%. Each fold loads only the subjects needed for that fold.

## Quick Start

### Step 1: Run All Folds

```bash
python run_crosssubject_folds.py \
    --model cnn_ncp \
    --dataset BNCI2014_001 \
    --subjects 1 2 3 4 5 6 7 8 9 \
    --mode test_perturb \
    --seed 42 \
    --tune \
    --noise_type gaussian \
    --intensity 10.0
```

This will:
1. Determine which subjects go in each fold (using `ThreeFoldSubjectSplit`)
2. Run each fold separately as a subprocess
3. Show progress and summary

### Step 2: Aggregate Results

After all folds complete, combine the results:

```bash
python aggregate_crosssubject_results.py \
    --model cnn_ncp \
    --dataset BNCI2014_001 \
    --subjects 1 2 3 4 5 6 7 8 9 \
    --mode test_perturb \
    --seed 42 \
    --tune \
    --noise_type gaussian \
    --intensity 10.0
```

## Advanced Usage

### Run a Single Fold

To run only one fold (useful for debugging or resuming):

```bash
python run_crosssubject_folds.py \
    --model cnn_ncp \
    --dataset BNCI2014_001 \
    --subjects 1 2 3 4 5 6 7 8 9 \
    --mode test_perturb \
    --seed 42 \
    --fold_idx 0
```

### Save/Load Fold Configuration

Save fold configuration for later use:

```bash
python run_crosssubject_folds.py \
    --model cnn_ncp \
    --dataset BNCI2014_001 \
    --subjects 1 2 3 4 5 6 7 8 9 \
    --mode test_perturb \
    --seed 42 \
    --save_config fold_config.json
```

Load saved configuration:

```bash
python run_crosssubject_folds.py \
    --model cnn_ncp \
    --dataset BNCI2014_001 \
    --subjects 1 2 3 4 5 6 7 8 9 \
    --mode test_perturb \
    --seed 42 \
    --load_config fold_config.json
```

### Direct Fold Execution

You can also run a fold directly using `unified_experiment_runner.py`:

```bash
python evaluation/unified_experiment_runner.py \
    --model cnn_ncp \
    --dataset BNCI2014_001 \
    --subjects 1 2 3 4 5 6 7 8 9 \
    --mode test_perturb \
    --eval_mode CrossSubject \
    --seed 42 \
    --fold_idx 0 \
    --train_subjects 4 5 6 7 8 9 \
    --eval_subjects 1 2 3 \
    --tune
```

## Memory Benefits

**Before (Legacy Mode)**:
- Loads all 9 subjects: ~11.6 GB (float64) or ~5.8 GB (float32)
- Creates train/valid splits: Additional copies
- **Peak memory**: ~15-20 GB

**After (Fold-by-Fold Mode)**:
- Loads only 6 train + 3 eval subjects: ~7.7 GB + ~3.9 GB
- **Peak memory**: ~8-10 GB (50% reduction)

## How It Works

1. **Fold Planning**: `ThreeFoldSubjectSplit.get_fold_subjects()` determines which subjects go in each fold
2. **Per-Fold Execution**: Each fold runs as a separate process, loading only needed subjects
3. **Result Storage**: Each fold saves results to its own directory (compatible with existing structure)
4. **Aggregation**: `aggregate_crosssubject_results.py` collects and combines all fold results

## Backward Compatibility

The legacy mode (loading all subjects at once) still works if you don't provide `--fold_idx`:

```bash
python evaluation/unified_experiment_runner.py \
    --model cnn_ncp \
    --dataset BNCI2014_001 \
    --subjects 1 2 3 4 5 6 7 8 9 \
    --mode test_perturb \
    --eval_mode CrossSubject \
    --seed 42 \
    --tune
```

This is useful for smaller datasets where memory is not a concern.

## Troubleshooting

### Missing Fold Results

If aggregation fails because a fold is missing:

1. Check if the fold completed: Look for `outputs/{model}/{seed}/subject_*/session_fold_{idx}_*/results.csv`
2. Re-run the missing fold using `--fold_idx {idx}`
3. Re-run aggregation

### Memory Still Too High

If you still get OOM errors:

1. Ensure you're using fold-by-fold mode (check for `[MEMORY] Fold-by-fold mode` in logs)
2. Check that SliceDataset is being used (should see `[MEMORY] Using SliceDataset` in logs)
3. Consider reducing the number of subjects per fold
4. Use `PYTHON_MAX_MEMORY_GB` environment variable to monitor memory

### Parallel Execution

The wrapper script runs folds sequentially. To run folds in parallel:

1. Use `--fold_idx` to run specific folds in separate terminals/processes
2. Or modify `run_crosssubject_folds.py` to use `subprocess.Popen` with parallel execution
3. Make sure each process has enough memory allocated

## Example Workflow

```bash
# 1. Run all folds
python run_crosssubject_folds.py \
    --model cnn_ncp \
    --dataset BNCI2014_001 \
    --subjects 1 2 3 4 5 6 7 8 9 \
    --mode test_perturb \
    --seed 42 \
    --tune \
    --noise_type gaussian \
    --intensity 10.0

# 2. Verify all folds completed
ls outputs/cnn_ncp/42/subject_*/session_fold_*/results.csv

# 3. Aggregate results
python aggregate_crosssubject_results.py \
    --model cnn_ncp \
    --dataset BNCI2014_001 \
    --subjects 1 2 3 4 5 6 7 8 9 \
    --mode test_perturb \
    --seed 42 \
    --tune \
    --noise_type gaussian \
    --intensity 10.0

# 4. Results are now in the standard location and can be analyzed normally
```
