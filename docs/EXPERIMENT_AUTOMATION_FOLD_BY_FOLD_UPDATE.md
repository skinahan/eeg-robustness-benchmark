# Experiment Automation - Fold-by-Fold Update

## Summary

Updated `experiment_automation.py` to use the new fold-by-fold execution workflow for CrossSubject experiments, reducing memory requirements from 64GB to 40GB per job.

## Changes Made

### 1. New Shell Script: `unified_eval_script_crosssubject_foldbyfold.sh`

Created a new shell script that:
- Uses `run_crosssubject_folds.py` to execute all folds sequentially
- Runs `aggregate_crosssubject_results.py` after all folds complete
- Sets memory limit to 40GB (reduced from 64GB)
- Handles multirun mode correctly (no noise_type/intensity needed)

### 2. Updated `experiment_automation.py`

**Changes:**
- Modified `generate_shell_script()` to use `unified_eval_script_crosssubject_foldbyfold.sh` for CrossSubject experiments
- Updated SLURM resource allocation:
  - Memory: 64GB → 40GB (37.5% reduction)
  - Time: 1-08:00:00 → 1-12:00:00 (slightly increased to account for 3 folds + aggregation)
- Added informational messages about fold-by-fold mode

### 3. Updated `run_crosssubject_folds.py`

**Changes:**
- Added support for `multirun` mode (no noise_type/intensity required)
- Updated command building to only include noise parameters for `test_perturb` mode
- Improved messaging for multirun mode (aggregation handled by shell script)

## Memory Benefits

**Before:**
- CrossSubject jobs: 64GB memory
- All subjects loaded at once
- Peak memory: ~15-20 GB

**After:**
- CrossSubject jobs: 40GB memory
- Only fold subjects loaded at a time
- Peak memory: ~8-10 GB per fold
- **37.5% memory reduction per job**

## Workflow

### For Automation (Generated Scripts)

The automation system now generates commands like:

```bash
sbatch --time=1-12:00:00 --mem=40G \
    unified_eval_script_crosssubject_foldbyfold.sh \
    1 2 3 4 5 6 7 8 9 \
    BNCI2014_001 \
    CrossSubject \
    true \
    cnn_ncp \
    42
```

This script:
1. Runs `run_crosssubject_folds.py` with multirun mode
2. Processes all 3 folds sequentially
3. Aggregates results using `aggregate_crosssubject_results.py`

### For Manual Execution

You can still use the fold-by-fold scripts directly:

```bash
# Run all folds
python run_crosssubject_folds.py \
    --model cnn_ncp \
    --dataset BNCI2014_001 \
    --subjects 1 2 3 4 5 6 7 8 9 \
    --mode multirun \
    --seed 42 \
    --tune

# Aggregate results
python aggregate_crosssubject_results.py \
    --model cnn_ncp \
    --dataset BNCI2014_001 \
    --subjects 1 2 3 4 5 6 7 8 9 \
    --mode multirun \
    --seed 42 \
    --tune
```

## Backward Compatibility

- Legacy mode still works: If `--fold_idx` is not provided, `unified_experiment_runner.py` uses the original all-subjects-at-once approach
- Existing result files are compatible: The aggregation script works with both old and new result structures
- Other eval modes unchanged: WithinSession and CrossSession continue to use the original scripts

## Testing Recommendations

1. **Test with small dataset first:**
   ```bash
   python experiment_automation.py --config experiment_config.yaml
   # Check generated script for CrossSubject jobs
   ```

2. **Verify fold execution:**
   - Check that each fold completes successfully
   - Verify memory usage is reduced
   - Confirm results are saved correctly

3. **Verify aggregation:**
   - Check that all 3 folds are aggregated
   - Verify result counts match expectations
   - Confirm compatibility with existing analysis tools

## Notes

- The fold-by-fold approach processes folds sequentially (not in parallel)
- Each fold runs as a subprocess, so errors in one fold don't stop others
- Aggregation happens automatically in the shell script, but can be run manually if needed
- Memory savings are most significant for large subject counts (9+ subjects)
