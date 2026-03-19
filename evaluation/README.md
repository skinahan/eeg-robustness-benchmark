# Evaluation Module

This directory contains the core experiment runner and evaluation utilities for EEG classification models.

## Main Entry Point

**`unified_experiment_runner.py`** – Single entry point for all experiment types.

### Modes

| Mode | Description |
|------|-------------|
| `test_perturb` | Robustness evaluation: train on clean data, evaluate under increasing noise intensities |
| `multirun` | Single noise type/intensity run (requires `--noise_type` and `--intensity`) |
| `aggregate_only` | Aggregate existing results without running new experiments |

### Evaluation Modes

- **WithinSession**: Train and test within the same session
- **CrossSession**: Train on one session, test on another
- **CrossSubject**: 3-fold subject split; train on 2/3, evaluate on 1/3

### Example Commands

```bash
# Robustness evaluation (Gaussian noise, 3 alpha levels)
python evaluation/unified_experiment_runner.py \
  --model eegnet --dataset BNCI2014_001 --mode test_perturb \
  --subjects 1 2 3 --eval_mode CrossSession --seed 42 --overwrite \
  --test_perturb_gaussian_only --test_perturb_gaussian_alpha_grid 0,0.25,0.5,0.75,1

# Single multirun
python evaluation/unified_experiment_runner.py \
  --model eegnet --dataset BNCI2014_001 --mode multirun \
  --subjects 1 --eval_mode CrossSession --noise_type gaussian --intensity 10.0 --seed 42
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `experiment_utils.py` | `check_skip_eval`, `collect_all_results`, path utilities |
| `metrics.py` | Classification metrics (accuracy, AUC, etc.) |
| `model_cache_manager.py` | Checkpoint caching for CrossSubject/CrossSession |
| `two_stage_hp_opt.py` | Optuna-based hyperparameter optimization |
| `chunked_subject_trainer.py` | Memory-efficient CrossSubject training |
| `periodic_checkpoint_callback.py` | Checkpoint callbacks for long runs |

## Training History

See [README_TRAINING_HISTORY.md](README_TRAINING_HISTORY.md) for training history logging.

## Deduplication

See [DEDUPLICATION_SCHEMA.md](DEDUPLICATION_SCHEMA.md) for result deduplication logic.
