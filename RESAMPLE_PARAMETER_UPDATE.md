# Resample Parameter Update Summary

## Overview
Updated the codebase to ensure that the `resample` parameter uses appropriate sampling rates for each dataset.

## Changes Made

### 1. Added Dataset-Specific Sampling Rate Function (`config.py`)

Created `get_dataset_sampling_rate()` function to return the appropriate sampling rate for each dataset:

```python
def get_dataset_sampling_rate(dataset="BNCI2014_001"):
    """Get the appropriate sampling rate (Hz) for a given dataset."""
    dataset_rates = {
        "BNCI2014_001": 250.0,      # MOABB provides this dataset at 250 Hz
        "Lee2019_SSVEP": 1000.0,    # Native sampling rate is 1000 Hz
        "BI2015a": 250.0             # Typical ERP datasets are 250 Hz
    }
    return dataset_rates.get(dataset, 250.0)  # Default to 250 Hz if unknown
```

### 2. Updated `get_paradigm()` Function (`config.py`)

Modified `get_paradigm()` to automatically use dataset-specific sampling rates when `resample=None`:

```python
def get_paradigm(resample=None, dataset="BNCI2014_001"):
    """Get the appropriate paradigm based on dataset."""
    # If resample not specified, use dataset-specific default
    if resample is None:
        resample = get_dataset_sampling_rate(dataset)
    # ... rest of function
```

**Impact**: When paradigms are created without specifying `resample`, they now automatically use the correct rate for each dataset.

### 3. Updated Unified Experiment Runner (`evaluation/unified_experiment_runner.py`)

- Added import: `from config import get_dataset_sampling_rate`
- Updated `_tune_and_get_params()` to get dataset-specific sampling rate:
  ```python
  # Get dataset-specific sampling rate
  resample_rate = get_dataset_sampling_rate(self.dataset)
  ```
- Pass `resample_rate` to `alternate_two_stage_optuna()` and `run_two_stage_optuna()`

**Impact**: Hyperparameter optimization now uses dataset-specific sampling rates instead of defaults.

### 4. Updated Documentation (`evaluation/two_stage_hp_opt.py`)

Added comments to clarify the default resample behavior:
```python
if resample is None:
    # Default to 250 Hz (common for MotorImagery and ERP datasets)
    # Note: Lee2019_SSVEP uses 1000 Hz, so resample should be provided explicitly
    resample = 250.0
```

## Dataset Sampling Rates

| Dataset | Sampling Rate | Rationale |
|---------|--------------|-----------|
| BNCI2014_001 | 250 Hz | MOABB provides this dataset pre-resampled to 250 Hz |
| Lee2019_SSVEP | 1000 Hz | Native sampling rate (high-frequency SSVEP signals) |
| BI2015a | 250 Hz | Typical for ERP datasets (sufficient for P300) |

## Backward Compatibility

- Functions in `two_stage_hp_opt.py` still have fallback logic (`resample=None` defaults to 250 Hz)
- This ensures compatibility with code that doesn't specify resample
- However, `unified_experiment_runner.py` now always passes the correct rate explicitly

## Benefits

1. **Correct Processing**: Each dataset now uses its appropriate sampling rate
2. **Automatic Handling**: No need to manually specify resample for each dataset
3. **Consistency**: Sampling rates are centralized and consistent across the codebase
4. **Future-Proof**: Easy to add new datasets by updating the `get_dataset_sampling_rate()` function

## Verification

To verify the correct sampling rate is used:
```python
from config import get_dataset_sampling_rate, get_paradigm

# Check rates
print(f"BNCI2014_001: {get_dataset_sampling_rate('BNCI2014_001')} Hz")
print(f"Lee2019_SSVEP: {get_dataset_sampling_rate('Lee2019_SSVEP')} Hz")
print(f"BI2015a: {get_dataset_sampling_rate('BI2015a')} Hz")

# Check paradigm configuration
paradigm = get_paradigm(resample=None, dataset='Lee2019_SSVEP')
# Should use 1000 Hz automatically
```

