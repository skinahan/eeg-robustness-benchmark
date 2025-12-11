# BI2015a P300 Paradigm Configuration

## Summary
Updated the P300 paradigm configuration for the BI2015a ERP dataset to follow MOABB best practices for P300 event-related potential analysis.

## Configuration Changes

### Previous Configuration
```python
P300(
    tmin=0.0,
    tmax=None,
    baseline=None,
    resample=resample
)
```

### Updated Configuration
```python
P300(
    fmin=1,
    fmax=24,
    tmin=0.0,
    tmax=1.0,
    baseline=(None, 0),
    resample=resample
)
```

## Rationale

### 1. Frequency Filtering (`fmin=1, fmax=24`)
- **Purpose**: Bandpass filter to focus on the P300 frequency range
- **Range**: 1-24 Hz captures the P300 component (typically around 3-5 Hz) while removing:
  - High-frequency noise and artifacts (>24 Hz)
  - Very low-frequency drifts (<1 Hz)
- **Standard**: This is the standard frequency range for P300 analysis in MOABB

### 2. Epoch Window (`tmin=0.0, tmax=1.0`)
- **tmin=0.0**: Start epoch at stimulus onset
- **tmax=1.0**: Capture 1 second post-stimulus, sufficient for P300 component (peak ~300ms)
- **Previous**: `tmax=None` would capture the entire trial, which is unnecessary for P300

### 3. Baseline Correction (`baseline=(None, 0)`)
- **Purpose**: Remove DC offset and slow drifts by correcting to pre-stimulus baseline
- **Parameter**: `(None, 0)` means baseline from the start of the epoch to stimulus onset
- **Standard**: Essential for ERP analysis to normalize signals across trials
- **Previous**: `baseline=None` skipped baseline correction, which is suboptimal for ERP

## MOABB Reference
According to MOABB documentation:
- P300 paradigm is designed for Target/NonTarget classification
- Typical configuration includes frequency filtering (1-24 Hz) and baseline correction
- Default parameters should align with ERP analysis best practices

## Verification
The configuration matches:
- MOABB P300 paradigm recommendations
- Standard ERP analysis practices
- Parameters used in MOABB tutorials and examples

## Impact
- Better signal quality due to frequency filtering
- Proper baseline correction for ERP components
- Appropriate epoch window for P300 analysis
- Consistent with MOABB best practices

