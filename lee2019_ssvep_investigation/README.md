# Lee2019 SSVEP Robustness Investigation

This directory contains the complete investigation of suspiciously high performance and low noise sensitivity observed in the Lee2019_SSVEP dataset evaluation.

## Investigation Overview

**Problem**: Lee2019_SSVEP showed unexpectedly high performance (85% accuracy) with minimal degradation under noise perturbation (0% drop at 10% intensity), raising concerns about methodological issues.

**Resolution**: SSVEP paradigm is inherently more robust to noise than motor imagery paradigm due to signal characteristics and generation mechanisms.

## Files in This Directory

### Documentation
- `INVESTIGATION_SUMMARY.md` - Complete investigation summary and findings
- `TECHNICAL_EVIDENCE.md` - Detailed technical evidence and analysis
- `README.md` - This file

### Analysis Scripts
- `run_lee2019_debug.py` - Comprehensive debugging script for Lee2019_SSVEP evaluation
- `dataset_noise_scaling_analysis.py` - Detailed analysis of noise scaling across datasets
- `debug_channels.py` - Channel count verification script

### Related Documentation (moved from root)
- `lee2019_investigation_report.md` - Initial investigation report
- `WITHINSESSION_BUG_FIX_REPORT.md` - Related bug fix documentation
- `WITHINSESSION_METRICS_UPDATE.md` - Metrics update documentation

## Key Findings

1. **No Methodological Issues**: Class balance, signal quality, noise application, and model training are all correct
2. **Paradigm Robustness**: SSVEP is inherently more robust to noise than motor imagery
3. **Expected Behavior**: High performance and low noise sensitivity are expected for SSVEP paradigm
4. **Technical Verification**: Noise scaling method works correctly with consistent noise-to-signal ratios

## Quick Start

To reproduce the investigation:

```bash
# Run comprehensive debugging
python run_lee2019_debug.py

# Run noise scaling analysis
python dataset_noise_scaling_analysis.py

# Check channel counts
python debug_channels.py
```

## Investigation Timeline

1. **Initial Observation**: High saturation points and minimal performance drop
2. **Data Quality Check**: Verified class balance, signal characteristics, and data integrity
3. **Noise Method Analysis**: Confirmed noise application and scaling are working correctly
4. **Paradigm Comparison**: Identified inherent robustness differences between SSVEP and motor imagery
5. **Resolution**: Concluded that high performance is expected behavior for SSVEP paradigm

## Conclusions

The investigation revealed that the observed high performance and low noise sensitivity in Lee2019_SSVEP is **expected behavior** due to the inherent robustness of the SSVEP paradigm, not a methodological issue. This finding has important implications for:

- Paradigm-specific performance expectations
- Noise calibration requirements
- Evaluation criteria adjustments
- BCI application design considerations

## References

- Lee, S., et al. (2019). "EEG dataset and processing pipeline for SSVEP-based brain-computer interfaces"
- BNCI2014_001 dataset documentation
- MOABB framework documentation
- SSVEP paradigm literature on signal robustness
