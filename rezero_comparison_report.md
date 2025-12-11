# ReZero Initialization Strategy Comparison

*Generated: 2025-12-05 12:24:21*

**Dataset**: BNCI2014_001
**Architecture**: Architecture 4
**Seeds**: 42, 123, 456

---

## Summary Statistics

### Backwards Rezero

- **Total measurements**: 180
- **Mean clean ROC-AUC**: 0.9512 ± 0.0146
- **Mean corrupted ROC-AUC**: 0.8006 ± 0.1392
- **Mean retention**: 84.18% ± 14.66%

**Performance by noise type**:

- **EOG**:
  - Mean ROC-AUC: 0.8848
  - Mean retention: 93.01%
  - Best retention: 99.88%
  - Worst retention: 76.07%

- **GAUSSIAN**:
  - Mean ROC-AUC: 0.8525
  - Mean retention: 89.66%
  - Best retention: 103.00%
  - Worst retention: 60.12%

- **DROPOUT**:
  - Mean ROC-AUC: 0.6644
  - Mean retention: 69.87%
  - Best retention: 94.27%
  - Worst retention: 39.39%

### Correct Rezero

- **Total measurements**: 180
- **Mean clean ROC-AUC**: 0.8874 ± 0.0532
- **Mean corrupted ROC-AUC**: 0.7384 ± 0.1371
- **Mean retention**: 83.53% ± 16.27%

**Performance by noise type**:

- **EOG**:
  - Mean ROC-AUC: 0.7826
  - Mean retention: 88.76%
  - Best retention: 109.04%
  - Worst retention: 50.25%

- **GAUSSIAN**:
  - Mean ROC-AUC: 0.7955
  - Mean retention: 89.83%
  - Best retention: 107.32%
  - Worst retention: 60.52%

- **DROPOUT**:
  - Mean ROC-AUC: 0.6371
  - Mean retention: 71.99%
  - Best retention: 95.98%
  - Worst retention: 45.73%

---

## Direct Comparison

### Performance at Key Intensity Levels

| Noise Type | Intensity | Backwards ReZero | Correct ReZero | Difference |
|------------|-----------|------------------|----------------|------------|
| eog | 25% | 94.33% | 86.76% | +7.56% |
| eog | 50% | 93.92% | 85.65% | +8.27% |
| eog | 75% | 91.89% | 85.72% | +6.17% |
| gaussian | 25% | 99.10% | 98.05% | +1.05% |
| gaussian | 50% | 89.95% | 92.86% | -2.91% |
| gaussian | 75% | 79.50% | 78.43% | +1.07% |
| gaussian | 100% | 82.85% | 77.30% | +5.55% |
| dropout | 25% | 72.93% | 76.77% | -3.84% |
| dropout | 50% | 63.35% | 63.44% | -0.10% |
| dropout | 75% | 51.19% | 52.70% | -1.52% |

### Overall Robustness Comparison

- **Backwards ReZero** mean retention: 84.18%
- **Correct ReZero** mean retention: 83.53%
- **Difference**: +0.65% (Backwards - Correct)

**Conclusion**: Backwards ReZero is **0.65% more robust** on average.

---

## Recommendations

Based on these results:

1. If backwards ReZero shows superior robustness, this suggests the accidental implementation may have beneficial properties.
2. If correct ReZero performs better, this supports implementing Fix Option 2.
3. Consider additional analysis of training dynamics and convergence speed.
