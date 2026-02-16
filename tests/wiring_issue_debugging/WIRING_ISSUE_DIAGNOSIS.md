# Wiring Construction Issue Diagnosis

## Problem Summary

The BranchedWiredCfC model has ~10.68x more parameters than CNN-NCP (67,775 vs 6,346), when they should be similar. The issue is in how the wiring graph is being used to construct the CfC cell.

## Key Findings

### 1. Wiring Matrix is Correct

From `diagnose_wiring_issue.py`:
- Wiring matrix shape: `(90, 90)` (I=16, H=58, O=16)
- Total connections: 1,106 (13.65% density)
- Region breakdown:
  - Input->Hidden: 348 connections
  - Hidden->Hidden: 662 connections  
  - Hidden->Output: 96 connections
  - Invalid connections (Input->Output, Output->Input): 0 ✓

**The wiring matrix itself is correctly constructed.**

### 2. CfC Cell Creates Dense Weight Matrices

The problem is that CfC in "wired mode" is creating **dense weight matrices** instead of sparse ones:

```
CfC Cell Parameters:
  layer_0.ff1.weight:     [16, 32] = 512 params  (dense!)
  layer_1.ff1.weight:     [58, 74] = 4,292 params (dense! should be sparse)
  layer_2.ff1.weight:      [16, 74] = 1,184 params (dense!)
  lstm.recurrent_map.weight: [360, 90] = 32,400 params (HUGE! should be sparse)
```

**Total CfC parameters: 63,104** (93.1% of total model)

### 3. Expected vs Actual

**Expected behavior**: The wiring matrix should be used to create sparse weight matrices where only the connections specified in the wiring matrix have learnable parameters.

**Actual behavior**: CfC creates dense weight matrices and (presumably) uses the wiring matrix as a mask during forward pass. However, PyTorch still counts all parameters, even masked ones.

### 4. Parameter Breakdown

```
BranchedWiredCfC: 67,775 parameters
  - Recurrent cell: 63,104 (93.1%)
    - LSTM recurrent_map: 32,400 (47.8% of total!)
    - Layer 1 gates: ~17,000 (4 gates × 4,292 each)
    - Other layers: ~13,700
  - Front-end: ~4,671 (6.9%)

CNN-NCP: 6,346 parameters
  - Recurrent cell: ~3,000 (47%)
  - Front-end: ~3,346 (53%)
```

## Root Cause Analysis

### Hypothesis 1: Wiring Matrix Format Mismatch

The `ArbitraryWiring` class may not be providing the wiring in the format that CfC's wired mode expects. CfC might expect:
- A different matrix structure
- Sparse matrix format instead of dense
- Different indexing scheme

### Hypothesis 2: CfC Wired Mode Doesn't Create Sparse Matrices

The CfC library's "wired mode" might work by:
1. Creating dense weight matrices
2. Applying wiring matrix as a mask during forward pass
3. But still storing all parameters

This would explain why parameter count is high even though `wired_mode=True`.

### Hypothesis 3: LSTM Component Not Using Wiring

The LSTM component (`lstm.recurrent_map`) has 32,400 parameters, which is:
- `360 × 90 = 32,400` (dense matrix)
- Should be sparse based on wiring

The LSTM might not be respecting the wiring sparsity at all.

## Evidence

1. **Wiring matrix is correct**: 1,106 connections, properly structured
2. **CfC reports `wired_mode=True`**: So it recognizes the wiring
3. **But creates dense matrices**: All weight matrices are dense
4. **LSTM is the biggest problem**: 32,400 params vs expected ~1,000-2,000

## Next Steps for Investigation

1. **Check CfC source code**: Understand how wired mode actually works
2. **Check if wiring is applied as masks**: Look for mask buffers in CfC cell
3. **Compare with CNN-NCP wiring**: See how CNN-NCP creates sparse parameters
4. **Check LSTM wiring**: Investigate why LSTM component ignores wiring

## Potential Solutions

1. **Fix wiring format**: Ensure `ArbitraryWiring` provides wiring in format CfC expects
2. **Use sparse matrices**: Create actual sparse PyTorch matrices instead of dense + mask
3. **Fix LSTM wiring**: Ensure LSTM component respects wiring sparsity
4. **Post-process parameters**: Prune masked parameters after model creation (not ideal)

## Files to Investigate

- `ncps/torch/cfc.py` - CfC implementation (external library)
- `architecture_refinement/arbitrary_wiring.py` - Our wiring class
- `models/branched_wiredcfc.py` - How we use wiring
- `models/cnnncp.py` - How CNN-NCP uses wiring (for comparison)
