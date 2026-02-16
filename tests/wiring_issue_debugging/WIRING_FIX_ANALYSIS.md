# Wiring Fix Analysis

## Current Status

After implementing the fix, we've achieved:
- **Correct behavior**: 54,079 parameters (8.5x CNN-NCP)
- **Legacy behavior**: 67,775 parameters (10.68x CNN-NCP)
- **Reduction**: ~20% fewer parameters with correct implementation

## What's Working

1. ✅ **Wiring matrices are correctly split**:
   - `sensory_adjacency_matrix`: [16, 74] with 348 connections (29.39% density)
   - `adjacency_matrix`: [74, 74] with 758 connections (13.84% density)

2. ✅ **Sparsity masks are correctly constructed**:
   - Layer 0: 37.50% density for input→hidden connections
   - Layer 1: 10.34% density for hidden→output connections
   - Self-connections are correctly dense (100%)

3. ✅ **Layer indexing is correct**:
   - Layer 0: Hidden neurons (indices 0-57)
   - Layer 1: Motor neurons (indices 58-73)

## Remaining Issues

### 1. PyTorch Stores Dense Matrices

The fundamental issue is that **PyTorch's `nn.Linear` creates dense weight matrices**, and the sparsity mask is applied during forward pass by element-wise multiplication. This means:
- All parameters are still stored in memory
- Parameter counting includes masked (zero) parameters
- The sparsity only affects computation, not storage

**This is how ncps works** - it's not a bug, it's a design choice. The sparsity masks reduce computation but don't reduce parameter storage.

### 2. LSTM Component is Dense

The LSTM component (`mixed_memory=True`) doesn't use wiring sparsity at all:
- `lstm.input_map`: [296, 16] = 4,736 parameters (dense)
- `lstm.recurrent_map`: [296, 74] = 21,904 parameters (dense)
- Total LSTM: ~26,700 parameters

This is expected behavior - the LSTM is a separate dense component that doesn't respect wiring.

### 3. Parameter Count Comparison

The remaining ~8.5x difference vs CNN-NCP is due to:

1. **LSTM component** (~26,700 params) - CNN-NCP also has this, but smaller
2. **Additional model components** in BranchedWiredCfC:
   - Multi-scale temporal block (~1,680 params)
   - SNR gate (~192 params)
   - Attention pooling (~512 params)
   - These don't exist in CNN-NCP

3. **CfC layer structure**:
   - Each layer has 4 linear layers (ff1, ff2, time_a, time_b)
   - Even with sparsity masks, all parameters are stored

## Expected vs Actual

### CNN-NCP Structure
- Simple CNN front-end
- Single CfC with AutoNCP wiring
- No multi-scale blocks, SNR gates, etc.
- Total: ~6,346 parameters

### BranchedWiredCfC Structure
- DIVA front-end with multi-scale temporal blocks
- SNR gate
- Branched CfC processing with attention pooling
- Total: ~54,079 parameters (correct) vs ~67,775 (legacy)

## Recommendations

### 1. For Parameter Counting

Use `count_sparse_parameters.py` which attempts to count only non-masked parameters. However, this is heuristic-based and may not be 100% accurate.

### 2. For Fair Comparison

When comparing to CNN-NCP, consider:
- CNN-NCP is a simpler architecture
- BranchedWiredCfC has additional components (multi-scale, SNR gate, attention)
- The ~8.5x difference is partially due to architectural differences, not just wiring issues

### 3. For Further Optimization

If you want to reduce parameters further:
- Consider disabling `mixed_memory` (removes LSTM, saves ~26K params)
- Use actual sparse PyTorch matrices (requires custom implementation)
- Prune masked parameters after training (post-processing)

## Conclusion

The wiring fix is **working correctly**. The sparsity masks are properly constructed and applied. The remaining parameter count difference is due to:
1. PyTorch storing dense matrices (by design)
2. Additional model components in BranchedWiredCfC
3. LSTM component being dense

The 20% reduction from legacy to correct behavior confirms the fix is working. The remaining ~8.5x difference vs CNN-NCP is expected given the architectural differences.
