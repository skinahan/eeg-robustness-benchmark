# Final Wiring Fix Summary

## Problem Statement

Wired CfC models created from ArbitraryWiring had ~10.68x more parameters than CNN-NCP (67,775 vs 6,346), when they should be similar given sparse wiring architectures.

## Root Causes Identified

### 1. Wrong `units` Definition
- **Issue**: Legacy code set `units = input_size + hidden_size + output_size` (90)
- **Fix**: Changed to `units = hidden_size + output_size` (74, then 59)
- **Impact**: Reduced parameters significantly

### 2. Wiring Matrix Interpretation Error
- **Issue**: JSON file contains full [I+H+O, I+H+O] matrix (58x58), but code treated it as hidden-only [H, H]
- **Impact**: Created wiring with 58 hidden neurons instead of 43
- **Fix**: Extract hidden portion: `wiring_matrix[I:I+H, I:I+H]`

### 3. Connection Generation Instead of Preservation
- **Issue**: `WsFlexHiddenWiring.full_wiring_matrix()` generates NEW I->H and H->O connections using strategies
- **Impact**: Lost the original optimized connections from architecture search
- **Fix**: Create `ArbitraryWiring` directly from full matrix to preserve original connections

### 4. Size Mismatch Handling
- **Issue**: Architecture file has I=8, O=7, but model needs I=16, O=16
- **Solution**: Use linear projection layers (I=16→8, O=7→16) instead of adapting wiring matrix
- **Benefit**: Preserves exact wiring graph while handling size differences

## Fixes Implemented

### 1. Fixed `ArbitraryWiring` Class
- Correct `units` calculation: `hidden_size + output_size` (not including inputs)
- Split wiring matrix correctly into `sensory_adjacency_matrix` and `adjacency_matrix`
- Fixed layer indexing to be relative to `units`
- Added `use_legacy_behavior` flag for testing

### 2. Fixed `load_architecture_from_file`
- Detects if wiring_matrix is full [I+H+O, I+H+O] or hidden-only [H, H]
- If full, creates `ArbitraryWiring` directly (preserves original connections)
- If hidden-only, uses `WsFlexHiddenWiring` (original behavior)

### 3. Added Projection Layers
- `input_proj`: Projects from model input size (F2=16) to wiring input size (8)
- `output_proj`: Projects from wiring output size (7) to model output size (16)
- Preserves exact wiring graph while handling size mismatches

## Results

### Parameter Counts:
- **Original (before fixes)**: 67,775 parameters (10.68x CNN-NCP)
- **After fix #1 (units)**: 54,079 parameters (8.5x CNN-NCP)
- **After fix #2 (hidden extraction)**: 37,039 parameters (5.84x CNN-NCP)
- **After fix #3 (preserve connections + projections)**: 27,139 parameters (4.28x CNN-NCP)
- **CNN-NCP baseline**: 6,346 parameters

### Reduction:
- **60% reduction** from original (67,775 → 27,139)
- **14% reduction** from legacy (31,427 → 27,139)

## Remaining Difference

The 4.28x difference vs CNN-NCP is due to:

1. **Architecture 4 is genuinely larger**:
   - Units: 59 vs 32 (1.84x)
   - Connections: ~750 vs 115 (6.5x)
   - This is the actual size of the optimized architecture

2. **LSTM component scales with state_size**:
   - Architecture 4: state_size=59 → LSTM ~22K params
   - AutoNCP: state_size=32 → LSTM ~6K params
   - Ratio: ~3.7x (matches state_size ratio)

3. **Different layer structures**:
   - Architecture 4: 2 layers (43, 16)
   - AutoNCP: 3 layers (15, 9, 8)
   - May affect parameter efficiency

## Key Insights

1. **Preserve the optimized wiring graph exactly** - don't regenerate connections
2. **Use projection layers for size mismatches** - simpler and preserves wiring
3. **Understand ncps conventions** - `units` = internal neurons only, inputs are external
4. **Wiring matrix format matters** - full matrix vs hidden-only requires different handling

## Files Modified

1. `architecture_refinement/arbitrary_wiring.py`:
   - Fixed `units` calculation
   - Fixed wiring matrix splitting
   - Fixed layer indexing
   - Added legacy flag
   - Fixed `load_architecture_from_file` to handle full matrices

2. `models/branched_wiredcfc.py`:
   - Added projection layers for size mismatches
   - Updated `_create_recurrent_cell` to preserve original wiring sizes
   - Updated `_process_bins` to use projections

## Testing

Run `test_wiring_fix.py` to compare:
- Correct behavior: Uses original wiring sizes with projections
- Legacy behavior: Uses adapted wiring matrix (for comparison)
- CNN-NCP: Baseline for comparison

## Conclusion

The wiring fix is **working correctly**. We've achieved a 60% reduction in parameters by:
1. Correctly interpreting the wiring matrix
2. Preserving the exact optimized wiring graph
3. Using projection layers for size mismatches

The remaining 4.28x difference appears to be due to Architecture 4 being genuinely larger than AutoNCP, which is expected given the architecture search process found a larger, more complex wiring structure.
