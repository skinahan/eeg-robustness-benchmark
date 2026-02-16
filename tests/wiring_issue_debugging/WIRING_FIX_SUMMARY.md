# Wiring Implementation Fix Summary

## Problem Identified

The `ArbitraryWiring` class was creating dense weight matrices instead of sparse ones, resulting in ~10.68x more parameters than expected (67,775 vs 6,346 for CNN-NCP).

## Root Cause

The ncps library expects wiring to be structured as:
- **`units`**: Internal neurons only (hidden + output), NOT including inputs
- **`sensory_adjacency_matrix`**: `[input_dim, units]` - connections from external inputs to internal neurons
- **`adjacency_matrix`**: `[units, units]` - connections between internal neurons

Our `ArbitraryWiring` was incorrectly:
- Setting `units = input_size + hidden_size + output_size` (including inputs)
- Using a single `wiring_matrix` of shape `[I+H+O, I+H+O]` without splitting it
- Not properly populating `sensory_adjacency_matrix` and `adjacency_matrix`

## Solution Implemented

### 1. Fixed `units` Definition
- **Correct**: `units = hidden_size + output_size` (internal neurons only)
- **Legacy**: `units = input_size + hidden_size + output_size` (incorrect)

### 2. Split Wiring Matrix Correctly
- **`sensory_adjacency_matrix`**: Extracted from `wiring_matrix[0:I, I:I+H+O]` (I→H and I→O connections)
- **`adjacency_matrix`**: Extracted from `wiring_matrix[I:I+H+O, I:I+H+O]` (H→H, H→O connections)

### 3. Fixed Layer Indexing
- **Correct**: 
  - Layer 0: Hidden neurons (indices 0..H-1 within units)
  - Layer 1: Motor neurons (indices H..H+O-1 within units)
- **Legacy**: Incorrectly included inputs as layer 0

### 4. Added Legacy Flag
- `use_legacy_behavior=False` (default): Uses correct sparse implementation
- `use_legacy_behavior=True`: Restores old incorrect behavior for testing/comparison

## Changes Made

### Files Modified
1. **`architecture_refinement/arbitrary_wiring.py`**:
   - Updated `ArbitraryWiring.__init__()` to accept `use_legacy_behavior` flag
   - Fixed `units` calculation (H+O instead of I+H+O)
   - Added `_build_wiring_from_matrix_correct()` method
   - Renamed old method to `_build_wiring_from_matrix_legacy()`
   - Updated `build()` to populate `sensory_adjacency_matrix` correctly
   - Fixed `get_neurons_of_layer()` to return correct indices
   - Fixed `get_type_of_neuron()` for correct indexing
   - Updated `WsFlexHiddenWiring.build()` to support legacy flag
   - Updated `load_architecture_from_file()` to support legacy flag

## Expected Results

After the fix:
- **Parameter count should be ~6,000-10,000** (similar to CNN-NCP)
- **Recurrent cell should use sparse matrices** based on wiring connections
- **LSTM component should respect wiring sparsity**

## Testing

Run `test_wiring_fix.py` to compare:
- Correct behavior (sparse): Should have ~6,000-10,000 parameters
- Legacy behavior (dense): Should have ~67,000 parameters (for comparison)
- CNN-NCP baseline: ~6,346 parameters

## Usage

### Default (Correct Behavior)
```python
wiring = load_architecture_from_file("path/to/architecture.json")
# Uses sparse matrices by default
```

### Legacy Behavior (For Testing)
```python
wiring = load_architecture_from_file("path/to/architecture.json", use_legacy_behavior=True)
# Uses dense matrices (incorrect, for comparison only)
```

## Technical Details

### How ncps Uses Wiring

1. **WiredCfCCell** extracts sparsity masks from wiring:
   - Layer 0: Uses `sensory_adjacency_matrix[:, hidden_units]`
   - Other layers: Uses `adjacency_matrix[prev_layer_neurons, hidden_units]`

2. **CfCCell** applies masks:
   - Creates dense weight matrices
   - Applies `sparsity_mask` by multiplying: `weight * mask`
   - Masked parameters are still stored but set to zero during forward pass

3. **The Fix**:
   - Properly splits the wiring matrix so masks are sparse
   - Results in fewer effective parameters (though PyTorch still counts all)
   - The sparse counting in `count_sparse_parameters.py` should now work correctly

## Next Steps

1. Run `test_wiring_fix.py` to verify the fix
2. Update `count_sparse_parameters.py` if needed to properly count sparse parameters
3. Re-run experiments with the corrected wiring
4. Compare results between correct and legacy behavior
