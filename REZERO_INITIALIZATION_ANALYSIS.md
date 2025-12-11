# ReZero Initialization Analysis

## Executive Summary

**CRITICAL FINDING**: The current implementation does **NOT** initialize the recurrent compartment to the identity function in ReZero style. The residual connection logic appears to be implemented backwards.

### Quick Reference

| Model | Formula | Init Value | At Init: Output = | Status |
|-------|---------|------------|-------------------|--------|
| `branched_diva_base.py` | `recurrent * (1-α) + residual * α` | 0.0 | **recurrent** ❌ | ReZero backwards |
| `diva_ncp.py` | `recurrent * (1-α) + residual * α` | random(0.1-0.9) | varies | Not ReZero |
| `diva_full.py` | `recurrent * (1-w) + residual * w` | 0.0 | **recurrent** ❌ | ReZero backwards |

**Expected for ReZero**: At initialization, `output = residual` (identity function)  
**Actual**: At initialization, `output = recurrent_output` (NOT identity)

### Affected Models (All inherit from `BranchedDIVABase`):
- `BranchedWiredCfC`
- `BranchedLSTM`
- `BranchedDIVANCP`

## Current Implementation Analysis

### 1. `branched_diva_base.py`

**Location**: Line 335  
**Residual Formula**:
```python
x_seq = (x_seq * (1 - self.weight_residual)) + (residual * self.weight_residual)
```

**Initialization** (Line 189):
```python
self.weight_residual = nn.Parameter(torch.zeros(1))  # ReZero initialization to 0.0
```

**Behavior When `weight_residual = 0.0`**:
- `x_seq = (x_seq * (1 - 0.0)) + (residual * 0.0)`
- `x_seq = (x_seq * 1.0) + (residual * 0.0)`
- `x_seq = x_seq` (only recurrent output, residual path is zeroed)

**Result**: At initialization, the output is **100% recurrent output** and **0% residual (identity)**. This is the **opposite** of ReZero initialization.

---

### 2. `diva_ncp.py`

**Location**: Line 231  
**Residual Formula**:
```python
x = (x * (1 - self.weight_residual)) + (residual * self.weight_residual)
```

**Initialization** (Line 180):
```python
self.weight_residual = nn.Parameter(torch.from_numpy(rng.uniform(0.1, 0.9, (1,))).float())
```

**Behavior**: Uses random initialization (0.1 to 0.9), so not claiming ReZero initialization, but follows the same formula pattern.

---

### 3. `diva_full.py`

**Location**: Line 382 (when uncertainty mixer is disabled)  
**Residual Formula**:
```python
z_fused = ((1.0 - w) * z_corr) + (w * z_res)
```
Where `z_corr` is the recurrent output and `z_res` is the residual.

**Initialization** (Line 282):
```python
self.weight_residual = nn.Parameter(torch.tensor(float(init_residual_weight)).clamp(0.0,1.0))
```
Default `init_residual_weight = 0.0` (Line 158).

**Behavior When `w = 0.0`**:
- `z_fused = ((1.0 - 0.0) * z_corr) + (0.0 * z_res)`
- `z_fused = z_corr` (only recurrent output, residual path is zeroed)

**Result**: Same issue as `branched_diva_base.py` - starts with 100% recurrent, 0% residual.

---

## What ReZero Initialization Should Do

ReZero (from "ReZero is All You Need" by Bachlechner et al., 2020) initializes residual connections such that:

1. **At initialization**: The residual block acts as the **identity function** (passes input through unchanged)
2. **During training**: The learnable scaling parameter gradually enables the transformation

The standard ReZero formulation is:
```
output = input + α * transformation(input)
```

Where `α` is initialized to 0, so:
- At init: `output = input + 0 * transformation(input) = input` (identity)
- After training: `α` learns an optimal scaling factor

---

## The Problem: Formula Interpretation

The current formula in all models is:
```python
output = recurrent_output * (1 - α) + residual * α
```

With `α = 0.0` (ReZero initialization):
- `output = recurrent_output * 1.0 + residual * 0.0`
- `output = recurrent_output` (100% recurrent, 0% residual/identity)

**This means the recurrent compartment starts at full strength, not as identity!**

---

## What Should Happen (Correct ReZero)

For ReZero to work correctly, the formula should ensure that when the scaling parameter is at its initial value, the output equals the input (identity).

### Option 1: Swap the coefficients
```python
output = recurrent_output * α + residual * (1 - α)
```
With `α = 0.0`: `output = residual` (identity/residual path passes through)

### Option 2: Keep formula, initialize to 1.0
```python
output = recurrent_output * (1 - α) + residual * α
```
With `α = 1.0`: `output = residual` (identity/residual path passes through)

---

## Analysis by Model

### Models with ReZero-style initialization (but backwards):
1. **`branched_diva_base.py`** (Base class): 
   - Formula: `x_seq * (1 - α) + residual * α`
   - Initialization: `α = 0.0`
   - **Result**: Starts with recurrent only (NOT identity)
   - **Impact**: All models inheriting from this base class are affected:
     - `BranchedWiredCfC` (models/branched_wiredcfc.py)
     - `BranchedLSTM` (models/branched_lstm.py)
     - `BranchedDIVANCP` (models/branched_diva_ncp.py)

### Models with different initialization:
1. **`diva_ncp.py`**: 
   - Formula: Same as above
   - Initialization: Random (0.1 to 0.9)
   - **Result**: Not claiming ReZero, but formula still backwards for ReZero purposes

2. **`diva_full.py`**: 
   - Formula: `z_corr * (1 - w) + z_res * w`
   - Initialization: `w = 0.0` (when mixer disabled)
   - **Result**: Starts with recurrent only (NOT identity)

---

## Expected Behavior vs. Actual Behavior

### Expected (Correct ReZero):
- **At initialization**: `output = residual` (identity function - passes input through)
- **Recurrent compartment**: Starts at zero contribution
- **Training**: Recurrent compartment gradually contributes more as `α` increases

### Actual (Current Implementation):
- **At initialization**: `output = recurrent_output` (recurrent compartment at full strength)
- **Residual path**: Starts at zero contribution
- **Training**: Residual path gradually contributes more as `α` increases

---

## Impact Assessment

1. **Training Dynamics**: Models start learning immediately from the recurrent output, not from identity. This may cause different training dynamics than intended.

2. **Gradient Flow**: ReZero is designed to improve gradient flow at initialization. The backwards implementation may not provide the same benefits.

3. **Convergence**: The models may still converge, but the initialization strategy is not achieving the ReZero objective.

4. **All Branched Models**: Any model inheriting from `BranchedDIVABase` (e.g., `BranchedWiredCfC`, `BranchedLSTM`, etc.) inherits this issue.

---

## Code References

### Files Affected:
1. `models/branched_diva_base.py` - Lines 189, 335
2. `models/diva_ncp.py` - Lines 180, 231
3. `models/diva_full.py` - Lines 158, 282, 382

### Documentation:
- `BRANCHED_WIREDCFC_ANALYSIS.md` - Line 492 incorrectly states: "allowing the model to start with identity mapping"

---

## Recommendations

To fix ReZero initialization, choose one of the following approaches:

### Fix Option 1: Change initialization value
```python
# In branched_diva_base.py, line 189
self.weight_residual = nn.Parameter(torch.ones(1))  # Changed from zeros(1) to ones(1)
```

### Fix Option 2: Swap the formula
```python
# In branched_diva_base.py, line 335
x_seq = (x_seq * self.weight_residual) + (residual * (1 - self.weight_residual))
# Keep initialization at 0.0
```

### Recommended Approach:
Use **Fix Option 2** (swap formula, keep init at 0.0) because:
- It's more intuitive: `α=0` means zero contribution from recurrent, full from residual
- Consistent with standard ReZero literature where α=0 means identity
- Easier to reason about during debugging

---

## Verification

To verify the fix, check that at initialization:
```python
output ≈ residual  # Should be approximately equal (identity function)
```

With the current (incorrect) implementation:
```python
output = recurrent_output  # Does NOT equal residual
```

---

## Conclusion

The current implementation initializes the residual connection backwards from the intended ReZero behavior. The recurrent compartment starts at full strength rather than being initialized to the identity function. This affects all models using this pattern, particularly those inheriting from `BranchedDIVABase`.

**Status**: ✅ **INTENTIONALLY using backwards ReZero (empirically validated as superior)**

**Decision**: After empirical validation, we have chosen to keep "backwards_rezero" as the default initialization strategy because:
- It achieves ~6.4% better clean performance (0.9512 vs 0.8874 ROC-AUC)
- It maintains similar or slightly better robustness (84.18% vs 83.53% retention)
- It provides better training stability (lower variance: σ = 0.0146 vs 0.0532)

See `REZERO_BACKWARDS_ANALYSIS.md` for detailed theoretical justification explaining why this counterintuitive approach performs better for temporal modeling tasks.
