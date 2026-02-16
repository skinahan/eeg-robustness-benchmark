"""
Diagnose how the wiring matrix from JSON is being interpreted.
"""

import json
import numpy as np
from pathlib import Path

arch4_path = Path("outputs/architectures/best_architecture_4_trial_178.json")
with open(arch4_path, 'r') as f:
    data = json.load(f)

print("=" * 80)
print("ARCHITECTURE FILE ANALYSIS")
print("=" * 80)
print(f"input_size: {data['input_size']}")
print(f"hidden_size: {data['hidden_size']}")
print(f"output_size: {data['output_size']}")
print(f"Total (I+H+O): {data['input_size'] + data['hidden_size'] + data['output_size']}")
print(f"wiring_matrix shape: {len(data['wiring_matrix'])} x {len(data['wiring_matrix'][0]) if data['wiring_matrix'] else 0}")

wiring_matrix = np.array(data['wiring_matrix'])
print(f"\nWiring matrix analysis:")
print(f"  Shape: {wiring_matrix.shape}")
print(f"  Non-zero: {np.count_nonzero(wiring_matrix)}")
print(f"  Density: {100 * np.count_nonzero(wiring_matrix) / wiring_matrix.size:.2f}%")

# Check if it's the full matrix or just hidden
I_orig = data['input_size']
H_orig = data['hidden_size']
O_orig = data['output_size']
total_orig = I_orig + H_orig + O_orig

print(f"\nOriginal dimensions: I={I_orig}, H={H_orig}, O={O_orig}, Total={total_orig}")
print(f"Wiring matrix size: {wiring_matrix.shape[0]}")

if wiring_matrix.shape[0] == total_orig:
    print("✓ Wiring matrix is FULL [I+H+O, I+H+O] matrix")
    
    # Analyze regions
    print(f"\nRegion analysis (assuming I={I_orig}, H={H_orig}, O={O_orig}):")
    I_end = I_orig
    H_end = I_orig + H_orig
    O_end = I_orig + H_orig + O_orig
    
    print(f"  Input->Hidden: {np.count_nonzero(wiring_matrix[0:I_end, I_end:H_end])} connections")
    print(f"  Input->Output: {np.count_nonzero(wiring_matrix[0:I_end, H_end:O_end])} connections")
    print(f"  Hidden->Hidden: {np.count_nonzero(wiring_matrix[I_end:H_end, I_end:H_end])} connections")
    print(f"  Hidden->Output: {np.count_nonzero(wiring_matrix[I_end:H_end, H_end:O_end])} connections")
    print(f"  Output->Hidden: {np.count_nonzero(wiring_matrix[H_end:O_end, I_end:H_end])} connections")
    print(f"  Output->Output: {np.count_nonzero(wiring_matrix[H_end:O_end, H_end:O_end])} connections")
    
elif wiring_matrix.shape[0] == H_orig:
    print("✓ Wiring matrix is HIDDEN-ONLY [H, H] matrix")
else:
    print(f"⚠ Wiring matrix size ({wiring_matrix.shape[0]}) doesn't match expected sizes!")
