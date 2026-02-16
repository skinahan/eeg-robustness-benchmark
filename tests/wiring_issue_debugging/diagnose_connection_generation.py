"""
Diagnose how connections are being generated vs the original wiring matrix.
"""

import json
import numpy as np
from pathlib import Path
from architecture_refinement.arbitrary_wiring import load_architecture_from_file

arch4_path = Path("outputs/architectures/best_architecture_4_trial_178.json")

print("=" * 80)
print("CONNECTION GENERATION ANALYSIS")
print("=" * 80)

# Load original data
with open(arch4_path, 'r') as f:
    data = json.load(f)

orig_input = data['input_size']
orig_hidden = data['hidden_size']
orig_output = data['output_size']
orig_matrix = np.array(data['wiring_matrix'])

print(f"\n1. Original architecture file:")
print(f"  input_size: {orig_input}")
print(f"  hidden_size: {orig_hidden}")
print(f"  output_size: {orig_output}")
print(f"  wiring_matrix: {orig_matrix.shape}, {np.count_nonzero(orig_matrix)} connections")

# Extract original connections by region
I, H, O = orig_input, orig_hidden, orig_output
print(f"\n  Original connections by region:")
print(f"    Input->Hidden: {np.count_nonzero(orig_matrix[0:I, I:I+H])}")
print(f"    Input->Output: {np.count_nonzero(orig_matrix[0:I, I+H:I+H+O])}")
print(f"    Hidden->Hidden: {np.count_nonzero(orig_matrix[I:I+H, I:I+H])}")
print(f"    Hidden->Output: {np.count_nonzero(orig_matrix[I:I+H, I+H:I+H+O])}")
print(f"    Output->Hidden: {np.count_nonzero(orig_matrix[I+H:I+H+O, I:I+H])}")
print(f"    Output->Output: {np.count_nonzero(orig_matrix[I+H:I+H+O, I+H:I+H+O])}")

# Load wiring and change sizes
print(f"\n2. Loading wiring and changing to input_size=16, output_size=16:")
wiring = load_architecture_from_file(str(arch4_path), use_legacy_behavior=False)
wiring.input_size = 16
wiring.output_size = 16

# Build the wiring to get the final matrix
built = wiring.build(16)
built.build(16)

# Get the full matrix from the built wiring
if hasattr(built, 'wiring_matrix'):
    full_matrix = built.wiring_matrix
    print(f"  Final wiring matrix: {full_matrix.shape}, {np.count_nonzero(full_matrix)} connections")
    print(f"  Expected: I=16, H={built.hidden_size}, O=16, Total={16 + built.hidden_size + 16}")
    
    I_new, H_new, O_new = 16, built.hidden_size, 16
    print(f"\n  Final connections by region:")
    print(f"    Input->Hidden: {np.count_nonzero(full_matrix[0:I_new, I_new:I_new+H_new])}")
    print(f"    Input->Output: {np.count_nonzero(full_matrix[0:I_new, I_new+H_new:I_new+H_new+O_new])}")
    print(f"    Hidden->Hidden: {np.count_nonzero(full_matrix[I_new:I_new+H_new, I_new:I_new+H_new])}")
    print(f"    Hidden->Output: {np.count_nonzero(full_matrix[I_new:I_new+H_new, I_new+H_new:I_new+H_new+O_new])}")
    print(f"    Output->Hidden: {np.count_nonzero(full_matrix[I_new+H_new:I_new+H_new+O_new, I_new:I_new+H_new])}")
    print(f"    Output->Output: {np.count_nonzero(full_matrix[I_new+H_new:I_new+H_new+O_new, I_new+H_new:I_new+H_new+O_new])}")
else:
    print("  Could not access wiring_matrix from built wiring")

# Build and check final wiring
built = wiring.build(16)
built.build(16)

print(f"\n3. Final ArbitraryWiring:")
print(f"  units: {built.units}")
print(f"  adjacency_matrix: {built.adjacency_matrix.shape}, {np.count_nonzero(built.adjacency_matrix)} connections")
print(f"  sensory_adjacency_matrix: {built.sensory_adjacency_matrix.shape if built.sensory_adjacency_matrix is not None else None}, {np.count_nonzero(built.sensory_adjacency_matrix) if built.sensory_adjacency_matrix is not None else 0} connections")
print(f"  Total: {np.count_nonzero(built.adjacency_matrix) + (np.count_nonzero(built.sensory_adjacency_matrix) if built.sensory_adjacency_matrix is not None else 0)} connections")

# Compare with original
print(f"\n4. Comparison:")
print(f"  Original total connections: {np.count_nonzero(orig_matrix)}")
print(f"  Generated total connections: {np.count_nonzero(full_matrix)}")
print(f"  Final wiring connections: {np.count_nonzero(built.adjacency_matrix) + (np.count_nonzero(built.sensory_adjacency_matrix) if built.sensory_adjacency_matrix is not None else 0)}")
print(f"  Ratio (generated/original): {np.count_nonzero(full_matrix) / np.count_nonzero(orig_matrix):.2f}x")
