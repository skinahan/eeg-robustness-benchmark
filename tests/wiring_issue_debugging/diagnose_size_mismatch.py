"""
Diagnose what happens when we change input_size and output_size after loading.
"""

import numpy as np
from pathlib import Path
from architecture_refinement.arbitrary_wiring import load_architecture_from_file

arch4_path = Path("outputs/architectures/best_architecture_4_trial_178.json")

print("=" * 80)
print("SIZE MISMATCH DIAGNOSIS")
print("=" * 80)

# Load with original sizes
print("\n1. Loading with ORIGINAL sizes from JSON:")
wiring_orig = load_architecture_from_file(str(arch4_path), use_legacy_behavior=False)
print(f"  Original: input_size={wiring_orig.input_size}, hidden_size={wiring_orig._hidden_size()}, output_size={wiring_orig.output_size}")

# Build with original sizes
built_orig = wiring_orig.build(wiring_orig.input_size)
built_orig.build(wiring_orig.input_size)
print(f"  Built units: {built_orig.units}")
print(f"  Adjacency matrix: {built_orig.adjacency_matrix.shape}, nnz: {np.count_nonzero(built_orig.adjacency_matrix)}")
print(f"  Sensory matrix: {built_orig.sensory_adjacency_matrix.shape if built_orig.sensory_adjacency_matrix is not None else None}, nnz: {np.count_nonzero(built_orig.sensory_adjacency_matrix) if built_orig.sensory_adjacency_matrix is not None else None}")

# Now change sizes and rebuild
print("\n2. Changing sizes to input_size=16, output_size=16:")
wiring_orig.input_size = 16
wiring_orig.output_size = 16
print(f"  Changed to: input_size={wiring_orig.input_size}, hidden_size={wiring_orig._hidden_size()}, output_size={wiring_orig.output_size}")

# Build with new sizes
built_new = wiring_orig.build(16)
built_new.build(16)
print(f"  Built units: {built_new.units}")
print(f"  Adjacency matrix: {built_new.adjacency_matrix.shape}, nnz: {np.count_nonzero(built_new.adjacency_matrix)}")
print(f"  Sensory matrix: {built_new.sensory_adjacency_matrix.shape if built_new.sensory_adjacency_matrix is not None else None}, nnz: {np.count_nonzero(built_new.sensory_adjacency_matrix) if built_new.sensory_adjacency_matrix is not None else None}")

# Check what WsFlexHiddenWiring.full_wiring_matrix() creates
print("\n3. Checking what full_wiring_matrix() creates with new sizes:")
full_matrix = wiring_orig.full_wiring_matrix()
print(f"  Full matrix shape: {full_matrix.shape}")
print(f"  Full matrix nnz: {np.count_nonzero(full_matrix)}")
print(f"  Expected: I=16, H={wiring_orig._hidden_size()}, O=16, Total={16 + wiring_orig._hidden_size() + 16}")

# Check the hidden graph that was extracted
print("\n4. Hidden graph that was extracted:")
print(f"  Hidden graph shape: {wiring_orig.hidden_graph.shape if hasattr(wiring_orig.hidden_graph, 'shape') else 'N/A'}")
if hasattr(wiring_orig.hidden_graph, 'shape'):
    print(f"  Hidden graph nnz: {np.count_nonzero(wiring_orig.hidden_graph)}")
