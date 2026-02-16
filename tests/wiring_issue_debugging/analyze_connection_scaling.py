"""
Analyze how connection scaling affects parameter counts.
"""

import numpy as np
from pathlib import Path
from architecture_refinement.arbitrary_wiring import load_architecture_from_file
from ncps.torch import CfC

arch4_path = Path("outputs/architectures/best_architecture_4_trial_178.json")

print("=" * 80)
print("CONNECTION SCALING ANALYSIS")
print("=" * 80)

# Original architecture
wiring_orig = load_architecture_from_file(str(arch4_path), use_legacy_behavior=False)
print(f"\n1. Original architecture:")
print(f"  input_size: {wiring_orig.input_size}")
print(f"  hidden_size: {wiring_orig._hidden_size()}")
print(f"  output_size: {wiring_orig.output_size}")

# Build with original sizes
built_orig = wiring_orig.build(wiring_orig.input_size)
built_orig.build(wiring_orig.input_size)
cfc_orig = CfC(input_size=wiring_orig.input_size, units=built_orig, proj_size=wiring_orig.output_size, 
               return_sequences=True, batch_first=True, mixed_memory=True)
params_orig = sum(p.numel() for p in cfc_orig.parameters() if p.requires_grad)
print(f"  CfC parameters: {params_orig:,}")

# Change sizes (as done in branched_wiredcfc)
wiring_scaled = load_architecture_from_file(str(arch4_path), use_legacy_behavior=False)
wiring_scaled.input_size = 16
wiring_scaled.output_size = 16

built_scaled = wiring_scaled.build(16)
built_scaled.build(16)
cfc_scaled = CfC(input_size=16, units=built_scaled, proj_size=16, 
                 return_sequences=True, batch_first=True, mixed_memory=True)
params_scaled = sum(p.numel() for p in cfc_scaled.parameters() if p.requires_grad)

print(f"\n2. After scaling to input_size=16, output_size=16:")
print(f"  input_size: {wiring_scaled.input_size}")
print(f"  hidden_size: {wiring_scaled._hidden_size()}")
print(f"  output_size: {wiring_scaled.output_size}")
print(f"  units: {built_scaled.units}")
print(f"  CfC parameters: {params_scaled:,}")

# Count connections
orig_connections = np.count_nonzero(built_orig.adjacency_matrix) + np.count_nonzero(built_orig.sensory_adjacency_matrix)
scaled_connections = np.count_nonzero(built_scaled.adjacency_matrix) + np.count_nonzero(built_scaled.sensory_adjacency_matrix)

print(f"\n3. Connection comparison:")
print(f"  Original connections: {orig_connections}")
print(f"  Scaled connections: {scaled_connections}")
print(f"  Ratio: {scaled_connections / orig_connections:.2f}x")

print(f"\n4. Parameter comparison:")
print(f"  Original parameters: {params_orig:,}")
print(f"  Scaled parameters: {params_scaled:,}")
print(f"  Ratio: {params_scaled / params_orig:.2f}x")

# The issue: scaling I=8→16 and O=7→16 doubles the I/O dimensions
# But we're creating 2x connections for inputs and ~2.3x for outputs
# This inflates the parameter count unnecessarily
