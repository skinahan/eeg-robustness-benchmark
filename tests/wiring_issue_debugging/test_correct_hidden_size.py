"""
Test if hidden_size is being preserved correctly when we change input/output sizes.
"""

import numpy as np
from pathlib import Path
from architecture_refinement.arbitrary_wiring import load_architecture_from_file
from ncps.torch import CfC

arch4_path = Path("outputs/architectures/best_architecture_4_trial_178.json")

print("=" * 80)
print("HIDDEN SIZE PRESERVATION TEST")
print("=" * 80)

# Load architecture
wiring = load_architecture_from_file(str(arch4_path), use_legacy_behavior=False)
print(f"\n1. After loading:")
print(f"  input_size: {wiring.input_size}")
print(f"  hidden_size (from _hidden_size()): {wiring._hidden_size()}")
print(f"  output_size: {wiring.output_size}")
print(f"  hidden_graph shape: {wiring.hidden_graph.shape if hasattr(wiring.hidden_graph, 'shape') else 'N/A'}")

# Change sizes (as done in branched_wiredcfc.py)
wiring.input_size = 16
wiring.output_size = 16
print(f"\n2. After changing to input_size=16, output_size=16:")
print(f"  input_size: {wiring.input_size}")
print(f"  hidden_size (from _hidden_size()): {wiring._hidden_size()}")
print(f"  output_size: {wiring.output_size}")
print(f"  hidden_graph shape: {wiring.hidden_graph.shape if hasattr(wiring.hidden_graph, 'shape') else 'N/A'}")

# Build
built = wiring.build(16)
built.build(16)
print(f"\n3. After build(16):")
print(f"  units: {built.units}")
print(f"  hidden_size: {built.hidden_size}")
print(f"  input_size: {built.input_size}")
print(f"  output_size: {built.output_size}")
print(f"  Expected units: {built.hidden_size + built.output_size}")

# Create CfC and check
cfc = CfC(input_size=16, units=built, proj_size=16, return_sequences=True, batch_first=True, mixed_memory=True)
print(f"\n4. CfC cell state_size: {cfc.state_size}")
print(f"  Expected: {built.units}")

# Count parameters
total_params = sum(p.numel() for p in cfc.parameters() if p.requires_grad)
print(f"\n5. CfC parameters: {total_params:,}")

# Compare with what we'd expect for a 43-hidden, 16-output wiring
print(f"\n6. Expected for H=43, O=16:")
expected_units = 43 + 16
print(f"  units: {expected_units}")
print(f"  If we had H=43, O=16, we'd expect ~{expected_units} units")
