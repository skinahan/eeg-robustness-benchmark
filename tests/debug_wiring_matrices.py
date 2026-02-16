"""
Debug script to check wiring matrices after build.
"""

import numpy as np
from pathlib import Path
from architecture_refinement.arbitrary_wiring import load_architecture_from_file
from ncps.torch import CfC

# Load wiring
arch4_path = Path("outputs/architectures/best_architecture_4_trial_178.json")
wiring_original = load_architecture_from_file(str(arch4_path), use_legacy_behavior=False)
wiring_original.input_size = 16
wiring_original.output_size = 16

# Build wiring
built_wiring = wiring_original.build(16)

# Now build it (this initializes sensory_adjacency_matrix)
built_wiring.build(16)

print("=" * 80)
print("WIRING MATRIX ANALYSIS")
print("=" * 80)
print(f"units: {built_wiring.units}")
print(f"input_dim: {built_wiring.input_dim}")
print(f"output_dim: {built_wiring.output_dim}")
print(f"hidden_size: {built_wiring.hidden_size}")
print(f"input_size: {built_wiring.input_size}")
print(f"output_size: {built_wiring.output_size}")

print(f"\nadjacency_matrix shape: {built_wiring.adjacency_matrix.shape}")
print(f"adjacency_matrix nnz: {np.count_nonzero(built_wiring.adjacency_matrix)}")
print(f"adjacency_matrix density: {100 * np.count_nonzero(built_wiring.adjacency_matrix) / built_wiring.adjacency_matrix.size:.2f}%")

print(f"\nsensory_adjacency_matrix shape: {built_wiring.sensory_adjacency_matrix.shape if built_wiring.sensory_adjacency_matrix is not None else None}")
if built_wiring.sensory_adjacency_matrix is not None:
    print(f"sensory_adjacency_matrix nnz: {np.count_nonzero(built_wiring.sensory_adjacency_matrix)}")
    print(f"sensory_adjacency_matrix density: {100 * np.count_nonzero(built_wiring.sensory_adjacency_matrix) / built_wiring.sensory_adjacency_matrix.size:.2f}%")

# Check layers
print(f"\nnum_layers: {built_wiring.num_layers}")
for l in range(built_wiring.num_layers):
    neurons = built_wiring.get_neurons_of_layer(l)
    print(f"  Layer {l}: {len(neurons)} neurons, indices: {neurons[:5]}..." if len(neurons) > 5 else f"  Layer {l}: {len(neurons)} neurons, indices: {neurons}")

# Create CfC and check what it does
print("\n" + "=" * 80)
print("CFC CONSTRUCTION")
print("=" * 80)
cfc = CfC(
    input_size=16,
    units=built_wiring,
    proj_size=16,
    return_sequences=True,
    batch_first=True,
    mixed_memory=True,
)

# Check the sparsity masks in each layer
import torch
print("\nLayer sparsity masks:")
for i, layer in enumerate(cfc.rnn_cell._layers):
    if hasattr(layer, 'sparsity_mask') and layer.sparsity_mask is not None:
        mask = layer.sparsity_mask.data
        print(f"  Layer {i}: mask shape {list(mask.shape)}, nnz: {torch.count_nonzero(mask).item()}, density: {100 * torch.count_nonzero(mask).item() / mask.numel():.2f}%")
        print(f"    Layer input_size: {layer.input_size}, hidden_size: {layer.hidden_size}")
        # Check the sparsity of the first part (connections from previous layer)
        if mask.shape[1] > mask.shape[0]:
            prev_layer_part = mask[:, :mask.shape[1] - mask.shape[0]]
            self_conn_part = mask[:, mask.shape[1] - mask.shape[0]:]
            print(f"    Prev layer connections: {torch.count_nonzero(prev_layer_part).item()} / {prev_layer_part.numel()} ({100 * torch.count_nonzero(prev_layer_part).item() / prev_layer_part.numel():.2f}% density)")
            print(f"    Self connections: {torch.count_nonzero(self_conn_part).item()} / {self_conn_part.numel()} (should be 100%)")
    else:
        print(f"  Layer {i}: No sparsity mask")
