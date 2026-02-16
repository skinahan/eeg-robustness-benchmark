"""
Compare wiring sizes between AutoNCP (CNN-NCP) and Architecture 4 (HYDRA).
"""

import numpy as np
import torch
from pathlib import Path
from ncps.wirings import AutoNCP
from architecture_refinement.arbitrary_wiring import load_architecture_from_file
from ncps.torch import CfC

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_wiring(wiring, name):
    """Analyze a wiring structure."""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    
    print(f"units: {wiring.units}")
    print(f"input_dim: {wiring.input_dim if hasattr(wiring, 'input_dim') else 'N/A'}")
    print(f"output_dim: {wiring.output_dim}")
    
    if hasattr(wiring, 'adjacency_matrix'):
        adj = wiring.adjacency_matrix
        print(f"adjacency_matrix shape: {adj.shape}")
        print(f"adjacency_matrix nnz: {np.count_nonzero(adj)}")
        print(f"adjacency_matrix density: {100 * np.count_nonzero(adj) / adj.size:.2f}%")
    
    if hasattr(wiring, 'sensory_adjacency_matrix') and wiring.sensory_adjacency_matrix is not None:
        sens = wiring.sensory_adjacency_matrix
        print(f"sensory_adjacency_matrix shape: {sens.shape}")
        print(f"sensory_adjacency_matrix nnz: {np.count_nonzero(sens)}")
        print(f"sensory_adjacency_matrix density: {100 * np.count_nonzero(sens) / sens.size:.2f}%")
    
    if hasattr(wiring, 'num_layers'):
        print(f"num_layers: {wiring.num_layers}")
        for l in range(wiring.num_layers):
            neurons = wiring.get_neurons_of_layer(l)
            print(f"  Layer {l}: {len(neurons)} neurons")
    
    # Count total connections
    total_connections = 0
    if hasattr(wiring, 'adjacency_matrix'):
        total_connections += np.count_nonzero(wiring.adjacency_matrix)
    if hasattr(wiring, 'sensory_adjacency_matrix') and wiring.sensory_adjacency_matrix is not None:
        total_connections += np.count_nonzero(wiring.sensory_adjacency_matrix)
    print(f"Total connections: {total_connections}")

def analyze_cfc_parameters(cfc, name):
    """Analyze CfC parameters."""
    print(f"\n{'='*80}")
    print(f"{name} - CFC PARAMETERS")
    print(f"{'='*80}")
    
    total = 0
    for name_param, param in cfc.named_parameters():
        if param.requires_grad:
            count = param.numel()
            total += count
            if 'rnn_cell' in name_param or 'lstm' in name_param or 'fc' in name_param:
                print(f"  {name_param:60s} {count:10,}  {str(list(param.shape)):30s}")
    
    print(f"{'TOTAL':60s} {total:10,}")
    return total

def main():
    print("=" * 80)
    print("WIRING SIZE COMPARISON")
    print("=" * 80)
    
    # CNN-NCP wiring (AutoNCP)
    print("\n1. Creating CNN-NCP wiring (AutoNCP)...")
    ncp_hidden_dim = 32  # From CNNNCPv3 default
    ncp_output_size = 8  # F1 = 8
    sparsity = 0.85
    
    autoncp_wiring = AutoNCP(
        units=ncp_hidden_dim,
        output_size=ncp_output_size,
        sparsity_level=sparsity,
        seed=17
    )
    autoncp_wiring.build(16)  # input_size = F2 = 16
    
    analyze_wiring(autoncp_wiring, "AutoNCP (CNN-NCP)")
    
    # Create CfC with AutoNCP
    cfc_autoncp = CfC(
        input_size=16,
        units=autoncp_wiring,
        proj_size=8,
        return_sequences=True,
        batch_first=True,
        mixed_memory=True,
    )
    autoncp_params = analyze_cfc_parameters(cfc_autoncp, "AutoNCP")
    
    # Architecture 4 wiring
    print("\n2. Loading Architecture 4 wiring...")
    arch4_path = Path("outputs/architectures/best_architecture_4_trial_178.json")
    if not arch4_path.exists():
        print(f"ERROR: {arch4_path} not found")
        return
    
    wiring_arch4 = load_architecture_from_file(str(arch4_path), use_legacy_behavior=False)
    wiring_arch4.input_size = 16
    wiring_arch4.output_size = 16
    built_arch4 = wiring_arch4.build(16)
    built_arch4.build(16)  # Initialize sensory_adjacency_matrix
    
    analyze_wiring(built_arch4, "Architecture 4 (HYDRA)")
    
    # Create CfC with Architecture 4
    cfc_arch4 = CfC(
        input_size=16,
        units=built_arch4,
        proj_size=16,
        return_sequences=True,
        batch_first=True,
        mixed_memory=True,
    )
    arch4_params = analyze_cfc_parameters(cfc_arch4, "Architecture 4")
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    print(f"AutoNCP connections: {np.count_nonzero(autoncp_wiring.adjacency_matrix) + np.count_nonzero(autoncp_wiring.sensory_adjacency_matrix)}")
    print(f"Architecture 4 connections: {np.count_nonzero(built_arch4.adjacency_matrix) + np.count_nonzero(built_arch4.sensory_adjacency_matrix)}")
    print(f"\nAutoNCP CfC parameters: {autoncp_params:,}")
    print(f"Architecture 4 CfC parameters: {arch4_params:,}")
    print(f"Ratio: {arch4_params / autoncp_params:.2f}x")
    
    # Check layer structures
    print(f"\n{'='*80}")
    print("LAYER STRUCTURE COMPARISON")
    print(f"{'='*80}")
    print("AutoNCP layers:")
    for l in range(autoncp_wiring.num_layers):
        neurons = autoncp_wiring.get_neurons_of_layer(l)
        print(f"  Layer {l}: {len(neurons)} neurons")
    
    print("Architecture 4 layers:")
    for l in range(built_arch4.num_layers):
        neurons = built_arch4.get_neurons_of_layer(l)
        print(f"  Layer {l}: {len(neurons)} neurons")
    
    # Check sparsity masks
    print(f"\n{'='*80}")
    print("SPARSITY MASK ANALYSIS")
    print(f"{'='*80}")
    
    print("AutoNCP sparsity masks:")
    for i, layer in enumerate(cfc_autoncp.rnn_cell._layers):
        if hasattr(layer, 'sparsity_mask') and layer.sparsity_mask is not None:
            mask = layer.sparsity_mask.data
            prev_part = mask[:, :mask.shape[1] - mask.shape[0]]
            self_part = mask[:, mask.shape[1] - mask.shape[0]:]
            print(f"  Layer {i}: mask {list(mask.shape)}, prev_layer density: {100 * torch.count_nonzero(prev_part).item() / prev_part.numel():.2f}%")
    
    print("Architecture 4 sparsity masks:")
    for i, layer in enumerate(cfc_arch4.rnn_cell._layers):
        if hasattr(layer, 'sparsity_mask') and layer.sparsity_mask is not None:
            mask = layer.sparsity_mask.data
            prev_part = mask[:, :mask.shape[1] - mask.shape[0]]
            self_part = mask[:, mask.shape[1] - mask.shape[0]:]
            print(f"  Layer {i}: mask {list(mask.shape)}, prev_layer density: {100 * torch.count_nonzero(prev_part).item() / prev_part.numel():.2f}%")

if __name__ == "__main__":
    main()
