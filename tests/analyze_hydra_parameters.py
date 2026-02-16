"""
Script to analyze HYDRA parameter breakdown and identify what's causing the large parameter count.
"""

import torch
import torch.nn as nn
from models.branched_wiredcfc import create_branched_wiredcfc_classifier
from architecture_refinement.arbitrary_wiring import load_architecture_from_file
from pathlib import Path

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_by_module(model, prefix=""):
    """Recursively count parameters by module."""
    results = {}
    total = 0
    
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        if module_params > 0:
            results[full_name] = module_params
            total += module_params
        
        # Recursively count submodules
        sub_results, sub_total = count_parameters_by_module(module, full_name)
        results.update(sub_results)
        total += sub_total
    
    return results, total

def main():
    # Standard input dimensions for BNCI2014_001
    n_chans = 22
    n_times = 1001
    n_outputs = 2
    
    print("=" * 80)
    print("HYDRA PARAMETER BREAKDOWN ANALYSIS")
    print("=" * 80)
    print(f"Input dimensions: {n_chans} channels, {n_times} timepoints, {n_outputs} outputs\n")
    
    # Load architecture
    arch4_path = Path("outputs/architectures/best_architecture_4_trial_178.json")
    wiring = load_architecture_from_file(str(arch4_path))
    
    print(f"Architecture 4 wiring info:")
    print(f"  Input size: {wiring.input_size}")
    hidden_size = getattr(wiring, 'hidden_size', getattr(wiring, '_hidden_size', 'N/A'))
    output_size = getattr(wiring, 'output_size', getattr(wiring, '_output_size', 'N/A'))
    print(f"  Hidden size: {hidden_size}")
    print(f"  Output size: {output_size}")
    if isinstance(hidden_size, int) and isinstance(output_size, int):
        print(f"  Total units: {wiring.input_size + hidden_size + output_size}\n")
    else:
        print()
    
    # Create model
    print("Creating HYDRA model...")
    model = create_branched_wiredcfc_classifier(
        n_chans, n_times, n_outputs, wiring=wiring
    )
    model.initialize()
    
    # Get the actual module
    module = model.module_
    
    # Count total parameters
    total_params = count_parameters(module)
    print(f"\nTotal trainable parameters: {total_params:,}\n")
    
    # Break down by major components
    print("=" * 80)
    print("PARAMETER BREAKDOWN BY COMPONENT")
    print("=" * 80)
    
    # Front-end components
    frontend_params = 0
    if hasattr(module, 'conv1'):
        conv1_params = count_parameters(module.conv1)
        frontend_params += conv1_params
        print(f"conv1 (Temporal Conv): {conv1_params:,}")
    
    if hasattr(module, 'bn1'):
        bn1_params = count_parameters(module.bn1)
        frontend_params += bn1_params
        print(f"bn1: {bn1_params:,}")
    
    if hasattr(module, 'depthwise_conv'):
        depthwise_params = count_parameters(module.depthwise_conv)
        frontend_params += depthwise_params
        print(f"depthwise_conv: {depthwise_params:,}")
    
    if hasattr(module, 'bn2'):
        bn2_params = count_parameters(module.bn2)
        frontend_params += bn2_params
        print(f"bn2: {bn2_params:,}")
    
    print(f"Front-end subtotal: {frontend_params:,}\n")
    
    # Multi-scale block
    ms_params = 0
    if hasattr(module, 'multi_scale_block'):
        ms_params = count_parameters(module.multi_scale_block)
        print(f"multi_scale_block: {ms_params:,}\n")
    
    # SNR gate
    snr_params = 0
    if hasattr(module, 'snr_gate'):
        snr_params = count_parameters(module.snr_gate)
        print(f"snr_gate: {snr_params:,}\n")
    
    # Temporal downsampler
    downsample_params = 0
    if hasattr(module, 'temporal_downsampler'):
        downsample_params = count_parameters(module.temporal_downsampler)
        print(f"temporal_downsampler: {downsample_params:,}\n")
    
    # Recurrent cell (CfC) - THIS IS LIKELY THE CULPRIT
    recurrent_params = 0
    if hasattr(module, 'recurrent_cell'):
        recurrent_params = count_parameters(module.recurrent_cell)
        print(f"recurrent_cell (CfC): {recurrent_params:,}")
        
        # Try to break down the CfC cell further
        if hasattr(module.recurrent_cell, 'cell'):
            cell_params = count_parameters(module.recurrent_cell.cell)
            print(f"  -> cell: {cell_params:,}")
            
            # Check for projection layers
            if hasattr(module.recurrent_cell, 'proj'):
                proj_params = count_parameters(module.recurrent_cell.proj)
                print(f"  -> proj: {proj_params:,}")
        
        print()
    
    # Attention mechanisms
    attn_params = 0
    if hasattr(module, 'intra_bin_attn'):
        intra_attn_params = count_parameters(module.intra_bin_attn)
        attn_params += intra_attn_params
        print(f"intra_bin_attn: {intra_attn_params:,}")
    
    if hasattr(module, 'inter_bin_fusion'):
        inter_fusion_params = count_parameters(module.inter_bin_fusion)
        attn_params += inter_fusion_params
        print(f"inter_bin_fusion: {inter_fusion_params:,}")
    
    if attn_params > 0:
        print(f"Attention subtotal: {attn_params:,}\n")
    
    # Classification head
    head_params = 0
    if hasattr(module, 'classifier'):
        head_params = count_parameters(module.classifier)
        print(f"classifier: {head_params:,}\n")
    
    # Other components
    other_params = total_params - (frontend_params + ms_params + snr_params + 
                                  downsample_params + recurrent_params + 
                                  attn_params + head_params)
    
    print(f"Other components: {other_params:,}\n")
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Front-end: {frontend_params:,} ({100*frontend_params/total_params:.1f}%)")
    print(f"Multi-scale block: {ms_params:,} ({100*ms_params/total_params:.1f}%)")
    print(f"SNR gate: {snr_params:,} ({100*snr_params/total_params:.1f}%)")
    print(f"Temporal downsampler: {downsample_params:,} ({100*downsample_params/total_params:.1f}%)")
    print(f"Recurrent cell (CfC): {recurrent_params:,} ({100*recurrent_params/total_params:.1f}%)")
    print(f"Attention mechanisms: {attn_params:,} ({100*attn_params/total_params:.1f}%)")
    print(f"Classification head: {head_params:,} ({100*head_params/total_params:.1f}%)")
    print(f"Other: {other_params:,} ({100*other_params/total_params:.1f}%)")
    print(f"\nTotal: {total_params:,}")
    
    # Detailed breakdown of recurrent cell if possible
    if hasattr(module, 'recurrent_cell'):
        print("\n" + "=" * 80)
        print("DETAILED RECURRENT CELL BREAKDOWN")
        print("=" * 80)
        cell = module.recurrent_cell
        cell_breakdown, _ = count_parameters_by_module(cell)
        for name, params in sorted(cell_breakdown.items(), key=lambda x: -x[1]):
            print(f"{name}: {params:,}")

if __name__ == "__main__":
    main()
