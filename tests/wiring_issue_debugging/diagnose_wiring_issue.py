"""
Diagnostic script to investigate the wiring construction issue.

This script will:
1. Load the architecture file
2. Build the wiring step by step
3. Inspect the wiring matrix dimensions and density
4. Create a model and count parameters in detail
5. Compare with CNN-NCP parameter counts
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from models.branched_wiredcfc import create_branched_wiredcfc_classifier
from models.cnnncp import create_cnnncp_classifier
from architecture_refinement.arbitrary_wiring import load_architecture_from_file

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameter_breakdown(model, prefix=""):
    """Print detailed parameter breakdown."""
    print(f"\n{prefix}Parameter Breakdown:")
    print("-" * 80)
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            count = param.numel()
            total += count
            print(f"  {name:60s} {count:10,} params  {str(list(param.shape)):30s}")
    print(f"{'TOTAL':60s} {total:10,} params")
    return total

def inspect_wiring(wiring, step_name=""):
    """Inspect wiring structure."""
    print(f"\n{'='*80}")
    print(f"WIRING INSPECTION: {step_name}")
    print(f"{'='*80}")
    
    print(f"Wiring type: {type(wiring)}")
    print(f"Wiring class: {wiring.__class__.__name__}")
    
    # Check attributes
    attrs = ['input_size', 'output_size', 'hidden_size', '_hidden_size', 
             'wiring_matrix', '_wiring_matrix', 'total_units', 'units']
    for attr in attrs:
        if hasattr(wiring, attr):
            val = getattr(wiring, attr)
            if isinstance(val, np.ndarray):
                print(f"  {attr}: shape={val.shape}, dtype={val.dtype}, nnz={np.count_nonzero(val)}")
            else:
                print(f"  {attr}: {val}")
    
    # Try to get wiring matrix
    wiring_matrix = None
    if hasattr(wiring, 'wiring_matrix'):
        wiring_matrix = wiring.wiring_matrix
    elif hasattr(wiring, '_wiring_matrix'):
        wiring_matrix = wiring._wiring_matrix
    
    if wiring_matrix is not None:
        wiring_matrix = np.array(wiring_matrix)
        print(f"\nWiring Matrix Analysis:")
        print(f"  Shape: {wiring_matrix.shape}")
        print(f"  Total elements: {wiring_matrix.size:,}")
        print(f"  Non-zero elements: {np.count_nonzero(wiring_matrix):,}")
        print(f"  Density: {100 * np.count_nonzero(wiring_matrix) / wiring_matrix.size:.2f}%")
        
        # Break down by regions
        if hasattr(wiring, 'input_size') and hasattr(wiring, 'hidden_size') and hasattr(wiring, 'output_size'):
            I = wiring.input_size
            H = wiring.hidden_size
            O = wiring.output_size
            T = I + H + O
            
            if wiring_matrix.shape == (T, T):
                print(f"\n  Region breakdown (I={I}, H={H}, O={O}):")
                print(f"    Input->Hidden:  {np.count_nonzero(wiring_matrix[0:I, I:I+H]):,} connections")
                print(f"    Hidden->Hidden: {np.count_nonzero(wiring_matrix[I:I+H, I:I+H]):,} connections")
                print(f"    Hidden->Output: {np.count_nonzero(wiring_matrix[I:I+H, I+H:I+H+O]):,} connections")
                print(f"    Input->Output:  {np.count_nonzero(wiring_matrix[0:I, I+H:I+H+O]):,} connections (should be 0)")
                print(f"    Output->Input:  {np.count_nonzero(wiring_matrix[I+H:I+H+O, 0:I]):,} connections (should be 0)")
    
    # Try to get wiring summary
    if hasattr(wiring, 'get_wiring_summary'):
        try:
            summary = wiring.get_wiring_summary()
            print(f"\nWiring Summary:")
            for key, val in summary.items():
                if isinstance(val, dict):
                    print(f"  {key}:")
                    for k, v in val.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {val}")
        except Exception as e:
            print(f"  Could not get wiring summary: {e}")

def inspect_cfc_cell(cfc_cell, wiring):
    """Inspect CfC cell parameters."""
    print(f"\n{'='*80}")
    print(f"CFC CELL INSPECTION")
    print(f"{'='*80}")
    
    print(f"CfC cell type: {type(cfc_cell)}")
    print(f"Wired mode: {getattr(cfc_cell, 'wired_mode', 'N/A')}")
    
    # Get wiring from cell if possible
    if hasattr(cfc_cell, 'wiring'):
        print(f"\nCell has wiring attribute:")
        inspect_wiring(cfc_cell.wiring, "From CfC cell")
    
    # Count parameters by type
    print(f"\nCfC Cell Parameters:")
    total = 0
    for name, param in cfc_cell.named_parameters():
        if param.requires_grad:
            count = param.numel()
            total += count
            print(f"  {name:60s} {count:10,} params  {str(list(param.shape)):30s}")
    print(f"{'TOTAL':60s} {total:10,} params")
    
    return total

def main():
    # Standard input dimensions
    n_chans = 22
    n_times = 1001
    n_outputs = 2
    
    print("=" * 80)
    print("WIRING CONSTRUCTION DIAGNOSTIC")
    print("=" * 80)
    
    # 1. Load architecture file
    arch4_path = Path("outputs/architectures/best_architecture_4_trial_178.json")
    if not arch4_path.exists():
        print(f"ERROR: Architecture file not found: {arch4_path}")
        return
    
    print(f"\n1. Loading architecture from: {arch4_path}")
    wiring_original = load_architecture_from_file(str(arch4_path))
    inspect_wiring(wiring_original, "After load_architecture_from_file")
    
    # 2. Check what happens when we set input_size and output_size
    print(f"\n2. Setting input_size and output_size (F2=16, recurrent_output_size=16)")
    F2 = 16  # F1 * D = 8 * 2
    recurrent_output_size = 16
    
    # Check if wiring has these attributes
    if hasattr(wiring_original, 'input_size'):
        print(f"  Original input_size: {wiring_original.input_size}")
    if hasattr(wiring_original, 'output_size'):
        print(f"  Original output_size: {wiring_original.output_size}")
    if hasattr(wiring_original, 'hidden_size'):
        print(f"  Original hidden_size: {wiring_original.hidden_size}")
    elif hasattr(wiring_original, '_hidden_size'):
        print(f"  Original _hidden_size: {wiring_original._hidden_size}")
    
    # Set them (as done in branched_wiredcfc.py)
    wiring_original.input_size = F2
    wiring_original.output_size = recurrent_output_size
    print(f"  Set input_size to: {wiring_original.input_size}")
    print(f"  Set output_size to: {wiring_original.output_size}")
    
    # 3. Build the wiring
    print(f"\n3. Building wiring with build({F2})")
    built_wiring = wiring_original.build(F2)
    inspect_wiring(built_wiring, "After build()")
    
    # 4. Create a model and inspect
    print(f"\n4. Creating BranchedWiredCfC model")
    branched_wiredcfc = create_branched_wiredcfc_classifier(
        n_chans, n_times, n_outputs, wiring=wiring_original  # Pass original wiring
    )
    branched_wiredcfc.initialize()
    
    # Get the recurrent cell
    if hasattr(branched_wiredcfc.module_, 'recurrent_cell'):
        cfc_cell = branched_wiredcfc.module_.recurrent_cell
        cfc_params = inspect_cfc_cell(cfc_cell, built_wiring)
    
    # Print full model breakdown
    print(f"\n5. Full Model Parameter Breakdown")
    total_params = print_parameter_breakdown(branched_wiredcfc.module_, "BranchedWiredCfC")
    
    # 6. Compare with CNN-NCP
    print(f"\n6. Creating CNN-NCP for comparison")
    cnn_ncp = create_cnnncp_classifier(n_chans, n_times, n_outputs)
    cnn_ncp.initialize()
    cnn_ncp_params = count_parameters(cnn_ncp.module_)
    print(f"  CNN-NCP total parameters: {cnn_ncp_params:,}")
    
    # Print CNN-NCP breakdown
    print_parameter_breakdown(cnn_ncp.module_, "CNN-NCP")
    
    # 7. Ratio analysis
    print(f"\n7. Comparison")
    print(f"  BranchedWiredCfC: {total_params:,} parameters")
    print(f"  CNN-NCP:          {cnn_ncp_params:,} parameters")
    print(f"  Ratio:            {total_params / cnn_ncp_params:.2f}x")
    
    # 8. Focus on recurrent cell parameters
    print(f"\n8. Recurrent Cell Analysis")
    if hasattr(branched_wiredcfc.module_, 'recurrent_cell'):
        print(f"  Recurrent cell parameters: {cfc_params:,}")
        print(f"  Percentage of total: {100 * cfc_params / total_params:.1f}%")
        
        # Estimate expected parameters based on wiring
        if hasattr(built_wiring, 'wiring_matrix'):
            wiring_matrix = np.array(built_wiring.wiring_matrix)
            connections = np.count_nonzero(wiring_matrix)
            print(f"  Wiring matrix connections: {connections:,}")
            print(f"  Parameters per connection (approx): {cfc_params / connections:.2f}")

if __name__ == "__main__":
    main()
