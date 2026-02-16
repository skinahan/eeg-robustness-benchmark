"""
Deep dive into how CfC uses the wiring object.

The diagnostic showed that CfC creates dense weight matrices even in wired mode.
This script investigates how the wiring is actually used by CfC.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from ncps.torch import CfC
from architecture_refinement.arbitrary_wiring import load_architecture_from_file

def inspect_cfc_internals(cfc_cell):
    """Inspect the internal structure of a CfC cell."""
    print(f"\n{'='*80}")
    print(f"CFC INTERNAL STRUCTURE")
    print(f"{'='*80}")
    
    print(f"CfC type: {type(cfc_cell)}")
    print(f"Wired mode: {getattr(cfc_cell, 'wired_mode', 'N/A')}")
    
    # Check for rnn_cell
    if hasattr(cfc_cell, 'rnn_cell'):
        print(f"\nHas rnn_cell: {type(cfc_cell.rnn_cell)}")
        
        # Check if it's a WiredCfCCell
        if hasattr(cfc_cell.rnn_cell, '__class__'):
            print(f"  rnn_cell class: {cfc_cell.rnn_cell.__class__.__name__}")
        
        # Check for layers
        if hasattr(cfc_cell.rnn_cell, 'layer_0'):
            print(f"\n  Layer 0 (input layer):")
            layer0 = cfc_cell.rnn_cell.layer_0
            for attr in dir(layer0):
                if not attr.startswith('_') and hasattr(layer0, attr):
                    val = getattr(layer0, attr)
                    if isinstance(val, nn.Parameter) or isinstance(val, nn.Module):
                        if isinstance(val, nn.Linear):
                            print(f"    {attr}: Linear({val.in_features}, {val.out_features})")
                        elif isinstance(val, nn.Parameter):
                            print(f"    {attr}: Parameter{list(val.shape)}")
        
        if hasattr(cfc_cell.rnn_cell, 'layer_1'):
            print(f"\n  Layer 1 (hidden layer):")
            layer1 = cfc_cell.rnn_cell.layer_1
            for attr in dir(layer1):
                if not attr.startswith('_') and hasattr(layer1, attr):
                    val = getattr(layer1, attr)
                    if isinstance(val, nn.Parameter) or isinstance(val, nn.Module):
                        if isinstance(val, nn.Linear):
                            print(f"    {attr}: Linear({val.in_features}, {val.out_features})")
                            # Check if it has a mask
                            if hasattr(val, 'weight'):
                                weight = val.weight
                                print(f"      Weight shape: {list(weight.shape)}")
                                if hasattr(val, 'sparsity_mask') or hasattr(val, '_mask'):
                                    mask = getattr(val, 'sparsity_mask', getattr(val, '_mask', None))
                                    if mask is not None:
                                        print(f"      Has sparsity mask: {mask.shape if hasattr(mask, 'shape') else 'yes'}")
                        elif isinstance(val, nn.Parameter):
                            print(f"    {attr}: Parameter{list(val.shape)}")
        
        # Check for LSTM component
        if hasattr(cfc_cell.rnn_cell, 'lstm'):
            print(f"\n  LSTM component:")
            lstm = cfc_cell.rnn_cell.lstm
            print(f"    Type: {type(lstm)}")
            for attr in dir(lstm):
                if not attr.startswith('_') and hasattr(lstm, attr):
                    val = getattr(lstm, attr)
                    if isinstance(val, (nn.Parameter, nn.Module)):
                        if isinstance(val, nn.Linear):
                            print(f"      {attr}: Linear({val.in_features}, {val.out_features})")
                        elif isinstance(val, nn.Parameter):
                            print(f"      {attr}: Parameter{list(val.shape)}")
    
    # Check wiring attribute
    if hasattr(cfc_cell, 'wiring'):
        print(f"\nCfC wiring attribute:")
        wiring = cfc_cell.wiring
        print(f"  Type: {type(wiring)}")
        if hasattr(wiring, 'wiring_matrix'):
            wm = wiring.wiring_matrix
            print(f"  Wiring matrix shape: {wm.shape if hasattr(wm, 'shape') else 'N/A'}")
    
    # Check for any mask buffers
    print(f"\nBuffer analysis:")
    for name, buffer in cfc_cell.named_buffers():
        print(f"  {name}: {list(buffer.shape) if hasattr(buffer, 'shape') else buffer}")

def check_wiring_usage_in_forward():
    """Check if wiring is used as a mask during forward pass."""
    print(f"\n{'='*80}")
    print(f"CHECKING WIRING USAGE IN FORWARD PASS")
    print(f"{'='*80}")
    
    # Create a simple test
    arch4_path = Path("outputs/architectures/best_architecture_4_trial_178.json")
    if not arch4_path.exists():
        print("Architecture file not found")
        return
    
    wiring_original = load_architecture_from_file(str(arch4_path))
    wiring_original.input_size = 16
    wiring_original.output_size = 16
    built_wiring = wiring_original.build(16)
    
    cfc = CfC(
        input_size=16,
        units=built_wiring,
        proj_size=16,
        return_sequences=True,
        batch_first=True,
        mixed_memory=True,
    )
    
    # Create a test input
    test_input = torch.randn(2, 10, 16)  # batch=2, seq_len=10, features=16
    
    # Hook into forward pass to see what happens
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(input, tuple):
                print(f"  {name}: input shapes = {[x.shape if hasattr(x, 'shape') else type(x) for x in input]}")
            else:
                print(f"  {name}: input shape = {input.shape if hasattr(input, 'shape') else type(input)}")
            if isinstance(output, tuple):
                print(f"  {name}: output shapes = {[x.shape if hasattr(x, 'shape') else type(x) for x in output]}")
            else:
                print(f"  {name}: output shape = {output.shape if hasattr(output, 'shape') else type(output)}")
        return hook
    
    # Register hooks on linear layers
    for name, module in cfc.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    print("Running forward pass...")
    with torch.no_grad():
        output, _ = cfc(test_input)
    
    print(f"Output shape: {output.shape}")
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()

def main():
    print("=" * 80)
    print("CFC WIRING USAGE DIAGNOSTIC")
    print("=" * 80)
    
    # 1. Load and build wiring
    arch4_path = Path("outputs/architectures/best_architecture_4_trial_178.json")
    if not arch4_path.exists():
        print(f"ERROR: Architecture file not found: {arch4_path}")
        return
    
    wiring_original = load_architecture_from_file(str(arch4_path))
    wiring_original.input_size = 16
    wiring_original.output_size = 16
    built_wiring = wiring_original.build(16)
    
    print(f"\n1. Built wiring info:")
    if hasattr(built_wiring, 'wiring_matrix'):
        wm = np.array(built_wiring.wiring_matrix)
        print(f"   Wiring matrix: {wm.shape}, {np.count_nonzero(wm)} non-zero entries")
        print(f"   Density: {100 * np.count_nonzero(wm) / wm.size:.2f}%")
    
    # 2. Create CfC with wiring
    print(f"\n2. Creating CfC with wiring...")
    cfc = CfC(
        input_size=16,
        units=built_wiring,
        proj_size=16,
        return_sequences=True,
        batch_first=True,
        mixed_memory=True,
    )
    
    print(f"   Wired mode: {cfc.wired_mode}")
    
    # 3. Inspect internal structure
    inspect_cfc_internals(cfc)
    
    # 4. Check parameter counts
    print(f"\n3. Parameter analysis:")
    total_params = sum(p.numel() for p in cfc.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    
    # Count by component
    layer_params = {}
    for name, param in cfc.named_parameters():
        if param.requires_grad:
            component = name.split('.')[0] if '.' in name else name
            if component not in layer_params:
                layer_params[component] = 0
            layer_params[component] += param.numel()
    
    print(f"\n   Parameters by component:")
    for comp, count in sorted(layer_params.items()):
        print(f"     {comp}: {count:,}")
    
    # 5. Check if wiring is used as mask
    print(f"\n4. Checking for sparsity masks...")
    has_masks = False
    for name, module in cfc.named_modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, 'sparsity_mask') or hasattr(module, '_mask') or hasattr(module, 'mask'):
                mask = getattr(module, 'sparsity_mask', getattr(module, '_mask', getattr(module, 'mask', None)))
                if mask is not None:
                    has_masks = True
                    mask_nnz = torch.count_nonzero(mask).item() if isinstance(mask, torch.Tensor) else 'N/A'
                    print(f"   {name}: Has mask (non-zero: {mask_nnz})")
    
    if not has_masks:
        print(f"   No sparsity masks found - wiring may not be applied as masks!")
    
    # 6. Try forward pass inspection
    check_wiring_usage_in_forward()

if __name__ == "__main__":
    main()
