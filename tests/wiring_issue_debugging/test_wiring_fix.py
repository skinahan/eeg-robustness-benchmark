"""
Test script to verify the wiring fix creates sparse matrices correctly.
"""

import torch
import numpy as np
from pathlib import Path
from models.branched_wiredcfc import create_branched_wiredcfc_classifier
from models.cnnncp import create_cnnncp_classifier
from architecture_refinement.arbitrary_wiring import load_architecture_from_file

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Standard input dimensions
    n_chans = 22
    n_times = 1001
    n_outputs = 2
    
    print("=" * 80)
    print("TESTING WIRING FIX")
    print("=" * 80)
    
    arch4_path = Path("outputs/architectures/best_architecture_4_trial_178.json")
    if not arch4_path.exists():
        print(f"ERROR: Architecture file not found: {arch4_path}")
        return
    
    # Test 1: Correct behavior (default)
    print("\n1. Testing CORRECT behavior (sparse matrices):")
    print("-" * 80)
    wiring_correct = load_architecture_from_file(str(arch4_path), use_legacy_behavior=False)
    
    branched_wiredcfc_correct = create_branched_wiredcfc_classifier(
        n_chans, n_times, n_outputs, wiring=wiring_correct
    )
    branched_wiredcfc_correct.initialize()
    
    params_correct = count_parameters(branched_wiredcfc_correct.module_)
    print(f"  Total parameters: {params_correct:,}")
    
    # Check recurrent cell parameters
    if hasattr(branched_wiredcfc_correct.module_, 'recurrent_cell'):
        cfc_cell = branched_wiredcfc_correct.module_.recurrent_cell
        cfc_params = sum(p.numel() for p in cfc_cell.parameters() if p.requires_grad)
        print(f"  Recurrent cell parameters: {cfc_params:,}")
        print(f"  Percentage of total: {100 * cfc_params / params_correct:.1f}%")
    
    # Test 2: Legacy behavior (for comparison)
    print("\n2. Testing LEGACY behavior (dense matrices - incorrect):")
    print("-" * 80)
    wiring_legacy = load_architecture_from_file(str(arch4_path), use_legacy_behavior=True)
    
    branched_wiredcfc_legacy = create_branched_wiredcfc_classifier(
        n_chans, n_times, n_outputs, wiring=wiring_legacy
    )
    branched_wiredcfc_legacy.initialize()
    
    params_legacy = count_parameters(branched_wiredcfc_legacy.module_)
    print(f"  Total parameters: {params_legacy:,}")
    
    # Check recurrent cell parameters
    if hasattr(branched_wiredcfc_legacy.module_, 'recurrent_cell'):
        cfc_cell = branched_wiredcfc_legacy.module_.recurrent_cell
        cfc_params = sum(p.numel() for p in cfc_cell.parameters() if p.requires_grad)
        print(f"  Recurrent cell parameters: {cfc_params:,}")
        print(f"  Percentage of total: {100 * cfc_params / params_legacy:.1f}%")
    
    # Test 3: CNN-NCP for comparison
    print("\n3. CNN-NCP (baseline):")
    print("-" * 80)
    cnn_ncp = create_cnnncp_classifier(n_chans, n_times, n_outputs)
    cnn_ncp.initialize()
    cnn_ncp_params = count_parameters(cnn_ncp.module_)
    print(f"  Total parameters: {cnn_ncp_params:,}")
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"  Correct (sparse):  {params_correct:,} parameters")
    print(f"  Legacy (dense):    {params_legacy:,} parameters")
    print(f"  CNN-NCP:           {cnn_ncp_params:,} parameters")
    print(f"\n  Ratio (Correct/CNN-NCP):   {params_correct / cnn_ncp_params:.2f}x")
    print(f"  Ratio (Legacy/CNN-NCP):    {params_legacy / cnn_ncp_params:.2f}x")
    print(f"  Ratio (Legacy/Correct):    {params_legacy / params_correct:.2f}x")
    
    # Expected: Correct should be much closer to CNN-NCP than Legacy
    if params_correct < params_legacy * 0.5:
        print("\n  ✓ SUCCESS: Correct behavior has significantly fewer parameters!")
    else:
        print("\n  ✗ WARNING: Correct behavior still has too many parameters")
    
    if abs(params_correct / cnn_ncp_params - 1.0) < 2.0:
        print("  ✓ SUCCESS: Correct behavior is close to CNN-NCP!")
    else:
        print(f"  ⚠ NOTE: Correct behavior is {params_correct / cnn_ncp_params:.1f}x CNN-NCP (should be ~1x)")

if __name__ == "__main__":
    main()
