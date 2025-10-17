#!/usr/bin/env python3
"""
Debug script to check the actual number of channels in Lee2019_SSVEP dataset.
"""

import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def debug_lee2019_channels():
    """Debug the actual channel configuration of Lee2019_SSVEP."""
    try:
        from moabb.datasets import Lee2019_SSVEP
        from moabb.paradigms import SSVEP
        
        # Create dataset and paradigm
        dataset = Lee2019_SSVEP()
        paradigm = SSVEP(
            n_classes=4,
            tmin=0.0,
            tmax=4.0,
            baseline=None,
            resample=None
        )
        
        # Get data for one subject
        X, y, metadata = paradigm.get_data(dataset, subjects=[1])
        
        print(f"Dataset: {dataset.code}")
        print(f"Data shape: {X.shape}")
        print(f"Number of channels: {X.shape[1]}")
        print(f"Number of time points: {X.shape[2]}")
        print(f"Number of epochs: {X.shape[0]}")
        print(f"Number of classes: {len(set(y))}")
        print(f"Classes: {set(y)}")
        
        # Check if it's MNE epochs object
        if hasattr(X, 'info'):
            print(f"MNE info channels: {len(X.info['ch_names'])}")
            print(f"Channel names: {X.info['ch_names']}")
        else:
            print("Data is numpy array, not MNE epochs")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_lee2019_channels()

