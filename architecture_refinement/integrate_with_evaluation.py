#!/usr/bin/env python3
"""
Integration script for CNNWiredCfC models with existing evaluation framework.

This script demonstrates how to use the new CNNWiredCfC models with optimized
architectures from the architecture search in the existing evaluation pipeline.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import numpy as np
from architecture_refinement.arbitrary_wiring import load_architecture_from_file, create_wiring_from_architecture_data
from models.cnnncp import create_cnnwiredcfc_classifier


def create_wiredcfc_model_from_architecture(architecture_file: str, 
                                          n_chans: int, 
                                          n_times: int, 
                                          n_outputs: int,
                                          logger: Optional[logging.Logger] = None) -> Any:
    """
    Create a CNNWiredCfC model from an architecture file.
    
    This function can be used to integrate with the existing evaluation framework
    by creating model instances with specific optimized architectures.
    
    Args:
        architecture_file: Path to the architecture JSON file
        n_chans: Number of EEG channels
        n_times: Number of time points
        n_outputs: Number of output classes
        logger: Optional logger for output
        
    Returns:
        CNNWiredCfC classifier instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        # Load the architecture
        logger.info(f"Loading architecture from {architecture_file}")
        wiring = load_architecture_from_file(architecture_file, logger)
        
        
        # Create the model
        logger.info("Creating CNNWiredCfC model")
        model = create_cnnwiredcfc_classifier(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            wiring=wiring,
            drop_prob=0.15,
            lr=1e-3,
            batch_size=64,
            weight_decay=1e-3,
            F1=8,
            D=2,
            kernel_length=128,
            temporal_kernel_size=3,
            temporal_stride=4,
            max_seq_length=250
        )
        
        logger.info("Model created successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error creating model from architecture {architecture_file}: {e}")
        raise


def create_model_factory_from_architecture(architecture_file: str, 
                                         logger: Optional[logging.Logger] = None):
    """
    Create a model factory function from an architecture file.
    
    This returns a function that matches the signature expected by the evaluation
    framework: f(n_chans, n_times, n_outputs) -> model
    
    Args:
        architecture_file: Path to the architecture JSON file
        logger: Optional logger for output
        
    Returns:
        Model factory function
    """
    def model_factory(n_chans, n_times, n_outputs):
        return create_wiredcfc_model_from_architecture(
            architecture_file, n_chans, n_times, n_outputs, logger
        )
    
    return model_factory


def get_architecture_summary(architecture_file: str) -> Dict[str, Any]:
    """
    Get a summary of an architecture without creating the full model.
    
    Args:
        architecture_file: Path to the architecture JSON file
        
    Returns:
        Dictionary containing architecture summary
    """
    try:
        wiring = load_architecture_from_file(architecture_file)
        return wiring.get_wiring_summary()
    except Exception as e:
        return {'error': str(e)}


def list_available_architectures(architectures_dir: str = "outputs/architectures") -> Dict[str, Dict[str, Any]]:
    """
    List all available architectures with their summaries.
    
    Args:
        architectures_dir: Directory containing architecture files
        
    Returns:
        Dictionary mapping filename to architecture summary
    """
    arch_path = Path(architectures_dir)
    if not arch_path.exists():
        return {}
    
    architectures = {}
    
    for json_file in arch_path.glob("*.json"):
        try:
            summary = get_architecture_summary(str(json_file))
            architectures[json_file.name] = summary
        except Exception as e:
            architectures[json_file.name] = {'error': str(e)}
    
    return architectures


def create_architecture_registry(architectures_dir: str = "outputs/architectures") -> Dict[str, Any]:
    """
    Create a registry of all available architectures for easy access.
    
    Args:
        architectures_dir: Directory containing architecture files
        
    Returns:
        Dictionary mapping architecture names to model factories
    """
    arch_path = Path(architectures_dir)
    if not arch_path.exists():
        return {}
    
    registry = {}
    
    for json_file in arch_path.glob("*.json"):
        try:
            # Create a model factory for this architecture
            model_factory = create_model_factory_from_architecture(str(json_file))
            
            # Get architecture summary
            summary = get_architecture_summary(str(json_file))
            
            # Register with a descriptive name
            arch_name = f"wiredcfc_{json_file.stem}"
            registry[arch_name] = {
                'factory': model_factory,
                'architecture_file': str(json_file),
                'summary': summary
            }
            
        except Exception as e:
            print(f"Warning: Could not register architecture {json_file.name}: {e}")
            continue
    
    return registry


def example_usage():
    """Example of how to use the integration functions."""
    print("CNNWiredCfC Architecture Integration Example")
    print("=" * 50)
    
    # List available architectures
    print("\n1. Available architectures:")
    architectures = list_available_architectures()
    for filename, summary in architectures.items():
        if 'error' not in summary:
            print(f"  {filename}: {summary['input_size']}->{summary['hidden_size']}->{summary['output_size']} "
                  f"({summary['total_connections']} connections)")
        else:
            print(f"  {filename}: ERROR - {summary['error']}")
    
    # Create architecture registry
    print("\n2. Creating architecture registry...")
    registry = create_architecture_registry()
    print(f"  Registered {len(registry)} architectures")
    
    # Show registry contents
    for name, info in registry.items():
        summary = info['summary']
        if 'error' not in summary:
            print(f"  {name}: {summary['input_size']}->{summary['hidden_size']}->{summary['output_size']}")
    
    # Example of creating a specific model
    if registry:
        print("\n3. Example: Creating a model from first architecture...")
        first_arch = list(registry.keys())[0]
        arch_info = registry[first_arch]
        
        print(f"  Using architecture: {first_arch}")
        print(f"  File: {arch_info['architecture_file']}")
        
        # This would create the actual model (commented out to avoid loading data)
        # model = arch_info['factory'](22, 1000, 2)  # Example dimensions
        # print(f"  Model created: {type(model)}")
    
    print("\n4. Integration with evaluation framework:")
    print("  - Use create_model_factory_from_architecture() to create model factories")
    print("  - These factories can be registered in config.py MODEL_REGISTRY")
    print("  - Or used directly in evaluation scripts")
    print("  - Each architecture becomes a separate model variant")


if __name__ == "__main__":
    example_usage()
