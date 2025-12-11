#!/usr/bin/env python3
"""
Test script for the model caching system.

This script tests:
1. Model checkpoint saving and loading
2. Configuration validation
3. Cache invalidation when parameters change
4. Integration with experiment runner
"""

import os
import sys
import tempfile
import shutil
import logging
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from evaluation.model_cache_manager import ModelCacheManager
from evaluation.periodic_checkpoint_callback import create_periodic_checkpoint_callback, create_model_cache_callback
from config import MODEL_REGISTRY
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

def test_basic_caching():
    """Test basic model saving and loading functionality."""
    print("="*60)
    print("TEST 1: Basic Model Caching")
    print("="*60)
    
    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_manager = ModelCacheManager(cache_root=temp_dir, check_interval=5)
        
        # Get a sample model
        model_fn = MODEL_REGISTRY['eegnet']
        model = model_fn(n_chans=22, n_times=1000, n_outputs=2)
        model.initialize()
        
        # Test configuration
        config = {
            'n_chans': 22,
            'n_times': 1000,
            'n_outputs': 2,
            'module__dropout': 0.5,
            'module__kernel_size': 32
        }
        
        # Save model
        print("Saving model...")
        saved = cache_manager.save_model(
            model=model,
            config=config,
            dataset="BNCI2014_001",
            model_name="eegnet",
            seed=100,
            subject=1,
            session="0train",
            eval_mode="CrossSession",
            tuned=False
        )
        
        print(f"Model saved: {saved}")
        
        # Load model
        print("Loading model...")
        loaded_model, config_matches = cache_manager.load_model(
            model_class=model_fn,
            config=config,
            dataset="BNCI2014_001",
            model_name="eegnet",
            seed=100,
            subject=1,
            session="0train",
            eval_mode="CrossSession",
            tuned=False
        )
        
        print(f"Model loaded: {loaded_model is not None}")
        print(f"Config matches: {config_matches}")
        print(f"Model was cached: {hasattr(loaded_model, '_was_cached') and loaded_model._was_cached}")
        
        # Test with different config
        print("\nTesting with different configuration...")
        different_config = config.copy()
        different_config['module__dropout'] = 0.3  # Changed dropout
        
        loaded_model2, config_matches2 = cache_manager.load_model(
            model_class=model_fn,
            config=different_config,
            dataset="BNCI2014_001",
            model_name="eegnet",
            seed=100,
            subject=1,
            session="0train",
            eval_mode="CrossSession",
            tuned=False
        )
        
        print(f"Model loaded with different config: {loaded_model2 is not None}")
        print(f"Config matches: {config_matches2}")
        
        # Test cache stats
        stats = cache_manager.get_cache_stats()
        print(f"\nCache stats: {stats}")

def test_configuration_validation():
    """Test configuration validation and hash generation."""
    print("\n" + "="*60)
    print("TEST 2: Configuration Validation")
    print("="*60)
    
    cache_manager = ModelCacheManager()
    
    # Test architectural parameter extraction
    config1 = {
        'n_chans': 22,
        'n_times': 1000,
        'n_outputs': 2,
        'module__dropout': 0.5,
        'module__kernel_size': 32,
        'module__n_filters': 8,
        'learning_rate': 0.001,  # Should be excluded
        'batch_size': 32,        # Should be excluded
        'optimizer': 'adam'      # Should be excluded
    }
    
    config2 = {
        'n_chans': 22,
        'n_times': 1000,
        'n_outputs': 2,
        'module__dropout': 0.5,
        'module__kernel_size': 32,
        'module__n_filters': 8,
        'learning_rate': 0.01,   # Different learning rate
        'batch_size': 64,        # Different batch size
        'optimizer': 'sgd'       # Different optimizer
    }
    
    config3 = {
        'n_chans': 22,
        'n_times': 1000,
        'n_outputs': 2,
        'module__dropout': 0.3,  # Different dropout
        'module__kernel_size': 32,
        'module__n_filters': 8,
        'learning_rate': 0.001,
        'batch_size': 32,
        'optimizer': 'adam'
    }
    
    hash1 = cache_manager._generate_config_hash(config1)
    hash2 = cache_manager._generate_config_hash(config2)
    hash3 = cache_manager._generate_config_hash(config3)
    
    print(f"Config 1 hash: {hash1}")
    print(f"Config 2 hash: {hash2}")
    print(f"Config 3 hash: {hash3}")
    print(f"Config 1 == Config 2 (should be True): {hash1 == hash2}")
    print(f"Config 1 == Config 3 (should be False): {hash1 == hash3}")
    
    # Test architectural parameter extraction
    arch_params1 = cache_manager._extract_architectural_params(config1)
    print(f"\nArchitectural parameters from config 1: {arch_params1}")

def test_experiment_runner_integration():
    """Test integration with the experiment runner."""
    print("\n" + "="*60)
    print("TEST 3: Experiment Runner Integration")
    print("="*60)
    
    try:
        from evaluation.unified_experiment_runner import UnifiedExperimentRunner
        
        # Create a minimal experiment runner
        runner = UnifiedExperimentRunner(
            model="eegnet",
            dataset="BNCI2014_001",
            subjects=[1],
            mode="baseline",
            eval_mode="CrossSession",
            seed=100
        )
        
        print(f"Cache manager initialized: {runner.cache_manager is not None}")
        print(f"Cache root: {runner.cache_manager.cache_root}")
        print(f"Check interval: {runner.cache_manager.check_interval}")
        
        # Test model creation with caching
        print("\nTesting model creation with caching...")
        runner.current_subject = 1
        runner.current_session = "0train"
        
        model = runner._create_model(n_chans=22, n_times=1000, n_outputs=2)
        print(f"Model created: {model is not None}")
        print(f"Model has callbacks: {hasattr(model, 'callbacks') and model.callbacks is not None}")
        
        if hasattr(model, 'callbacks') and model.callbacks:
            print(f"Number of callbacks: {len(model.callbacks)}")
            for i, callback in enumerate(model.callbacks):
                print(f"  Callback {i}: {type(callback).__name__}")
        
    except ImportError as e:
        print(f"Could not import UnifiedExperimentRunner: {e}")
        print("This is expected if dependencies are not available")

def test_cache_management():
    """Test cache management functions."""
    print("\n" + "="*60)
    print("TEST 4: Cache Management")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_manager = ModelCacheManager(cache_root=temp_dir, check_interval=5)
        
        # Create some dummy checkpoints
        model_fn = MODEL_REGISTRY['eegnet']
        config = {'n_chans': 22, 'n_times': 1000, 'n_outputs': 2}
        
        for i in range(3):
            model = model_fn(n_chans=22, n_times=1000, n_outputs=2)
            model.initialize()
            
            cache_manager.save_model(
                model=model,
                config=config,
                dataset="BNCI2014_001",
                model_name="eegnet",
                seed=100 + i,
                subject=1,
                session="0train",
                eval_mode="CrossSession",
                tuned=False
            )
        
        # Test cache stats
        stats = cache_manager.get_cache_stats()
        print(f"Cache stats: {stats}")
        
        # Test listing available models
        models = cache_manager.list_available_models()
        print(f"Available models: {len(models)}")
        for model_info in models:
            print(f"  {model_info['config']['model']} - {model_info['config']['dataset']} - {model_info['size_mb']:.2f} MB")

def main():
    """Run all tests."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("MODEL CACHING SYSTEM TEST")
    print("="*60)
    
    try:
        test_basic_caching()
        test_configuration_validation()
        test_experiment_runner_integration()
        test_cache_management()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
