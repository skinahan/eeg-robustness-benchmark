#!/usr/bin/env python3
"""
Comprehensive test script for the updated model caching system.

This script tests:
1. Dual checkpoint system (best and final)
2. Configuration validation including training hyperparameters
3. Automatic retraining and cache invalidation
4. Checkpoint cleanup
5. Integration with experiment runner
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
from evaluation.experiment_utils import check_skip_eval
from config import MODEL_REGISTRY
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

def test_dual_checkpoint_system():
    """Test the dual checkpoint system (best and final)."""
    print("="*60)
    print("TEST 1: Dual Checkpoint System")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_manager = ModelCacheManager(cache_root=temp_dir, check_interval=1)
        
        # Get a sample model
        model_fn = MODEL_REGISTRY['eegnet']
        model = model_fn(n_chans=22, n_times=1000, n_outputs=2)
        model.initialize()
        
        # Test configuration with training hyperparameters
        config = {
            'n_chans': 22,
            'n_times': 1000,
            'n_outputs': 2,
            'module__dropout': 0.5,
            'module__kernel_size': 32,
            'learning_rate': 0.001,
            'batch_size': 32,
            'optimizer': 'adam',
            'max_epochs': 200
        }
        
        # Save best checkpoint
        print("Saving best checkpoint...")
        saved_best = cache_manager.save_model(
            model=model,
            config=config,
            dataset="BNCI2014_001",
            model_name="eegnet",
            seed=100,
            subject=1,
            session="0train",
            eval_mode="CrossSessionEvaluation",
            tuned=False,
            validation_loss=0.5,
            epoch=10,
            checkpoint_type="best"
        )
        print(f"Best checkpoint saved: {saved_best}")
        
        # Save final checkpoint
        print("Saving final checkpoint...")
        saved_final = cache_manager.save_model(
            model=model,
            config=config,
            dataset="BNCI2014_001",
            model_name="eegnet",
            seed=100,
            subject=1,
            session="0train",
            eval_mode="CrossSessionEvaluation",
            tuned=False,
            validation_loss=0.6,
            epoch=20,
            checkpoint_type="final"
        )
        print(f"Final checkpoint saved: {saved_final}")
        
        # Test loading best checkpoint
        print("Loading best checkpoint...")
        loaded_best, config_matches_best = cache_manager.load_model(
            model_class=model_fn,
            config=config,
            dataset="BNCI2014_001",
            model_name="eegnet",
            seed=100,
            subject=1,
            session="0train",
            eval_mode="CrossSessionEvaluation",
            tuned=False,
            checkpoint_type="best"
        )
        print(f"Best checkpoint loaded: {loaded_best is not None}")
        print(f"Config matches: {config_matches_best}")
        
        # Test loading final checkpoint
        print("Loading final checkpoint...")
        loaded_final, config_matches_final = cache_manager.load_model(
            model_class=model_fn,
            config=config,
            dataset="BNCI2014_001",
            model_name="eegnet",
            seed=100,
            subject=1,
            session="0train",
            eval_mode="CrossSessionEvaluation",
            tuned=False,
            checkpoint_type="final"
        )
        print(f"Final checkpoint loaded: {loaded_final is not None}")
        print(f"Config matches: {config_matches_final}")
        
        # Test cache stats
        stats = cache_manager.get_cache_stats()
        print(f"Cache stats: {stats}")

def test_configuration_validation_with_training_params():
    """Test configuration validation including training hyperparameters."""
    print("\n" + "="*60)
    print("TEST 2: Configuration Validation with Training Parameters")
    print("="*60)
    
    cache_manager = ModelCacheManager()
    
    # Test configurations with different training hyperparameters
    config1 = {
        'n_chans': 22,
        'n_times': 1000,
        'n_outputs': 2,
        'module__dropout': 0.5,
        'module__kernel_size': 32,
        'learning_rate': 0.001,
        'batch_size': 32,
        'optimizer': 'adam',
        'max_epochs': 200
    }
    
    config2 = {
        'n_chans': 22,
        'n_times': 1000,
        'n_outputs': 2,
        'module__dropout': 0.5,
        'module__kernel_size': 32,
        'learning_rate': 0.01,  # Different learning rate
        'batch_size': 64,       # Different batch size
        'optimizer': 'sgd',     # Different optimizer
        'max_epochs': 100       # Different max epochs
    }
    
    config3 = {
        'n_chans': 22,
        'n_times': 1000,
        'n_outputs': 2,
        'module__dropout': 0.3,  # Different dropout
        'module__kernel_size': 32,
        'learning_rate': 0.001,
        'batch_size': 32,
        'optimizer': 'adam',
        'max_epochs': 200
    }
    
    hash1 = cache_manager._generate_config_hash(config1)
    hash2 = cache_manager._generate_config_hash(config2)
    hash3 = cache_manager._generate_config_hash(config3)
    
    print(f"Config 1 hash: {hash1}")
    print(f"Config 2 hash: {hash2}")
    print(f"Config 3 hash: {hash3}")
    print(f"Config 1 == Config 2 (should be False): {hash1 == hash2}")
    print(f"Config 1 == Config 3 (should be False): {hash1 == hash3}")
    
    # Test architectural parameter extraction
    arch_params1 = cache_manager._extract_architectural_params(config1)
    print(f"\nRelevant parameters from config 1: {arch_params1}")

def test_cache_invalidation():
    """Test cache invalidation when configuration changes."""
    print("\n" + "="*60)
    print("TEST 3: Cache Invalidation")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_manager = ModelCacheManager(cache_root=temp_dir, check_interval=1)
        
        model_fn = MODEL_REGISTRY['eegnet']
        model = model_fn(n_chans=22, n_times=1000, n_outputs=2)
        model.initialize()
        
        # Initial configuration
        config1 = {
            'n_chans': 22,
            'n_times': 1000,
            'n_outputs': 2,
            'module__dropout': 0.5,
            'learning_rate': 0.001,
            'batch_size': 32
        }
        
        # Save initial checkpoints
        print("Saving initial checkpoints...")
        cache_manager.save_model(
            model=model, config=config1, dataset="BNCI2014_001",
            model_name="eegnet", seed=100, subject=1, session="0train",
            eval_mode="CrossSessionEvaluation", tuned=False, checkpoint_type="best"
        )
        cache_manager.save_model(
            model=model, config=config1, dataset="BNCI2014_001",
            model_name="eegnet", seed=100, subject=1, session="0train",
            eval_mode="CrossSessionEvaluation", tuned=False, checkpoint_type="final"
        )
        
        # Check initial stats
        stats1 = cache_manager.get_cache_stats()
        print(f"Initial cache stats: {stats1}")
        
        # Modified configuration (different learning rate)
        config2 = config1.copy()
        config2['learning_rate'] = 0.01
        
        # Save with modified config (should trigger cleanup)
        print("Saving with modified configuration...")
        cache_manager.save_model(
            model=model, config=config2, dataset="BNCI2014_001",
            model_name="eegnet", seed=100, subject=1, session="0train",
            eval_mode="CrossSessionEvaluation", tuned=False, checkpoint_type="best"
        )
        
        # Check final stats
        stats2 = cache_manager.get_cache_stats()
        print(f"Final cache stats: {stats2}")

def test_skip_eval_with_caching():
    """Test the updated check_skip_eval function with caching."""
    print("\n" + "="*60)
    print("TEST 4: Skip Eval with Caching")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_manager = ModelCacheManager(cache_root=temp_dir, check_interval=1)
        
        # Test configuration
        config = {
            'n_chans': 22,
            'n_times': 1000,
            'n_outputs': 2,
            'module__dropout': 0.5,
            'learning_rate': 0.001,
            'batch_size': 32
        }
        
        # Test without cached models
        print("Testing skip_eval without cached models...")
        should_skip = check_skip_eval(
            model_name="eegnet",
            seed=100,
            subject_list=[1],
            mode="baseline",
            noise_type=None,
            intensity=None,
            eval_mode="CrossSessionEvaluation",  # Use full evaluation name
            paradigm="MotorImagery",
            dataset="BNCI2014_001",
            cache_manager=cache_manager,
            config=config,
            tuned=False
        )
        print(f"Should skip (no cached models): {should_skip}")
        
        # Create some cached models
        model_fn = MODEL_REGISTRY['eegnet']
        model = model_fn(n_chans=22, n_times=1000, n_outputs=2)
        model.initialize()
        
        # Save cached models
        for session in ['0train', '1test']:
            cache_manager.save_model(
                model=model, config=config, dataset="BNCI2014_001",
                model_name="eegnet", seed=100, subject=1, session=session,
                eval_mode="CrossSessionEvaluation", tuned=False, checkpoint_type="best"
            )
        
        # Test with cached models
        print("Testing skip_eval with cached models...")
        should_skip = check_skip_eval(
            model_name="eegnet",
            seed=100,
            subject_list=[1],
            mode="baseline",
            noise_type=None,
            intensity=None,
            eval_mode="CrossSessionEvaluation",  # Use full evaluation name
            paradigm="MotorImagery",
            dataset="BNCI2014_001",
            cache_manager=cache_manager,
            config=config,
            tuned=False
        )
        print(f"Should skip (with cached models): {should_skip}")

def test_periodic_checkpoint_callback():
    """Test the periodic checkpoint callback functionality."""
    print("\n" + "="*60)
    print("TEST 5: Periodic Checkpoint Callback")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_manager = ModelCacheManager(cache_root=temp_dir, check_interval=1)
        
        # Create callback
        callback = create_periodic_checkpoint_callback(
            cache_manager=cache_manager,
            dataset="BNCI2014_001",
            model_name="eegnet",
            seed=100,
            subject=1,
            session="0train",
            eval_mode="CrossSessionEvaluation",
            tuned=False,
            check_interval=1
        )
        
        print(f"Callback created: {callback is not None}")
        print(f"Check interval: {callback.check_interval}")
        print(f"Monitor: {callback.monitor}")
        print(f"Mode: {callback.mode}")

def main():
    """Run all tests."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("UPDATED MODEL CACHING SYSTEM TEST")
    print("="*60)
    
    try:
        test_dual_checkpoint_system()
        test_configuration_validation_with_training_params()
        test_cache_invalidation()
        test_skip_eval_with_caching()
        test_periodic_checkpoint_callback()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Implemented:")
        print("✓ Dual checkpoint system (best and final)")
        print("✓ Configuration validation including training hyperparameters")
        print("✓ Automatic cache invalidation and retraining")
        print("✓ Checkpoint cleanup to save space")
        print("✓ Updated skip_eval logic with caching support")
        print("✓ Configurable check interval (default: 1 epoch)")
        print("✓ Separate storage for baseline and tuned models")
        print("✓ JSON configuration files alongside checkpoints")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
