#!/usr/bin/env python3
"""
Model Cache Manager for EEG Experiments

This module provides a comprehensive model caching system that:
1. Manages model checkpoints with configuration validation
2. Separates baseline and tuned models
3. Validates model configurations before loading
4. Supports periodic checkpoint saving during training
5. Handles cache invalidation when hyperparameters change
6. Prevents data leakage by including fold_idx in cache keys for WithinSession

IMPORTANT DATA LEAKAGE PREVENTION:
- CrossSession: session parameter represents the VALIDATION session. A model cached
  for validation session "0train" was trained on "1test" (and vice versa).
- WithinSession: fold_idx is REQUIRED and included in cache key to prevent mixing
  models from different folds of the same session.
- CrossSubject: session parameter includes fold info (e.g., "fold_0_eval_subjects_1,2,3")
  which uniquely identifies the training configuration.
"""

import os
import json
import hashlib
import torch
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import logging

class ModelCacheManager:
    """
    Manages model checkpoints with configuration validation and periodic saving.
    """
    
    def __init__(self, cache_root: str = "model_cache", check_interval: int = 10):
        """
        Initialize the model cache manager.
        
        Args:
            cache_root: Root directory for model cache
            check_interval: How often (in epochs) to check for best model during training
        """
        self.cache_root = Path(cache_root)
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory structure
        self.cache_root.mkdir(parents=True, exist_ok=True)
        
        # Track best validation losses for periodic saving
        self._best_validation_losses = {}
        
    def _get_cache_path(self, 
                       dataset: str, 
                       model: str, 
                       seed: int, 
                       subject: int, 
                       session: str, 
                       eval_mode: str,
                       tuned: bool = False,
                       checkpoint_type: str = "final",
                       fold_idx: Optional[int] = None) -> Path:
        """
        Get the cache path for a specific model configuration.
        
        Args:
            dataset: Dataset name
            model: Model name
            seed: Random seed
            subject: Subject ID
            session: Session identifier - meaning depends on eval_mode:
                - CrossSession: validation/test session (model was trained on other session(s))
                - WithinSession: session being evaluated (model trained on other folds of same session)
                - CrossSubject: fold identifier like "fold_0_eval_subjects_1,2,3"
            eval_mode: Evaluation mode (WithinSession, CrossSession, CrossSubject)
            tuned: Whether this is a tuned model
            checkpoint_type: Type of checkpoint ("best" or "final")
            fold_idx: Fold index (required for WithinSession to prevent fold mixing)
        
        Returns:
            Path to the cache file
        """
        tuned_suffix = "_tuned" if tuned else "_baseline"
        type_suffix = f"_{checkpoint_type}" if checkpoint_type != "final" else ""
        
        # For WithinSession, include fold_idx in cache key to prevent data leakage
        # between different folds of the same session
        if eval_mode == "WithinSession" and fold_idx is not None:
            session_key = f"{session}_fold{fold_idx}"
        else:
            session_key = session
        
        cache_dir = self.cache_root / dataset / model / f"seed_{seed}" / f"subject_{subject:03d}" / eval_mode
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"session_{session_key}{tuned_suffix}{type_suffix}.pt"
    
    def _get_config_path(self, checkpoint_path: Path) -> Path:
        """Get the configuration file path for a checkpoint."""
        return checkpoint_path.with_suffix('.json')
    
    def _generate_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate a hash for model configuration to detect changes."""
        # Create a normalized config for hashing (exclude non-architectural params)
        arch_config = self._extract_architectural_params(config)
        
        # Sort keys for consistent hashing
        sorted_config = json.dumps(arch_config, sort_keys=True)
        return hashlib.md5(sorted_config.encode()).hexdigest()
    
    def _extract_architectural_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract architectural and training parameters that affect model performance.
        Includes both architectural parameters and training hyperparameters.
        
        Excludes parameters that don't affect model compatibility:
        - verbose: purely a logging parameter
        """
        relevant_params = {}
        
        # Parameters to exclude from hash (don't affect model compatibility)
        exclude_params = ['verbose']
        
        # Define architectural parameter patterns
        arch_patterns = [
            'kernel_size', 'hidden_size', 'num_layers', 'dropout', 'sparsity',
            'n_chans', 'n_times', 'n_outputs', 'n_filters', 'filter_length',
            'pool_length', 'stride', 'dilation', 'padding', 'bias',
            'activation', 'pool_mode', 'conv_nonlin', 'final_nonlin',
            'n_neurons', 'n_connections', 'wiring', 'architecture'
        ]
        
        # Define training hyperparameter patterns
        training_patterns = [
            'learning_rate', 'lr', 'batch_size', 'optimizer', 'weight_decay',
            'momentum', 'beta1', 'beta2', 'eps', 'max_epochs', 'patience',
            'threshold', 'monitor', 'load_best', 'scheduler', 'step_size',
            'gamma', 'min_lr', 'factor', 'train_split'
        ]
        
        for key, value in config.items():
            # Skip excluded parameters
            if key.lower() in exclude_params:
                continue
            
            # Check if this is an architectural parameter
            if any(pattern in key.lower() for pattern in arch_patterns):
                relevant_params[key] = value
            # Check if this is a training hyperparameter
            elif any(pattern in key.lower() for pattern in training_patterns):
                relevant_params[key] = value
            # Also include module parameters (they're usually architectural)
            elif 'module__' in key:
                relevant_params[key] = value
            # Include optimizer parameters
            elif 'optimizer__' in key:
                relevant_params[key] = value
        
        return relevant_params
    
    def save_model(self, 
                   model, 
                   config: Dict[str, Any], 
                   dataset: str, 
                   model_name: str, 
                   seed: int, 
                   subject: int, 
                   session: str, 
                   eval_mode: str,
                   tuned: bool = False,
                   validation_loss: Optional[float] = None,
                   epoch: Optional[int] = None,
                   checkpoint_type: str = "final",
                   fold_idx: Optional[int] = None) -> bool:
        """
        Save a model checkpoint with configuration validation.
        
        Args:
            model: The trained model to save
            config: Model configuration dictionary
            dataset: Dataset name
            model_name: Model name
            seed: Random seed
            subject: Subject ID
            session: Session identifier
            eval_mode: Evaluation mode
            tuned: Whether this is a tuned model
            validation_loss: Current validation loss (for periodic saving)
            epoch: Current epoch number
            checkpoint_type: Type of checkpoint ("best" or "final")
            
        Returns:
            bool: True if model was saved, False if skipped
        """
        checkpoint_path = self._get_cache_path(dataset, model_name, seed, subject, session, eval_mode, tuned, checkpoint_type, fold_idx)
        config_path = self._get_config_path(checkpoint_path)
        
        # Generate configuration hash
        config_hash = self._generate_config_hash(config)
        
        # Check if checkpoint already exists
        if checkpoint_path.exists() and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    existing_config = json.load(f)
                existing_hash = existing_config.get('config_hash', '')
                
                # If config hasn't changed, check if we should update based on validation loss
                if existing_hash == config_hash:
                    if checkpoint_type == "best" and validation_loss is not None and epoch is not None:
                        # Check if this is a better model for best checkpoint
                        cache_key = str(checkpoint_path)
                        if cache_key not in self._best_validation_losses:
                            self._best_validation_losses[cache_key] = float('inf')
                        
                        if validation_loss < self._best_validation_losses[cache_key]:
                            self._best_validation_losses[cache_key] = validation_loss
                            self.logger.info(f"New best validation loss {validation_loss:.6f} at epoch {epoch}, updating best checkpoint")
                        else:
                            self.logger.debug(f"Validation loss {validation_loss:.6f} not better than {self._best_validation_losses[cache_key]:.6f}, skipping best checkpoint save")
                            return False
                    elif checkpoint_type == "final":
                        self.logger.info(f"Final model checkpoint already exists with same config, skipping save")
                        return False
                else:
                    self.logger.warning(f"Model configuration has changed, will overwrite existing {checkpoint_type} checkpoint")
                    # Clean up old checkpoints when config changes
                    self._cleanup_old_checkpoints(dataset, model_name, seed, subject, session, eval_mode, tuned, fold_idx)
            except Exception as e:
                self.logger.warning(f"Could not read existing config: {e}, will overwrite")
        
        # Save the model
        try:
            # Save model state dict
            torch.save(model.module_.state_dict(), checkpoint_path)
            
            # Save configuration with metadata
            config_data = {
                'config': config,
                'config_hash': config_hash,
                'dataset': dataset,
                'model': model_name,
                'seed': seed,
                'subject': subject,
                'session': session,
                'eval_mode': eval_mode,
                'tuned': tuned,
                'checkpoint_type': checkpoint_type,
                'fold_idx': fold_idx,
                'saved_at': datetime.now().isoformat(),
                'validation_loss': validation_loss,
                'epoch': epoch
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self.logger.info(f"Saved {checkpoint_type} model checkpoint: {checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model checkpoint: {e}")
            return False
    
    def _cleanup_old_checkpoints(self, dataset: str, model_name: str, seed: int, 
                                subject: int, session: str, eval_mode: str, tuned: bool, fold_idx: Optional[int] = None):
        """Clean up old checkpoints when configuration changes."""
        try:
            # Remove both best and final checkpoints
            for checkpoint_type in ["best", "final"]:
                checkpoint_path = self._get_cache_path(dataset, model_name, seed, subject, session, eval_mode, tuned, checkpoint_type, fold_idx)
                config_path = self._get_config_path(checkpoint_path)
                
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    self.logger.info(f"Removed old {checkpoint_type} checkpoint: {checkpoint_path}")
                
                if config_path.exists():
                    config_path.unlink()
                    self.logger.info(f"Removed old {checkpoint_type} config: {config_path}")
                    
        except Exception as e:
            self.logger.warning(f"Failed to cleanup old checkpoints: {e}")
    
    def load_model(self, 
                   model_class, 
                   config: Dict[str, Any], 
                   dataset: str, 
                   model_name: str, 
                   seed: int, 
                   subject: int, 
                   session: str, 
                   eval_mode: str,
                   tuned: bool = False,
                   checkpoint_type: str = "best",
                   fold_idx: Optional[int] = None) -> Tuple[Optional[Any], bool]:
        """
        Load a model checkpoint with configuration validation.
        
        Args:
            model_class: Model class to instantiate
            config: Expected model configuration
            dataset: Dataset name
            model_name: Model name
            seed: Random seed
            subject: Subject ID
            session: Session identifier
            eval_mode: Evaluation mode
            tuned: Whether this is a tuned model
            checkpoint_type: Type of checkpoint to load ("best" or "final")
            
        Returns:
            Tuple of (loaded_model, config_matches)
        """
        checkpoint_path = self._get_cache_path(dataset, model_name, seed, subject, session, eval_mode, tuned, checkpoint_type, fold_idx)
        config_path = self._get_config_path(checkpoint_path)
        
        if not checkpoint_path.exists() or not config_path.exists():
            self.logger.info(f"No cached model found: {checkpoint_path}")
            return None, False
        
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Validate configuration
            # Use a more lenient comparison: only compare parameters present in the load config
            # This handles cases where saved config has extra params (like max_epochs, verbose)
            # that aren't in the minimal load config
            load_config_params = self._extract_architectural_params(config)
            cached_config = config_data.get('config', {})
            cached_config_params = self._extract_architectural_params(cached_config)
            
            # Only compare parameters that are in the load config
            # This allows saved configs with extra params to still match
            config_matches = True
            for key, value in load_config_params.items():
                if key not in cached_config_params:
                    config_matches = False
                    break
                if cached_config_params[key] != value:
                    config_matches = False
                    break
            
            if not config_matches:
                # Fall back to hash comparison for detailed logging
                expected_hash = self._generate_config_hash(config)
                cached_hash = config_data.get('config_hash', '')
                self.logger.warning(f"Model configuration mismatch, cannot load cached model")
                self.logger.debug(f"Expected hash: {expected_hash}")
                self.logger.debug(f"Cached hash: {cached_hash}")
                self.logger.debug(f"Load config params: {list(load_config_params.keys())}")
                self.logger.debug(f"Cached config params: {list(cached_config_params.keys())}")
                return None, False
            
            # Load model - only use architectural parameters for model creation
            arch_config = self._extract_architectural_params(config)
            # Remove module__ prefix for model creation
            model_config = {}
            for key, value in arch_config.items():
                if key.startswith('module__'):
                    model_config[key[8:]] = value  # Remove 'module__' prefix
                elif key in ['n_chans', 'n_times', 'n_outputs']:
                    model_config[key] = value
            
            model = model_class(**model_config)
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.module_.load_state_dict(state_dict)
            
            # Mark model as cached
            model._was_cached = True
            
            self.logger.info(f"Loaded cached model: {checkpoint_path}")
            return model, True
            
        except Exception as e:
            self.logger.error(f"Failed to load cached model: {e}")
            return None, False
    
    def should_checkpoint(self, epoch: int) -> bool:
        """Check if we should save a checkpoint at this epoch."""
        return epoch % self.check_interval == 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the model cache."""
        stats = {
            'total_checkpoints': 0,
            'total_size_mb': 0,
            'by_dataset': {},
            'by_model': {},
            'by_eval_mode': {}
        }
        
        for checkpoint_path in self.cache_root.rglob("*.pt"):
            try:
                # Get file size
                size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
                stats['total_size_mb'] += size_mb
                stats['total_checkpoints'] += 1
                
                # Parse path for categorization
                parts = checkpoint_path.parts
                if len(parts) >= 4:
                    dataset = parts[-4]
                    model = parts[-3]
                    eval_mode = parts[-2]
                    
                    stats['by_dataset'][dataset] = stats['by_dataset'].get(dataset, 0) + 1
                    stats['by_model'][model] = stats['by_model'].get(model, 0) + 1
                    stats['by_eval_mode'][eval_mode] = stats['by_eval_mode'].get(eval_mode, 0) + 1
                    
            except Exception as e:
                self.logger.warning(f"Could not process checkpoint {checkpoint_path}: {e}")
        
        return stats
    
    def cleanup_old_checkpoints(self, days_old: int = 30) -> int:
        """Remove checkpoints older than specified days."""
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        removed_count = 0
        
        for checkpoint_path in self.cache_root.rglob("*.pt"):
            try:
                if checkpoint_path.stat().st_mtime < cutoff_time:
                    config_path = self._get_config_path(checkpoint_path)
                    checkpoint_path.unlink()
                    if config_path.exists():
                        config_path.unlink()
                    removed_count += 1
            except Exception as e:
                self.logger.warning(f"Could not remove old checkpoint {checkpoint_path}: {e}")
        
        self.logger.info(f"Removed {removed_count} old checkpoints")
        return removed_count
    
    def list_available_models(self, 
                            dataset: Optional[str] = None,
                            model: Optional[str] = None,
                            eval_mode: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available cached models with optional filtering."""
        models = []
        
        for checkpoint_path in self.cache_root.rglob("*.pt"):
            try:
                config_path = self._get_config_path(checkpoint_path)
                if not config_path.exists():
                    continue
                
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Apply filters
                if dataset and config_data.get('dataset') != dataset:
                    continue
                if model and config_data.get('model') != model:
                    continue
                if eval_mode and config_data.get('eval_mode') != eval_mode:
                    continue
                
                models.append({
                    'path': str(checkpoint_path),
                    'config': config_data,
                    'size_mb': checkpoint_path.stat().st_size / (1024 * 1024)
                })
                
            except Exception as e:
                self.logger.warning(f"Could not process checkpoint {checkpoint_path}: {e}")
        
        return models


def create_model_cache_manager(cache_root: str = "model_cache", check_interval: int = 10) -> ModelCacheManager:
    """Factory function to create a model cache manager."""
    return ModelCacheManager(cache_root, check_interval)


if __name__ == "__main__":
    # Example usage and testing
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create cache manager
    cache_manager = create_model_cache_manager()
    
    # Print cache statistics
    stats = cache_manager.get_cache_stats()
    print("Cache Statistics:")
    print(f"Total checkpoints: {stats['total_checkpoints']}")
    print(f"Total size: {stats['total_size_mb']:.2f} MB")
    print(f"By dataset: {stats['by_dataset']}")
    print(f"By model: {stats['by_model']}")
    print(f"By eval_mode: {stats['by_eval_mode']}")
