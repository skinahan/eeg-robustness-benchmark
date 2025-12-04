#!/usr/bin/env python3
"""
Periodic Checkpoint Callback for Model Caching

This callback integrates with skorch/braindecode training to save checkpoints
periodically during training, checking for best validation loss.
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from skorch.callbacks import Callback
import torch

from evaluation.model_cache_manager import ModelCacheManager


class PeriodicCheckpointCallback(Callback):
    """
    Callback that saves model checkpoints periodically during training.
    
    This callback checks every N epochs if the current model has the best
    validation loss seen so far, and if so, saves it to the model cache.
    """
    
    def __init__(self, 
                 cache_manager: ModelCacheManager,
                 dataset: str,
                 model_name: str,
                 seed: int,
                 subject: int,
                 session: str,
                 eval_mode: str,
                 tuned: bool = False,
                 check_interval: int = 1,
                 monitor: str = 'valid_loss',
                 mode: str = 'min',
                 fold_idx: Optional[int] = None):
        """
        Initialize the periodic checkpoint callback.
        
        Args:
            cache_manager: ModelCacheManager instance
            dataset: Dataset name
            model_name: Model name
            seed: Random seed
            subject: Subject ID
            session: Session identifier
            eval_mode: Evaluation mode
            tuned: Whether this is a tuned model
            check_interval: How often (in epochs) to check for best model (default: 1)
            monitor: Metric to monitor (default: 'valid_loss')
            mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better)
            fold_idx: Fold index (required for WithinSession to prevent data leakage)
        """
        self.cache_manager = cache_manager
        self.dataset = dataset
        self.model_name = model_name
        self.seed = seed
        self.subject = subject
        self.session = session
        self.eval_mode = eval_mode
        self.tuned = tuned
        self.check_interval = check_interval
        self.monitor = monitor
        self.mode = mode
        self.fold_idx = fold_idx
        
        self.logger = logging.getLogger(__name__)
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        
    def on_epoch_end(self, net, **kwargs):
        """Called at the end of each epoch."""
        current_epoch = net.history[-1]['epoch']
        
        # Check if we should evaluate this epoch
        if current_epoch % self.check_interval != 0:
            return
        
        # Get current validation loss/score
        if self.monitor not in net.history[-1]:
            self.logger.warning(f"Metric '{self.monitor}' not found in history")
            return
        
        current_score = net.history[-1][self.monitor]
        
        # Check if this is the best score so far
        is_better = False
        if self.mode == 'min':
            is_better = current_score < self.best_score
        else:  # mode == 'max'
            is_better = current_score > self.best_score
        
        if is_better:
            self.best_score = current_score
            self.best_epoch = current_epoch
            
            # Save the model
            try:
                # Get model configuration
                config = net.get_params()
                
                # Save best checkpoint to cache
                saved = self.cache_manager.save_model(
                    model=net,
                    config=config,
                    dataset=self.dataset,
                    model_name=self.model_name,
                    seed=self.seed,
                    subject=self.subject,
                    session=self.session,
                    eval_mode=self.eval_mode,
                    tuned=self.tuned,
                    validation_loss=current_score,
                    epoch=current_epoch,
                    checkpoint_type="best",
                    fold_idx=self.fold_idx
                )
                
                if saved:
                    self.logger.info(f"Saved best model at epoch {current_epoch} with {self.monitor}={current_score:.6f}")
                else:
                    self.logger.debug(f"Model not saved at epoch {current_epoch} (not better or already exists)")
                    
            except Exception as e:
                self.logger.error(f"Failed to save model at epoch {current_epoch}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                print(f"Failed to save model at epoch {current_epoch}: {e}")
                traceback.print_exc()
        else:
            self.logger.debug(f"Epoch {current_epoch}: {self.monitor}={current_score:.6f} not better than {self.best_score:.6f}")
    
    def on_train_end(self, net, **kwargs):
        """Called at the end of training."""
        self.logger.info(f"Training completed. Best {self.monitor}={self.best_score:.6f} at epoch {self.best_epoch}")
        
        # Save final model checkpoint
        try:
            config = net.get_params()
            final_epoch = net.history[-1]['epoch'] if net.history else None
            final_validation_loss = net.history[-1].get('valid_loss', None) if net.history else None
            
            saved = self.cache_manager.save_model(
                model=net,
                config=config,
                dataset=self.dataset,
                model_name=self.model_name,
                seed=self.seed,
                subject=self.subject,
                session=self.session,
                eval_mode=self.eval_mode,
                tuned=self.tuned,
                validation_loss=final_validation_loss,
                epoch=final_epoch,
                checkpoint_type="final",
                fold_idx=self.fold_idx
            )
            
            if saved:
                self.logger.info("Saved final model checkpoint")
            else:
                self.logger.debug("Final model not saved (already exists with same config)")
            
            # Clean up unused checkpoint if we have both best and final
            self._cleanup_unused_checkpoint(net, config)
                
        except Exception as e:
            self.logger.error(f"Failed to save final model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            print(f"Failed to save final model: {e}")
            traceback.print_exc()
    
    def _cleanup_unused_checkpoint(self, net, config):
        """Clean up unused checkpoint to save space."""
        try:
            # Check if we have both best and final checkpoints
            best_path = self.cache_manager._get_cache_path(
                self.dataset, self.model_name, self.seed, self.subject, 
                self.session, self.eval_mode, self.tuned, "best", self.fold_idx
            )
            final_path = self.cache_manager._get_cache_path(
                self.dataset, self.model_name, self.seed, self.subject, 
                self.session, self.eval_mode, self.tuned, "final", self.fold_idx
            )
            
            if best_path.exists() and final_path.exists():
                # Check if best and final are the same epoch
                best_config_path = best_path.with_suffix('.json')
                final_config_path = final_path.with_suffix('.json')
                
                if best_config_path.exists() and final_config_path.exists():
                    with open(best_config_path, 'r') as f:
                        best_data = json.load(f)
                    with open(final_config_path, 'r') as f:
                        final_data = json.load(f)
                    
                    best_epoch = best_data.get('epoch', -1)
                    final_epoch = final_data.get('epoch', -1)
                    
                    # If best and final are from the same epoch, remove the final checkpoint
                    if best_epoch == final_epoch:
                        final_path.unlink()
                        final_config_path.unlink()
                        self.logger.info(f"Removed duplicate final checkpoint (same as best at epoch {best_epoch})")
                        
        except Exception as e:
            self.logger.warning(f"Failed to cleanup unused checkpoint: {e}")


class ModelCacheCallback(Callback):
    """
    Simplified callback that just saves the model at the end of training.
    Useful for non-early-stopping scenarios.
    """
    
    def __init__(self, 
                 cache_manager: ModelCacheManager,
                 dataset: str,
                 model_name: str,
                 seed: int,
                 subject: int,
                 session: str,
                 eval_mode: str,
                 tuned: bool = False,
                 fold_idx: Optional[int] = None):
        """
        Initialize the model cache callback.
        
        Args:
            cache_manager: ModelCacheManager instance
            dataset: Dataset name
            model_name: Model name
            seed: Random seed
            subject: Subject ID
            session: Session identifier
            eval_mode: Evaluation mode
            tuned: Whether this is a tuned model
            fold_idx: Fold index (required for WithinSession to prevent data leakage)
        """
        self.cache_manager = cache_manager
        self.dataset = dataset
        self.model_name = model_name
        self.seed = seed
        self.subject = subject
        self.session = session
        self.eval_mode = eval_mode
        self.tuned = tuned
        self.fold_idx = fold_idx
        
        self.logger = logging.getLogger(__name__)
    
    def on_train_end(self, net, **kwargs):
        """Called at the end of training."""
        try:
            config = net.get_params()
            validation_loss = None
            
            # Try to get the best validation loss from history
            if net.history:
                for epoch_data in reversed(net.history):
                    if 'valid_loss' in epoch_data:
                        validation_loss = epoch_data['valid_loss']
                        break
            
            saved = self.cache_manager.save_model(
                model=net,
                config=config,
                dataset=self.dataset,
                model_name=self.model_name,
                seed=self.seed,
                subject=self.subject,
                session=self.session,
                eval_mode=self.eval_mode,
                tuned=self.tuned,
                validation_loss=validation_loss,
                epoch=net.history[-1]['epoch'] if net.history else None,
                fold_idx=self.fold_idx
            )
            
            if saved:
                self.logger.info("Saved model checkpoint")
            else:
                self.logger.debug("Model not saved (already exists with same config)")
                
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")


def create_periodic_checkpoint_callback(cache_manager: ModelCacheManager,
                                      dataset: str,
                                      model_name: str,
                                      seed: int,
                                      subject: int,
                                      session: str,
                                      eval_mode: str,
                                      tuned: bool = False,
                                      check_interval: int = 1,
                                      fold_idx: Optional[int] = None) -> PeriodicCheckpointCallback:
    """Factory function to create a periodic checkpoint callback."""
    return PeriodicCheckpointCallback(
        cache_manager=cache_manager,
        dataset=dataset,
        model_name=model_name,
        seed=seed,
        subject=subject,
        session=session,
        eval_mode=eval_mode,
        tuned=tuned,
        check_interval=check_interval,
        fold_idx=fold_idx
    )


def create_model_cache_callback(cache_manager: ModelCacheManager,
                              dataset: str,
                              model_name: str,
                              seed: int,
                              subject: int,
                              session: str,
                              eval_mode: str,
                              tuned: bool = False,
                              fold_idx: Optional[int] = None) -> ModelCacheCallback:
    """Factory function to create a model cache callback."""
    return ModelCacheCallback(
        cache_manager=cache_manager,
        dataset=dataset,
        model_name=model_name,
        seed=seed,
        subject=subject,
        session=session,
        eval_mode=eval_mode,
        tuned=tuned,
        fold_idx=fold_idx
    )
