#!/usr/bin/env python3
"""
Test script for optimized architectures from architecture search.

This script loads the optimized architectures from the outputs/architectures directory
and tests them using the new CNNWiredCfC model on EEG data.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import numpy as np
import torch
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from architecture_refinement.arbitrary_wiring import load_architecture_from_file, create_wiring_from_architecture_data
from models.cnnncp import create_cnnwiredcfc_classifier
from globals import set_seeds


def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('optimized_architecture_test.log')
        ]
    )
    return logging.getLogger(__name__)


def load_architecture_files(architectures_dir: str) -> List[Dict[str, Any]]:
    """
    Load all architecture files from the specified directory.
    
    Args:
        architectures_dir: Path to architectures directory
        
    Returns:
        List of architecture data dictionaries
    """
    architectures = []
    arch_path = Path(architectures_dir)
    
    if not arch_path.exists():
        raise FileNotFoundError(f"Architectures directory not found: {architectures_dir}")
    
    # Find all JSON files
    json_files = list(arch_path.glob("*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {architectures_dir}")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Found {len(json_files)} architecture files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Add filename for reference
            data['filename'] = json_file.name
            data['filepath'] = str(json_file)
            architectures.append(data)
            
            logger.info(f"Loaded architecture from {json_file.name}")
            
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(architectures)} architectures")
    return architectures


def validate_architecture(architecture: Dict[str, Any]) -> List[str]:
    """
    Validate an architecture specification.
    
    Args:
        architecture: Architecture data dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    required_fields = ['input_size', 'hidden_size', 'output_size', 'wiring_matrix']
    for field in required_fields:
        if field not in architecture:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return errors
    
    # Check data types and values
    if not isinstance(architecture['input_size'], int) or architecture['input_size'] <= 0:
        errors.append("input_size must be a positive integer")
    
    if not isinstance(architecture['hidden_size'], int) or architecture['hidden_size'] <= 0:
        errors.append("hidden_size must be a positive integer")
    
    if not isinstance(architecture['output_size'], int) or architecture['output_size'] <= 0:
        errors.append("output_size must be a positive integer")
    
    # Check wiring matrix
    if not isinstance(architecture['wiring_matrix'], list):
        errors.append("wiring_matrix must be a list")
    else:
        try:
            wiring_matrix = np.array(architecture['wiring_matrix'])
            expected_shape = (architecture['input_size'] + architecture['hidden_size'] + architecture['output_size'], 
                           architecture['input_size'] + architecture['hidden_size'] + architecture['output_size'])
            
            if wiring_matrix.shape != expected_shape:
                errors.append(f"Wiring matrix shape mismatch: expected {expected_shape}, got {wiring_matrix.shape}")
            
        except Exception as e:
            errors.append(f"Error processing wiring matrix: {e}")
    
    return errors


def prepare_eeg_data(subject_id: int = 1, test_size: float = 0.3, random_state: int = 42):
    """
    Prepare EEG data for testing.
    
    Args:
        subject_id: Subject ID to use
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading EEG data for subject {subject_id}")
    
    # Load dataset
    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        events=["left_hand", "right_hand"],
        fmin=8, fmax=35,
        tmin=0.0, tmax=None,
        baseline=None,
        resample=250.0
    )
    
    # Get data for specific subject
    X, y, metadata = paradigm.get_data(dataset, subjects=[subject_id])
    
    logger.info(f"Loaded data: {X.shape}, labels: {y.shape}")
    logger.info(f"Number of samples: {len(X)}")
    logger.info(f"Number of channels: {X.shape[1]}")
    logger.info(f"Number of time points: {X.shape[2]}")
    logger.info(f"Classes: {np.unique(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def test_architecture_with_data(architecture: Dict[str, Any], 
                              X_train: np.ndarray, 
                              X_test: np.ndarray,
                              y_train: np.ndarray, 
                              y_test: np.ndarray,
                              logger: logging.Logger) -> Dict[str, Any]:
    """
    Test a specific architecture with EEG data.
    
    Args:
        architecture: Architecture data dictionary
        X_train, X_test, y_train, y_test: Training and test data
        logger: Logger instance
        
    Returns:
        Dictionary containing test results
    """
    logger.info(f"Testing architecture: {architecture['filename']}")
    
    try:
        # Create wiring from architecture
        wiring = create_wiring_from_architecture_data(architecture, logger)
        
        # Get data dimensions
        n_chans = X_train.shape[1]
        n_times = X_train.shape[2]
        n_outputs = len(np.unique(y_train))
        
        logger.info(f"Data dimensions: channels={n_chans}, times={n_times}, outputs={n_outputs}")
        logger.info(f"Architecture: {architecture['input_size']}->{architecture['hidden_size']}->{architecture['output_size']}")
        
        # Create model
        model = create_cnnwiredcfc_classifier(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            wiring=wiring,
            drop_prob=0.15,
            lr=1e-3,
            batch_size=32,  # Smaller batch size for testing
            max_epochs=10,   # Fewer epochs for testing
            F1=8,
            D=2,
            kernel_length=128,
            temporal_kernel_size=3,
            temporal_stride=4,
            max_seq_length=250
        )
        
        logger.info("Model created successfully")
        
        # Get wiring summary
        wiring_summary = wiring.get_wiring_summary()
        logger.info(f"Wiring summary: {wiring_summary['total_connections']} connections, "
                   f"density: {wiring_summary['connection_density']:.4f}")
        
        # Train the model
        logger.info("Starting training...")
        model.fit(X_train, y_train)
        logger.info("Training completed")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'architecture_file': architecture['filename'],
            'accuracy': accuracy,
            'classification_report': report,
            'wiring_summary': wiring_summary,
            'data_dimensions': {
                'n_chans': n_chans,
                'n_times': n_times,
                'n_outputs': n_outputs
            },
            'architecture_params': {
                'input_size': architecture['input_size'],
                'hidden_size': architecture['hidden_size'],
                'output_size': architecture['output_size']
            },
            'status': 'success'
        }
        
        logger.info(f"Test completed successfully. Accuracy: {accuracy:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"Error testing architecture {architecture['filename']}: {e}")
        return {
            'architecture_file': architecture['filename'],
            'status': 'error',
            'error_message': str(e)
        }


def main():
    """Main function to test optimized architectures."""
    parser = argparse.ArgumentParser(description='Test optimized architectures from architecture search')
    parser.add_argument('--architectures-dir', 
                       default='../outputs/architectures',
                       help='Directory containing architecture JSON files')
    parser.add_argument('--subject-id', type=int, default=1,
                       help='Subject ID to use for testing')
    parser.add_argument('--test-size', type=float, default=0.3,
                       help='Fraction of data to use for testing')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(log_level)
    
    # Set random seeds
    set_seeds(args.random_state)
    
    logger.info("Starting optimized architecture testing")
    logger.info(f"Architectures directory: {args.architectures_dir}")
    logger.info(f"Subject ID: {args.subject_id}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"Random state: {args.random_state}")
    
    try:
        # Load architectures
        architectures = load_architecture_files(args.architectures_dir)
        
        if not architectures:
            logger.error("No valid architectures found")
            return
        
        # Validate architectures
        valid_architectures = []
        for arch in architectures:
            errors = validate_architecture(arch)
            if errors:
                logger.warning(f"Architecture {arch['filename']} has validation errors: {errors}")
                continue
            valid_architectures.append(arch)
        
        logger.info(f"Found {len(valid_architectures)} valid architectures")
        
        if not valid_architectures:
            logger.error("No valid architectures to test")
            return
        
        # Prepare EEG data
        X_train, X_test, y_train, y_test = prepare_eeg_data(
            subject_id=args.subject_id,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # Test each architecture
        results = []
        for i, architecture in enumerate(valid_architectures):
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing architecture {i+1}/{len(valid_architectures)}: {architecture['filename']}")
            logger.info(f"{'='*60}")
            
            result = test_architecture_with_data(
                architecture, X_train, X_test, y_train, y_test, logger
            )
            results.append(result)
        
        # Save results
        results_file = f"architecture_test_results_subject_{args.subject_id}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\n{'='*60}")
        logger.info("TESTING COMPLETED")
        logger.info(f"{'='*60}")
        
        # Print summary
        successful_tests = [r for r in results if r['status'] == 'success']
        failed_tests = [r for r in results if r['status'] == 'error']
        
        logger.info(f"Total architectures tested: {len(results)}")
        logger.info(f"Successful tests: {len(successful_tests)}")
        logger.info(f"Failed tests: {len(failed_tests)}")
        
        if successful_tests:
            accuracies = [r['accuracy'] for r in successful_tests]
            best_accuracy = max(accuracies)
            best_arch = successful_tests[accuracies.index(best_accuracy)]
            
            logger.info(f"\nBest performing architecture: {best_arch['architecture_file']}")
            logger.info(f"Best accuracy: {best_accuracy:.4f}")
            logger.info(f"Average accuracy: {np.mean(accuracies):.4f}")
            logger.info(f"Accuracy std: {np.std(accuracies):.4f}")
        
        if failed_tests:
            logger.info(f"\nFailed architectures:")
            for failed in failed_tests:
                logger.info(f"  {failed['architecture_file']}: {failed['error_message']}")
        
        logger.info(f"\nDetailed results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
