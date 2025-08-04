#!/usr/bin/env python3
"""
HyperNEAT CfC Evolution Demo with Real EEG Data
Uses real subject EEG data from subject 1 in the BNCI2014_001 dataset.
Only uses data from the 0train session for training and validation.
"""

import numpy as np
import torch
import logging
import argparse
import os
import sys
from typing import Tuple

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from hyperneat_cfc import (
    CfCSubstrate,
    HyperNEATGenome,
    HyperNEATFitnessEvaluator,
    HyperNEATEvolutionEngine,
    CfCPhenotype
)

# Import MOABB and related modules
from moabb.datasets import BNCI2014_001
from sklearn.preprocessing import LabelEncoder
from config import get_paradigm

def load_real_eeg_data(subject_id=1, resample=250.0):
    """Load real EEG data from BNCI2014_001 dataset for subject 1, 0train session only."""
    print(f"Loading real EEG data for subject {subject_id} from BNCI2014_001 dataset...")
    
    # Load dataset
    dataset = BNCI2014_001()
    dataset.subject_list = [subject_id]
    
    # Get paradigm
    paradigm = get_paradigm(resample=resample)
    
    # Load data
    X, y, metadata = paradigm.get_data(dataset, subjects=[subject_id])
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Full data shape: {X.shape}")
    print(f"Labels: {np.unique(y_encoded)}")
    print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    print(f"Metadata columns: {metadata.columns.tolist()}")
    
    # Filter to training session only (0train)
    if 'session' in metadata.columns:
        train_mask = metadata['session'] == '0train'
        
        if train_mask.sum() > 0:
            X_train = X[train_mask]
            y_train = y_encoded[train_mask]
            train_metadata = metadata[train_mask]
            
            print(f"Training session (0train) data:")
            print(f"  Samples: {len(X_train)}")
            print(f"  Shape: {X_train.shape}")
            print(f"  Classes: {np.unique(y_train)}")
            print(f"  Class distribution: {np.bincount(y_train)}")
        else:
            raise ValueError("No training session data found")
    else:
        raise ValueError("No session column found in metadata")
    
    return X_train, y_train, train_metadata, label_encoder

def split_data(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8, val_ratio: float = 0.10):
    """Split data into train/validation/test sets."""
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    
    # Split indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Split data
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    print(f"Data split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def run_demo_evolution(population_size: int = 10, generations: int = 20, output_dir: str = "hyperneat_real_eeg_results"):
    """Run HyperNEAT evolution with real EEG data."""
    print("=" * 60)
    print("HyperNEAT CfC Evolution with Real EEG Data")
    print("=" * 60)
    
    # Load real EEG data
    X_full, y_full, metadata, label_encoder = load_real_eeg_data(subject_id=1)
    
    # Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X_full, y_full)
    
    # Get data dimensions
    n_samples, n_channels, n_times = X_train.shape
    n_classes = len(np.unique(y_train))
    
    print(f"\nData dimensions:")
    print(f"  Channels: {n_channels}")
    print(f"  Time points: {n_times}")
    print(f"  Classes: {n_classes}")
    
    # Create substrate
    print(f"\nCreating CfC substrate...")
    substrate = CfCSubstrate(
        input_size=n_channels,
        hidden_size=8,
        output_size=n_classes,
        layout_type="hierarchical"
    )
    
    # Create fitness evaluator
    print(f"Creating fitness evaluator...")
    fitness_evaluator = HyperNEATFitnessEvaluator(
        substrate=substrate,
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        test_data=(X_test, y_test),
        max_epochs=50,  # Reduced for demo
        batch_size=16,
        learning_rate=0.001,
        device='cpu'
    )
    
    # Log that we're using the improved HyperNEAT CfC implementation
    print(f"Using improved HyperNEAT CfC with evolved architecture")
    print(f"  - WiredCfCCell with custom HyperNEATWiring")
    print(f"  - Evolved connection patterns from CPPN")
    print(f"  - Proper adjacency matrix utilization")
    
    # Create evolution engine
    print(f"Creating evolution engine...")
    evolution_engine = HyperNEATEvolutionEngine(
        substrate=substrate,
        fitness_evaluator=fitness_evaluator,
        population_size=population_size,
        generations=generations,
        mutation_rate=0.30,
        crossover_rate=0.8,
        elitism_size=2,
        output_dir=output_dir
    )
    
    # Run evolution
    print(f"\nStarting evolution with {population_size} individuals for {generations} generations...")
    best_genome, evolution_history = evolution_engine.evolve()
    
    print(f"\nEvolution completed!")
    print(f"Best genome saved to: {output_dir}/best_genome_final.json")
    print(f"Evolution history saved to: {output_dir}/evolution_history.json")
    
    return best_genome, evolution_history

def test_single_genome():
    """Test a single genome with real EEG data."""
    print("=" * 60)
    print("Testing Single HyperNEAT Genome with Real EEG Data")
    print("=" * 60)
    
    # Load real EEG data
    X_full, y_full, metadata, label_encoder = load_real_eeg_data(subject_id=1)
    
    # Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X_full, y_full)
    
    # Get data dimensions
    n_samples, n_channels, n_times = X_train.shape
    n_classes = len(np.unique(y_train))
    
    # Create substrate
    substrate = CfCSubstrate(
        input_size=n_channels,
        hidden_size=32,  # Smaller for quick test
        output_size=n_classes,
        layout_type="grid"
    )
    
    # Create a random genome
    genome = HyperNEATGenome(
        input_nodes=4,
        hidden_nodes=6,
        output_nodes=1,
        max_connections=15
    )
    
    # Create phenotype
    phenotype = CfCPhenotype(substrate)
    
    # Develop the genome into a network
    print("Developing genome into CfC network...")
    model = phenotype.develop(genome)
    
    print(f"Model created:")
    print(f"  Input size: {n_channels}")
    print(f"  Hidden size: 32")
    print(f"  Output size: {n_classes}")
    print(f"  Parameters: {model.get_parameter_count():,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    # Create test data with correct shape: (batch_size, time_steps, channels)
    x_test = torch.randn(batch_size, n_times, n_channels)
    
    try:
        with torch.no_grad():
            output, hidden = model(x_test)
        print(f"Forward pass successful!")
        print(f"  Input shape: {x_test.shape}")
        print(f"  Output shape: {output.shape}")
        # print(f"  Hidden shape: {hidden.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        print(e.traceback)
        print(f"  Expected input shape: (batch_size, time_steps, channels) = ({batch_size}, {n_times}, {n_channels})")
        print(f"  Actual input shape: {x_test.shape}")
    
    print("\nSingle genome test completed!")

def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="HyperNEAT CfC Evolution Demo with Real EEG Data")
    parser.add_argument("--mode", choices=["evolution", "test"], default="evolution",
                       help="Run mode: evolution or test single genome")
    parser.add_argument("--population", type=int, default=10,
                       help="Population size for evolution")
    parser.add_argument("--generations", type=int, default=20,
                       help="Number of generations for evolution")
    parser.add_argument("--output-dir", type=str, default="hyperneat_real_eeg_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.mode == "evolution":
        run_demo_evolution(
            population_size=args.population,
            generations=args.generations,
            output_dir=args.output_dir
        )
    else:
        test_single_genome()

if __name__ == "__main__":
    main() 