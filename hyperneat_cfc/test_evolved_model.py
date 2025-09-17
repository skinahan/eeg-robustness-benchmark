#!/usr/bin/env python3
"""
Simple test script for evolved HyperNEAT model

This script loads the best evolved genome and tests basic functionality.
"""

import numpy as np
import torch
import json
import os
import sys

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from hyperneat_cfc import (
 CfCSubstrate,
 HyperNEATGenome,
 CfCPhenotype
)


def test_evolved_model(results_dir="../hyperneat_real_eeg_results"):
 """Test the evolved model with synthetic data."""
 print("Testing Evolved HyperNEAT CfC Model")
 print("=" * 40)
 
 # Load best genome
 best_genome_path = os.path.join(results_dir, "best_genome_final.json")
 if not os.path.exists(best_genome_path):
 print(f"Error: Best genome not found at {best_genome_path}")
 return
 
 genome = HyperNEATGenome.load(best_genome_path)
 print(f"Loaded genome with fitness: {genome.fitness:.4f}")
 
 # Print genome structure
 print(f"\nGenome Structure:")
 print(f" Nodes: {len(genome.nodes)}")
 print(f" Connections: {len(genome.connections)}")
 print(f" Enabled connections: {len([c for c in genome.connections if c.enabled])}")
 
 # Create substrate (same parameters as training)
 substrate = CfCSubstrate(
 input_size=22, # EEG channels
 hidden_size=6, # Hidden neurons
 output_size=4, # Classes
 layout_type="hierarchical"
 )
 
 # Create phenotype and develop model
 phenotype = CfCPhenotype(substrate)
 model = phenotype.develop(genome, n_chans=22, n_times=1000)
 
 print(f"\nModel Information:")
 print(f" Input size: {substrate.input_size}")
 print(f" Hidden size: {substrate.hidden_size}")
 print(f" Output size: {substrate.output_size}")
 print(f" Total parameters: {model.get_parameter_count():,}")
 
 # Test forward pass with synthetic data
 print(f"\nTesting forward pass...")
 batch_size = 4
 time_steps = 1000 # Typical EEG sequence length
 channels = 22
 
 # Create synthetic input data: (batch_size, time_steps, channels)
 x_test = torch.randn(batch_size, time_steps, channels)
 
 try:
 model.eval()
 with torch.no_grad():
 output, hidden = model(x_test)
 
 print(f"Forward pass successful!")
 print(f" Input shape: {x_test.shape}")
 print(f" Output shape: {output.shape}")
 print(f" Output sample: {output[0, :5].numpy()}") # First 5 time steps of first sample
 
 # Test with different sequence lengths
 for seq_len in [100, 500, 2000]:
 x_test_var = torch.randn(2, seq_len, channels)
 try:
 with torch.no_grad():
 output_var, _ = model(x_test_var)
 print(f" Variable length test ({seq_len}): Output shape: {output_var.shape}")
 except Exception as e:
 print(f" Variable length test ({seq_len}): Error: {e}")
 
 except Exception as e:
 print(f"Forward pass failed: {e}")
 import traceback
 traceback.print_exc()
 return
 
 print(f"\nModel test completed successfully!")
 return model


def analyze_model_connections(model, genome, substrate):
 """Analyze the connection patterns in the evolved model."""
 print(f"\nAnalyzing Model Connections:")
 
 # Get the wiring information
 wiring = model.rnn_cell.wiring
 
 print(f" Wiring type: {type(wiring).__name__}")
 print(f" Input dimension: {wiring.input_dim}")
 print(f" Output dimension: {wiring.output_dim}")
 print(f" Units: {wiring.units}")
 
 # Analyze CPPN structure
 print(f"\nCPPN Analysis:")
 node_types = {}
 activations = {}
 
 for node in genome.nodes:
 node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
 activations[node.activation] = activations.get(node.activation, 0) + 1
 
 print(f" Node types: {node_types}")
 print(f" Activations: {activations}")
 
 # Connection analysis
 enabled_conns = [c for c in genome.connections if c.enabled]
 weights = [c.weight for c in enabled_conns]
 
 if weights:
 print(f" Connection weights:")
 print(f" Count: {len(weights)}")
 print(f" Mean: {np.mean(weights):.3f}")
 print(f" Std: {np.std(weights):.3f}")
 print(f" Range: [{np.min(weights):.3f}, {np.max(weights):.3f}]")


if __name__ == "__main__":
 import argparse
 
 parser = argparse.ArgumentParser(description="Test evolved HyperNEAT model")
 parser.add_argument("--results-dir", type=str, default="../hyperneat_real_eeg_results",
 help="Directory containing evolution results")
 parser.add_argument("--analyze", action="store_true",
 help="Perform detailed connection analysis")
 
 args = parser.parse_args()
 
 # Test the model
 model = test_evolved_model(args.results_dir)
 
 if model and args.analyze:
 # Load genome for analysis
 genome = HyperNEATGenome.load(os.path.join(args.results_dir, "best_genome_final.json"))
 substrate = CfCSubstrate(input_size=22, hidden_size=6, output_size=4, layout_type="hierarchical")
 analyze_model_connections(model, genome, substrate)