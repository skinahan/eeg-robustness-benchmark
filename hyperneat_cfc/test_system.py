#!/usr/bin/env python3
"""
Test script for HyperNEAT CfC Evolution System

This script tests all components of the HyperNEAT CfC evolution system
to ensure everything works correctly.
"""

import numpy as np
import torch
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperneat_cfc import (
    CfCSubstrate,
    HyperNEATGenome,
    CfCPhenotype,
    HyperNEATFitnessEvaluator,
    HyperNEATEvolutionEngine
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_substrate():
    """Test substrate creation and visualization"""
    logger.info("Testing CfC Substrate...")
    
    # Test different layout types
    for layout_type in ["grid", "circular", "hierarchical"]:
        substrate = CfCSubstrate(
            input_size=22,
            hidden_size=32,
            output_size=2,
            layout_type=layout_type
        )
        
        # Check cell counts
        input_cells = substrate.get_input_cells()
        hidden_cells = substrate.get_hidden_cells()
        output_cells = substrate.get_output_cells()
        
        assert len(input_cells) == 22, f"Expected 22 input cells, got {len(input_cells)}"
        assert len(hidden_cells) == 32, f"Expected 32 hidden cells, got {len(hidden_cells)}"
        assert len(output_cells) == 2, f"Expected 2 output cells, got {len(output_cells)}"
        
        logger.info(f"✓ {layout_type} layout: {len(input_cells)} input, {len(hidden_cells)} hidden, {len(output_cells)} output cells")
    
    logger.info("✓ Substrate tests passed")


def test_genome():
    """Test genome creation and operations"""
    logger.info("Testing HyperNEAT Genome...")
    
    # Create genome
    genome = HyperNEATGenome()
    
    # Check basic properties
    assert len(genome.nodes) > 0, "Genome should have nodes"
    assert len(genome.connections) > 0, "Genome should have connections"
    
    # Test CPPN evaluation
    test_inputs = [1.0, 2.0, 3.0, 4.0]  # x1, y1, x2, y2
    output = genome.evaluate_cppn(test_inputs)
    assert isinstance(output, float), "CPPN output should be float"
    assert -1.0 <= output <= 1.0, "CPPN output should be in [-1, 1]"
    
    # Test mutation
    original_fitness = genome.fitness
    genome.mutate(mutation_rate=0.5)
    assert genome.fitness == original_fitness, "Mutation shouldn't change fitness"
    
    # Test crossover
    genome2 = HyperNEATGenome()
    child = genome.crossover(genome2)
    assert isinstance(child, HyperNEATGenome), "Crossover should return genome"
    
    # Test serialization
    genome_dict = genome.to_dict()
    genome_loaded = HyperNEATGenome.from_dict(genome_dict)
    assert len(genome_loaded.nodes) == len(genome.nodes), "Serialization should preserve nodes"
    
    logger.info("✓ Genome tests passed")


def test_phenotype():
    """Test phenotype development"""
    logger.info("Testing CfC Phenotype...")
    
    # Create substrate and phenotype
    substrate = CfCSubstrate(input_size=22, hidden_size=16, output_size=2)
    phenotype = CfCPhenotype(substrate)
    
    # Create genome
    genome = HyperNEATGenome()
    
    # Develop genome into network
    model = phenotype.develop(genome)
    
    # Check model properties
    assert hasattr(model, 'wiring'), "Model should have wiring"
    assert hasattr(model, 'cfc_cell'), "Model should have CfC cell"
    assert hasattr(model, 'output_projection'), "Model should have output projection"
    
    # Test forward pass with dummy data
    dummy_input = torch.randn(2, 100, 22)  # batch_size=2, seq_len=100, input_size=22
    output, hidden = model(dummy_input)
    
    assert output.shape[0] == 2, "Output batch size should match input"
    assert output.shape[1] == 2, "Output should have 2 classes"
    assert hidden.shape[0] == 2, "Hidden state batch size should match input"
    assert hidden.shape[1] == 16, "Hidden state should match hidden size"
    
    # Test parameter count
    param_count = model.get_parameter_count()
    assert param_count > 0, "Model should have parameters"
    
    logger.info(f"✓ Phenotype tests passed (model has {param_count} parameters)")


def test_fitness_evaluator():
    """Test fitness evaluator"""
    logger.info("Testing Fitness Evaluator...")
    
    # Generate small synthetic dataset
    X = np.random.randn(50, 100, 22)  # 50 samples, 100 timepoints, 22 channels
    y = np.random.randint(0, 2, 50)  # Binary classification
    
    # Split data
    train_size = 30
    val_size = 10
    test_size = 10
    
    train_data = (X[:train_size], y[:train_size])
    val_data = (X[train_size:train_size+val_size], y[train_size:train_size+val_size])
    test_data = (X[train_size+val_size:], y[train_size+val_size:])
    
    # Create substrate and evaluator
    substrate = CfCSubstrate(input_size=22, hidden_size=16, output_size=2)
    evaluator = HyperNEATFitnessEvaluator(
        substrate=substrate,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        max_epochs=3,  # Very few epochs for quick test
        batch_size=8,
        learning_rate=0.001,
        device='cpu'
    )
    
    # Test single genome evaluation
    genome = HyperNEATGenome()
    result = evaluator.evaluate_genome(genome)
    
    # Check result structure
    required_keys = ['clean_accuracy', 'train_accuracy', 'noise_resilience', 
                    'complexity_score', 'overall_fitness', 'parameter_count']
    for key in required_keys:
        assert key in result, f"Result should contain {key}"
        assert isinstance(result[key], (int, float)), f"{key} should be numeric"
    
    logger.info(f"✓ Fitness evaluator tests passed (fitness: {result['overall_fitness']:.4f})")


def test_evolution_engine():
    """Test evolution engine (minimal test)"""
    logger.info("Testing Evolution Engine...")
    
    # Generate small synthetic dataset
    X = np.random.randn(30, 50, 22)  # Very small dataset
    y = np.random.randint(0, 2, 30)
    
    train_data = (X[:20], y[:20])
    val_data = (X[20:25], y[20:25])
    test_data = (X[25:], y[25:])
    
    # Create components
    substrate = CfCSubstrate(input_size=22, hidden_size=8, output_size=2)
    evaluator = HyperNEATFitnessEvaluator(
        substrate=substrate,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        max_epochs=2,  # Very few epochs
        batch_size=4,
        learning_rate=0.001,
        device='cpu'
    )
    
    engine = HyperNEATEvolutionEngine(
        substrate=substrate,
        fitness_evaluator=evaluator,
        population_size=3,  # Very small population
        generations=2,  # Very few generations
        output_dir="test_results"
    )
    
    # Test population initialization
    engine.initialize_population()
    assert len(engine.population) == 3, "Population should have 3 individuals"
    
    # Test single generation
    engine._evaluate_population()
    assert all(hasattr(g, 'fitness') for g in engine.population), "All genomes should have fitness"
    
    logger.info("✓ Evolution engine tests passed")


def main():
    """Run all tests"""
    logger.info("Starting HyperNEAT CfC System Tests...")
    
    try:
        test_substrate()
        test_genome()
        test_phenotype()
        test_fitness_evaluator()
        test_evolution_engine()
        
        logger.info("🎉 All tests passed! System is working correctly.")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise
    
    # Clean up test results
    if os.path.exists("test_results"):
        import shutil
        shutil.rmtree("test_results")
        logger.info("Cleaned up test results directory")


if __name__ == "__main__":
    main() 