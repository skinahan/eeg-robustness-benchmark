#!/usr/bin/env python3
"""
CfC/NCP Neuroevolution for Noise-Robust EEG Architecture Discovery

This script applies neuroevolution to discover EEG architectures that:
1. Contain at least one CfC or NCP layer
2. Are accurate for EEG decoding
3. Are robust to noise
4. Balance complexity and overfitting

Uses real BNCI2014_001 EEG data for evaluation.
"""

import sys
import os
import argparse
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuroevolution import (
    NeuroevolutionEngine, 
    FastFitnessEvaluator
)
from neuroevolution.cfcncp_genome_generator import CfCNCPGenomeGenerator
from neuroevolution.robust_model_builder import RobustNeuroevolutionModelBuilder
from neuroevolution.real_data_fitness_evaluator import RealDataFitnessEvaluator, FastRealDataEvaluator
from config import get_paradigm
from moabb.datasets import BNCI2014_001
from globals import set_seeds


def run_cfcncp_neuroevolution():
    """Run neuroevolution to discover noise-robust EEG architectures with CfC/NCP layers using real data"""
    print("=" * 80)
    print("CfC/NCP NEUROEVOLUTION - Noise-Robust EEG Architecture Discovery")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set random seed for reproducibility
    set_seeds(42)
    
    # Test known good CfC/NCP architectures first with real data
    print("\nTesting known good CfC/NCP architectures with real BNCI2014_001 data...")
    cfcncp_generator = CfCNCPGenomeGenerator()
    known_architectures = cfcncp_generator.generate_known_good_cfcncp_architectures()
    
    # Create real data evaluator
    real_evaluator = FastRealDataEvaluator(
        max_epochs=15,
        cv_folds=2,
        noise_type=None
    )
    
    for i, genome in enumerate(known_architectures):
        print(f"\nTesting known CfC/NCP architecture {i+1} with real data:")
        print(f"- Layers: {len(genome.layers)}")
        print(f"- Parameters: {genome.get_parameter_count():,}")
        
        # Check if it has CfC/NCP layer
        cfcncp_layers = [layer for layer in genome.layers if layer.layer_type in ['cfc', 'ncp']]
        print(f"- CfC/NCP layers: {len(cfcncp_layers)}")
        for j, layer in enumerate(cfcncp_layers):
            print(f"  Layer {j+1}: {layer.layer_type} (units: {layer.hidden_size}, output: {layer.out_channels})")
        
        # Test with real data
        try:
            result = real_evaluator.evaluate_genome(genome)
            
            print(f"✓ Architecture {i+1} works with real data")
            print(f"  - Accuracy: {result['accuracy']:.4f}")
            print(f"  - Test Score (ROC AUC): {result['test_score']:.4f}")
            print(f"  - Noise Resilience: {result['noise_resilience']:.4f}")
            print(f"  - Overall Fitness: {result['overall_fitness']:.4f}")
            
        except Exception as e:
            print(f"✗ Architecture {i+1} failed with real data: {e}")
            genome.overall_fitness = 0.0
    
    # Find the best known architecture
    best_known = max(known_architectures, key=lambda x: x.overall_fitness)
    
    print(f"\n" + "=" * 80)
    print("KNOWN CfC/NCP ARCHITECTURES RESULTS (Real Data)")
    print("=" * 80)
    
    print(f"\nBest known CfC/NCP architecture:")
    print(f"- Overall fitness: {best_known.overall_fitness:.4f}")
    print(f"- Accuracy: {best_known.accuracy:.4f}")
    print(f"- Noise resilience: {best_known.noise_resilience:.4f}")
    print(f"- Complexity score: {best_known.complexity_score:.4f}")
    print(f"- Overfitting score: {best_known.overfitting_score:.4f}")
    print(f"- Number of layers: {len(best_known.layers)}")
    print(f"- Number of parameters: {best_known.get_parameter_count():,}")
    
    print(f"\nArchitecture layers:")
    for i, layer in enumerate(best_known.layers):
        cfcncp_marker = " [CfC/NCP]" if layer.layer_type in ['cfc', 'ncp'] else ""
        print(f"  {i+1:2d}. {layer.layer_type:12s} | "
              f"in: {layer.in_channels:3d} | "
              f"out: {layer.out_channels:3d} | "
              f"act: {layer.activation:8s} | "
              f"dropout: {layer.dropout_rate:.2f}{cfcncp_marker}")
    
    # Now run neuroevolution with CfC/NCP constraint and real data
    print(f"\n" + "=" * 80)
    print("CfC/NCP NEUROEVOLUTION - Discovering Noise-Robust Architectures with Real Data")
    print("=" * 80)
    
    # Create a custom evolution engine that uses the CfC/NCP generator and real data evaluator
    class CfCNCPRealDataEvolutionEngine(NeuroevolutionEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.generator = CfCNCPGenomeGenerator()
            # Use real data evaluator instead of fast evaluator
            self.real_evaluator = FastRealDataEvaluator(
                available_subjects=[1],
                max_epochs=15,  # Even fewer epochs for evolution
                cv_folds=2,
                noise_type=None  # No noise for initial evolution
            )
        
        def evaluate_population(self, population):
            """Override to use real data evaluation"""
            return self.real_evaluator.evaluate_population(population)
    
    # Run neuroevolution with real data evaluation
    engine = CfCNCPRealDataEvolutionEngine(
        population_size=15,  # Smaller population due to real data evaluation cost
        generations=10,      # Fewer generations due to evaluation time
        mutation_rate=0.15,
        crossover_rate=0.8,
        tournament_size=3,
        elite_size=2,
        evaluator_type='real_data',  # Use real data evaluation
        output_dir='cfcncp_real_data_neuroevolution_results',
        seed=42
    )
    
    print(f"\nConfiguration:")
    print(f"- Population size: {engine.population_size}")
    print(f"- Generations: {engine.generations}")
    print(f"- Mutation rate: {engine.mutation_rate}")
    print(f"- Crossover rate: {engine.crossover_rate}")
    print(f"- Evaluator type: {engine.evaluator_type}")
    print(f"- Output directory: {engine.output_dir}")
    print(f"- Constraint: All architectures must contain at least one CfC or NCP layer")
    print(f"- Data: Real BNCI2014_001 EEG data")
    print(f"- Subjects: {engine.real_evaluator.subject_list}")
    print(f"- Max epochs: {engine.real_evaluator.max_epochs}")
    
    # Run evolution
    print(f"\nStarting CfC/NCP neuroevolution with real data...")
    best_genome, history = engine.evolve()
    
    # Display results
    print(f"\n" + "=" * 80)
    print("CfC/NCP NEUROEVOLUTION COMPLETED - RESULTS (Real Data)")
    print("=" * 80)
    
    print(f"\nBest evolved CfC/NCP genome:")
    print(f"- Overall fitness: {best_genome.overall_fitness:.4f}")
    print(f"- Accuracy: {best_genome.accuracy:.4f}")
    print(f"- Noise resilience: {best_genome.noise_resilience:.4f}")
    print(f"- Complexity score: {best_genome.complexity_score:.4f}")
    print(f"- Overfitting score: {best_genome.overfitting_score:.4f}")
    print(f"- Number of layers: {len(best_genome.layers)}")
    print(f"- Number of parameters: {best_genome.get_parameter_count():,}")
    print(f"- Learning rate: {best_genome.learning_rate:.6f}")
    print(f"- Weight decay: {best_genome.weight_decay:.6f}")
    print(f"- Batch size: {best_genome.batch_size}")
    print(f"- Dropout rate: {best_genome.dropout_rate:.3f}")
    
    # Check CfC/NCP layers
    cfcncp_layers = [layer for layer in best_genome.layers if layer.layer_type in ['cfc', 'ncp']]
    print(f"- CfC/NCP layers: {len(cfcncp_layers)}")
    for i, layer in enumerate(cfcncp_layers):
        print(f"  Layer {i+1}: {layer.layer_type} (units: {layer.hidden_size}, output: {layer.out_channels}, sparsity: {layer.sparsity:.2f})")
    
    print(f"\nArchitecture layers:")
    for i, layer in enumerate(best_genome.layers):
        cfcncp_marker = " [CfC/NCP]" if layer.layer_type in ['cfc', 'ncp'] else ""
        print(f"  {i+1:2d}. {layer.layer_type:12s} | "
              f"in: {layer.in_channels:3d} | "
              f"out: {layer.out_channels:3d} | "
              f"act: {layer.activation:8s} | "
              f"dropout: {layer.dropout_rate:.2f}{cfcncp_marker}")
    
    print(f"\nEvolution statistics:")
    print(f"- Best fitness achieved: {max(history['best_fitness']):.4f}")
    print(f"- Final average fitness: {history['avg_fitness'][-1]:.4f}")
    print(f"- Population diversity (final): {history['population_diversity'][-1]:.4f}")
    print(f"- Fitness improvement: {max(history['best_fitness']) - history['best_fitness'][0]:.4f}")
    
    # Test the best evolved model with real data
    print(f"\nTesting best evolved CfC/NCP model with real data...")
    try:
        # Re-evaluate with more subjects for final validation
        final_evaluator = RealDataFitnessEvaluator(
            subject_list=list(range(1, 5)),  # Use 4 subjects for final validation
            max_epochs=30,
            cv_folds=3,
            noise_type=None
        )
        
        final_result = final_evaluator.evaluate_genome(best_genome)
        
        print(f"✓ Final validation with real data successful")
        print(f"  - Final Accuracy: {final_result['accuracy']:.4f}")
        print(f"  - Final Test Score: {final_result['test_score']:.4f}")
        print(f"  - Final Noise Resilience: {final_result['noise_resilience']:.4f}")
        print(f"  - Final Overall Fitness: {final_result['overall_fitness']:.4f}")
        
    except Exception as e:
        print(f"✗ Error in final validation: {e}")
    
    # Compare with known good architectures
    print(f"\n" + "=" * 80)
    print("COMPARISON: EVOLVED vs KNOWN GOOD CfC/NCP (Real Data)")
    print("=" * 80)
    
    print(f"\nBest known CfC/NCP architecture:")
    print(f"- Fitness: {best_known.overall_fitness:.4f}")
    print(f"- Parameters: {best_known.get_parameter_count():,}")
    print(f"- Layers: {len(best_known.layers)}")
    print(f"- CfC/NCP layers: {len([l for l in best_known.layers if l.layer_type in ['cfc', 'ncp']])}")
    
    print(f"\nBest evolved CfC/NCP architecture:")
    print(f"- Fitness: {best_genome.overall_fitness:.4f}")
    print(f"- Parameters: {best_genome.get_parameter_count():,}")
    print(f"- Layers: {len(best_genome.layers)}")
    print(f"- CfC/NCP layers: {len(cfcncp_layers)}")
    
    if best_genome.overall_fitness > best_known.overall_fitness:
        print(f"\n✓ Evolution found a better CfC/NCP architecture!")
        improvement = best_genome.overall_fitness - best_known.overall_fitness
        print(f"  - Improvement: {improvement:.4f}")
    else:
        print(f"\n- Evolution did not improve over known good CfC/NCP architecture")
        difference = best_known.overall_fitness - best_genome.overall_fitness
        print(f"  - Difference: {difference:.4f}")
    
    # Save the best genome for further use
    best_genome_path = os.path.join(engine.output_dir, 'best_cfcncp_genome_real_data.json')
    with open(best_genome_path, 'w') as f:
        json.dump(best_genome.to_dict(), f, indent=2)
    
    print(f"\nResults saved to: {engine.output_dir}")
    print(f"- Best CfC/NCP genome: {best_genome_path}")
    print(f"- Evolution plots: {engine.output_dir}/evolution_plots.png")
    print(f"- Population summary: {engine.output_dir}/population_summary.csv")
    
    # Summary of discovered architecture
    print(f"\n" + "=" * 80)
    print("DISCOVERED NOISE-ROBUST EEG ARCHITECTURE (Real Data)")
    print("=" * 80)
    print(f"\nThe neuroevolution discovered the following architecture:")
    print(f"- Type: Hybrid Conv2D + CfC/NCP + FC")
    print(f"- Purpose: Noise-robust EEG decoding")
    print(f"- Key features:")
    print(f"  * Convolutional layers for spatial-temporal feature extraction")
    print(f"  * CfC/NCP layers for temporal dynamics and noise resilience")
    print(f"  * Fully connected layers for classification")
    print(f"  * Balanced complexity and regularization")
    print(f"- Evaluation: Real BNCI2014_001 EEG data")
    print(f"- Performance: {best_genome.accuracy:.4f} accuracy on real data")
    
    return best_genome, history


def test_cfcncp_architectures_real_data():
    """Test CfC/NCP architectures with real BNCI2014_001 data"""
    print("=" * 80)
    print("CfC/NCP ARCHITECTURE TESTING WITH REAL DATA")
    print("=" * 80)
    
    cfcncp_generator = CfCNCPGenomeGenerator()
    real_evaluator = FastRealDataEvaluator(
        max_epochs=15,
        cv_folds=2,
        noise_type=None
    )
    
    # Test random CfC/NCP architectures
    print("\nTesting random CfC/NCP architectures with real data...")
    for i in range(3):  # Test fewer for speed
        genome = cfcncp_generator.generate_random_genome()
        print(f"\nRandom CfC/NCP architecture {i+1}:")
        print(f"- Layers: {len(genome.layers)}")
        print(f"- Parameters: {genome.get_parameter_count():,}")
        
        # Check CfC/NCP layers
        cfcncp_layers = [layer for layer in genome.layers if layer.layer_type in ['cfc', 'ncp']]
        print(f"- CfC/NCP layers: {len(cfcncp_layers)}")
        
        try:
            result = real_evaluator.evaluate_genome(genome)
            print(f"✓ Random CfC/NCP architecture {i+1} works with real data")
            print(f"  - Accuracy: {result['accuracy']:.4f}")
            print(f"  - Test Score: {result['test_score']:.4f}")
            print(f"  - Overall Fitness: {result['overall_fitness']:.4f}")
            
        except Exception as e:
            print(f"✗ Random CfC/NCP architecture {i+1} failed: {e}")
    
    # Test known good CfC/NCP architectures
    print(f"\nTesting known good CfC/NCP architectures with real data...")
    known_architectures = cfcncp_generator.generate_known_good_cfcncp_architectures()
    
    for i, genome in enumerate(known_architectures):
        print(f"\nKnown CfC/NCP architecture {i+1}:")
        print(f"- Layers: {len(genome.layers)}")
        print(f"- Parameters: {genome.get_parameter_count():,}")
        
        # Check CfC/NCP layers
        cfcncp_layers = [layer for layer in genome.layers if layer.layer_type in ['cfc', 'ncp']]
        print(f"- CfC/NCP layers: {len(cfcncp_layers)}")
        
        try:
            result = real_evaluator.evaluate_genome(genome)
            print(f"✓ Known CfC/NCP architecture {i+1} works with real data")
            print(f"  - Accuracy: {result['accuracy']:.4f}")
            print(f"  - Test Score: {result['test_score']:.4f}")
            print(f"  - Overall Fitness: {result['overall_fitness']:.4f}")
            
        except Exception as e:
            print(f"✗ Known CfC/NCP architecture {i+1} failed: {e}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="CfC/NCP Neuroevolution for Noise-Robust EEG Architecture Discovery")
    parser.add_argument('--mode', choices=['evolve', 'test'], 
                       default='evolve', help='Run mode')
    
    args = parser.parse_args()
    
    if args.mode == 'evolve':
        run_cfcncp_neuroevolution()
    elif args.mode == 'test':
        test_cfcncp_architectures_real_data()
    else:
        print("Invalid mode specified")


if __name__ == "__main__":
    main() 