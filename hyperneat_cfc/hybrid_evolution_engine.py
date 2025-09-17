"""
Hybrid Evolution Engine: HyperNEAT + Bayesian Optimization

This module combines traditional HyperNEAT evolution with Bayesian optimization
by reserving space for BO-suggested genomes in each generation.
"""

import numpy as np
import random
import json
import os
from typing import List, Dict, Any, Tuple, Optional
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import optuna

from .hyperneat_genome import HyperNEATGenome
from .cfc_substrate import CfCSubstrate
from .fitness_evaluator import HyperNEATFitnessEvaluator
from .bo_genome_generator import BOGenomeGenerator


class HybridEvolutionEngine:
    """
    Hybrid evolution engine combining HyperNEAT with Bayesian optimization.
    
    This class orchestrates evolution while reserving space for BO-suggested
    genomes in each generation, creating a hybrid approach that leverages
    both evolutionary search and Bayesian optimization.
    """
    
    def __init__(
        self,
        substrate: CfCSubstrate,
        fitness_evaluator: HyperNEATFitnessEvaluator,
        population_size: int = 20,
        generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism_size: int = 2,
        bo_ratio: float = 0.2,  # 20% of population from BO
        output_dir: str = "hybrid_evolution_results"
    ):
        self.substrate = substrate
        self.fitness_evaluator = fitness_evaluator
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_size = elitism_size
        self.bo_ratio = bo_ratio
        self.output_dir = output_dir
        
        # Calculate BO population size
        self.bo_population_size = max(1, int(population_size * bo_ratio))
        self.evolution_population_size = population_size - self.bo_population_size
        
        # Evolution tracking
        self.population: List[HyperNEATGenome] = []
        self.generation_stats: List[Dict[str, Any]] = []
        self.best_genome: Optional[HyperNEATGenome] = None
        self.best_fitness: float = 0.0
        
        # BO components
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        self.bo_generator = BOGenomeGenerator(
            study=self.study,
            substrate_dimensions=(
                substrate.input_size,
                substrate.hidden_size,
                substrate.output_size
            )
        )
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def initialize_population(self):
        """Initialize the population with diverse random genomes"""
        self.logger.info(f"Initializing hybrid population:")
        self.logger.info(f"  Evolution genomes: {self.evolution_population_size}")
        self.logger.info(f"  BO genomes: {self.bo_population_size}")
        
        # Initialize evolution population
        evolution_genomes = self._initialize_evolution_population()
        
        # Initialize BO population
        bo_genomes = self._initialize_bo_population()
        
        # Combine populations
        self.population = evolution_genomes + bo_genomes
        
        self.logger.info(f"Total population size: {len(self.population)}")
    
    def _initialize_evolution_population(self) -> List[HyperNEATGenome]:
        """Initialize traditional evolution population"""
        genomes = []
        
        # Determine CPPN input size based on substrate dimensions
        sample_cell = self.substrate.cells[0] if self.substrate.cells else None
        if sample_cell:
            if hasattr(sample_cell, 'z'):
                cppn_input_size = 6
            elif hasattr(sample_cell, 'y'):
                cppn_input_size = 4
            else:
                cppn_input_size = 2
        else:
            cppn_input_size = 4
        
        # Define parameter ranges for diverse initialization
        param_ranges = {
            'input_nodes': [cppn_input_size],
            'hidden_nodes': [2, 3, 4, 5, 6],
            'output_nodes': [1, 2],
            'max_connections': [5, 8, 10, 12, 15],
            'proj_size': [None, 2],
            'mode': ["default", "pure", "no_gate"],
            'mixed_memory': [True, False],
            'return_sequences': [False]
        }
        
        for i in range(self.evolution_population_size):
            genome = HyperNEATGenome(
                input_nodes=random.choice(param_ranges['input_nodes']),
                hidden_nodes=random.choice(param_ranges['hidden_nodes']),
                output_nodes=random.choice(param_ranges['output_nodes']),
                max_connections=random.choice(param_ranges['max_connections']),
                proj_size=random.choice(param_ranges['proj_size']),
                mode=random.choice(param_ranges['mode']),
                mixed_memory=random.choice(param_ranges['mixed_memory']),
                return_sequences=random.choice(param_ranges['return_sequences'])
            )
            genomes.append(genome)
        
        return genomes
    
    def _initialize_bo_population(self) -> List[HyperNEATGenome]:
        """Initialize BO-suggested population"""
        genomes = []
        
        for i in range(self.bo_population_size):
            genome = self.bo_generator.suggest_genome()
            genomes.append(genome)
        
        return genomes
    
    def evolve(self) -> Tuple[HyperNEATGenome, List[Dict[str, Any]]]:
        """Run hybrid evolution"""
        self.logger.info("Starting hybrid evolution...")
        
        # Initialize population
        self.initialize_population()
        
        # Evolution loop
        for generation in range(self.generations):
            self.logger.info(f"\nGeneration {generation + 1}/{self.generations}")
            
            # Evaluate population
            self._evaluate_population()
            
            # Record statistics
            stats = self._record_generation_stats(generation)
            self.generation_stats.append(stats)
            
            # Update best genome
            self._update_best_genome()
            
            # Save checkpoint
            if (generation + 1) % 10 == 0:
                self._save_checkpoint(generation)
            
            # Create next generation (except for last generation)
            if generation < self.generations - 1:
                self._create_next_generation()
        
        # Save final results
        self._save_final_results()
        
        return self.best_genome, self.generation_stats
    
    def _evaluate_population(self):
        """Evaluate the entire population"""
        self.logger.info("Evaluating population...")
        
        # Evaluate all genomes
        fitness_results = self.fitness_evaluator.evaluate_population(self.population)
        
        # Update fitness scores
        for genome, fitness in zip(self.population, fitness_results):
            genome.fitness = fitness['overall_fitness']
            
            # Update BO historical data
            self.bo_generator.update_historical_data(genome, fitness)
        
        # Log BO statistics
        bo_stats = self.bo_generator.get_bo_suggestions_summary()
        self.logger.info(f"BO Statistics: {bo_stats}")
    
    def _create_next_generation(self):
        """Create the next generation using hybrid approach"""
        self.logger.info("Creating next generation...")
        
        # Sort population by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        
        # Elitism: keep best genomes
        elite_genomes = self.population[:self.elitism_size]
        
        # Evolution: create new genomes through crossover and mutation
        evolution_genomes = self._evolve_population()
        
        # BO: create new genomes using Bayesian optimization
        bo_genomes = self._create_bo_genomes()
        
        # Combine all genomes
        self.population = elite_genomes + evolution_genomes + bo_genomes
        
        self.logger.info(f"New generation: {len(elite_genomes)} elite + {len(evolution_genomes)} evolved + {len(bo_genomes)} BO")
    
    def _evolve_population(self) -> List[HyperNEATGenome]:
        """Create new genomes through traditional evolution"""
        genomes = []
        
        # Tournament selection and reproduction
        while len(genomes) < self.evolution_population_size - self.elitism_size:
            if random.random() < self.crossover_rate:
                # Crossover
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child = parent1.crossover(parent2)
            else:
                # Mutation
                parent = self._tournament_selection()
                child = self._clone_genome(parent)
                child.mutate(self.mutation_rate)
            
            genomes.append(child)
        
        return genomes
    
    def _create_bo_genomes(self) -> List[HyperNEATGenome]:
        """Create new genomes using Bayesian optimization"""
        genomes = []
        
        for i in range(self.bo_population_size):
            genome = self.bo_generator.suggest_genome()
            genomes.append(genome)
        
        return genomes
    
    def _tournament_selection(self, tournament_size: int = 3) -> HyperNEATGenome:
        """Tournament selection"""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda g: g.fitness)
    
    def _clone_genome(self, genome: HyperNEATGenome) -> HyperNEATGenome:
        """Create a deep copy of a genome"""
        # Create new genome with same parameters
        clone = HyperNEATGenome(
            input_nodes=genome.input_nodes,
            hidden_nodes=genome.hidden_nodes,
            output_nodes=genome.output_nodes,
            max_connections=genome.max_connections,
            weight_range=genome.weight_range,
            bias_range=genome.bias_range,
            proj_size=genome.proj_size,
            mode=genome.mode,
            mixed_memory=genome.mixed_memory,
            return_sequences=genome.return_sequences
        )
        
        # Copy nodes
        clone.nodes = []
        for node in genome.nodes:
            clone.nodes.append(type(node)(
                node_id=node.node_id,
                node_type=node.node_type,
                activation=node.activation,
                bias=node.bias
            ))
        
        # Copy connections
        clone.connections = []
        for connection in genome.connections:
            clone.connections.append(type(connection)(
                from_node=connection.from_node,
                to_node=connection.to_node,
                weight=connection.weight,
                enabled=connection.enabled
            ))
        
        return clone
    
    def _record_generation_stats(self, generation: int) -> Dict[str, Any]:
        """Record statistics for the current generation"""
        fitnesses = [g.fitness for g in self.population]
        
        # Separate evolution and BO genomes
        evolution_genomes = self.population[:self.evolution_population_size]
        bo_genomes = self.population[self.evolution_population_size:]
        
        evolution_fitnesses = [g.fitness for g in evolution_genomes]
        bo_fitnesses = [g.fitness for g in bo_genomes]
        
        stats = {
            'generation': generation,
            'population_size': len(self.population),
            'evolution_population_size': len(evolution_genomes),
            'bo_population_size': len(bo_genomes),
            'best_fitness': max(fitnesses),
            'avg_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'evolution_best_fitness': max(evolution_fitnesses) if evolution_fitnesses else 0,
            'evolution_avg_fitness': np.mean(evolution_fitnesses) if evolution_fitnesses else 0,
            'bo_best_fitness': max(bo_fitnesses) if bo_fitnesses else 0,
            'bo_avg_fitness': np.mean(bo_fitnesses) if bo_fitnesses else 0,
            'bo_stats': self.bo_generator.get_bo_suggestions_summary()
        }
        
        self.logger.info(f"Generation {generation} stats:")
        self.logger.info(f"  Best fitness: {stats['best_fitness']:.4f}")
        self.logger.info(f"  Evolution best: {stats['evolution_best_fitness']:.4f}")
        self.logger.info(f"  BO best: {stats['bo_best_fitness']:.4f}")
        
        return stats
    
    def _update_best_genome(self):
        """Update the best genome found so far"""
        current_best = max(self.population, key=lambda g: g.fitness)
        if current_best.fitness > self.best_fitness:
            self.best_genome = self._clone_genome(current_best)
            self.best_fitness = current_best.fitness
            self.logger.info(f"New best genome found! Fitness: {self.best_fitness:.4f}")
    
    def _save_checkpoint(self, generation: int):
        """Save evolution checkpoint"""
        checkpoint = {
            'generation': generation,
            'best_genome': self.best_genome.to_dict() if self.best_genome else None,
            'best_fitness': self.best_fitness,
            'population': [g.to_dict() for g in self.population],
            'generation_stats': self.generation_stats
        }
        
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_gen_{generation}.json")
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_final_results(self):
        """Save final evolution results"""
        # Save best genome
        if self.best_genome:
            best_genome_path = os.path.join(self.output_dir, "best_genome_final.json")
            self.best_genome.save(best_genome_path)
        
        # Save evolution history
        history_path = os.path.join(self.output_dir, "evolution_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.generation_stats, f, indent=2)
        
        # Create evolution plots
        self._create_evolution_plots()
        
        self.logger.info(f"Final results saved to: {self.output_dir}")
    
    def _create_evolution_plots(self):
        """Create evolution visualization plots"""
        if not self.generation_stats:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        generations = [stats['generation'] for stats in self.generation_stats]
        
        # Overall fitness
        axes[0, 0].plot(generations, [stats['best_fitness'] for stats in self.generation_stats], 'b-', label='Best Fitness')
        axes[0, 0].plot(generations, [stats['avg_fitness'] for stats in self.generation_stats], 'r--', label='Average Fitness')
        axes[0, 0].set_title('Overall Fitness Evolution')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Fitness')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Evolution vs BO comparison
        axes[0, 1].plot(generations, [stats['evolution_best_fitness'] for stats in self.generation_stats], 'g-', label='Evolution Best')
        axes[0, 1].plot(generations, [stats['bo_best_fitness'] for stats in self.generation_stats], 'm-', label='BO Best')
        axes[0, 1].set_title('Evolution vs BO Performance')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Fitness')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Population diversity
        axes[1, 0].plot(generations, [stats['std_fitness'] for stats in self.generation_stats], 'c-')
        axes[1, 0].set_title('Population Diversity (Fitness Std)')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Fitness Standard Deviation')
        axes[1, 0].grid(True)
        
        # BO statistics
        bo_recent_avg = [stats['bo_stats'].get('recent_avg_fitness', 0) for stats in self.generation_stats]
        axes[1, 1].plot(generations, bo_recent_avg, 'orange', label='BO Recent Avg')
        axes[1, 1].set_title('BO Performance Over Time')
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Recent Average Fitness')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, "evolution_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Evolution plots saved: {plot_path}")
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution results"""
        if not self.generation_stats:
            return {'message': 'No evolution data available'}
        
        return {
            'total_generations': len(self.generation_stats),
            'best_fitness': self.best_fitness,
            'final_population_size': len(self.population),
            'bo_ratio': self.bo_ratio,
            'evolution_population_size': self.evolution_population_size,
            'bo_population_size': self.bo_population_size,
            'final_stats': self.generation_stats[-1] if self.generation_stats else None
        } 