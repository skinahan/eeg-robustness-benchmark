"""
HyperNEAT Evolution Engine for CfC Networks

This module implements the main HyperNEAT evolution engine that orchestrates
the evolution of CfC networks using HyperNEAT's indirect encoding approach.
"""

import numpy as np
import random
import json
import os
from typing import List, Dict, Any, Tuple, Optional
import logging
import matplotlib.pyplot as plt
from datetime import datetime

from .hyperneat_genome import HyperNEATGenome
from .cfc_substrate import CfCSubstrate
from .fitness_evaluator import HyperNEATFitnessEvaluator


class HyperNEATEvolutionEngine:
    """
    Main HyperNEAT evolution engine for CfC networks.
    
    This class orchestrates the entire evolution process, including:
    - Population management
    - Selection and reproduction
    - Fitness evaluation
    - Evolution statistics tracking
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
        output_dir: str = "hyperneat_results"
    ):
        self.substrate = substrate
        self.fitness_evaluator = fitness_evaluator
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_size = elitism_size
        self.output_dir = output_dir
        
        # Evolution tracking
        self.population: List[HyperNEATGenome] = []
        self.generation_stats: List[Dict[str, Any]] = []
        self.best_genome: Optional[HyperNEATGenome] = None
        self.best_fitness: float = 0.0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def initialize_population(self):
        """Initialize the population with random genomes"""
        self.logger.info(f"Initializing population of size {self.population_size}")
        
        self.population = []
        for i in range(self.population_size):
            genome = HyperNEATGenome()
            genome.generation = 0
            self.population.append(genome)
        
        self.logger.info("Population initialized successfully")
    
    def evolve(self) -> Tuple[HyperNEATGenome, List[Dict[str, Any]]]:
        """
        Run the complete evolution process.
        
        Returns:
            Tuple of (best_genome, evolution_history)
        """
        self.logger.info("Starting HyperNEAT evolution")
        
        # Initialize population
        self.initialize_population()
        
        # Evolution loop
        for generation in range(self.generations):
            self.logger.info(f"Generation {generation + 1}/{self.generations}")
            
            # Evaluate population
            self._evaluate_population()
            
            # Record statistics
            stats = self._record_generation_stats(generation)
            self.generation_stats.append(stats)
            
            # Check for best genome
            self._update_best_genome()
            
            # Save checkpoint
            if (generation + 1) % 10 == 0:
                self._save_checkpoint(generation)
            
            # Create next generation
            if generation < self.generations - 1:
                self._create_next_generation()
        
        # Final evaluation
        self._evaluate_population()
        self._update_best_genome()
        
        # Save final results
        self._save_final_results()
        
        return self.best_genome, self.generation_stats
    
    def _evaluate_population(self):
        """Evaluate the entire population"""
        self.logger.info("Evaluating population")
        
        # Evaluate all genomes
        results = self.fitness_evaluator.evaluate_population(self.population)
        
        # Update genome fitness values
        for genome, result in zip(self.population, results):
            genome.fitness = result['overall_fitness']
    
    def _create_next_generation(self):
        """Create the next generation through selection and reproduction"""
        # Sort population by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        
        # Elitism: keep best individuals
        new_population = self.population[:self.elitism_size].copy()
        
        # Generate remaining individuals
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Reproduction
            if random.random() < self.crossover_rate:
                # Crossover
                child = parent1.crossover(parent2)
            else:
                # Clone parent
                child = self._clone_genome(parent1)
            
            # Mutation
            child.mutate(self.mutation_rate)
            child.generation = self.population[0].generation + 1
            
            new_population.append(child)
        
        self.population = new_population
    
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
            bias_range=genome.bias_range
        )
        
        # Copy nodes and connections
        clone.nodes = []
        for node in genome.nodes:
            clone.nodes.append(type(node)(
                node_id=node.node_id,
                node_type=node.node_type,
                activation=node.activation,
                bias=node.bias
            ))
        
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
        
        stats = {
            'generation': generation,
            'best_fitness': max(fitnesses),
            'worst_fitness': min(fitnesses),
            'avg_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'population_size': len(self.population),
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Generation {generation}: Best={stats['best_fitness']:.4f}, "
                        f"Avg={stats['avg_fitness']:.4f}")
        
        return stats
    
    def _update_best_genome(self):
        """Update the best genome found so far"""
        current_best = max(self.population, key=lambda g: g.fitness)
        
        if self.best_genome is None or current_best.fitness > self.best_fitness:
            self.best_genome = self._clone_genome(current_best)
            self.best_fitness = current_best.fitness
            self.logger.info(f"New best genome found: {self.best_fitness:.4f}")
    
    def _save_checkpoint(self, generation: int):
        """Save a checkpoint of the evolution"""
        checkpoint = {
            'generation': generation,
            'best_genome': self.best_genome.to_dict() if self.best_genome else None,
            'best_fitness': self.best_fitness,
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
        
        self.logger.info(f"Final results saved to {self.output_dir}")
    
    def _create_evolution_plots(self):
        """Create plots showing evolution progress"""
        if not self.generation_stats:
            return
        
        generations = [stats['generation'] for stats in self.generation_stats]
        best_fitness = [stats['best_fitness'] for stats in self.generation_stats]
        avg_fitness = [stats['avg_fitness'] for stats in self.generation_stats]
        worst_fitness = [stats['worst_fitness'] for stats in self.generation_stats]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Fitness evolution
        ax1.plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
        ax1.plot(generations, avg_fitness, 'g-', label='Average Fitness', linewidth=2)
        ax1.plot(generations, worst_fitness, 'r-', label='Worst Fitness', linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Evolution Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Fitness distribution
        ax2.hist(best_fitness, bins=20, alpha=0.7, label='Best Fitness Distribution')
        ax2.set_xlabel('Fitness')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Fitness Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "evolution_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Evolution plots saved: {plot_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load evolution from a checkpoint"""
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        # Load best genome
        if checkpoint['best_genome']:
            self.best_genome = HyperNEATGenome.from_dict(checkpoint['best_genome'])
            self.best_fitness = checkpoint['best_fitness']
        
        # Load generation stats
        self.generation_stats = checkpoint['generation_stats']
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get a summary of the evolution results"""
        if not self.generation_stats:
            return {}
        
        final_stats = self.generation_stats[-1]
        
        summary = {
            'total_generations': len(self.generation_stats),
            'final_best_fitness': final_stats['best_fitness'],
            'final_avg_fitness': final_stats['avg_fitness'],
            'best_fitness_ever': max(stats['best_fitness'] for stats in self.generation_stats),
            'improvement': final_stats['best_fitness'] - self.generation_stats[0]['best_fitness'],
            'best_genome_parameter_count': self.best_genome.get_parameter_count() if self.best_genome else 0
        }
        
        return summary 