import random
import numpy as np
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

from .architecture_genome import ArchitectureGenome, GenomeGenerator
from .genetic_operators import GeneticOperators, TournamentSelection
from .fitness_evaluator import FitnessEvaluator, FastFitnessEvaluator, ParetoFitnessEvaluator
from .model_builder import NeuroevolutionModelBuilder


class NeuroevolutionEngine:
    """Main evolution engine for architecture search"""
    
    def __init__(
        self,
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        tournament_size: int = 3,
        elite_size: int = 2,
        evaluator_type: str = 'fast',  # 'fast', 'full', 'pareto'
        save_best_models: bool = True,
        output_dir: str = 'neuroevolution_results',
        seed: int = 42
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.evaluator_type = evaluator_type
        self.save_best_models = save_best_models
        self.output_dir = output_dir
        self.seed = seed
        
        # Initialize components
        self.generator = GenomeGenerator()
        self.operators = GeneticOperators(mutation_rate, crossover_rate)
        self.selector = TournamentSelection(tournament_size)
        
        # Initialize evaluator based on type
        if evaluator_type == 'fast':
            self.evaluator = FastFitnessEvaluator(seed=seed)
        elif evaluator_type == 'pareto':
            self.evaluator = ParetoFitnessEvaluator(seed=seed)
        else:
            self.evaluator = FitnessEvaluator(seed=seed)
        
        # Evolution history
        self.history = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'best_genomes': [],
            'population_diversity': []
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
    
    def evolve(self) -> Tuple[ArchitectureGenome, Dict[str, Any]]:
        """Run the complete evolution process"""
        print(f"Starting neuroevolution with {self.population_size} individuals for {self.generations} generations")
        print(f"Evaluator type: {self.evaluator_type}")
        
        # Initialize population
        population = self._initialize_population()
        
        # Evolution loop
        for generation in range(self.generations):
            print(f"\n--- Generation {generation + 1}/{self.generations} ---")
            
            # Evaluate population
            self._evaluate_population(population)
            
            # Record statistics
            self._record_generation_stats(population, generation)
            
            # Select parents and create offspring
            offspring = self._create_offspring(population)
            
            # Replace population
            population = self._replace_population(population, offspring)
            
            # Save intermediate results
            if generation % 5 == 0:
                self._save_intermediate_results(generation)
        
        # Final evaluation
        self._evaluate_population(population)
        
        # Get best genome
        best_genome = max(population, key=lambda x: x.overall_fitness)
        
        # Save final results
        self._save_final_results(best_genome, population)
        
        return best_genome, self.history
    
    def _initialize_population(self) -> List[ArchitectureGenome]:
        """Initialize the population with random genomes"""
        population = []
        
        print("Initializing population...")
        for i in range(self.population_size):
            genome = self.generator.generate_random_genome()
            population.append(genome)
            print(f"Created genome {i+1}/{self.population_size}")
        
        return population
    
    def _evaluate_population(self, population: List[ArchitectureGenome]):
        """Evaluate all genomes in the population"""
        print("Evaluating population...")
        
        for i, genome in enumerate(population):
            print(f"Evaluating genome {i+1}/{len(population)}")
            fitness = self.evaluator.evaluate_genome(genome)
            
            # Update genome with fitness scores
            genome.accuracy = fitness['accuracy']
            genome.noise_resilience = fitness['noise_resilience']
            genome.complexity_score = fitness['complexity_score']
            genome.overfitting_score = fitness['overfitting_score']
            genome.overall_fitness = fitness['overall_fitness']
    
    def _create_offspring(self, population: List[ArchitectureGenome]) -> List[ArchitectureGenome]:
        """Create offspring through selection, crossover, and mutation"""
        offspring = []
        
        # Elitism: keep best individuals
        sorted_population = sorted(population, key=lambda x: x.overall_fitness, reverse=True)
        elite = sorted_population[:self.elite_size]
        offspring.extend(elite)
        
        # Create remaining offspring
        while len(offspring) < self.population_size:
            # Select parents
            parents = self.selector.select_parents(population, num_parents=2)
            
            # Crossover
            child1, child2 = self.operators.crossover(parents[0], parents[1])
            
            # Mutation
            child1 = self.operators.mutate(child1)
            child2 = self.operators.mutate(child2)
            
            offspring.extend([child1, child2])
        
        # Trim to population size
        offspring = offspring[:self.population_size]
        
        return offspring
    
    def _replace_population(self, old_population: List[ArchitectureGenome], offspring: List[ArchitectureGenome]) -> List[ArchitectureGenome]:
        """Replace the old population with offspring"""
        return offspring
    
    def _record_generation_stats(self, population: List[ArchitectureGenome], generation: int):
        """Record statistics for the current generation"""
        fitnesses = [genome.overall_fitness for genome in population]
        accuracies = [genome.accuracy for genome in population]
        noise_resiliences = [genome.noise_resilience for genome in population]
        complexities = [genome.complexity_score for genome in population]
        
        best_genome = max(population, key=lambda x: x.overall_fitness)
        
        self.history['generations'].append(generation)
        self.history['best_fitness'].append(max(fitnesses))
        self.history['avg_fitness'].append(np.mean(fitnesses))
        self.history['best_genomes'].append(best_genome)
        
        # Calculate population diversity (standard deviation of fitness)
        self.history['population_diversity'].append(np.std(fitnesses))
        
        print(f"Best fitness: {max(fitnesses):.4f}")
        print(f"Average fitness: {np.mean(fitnesses):.4f}")
        print(f"Best accuracy: {max(accuracies):.4f}")
        print(f"Best noise resilience: {max(noise_resiliences):.4f}")
        print(f"Average complexity: {np.mean(complexities):.4f}")
    
    def _save_intermediate_results(self, generation: int):
        """Save intermediate results"""
        best_genome = self.history['best_genomes'][-1]
        
        # Save best genome
        genome_path = os.path.join(self.output_dir, f"best_genome_gen_{generation}.json")
        with open(genome_path, 'w') as f:
            json.dump(best_genome.to_dict(), f, indent=2)
        
        # Save statistics
        stats_path = os.path.join(self.output_dir, f"stats_gen_{generation}.json")
        stats = {
            'generation': generation,
            'best_fitness': self.history['best_fitness'][-1],
            'avg_fitness': self.history['avg_fitness'][-1],
            'population_diversity': self.history['population_diversity'][-1]
        }
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _save_final_results(self, best_genome: ArchitectureGenome, final_population: List[ArchitectureGenome]):
        """Save final results and create visualizations"""
        print("\nSaving final results...")
        
        # Save best genome
        best_genome_path = os.path.join(self.output_dir, "best_genome_final.json")
        with open(best_genome_path, 'w') as f:
            json.dump(best_genome.to_dict(), f, indent=2)
        
        # Save evolution history
        history_path = os.path.join(self.output_dir, "evolution_history.json")
        history_data = {
            'generations': self.history['generations'],
            'best_fitness': self.history['best_fitness'],
            'avg_fitness': self.history['avg_fitness'],
            'population_diversity': self.history['population_diversity']
        }
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        # Create visualizations
        self._create_evolution_plots()
        
        # Save population summary
        self._save_population_summary(final_population)
        
        print(f"Results saved to {self.output_dir}")
    
    def _create_evolution_plots(self):
        """Create evolution plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Fitness evolution
        axes[0, 0].plot(self.history['generations'], self.history['best_fitness'], 'b-', label='Best Fitness')
        axes[0, 0].plot(self.history['generations'], self.history['avg_fitness'], 'r--', label='Average Fitness')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Fitness')
        axes[0, 0].set_title('Fitness Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Population diversity
        axes[0, 1].plot(self.history['generations'], self.history['population_diversity'], 'g-')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Population Diversity (Std Dev)')
        axes[0, 1].set_title('Population Diversity')
        axes[0, 1].grid(True)
        
        # Best genome statistics
        best_genomes = self.history['best_genomes']
        accuracies = [g.accuracy for g in best_genomes]
        noise_resiliences = [g.noise_resilience for g in best_genomes]
        
        axes[1, 0].plot(self.history['generations'], accuracies, 'b-', label='Accuracy')
        axes[1, 0].plot(self.history['generations'], noise_resiliences, 'r-', label='Noise Resilience')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Best Genome Performance')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Complexity evolution
        complexities = [g.complexity_score for g in best_genomes]
        axes[1, 1].plot(self.history['generations'], complexities, 'm-')
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Complexity Score')
        axes[1, 1].set_title('Best Genome Complexity')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "evolution_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_population_summary(self, population: List[ArchitectureGenome]):
        """Save summary of final population"""
        summary_data = []
        
        for i, genome in enumerate(population):
            summary_data.append({
                'rank': i + 1,
                'fitness': genome.overall_fitness,
                'accuracy': genome.accuracy,
                'noise_resilience': genome.noise_resilience,
                'complexity': genome.complexity_score,
                'overfitting': genome.overfitting_score,
                'num_layers': len(genome.layers),
                'num_parameters': genome.get_parameter_count(),
                'learning_rate': genome.learning_rate,
                'dropout_rate': genome.dropout_rate
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.output_dir, "population_summary.csv")
        summary_df.to_csv(summary_path, index=False)
    
    def load_best_genome(self, genome_path: str) -> ArchitectureGenome:
        """Load a genome from file"""
        with open(genome_path, 'r') as f:
            genome_data = json.load(f)
        
        return ArchitectureGenome.from_dict(genome_data)
    
    def create_model_from_genome(self, genome: ArchitectureGenome):
        """Create a model from a genome"""
        return NeuroevolutionModelBuilder.create_classifier(genome)


class MultiObjectiveEvolutionEngine(NeuroevolutionEngine):
    """Evolution engine for multi-objective optimization"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluator = ParetoFitnessEvaluator(seed=self.seed)
    
    def _create_offspring(self, population: List[ArchitectureGenome]) -> List[ArchitectureGenome]:
        """Create offspring using Pareto-based selection"""
        offspring = []
        
        # Get Pareto front
        pareto_front = self.evaluator.get_pareto_front(population)
        
        # Elitism: keep Pareto front individuals
        offspring.extend(pareto_front[:self.elite_size])
        
        # Create remaining offspring
        while len(offspring) < self.population_size:
            # Select parents from Pareto front
            if len(pareto_front) >= 2:
                parents = random.sample(pareto_front, 2)
            else:
                parents = self.selector.select_parents(population, num_parents=2)
            
            # Crossover and mutation
            child1, child2 = self.operators.crossover(parents[0], parents[1])
            child1 = self.operators.mutate(child1)
            child2 = self.operators.mutate(child2)
            
            offspring.extend([child1, child2])
        
        # Trim to population size
        offspring = offspring[:self.population_size]
        
        return offspring


def run_quick_demo():
    """Run a quick demonstration of the neuroevolution system"""
    print("Running quick neuroevolution demo...")
    
    # Create evolution engine with fast evaluator
    engine = NeuroevolutionEngine(
        population_size=10,
        generations=5,
        evaluator_type='fast',
        output_dir='demo_results'
    )
    
    # Run evolution
    best_genome, history = engine.evolve()
    
    print(f"\nBest genome found:")
    print(f"Fitness: {best_genome.overall_fitness:.4f}")
    print(f"Accuracy: {best_genome.accuracy:.4f}")
    print(f"Noise Resilience: {best_genome.noise_resilience:.4f}")
    print(f"Complexity: {best_genome.complexity_score:.4f}")
    print(f"Number of layers: {len(best_genome.layers)}")
    print(f"Number of parameters: {best_genome.get_parameter_count()}")
    
    return best_genome, history


if __name__ == "__main__":
    # Run demo
    best_genome, history = run_quick_demo() 