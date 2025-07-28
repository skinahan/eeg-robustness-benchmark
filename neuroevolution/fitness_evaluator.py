import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch

from .architecture_genome import ArchitectureGenome
from .model_builder import NeuroevolutionModelBuilder
from augmentation.noise import EEGNoiseAugmentor, TrainOnlyNoiseClassifier
from config import get_paradigm
from moabb.datasets import BNCI2014_001
from moabb.evaluations import WithinSessionEvaluation
from globals import set_seeds, get_seed


class FitnessEvaluator:
    """Evaluates fitness of architecture genomes across multiple objectives"""
    
    def __init__(
        self,
        dataset=None,
        paradigm=None,
        noise_types: List[str] = ['dropout', 'gaussian', 'eog'],
        noise_intensities: List[float] = [10.0, 25.0, 50.0],
        subject_list: Optional[List[int]] = None,
        max_epochs: int = 50,  # Reduced for faster evaluation
        cv_folds: int = 3,
        seed: int = 42
    ):
        self.dataset = dataset or BNCI2014_001()
        self.paradigm = paradigm or get_paradigm()
        self.noise_types = noise_types
        self.noise_intensities = noise_intensities
        self.subject_list = subject_list
        self.max_epochs = max_epochs
        self.cv_folds = cv_folds
        self.seed = seed
        
        if subject_list is not None:
            self.dataset.subject_list = subject_list
        
        set_seeds(seed)
    
    def evaluate_genome(self, genome: ArchitectureGenome) -> Dict[str, float]:
        """Evaluate a single genome and return fitness scores"""
        
        # Validate genome first
        if not NeuroevolutionModelBuilder.validate_genome(genome):
            return self._get_invalid_fitness()
        
        try:
            # Update genome with evaluation parameters
            genome.max_epochs = self.max_epochs
            
            # Evaluate clean performance
            clean_accuracy = self._evaluate_clean_performance(genome)
            
            # Evaluate noise resilience
            noise_resilience = self._evaluate_noise_resilience(genome)
            
            # Calculate complexity score
            complexity_score = genome.get_complexity_score()
            
            # Calculate overfitting score (simplified)
            overfitting_score = self._evaluate_overfitting(genome, clean_accuracy)
            
            # Calculate overall fitness
            overall_fitness = self._calculate_overall_fitness(
                clean_accuracy, noise_resilience, complexity_score, overfitting_score
            )
            
            # Update genome with scores
            genome.accuracy = clean_accuracy
            genome.noise_resilience = noise_resilience
            genome.complexity_score = complexity_score
            genome.overfitting_score = overfitting_score
            genome.overall_fitness = overall_fitness
            
            return {
                'accuracy': clean_accuracy,
                'noise_resilience': noise_resilience,
                'complexity_score': complexity_score,
                'overfitting_score': overfitting_score,
                'overall_fitness': overall_fitness
            }
            
        except Exception as e:
            print(f"Error evaluating genome: {e}")
            return self._get_invalid_fitness()
    
    def _evaluate_clean_performance(self, genome: ArchitectureGenome) -> float:
        """Evaluate model performance on clean data"""
        try:
            # Create classifier
            classifier = NeuroevolutionModelBuilder.create_classifier(genome)
            
            # Simple evaluation using cross-validation
            # For faster evaluation, we'll use a simplified approach
            evaluation = WithinSessionEvaluation(
                paradigm=self.paradigm,
                datasets=[self.dataset],
                n_jobs=1,  # Single job for faster evaluation
                overwrite=True
            )
            
            # Run evaluation
            results = evaluation.process(classifier)
            
            # Extract accuracy
            if not results.empty:
                accuracy = results['score'].mean()
                return float(accuracy)
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error in clean performance evaluation: {e}")
            return 0.0
    
    def _evaluate_noise_resilience(self, genome: ArchitectureGenome) -> float:
        """Evaluate model resilience to different types of noise"""
        resilience_scores = []
        
        for noise_type in self.noise_types:
            for intensity in self.noise_intensities:
                try:
                    # Create classifier
                    classifier = NeuroevolutionModelBuilder.create_classifier(genome)
                    
                    # Create noisy classifier
                    noisy_classifier = TrainOnlyNoiseClassifier(
                        base_pipeline=classifier,
                        noise_type=noise_type,
                        intensity=intensity,
                        seed=self.seed
                    )
                    
                    # Evaluate
                    evaluation = WithinSessionEvaluation(
                        paradigm=self.paradigm,
                        datasets=[self.dataset],
                        n_jobs=1,
                        overwrite=True
                    )
                    
                    results = evaluation.process(noisy_classifier)
                    
                    if not results.empty:
                        accuracy = results['score'].mean()
                        resilience_scores.append(float(accuracy))
                    else:
                        resilience_scores.append(0.0)
                        
                except Exception as e:
                    print(f"Error evaluating noise resilience for {noise_type} {intensity}: {e}")
                    resilience_scores.append(0.0)
        
        # Return average resilience score
        return np.mean(resilience_scores) if resilience_scores else 0.0
    
    def _evaluate_overfitting(self, genome: ArchitectureGenome, clean_accuracy: float) -> float:
        """Evaluate overfitting tendency (simplified)"""
        # Simple heuristic: penalize very complex models with low accuracy
        complexity = genome.get_complexity_score()
        
        if clean_accuracy < 0.5 and complexity > 0.7:
            # High complexity with low accuracy suggests overfitting
            return 0.0
        elif clean_accuracy > 0.8 and complexity < 0.5:
            # High accuracy with low complexity is good
            return 1.0
        else:
            # Moderate score
            return 0.5
    
    def _calculate_overall_fitness(
        self,
        accuracy: float,
        noise_resilience: float,
        complexity_score: float,
        overfitting_score: float
    ) -> float:
        """Calculate overall fitness using weighted multi-objective optimization"""
        
        # Weights for different objectives
        weights = {
            'accuracy': 0.4,
            'noise_resilience': 0.3,
            'complexity': 0.2,
            'overfitting': 0.1
        }
        
        # Normalize complexity (lower is better)
        complexity_penalty = 1.0 - complexity_score
        
        # Calculate weighted fitness
        fitness = (
            weights['accuracy'] * accuracy +
            weights['noise_resilience'] * noise_resilience +
            weights['complexity'] * complexity_penalty +
            weights['overfitting'] * overfitting_score
        )
        
        return fitness
    
    def _get_invalid_fitness(self) -> Dict[str, float]:
        """Return fitness scores for invalid genomes"""
        return {
            'accuracy': 0.0,
            'noise_resilience': 0.0,
            'complexity_score': 1.0,  # High complexity penalty
            'overfitting_score': 0.0,
            'overall_fitness': 0.0
        }
    
    def evaluate_population(self, population: List[ArchitectureGenome]) -> List[Dict[str, float]]:
        """Evaluate a population of genomes"""
        results = []
        
        for i, genome in enumerate(population):
            print(f"Evaluating genome {i+1}/{len(population)}")
            fitness = self.evaluate_genome(genome)
            results.append(fitness)
        
        return results


class FastFitnessEvaluator(FitnessEvaluator):
    """Faster fitness evaluator for initial screening"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_epochs = 20  # Even fewer epochs for faster evaluation
        self.cv_folds = 2
    
    def _evaluate_clean_performance(self, genome: ArchitectureGenome) -> float:
        """Fast evaluation using simplified approach"""
        try:
            # Create a simple model and test with dummy data
            model = NeuroevolutionModelBuilder.create_model(genome)
            
            # Test with dummy input
            dummy_input = torch.randn(2, genome.input_channels, genome.input_times)
            output = model(dummy_input)
            
            # Simple heuristic based on model properties
            param_count = genome.get_parameter_count()
            layer_count = len(genome.layers)
            
            # Prefer models with reasonable complexity
            if param_count > 500000 or layer_count > 15:
                return 0.0
            
            # Simple scoring based on architecture characteristics
            score = 0.5  # Base score
            
            # Bonus for having conv layers early
            if genome.layers[0].layer_type in ['conv1d', 'conv2d']:
                score += 0.1
            
            # Bonus for reasonable dropout
            avg_dropout = np.mean([layer.dropout_rate for layer in genome.layers])
            if 0.1 <= avg_dropout <= 0.3:
                score += 0.1
            
            # Penalty for very high complexity
            complexity = genome.get_complexity_score()
            if complexity > 0.8:
                score -= 0.2
            
            return min(max(score, 0.0), 1.0)
            
        except Exception as e:
            print(f"Error in fast evaluation: {e}")
            return 0.0
    
    def _evaluate_noise_resilience(self, genome: ArchitectureGenome) -> float:
        """Fast noise resilience evaluation using heuristics"""
        # Simple heuristic: models with more regularization tend to be more noise resilient
        avg_dropout = np.mean([layer.dropout_rate for layer in genome.layers])
        batch_norm_count = sum(1 for layer in genome.layers if layer.batch_norm)
        
        resilience = 0.5  # Base score
        
        # Bonus for regularization
        if avg_dropout > 0.15:
            resilience += 0.2
        
        if batch_norm_count > 0:
            resilience += 0.1
        
        # Bonus for attention mechanisms
        attention_count = sum(1 for layer in genome.layers if layer.layer_type == 'attention')
        if attention_count > 0:
            resilience += 0.1
        
        return min(max(resilience, 0.0), 1.0)


class ParetoFitnessEvaluator(FitnessEvaluator):
    """Fitness evaluator that supports Pareto-optimal multi-objective optimization"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.objectives = ['accuracy', 'noise_resilience', 'complexity', 'overfitting']
    
    def evaluate_genome_pareto(self, genome: ArchitectureGenome) -> Dict[str, float]:
        """Evaluate genome for Pareto optimization (returns individual objectives)"""
        fitness = self.evaluate_genome(genome)
        
        return {
            'accuracy': fitness['accuracy'],
            'noise_resilience': fitness['noise_resilience'],
            'complexity': 1.0 - fitness['complexity_score'],  # Invert for minimization
            'overfitting': 1.0 - fitness['overfitting_score']  # Invert for minimization
        }
    
    def dominates(self, genome1: ArchitectureGenome, genome2: ArchitectureGenome) -> bool:
        """Check if genome1 dominates genome2 in Pareto sense"""
        obj1 = self.evaluate_genome_pareto(genome1)
        obj2 = self.evaluate_genome_pareto(genome2)
        
        # Check if genome1 is at least as good in all objectives
        at_least_as_good = all(obj1[obj] >= obj2[obj] for obj in self.objectives)
        
        # Check if genome1 is strictly better in at least one objective
        strictly_better = any(obj1[obj] > obj2[obj] for obj in self.objectives)
        
        return at_least_as_good and strictly_better
    
    def get_pareto_front(self, population: List[ArchitectureGenome]) -> List[ArchitectureGenome]:
        """Get the Pareto front of the population"""
        pareto_front = []
        
        for genome in population:
            is_dominated = False
            
            # Check if this genome is dominated by any other
            for other_genome in population:
                if genome != other_genome and self.dominates(other_genome, genome):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(genome)
        
        return pareto_front 