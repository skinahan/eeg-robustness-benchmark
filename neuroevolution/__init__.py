"""
Neuroevolution module for EEG architecture search.

This module provides a complete neuroevolution framework for automatically
designing EEG classification architectures that balance model complexity,
overfitting, and noise resilience.
"""

from .architecture_genome import ArchitectureGenome, LayerConfig, GenomeGenerator
from .genetic_operators import GeneticOperators, TournamentSelection
from .model_builder import NeuroevolutionModel, NeuroevolutionModelBuilder
from .fitness_evaluator import FitnessEvaluator, FastFitnessEvaluator, ParetoFitnessEvaluator
from .evolution_engine import NeuroevolutionEngine, MultiObjectiveEvolutionEngine

__all__ = [
    'ArchitectureGenome',
    'LayerConfig', 
    'GenomeGenerator',
    'GeneticOperators',
    'TournamentSelection',
    'NeuroevolutionModel',
    'NeuroevolutionModelBuilder',
    'FitnessEvaluator',
    'FastFitnessEvaluator',
    'ParetoFitnessEvaluator',
    'NeuroevolutionEngine',
    'MultiObjectiveEvolutionEngine'
]

__version__ = "1.0.0" 