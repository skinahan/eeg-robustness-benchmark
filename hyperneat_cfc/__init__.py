"""
HyperNEAT-based CfC Evolution for EEG Classification

This module provides a proof-of-concept implementation for using HyperNEAT
to evolve CfC (Closed-form Continuous) networks for EEG signal classification.
The system leverages the ncps framework to create explicit CfC network wirings
that can be evolved through HyperNEAT's indirect encoding approach.

Now includes hybrid evolution combining HyperNEAT with Bayesian optimization.
"""

from .hyperneat_genome import HyperNEATGenome
from .cfc_substrate import CfCSubstrate
from .hyperneat_engine import HyperNEATEvolutionEngine
from .cfc_phenotype import CfCPhenotype
from .fitness_evaluator import HyperNEATFitnessEvaluator
from .bo_genome_generator import BOGenomeGenerator
from .hybrid_evolution_engine import HybridEvolutionEngine

__version__ = "0.1.0"
__author__ = "EEG Noise Robustness Project"

__all__ = [
    "HyperNEATGenome",
    "CfCSubstrate", 
    "HyperNEATEvolutionEngine",
    "CfCPhenotype",
    "HyperNEATFitnessEvaluator",
    "BOGenomeGenerator",
    "HybridEvolutionEngine"
] 