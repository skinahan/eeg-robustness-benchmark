"""
Bayesian Optimization Genome Generator for HyperNEAT CfC

This module creates genomes using Optuna's Bayesian optimization
while maintaining full compatibility with the existing HyperNEAT evolution system.
"""

import optuna
import numpy as np
import random
from typing import Dict, Any, List, Optional, Tuple
import logging

from .hyperneat_genome import HyperNEATGenome, CPPNNode, CPPNConnection


class BOGenomeGenerator:
    """
    Generates HyperNEAT genomes using Bayesian optimization.
    
    This class creates genomes that are fully compatible with the existing
    HyperNEAT evolution system by combining BO-suggested parameters with
    randomly generated CPPN structures.
    """
    
    def __init__(
        self,
        study: optuna.Study,
        substrate_dimensions: Tuple[int, int, int],  # (input_size, hidden_size, output_size)
        historical_fitness_data: Optional[List[Dict[str, Any]]] = None
    ):
        self.study = study
        self.substrate_dimensions = substrate_dimensions
        self.historical_fitness_data = historical_fitness_data or []
        self.logger = logging.getLogger(__name__)
        
        # Determine CPPN input size based on substrate dimensions
        self.cppn_input_size = 4
    
    def suggest_genome(self) -> HyperNEATGenome:
        """
        Generate a genome using BO-suggested parameters.
        
        Returns:
            HyperNEATGenome that is fully compatible with evolution
        """
        # Get BO suggestions for parameters
        trial = self.study.ask()
        bo_params = self._suggest_parameters(trial)
        
        # Create genome with BO parameters
        genome = HyperNEATGenome(
            input_nodes=self.cppn_input_size,
            hidden_nodes=bo_params['hidden_nodes'],
            output_nodes=bo_params['output_nodes'],
            max_connections=bo_params['max_connections'],
            proj_size=bo_params['proj_size'],
            mode=bo_params['mode'],
            mixed_memory=bo_params['mixed_memory'],
            return_sequences=bo_params['return_sequences']
        )
        
        # Generate random CPPN structure to ensure compatibility
        self._generate_random_cppn_structure(genome, bo_params)
        
        self.logger.info(f"Generated BO genome with {len(genome.nodes)} nodes and {len(genome.connections)} connections")
        
        return genome
    
    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Use Optuna to suggest genome parameters.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested parameters
        """
        # Suggest CPPN architecture parameters
        hidden_nodes = trial.suggest_int('hidden_nodes', 2, 8)
        output_nodes = trial.suggest_int('output_nodes', 1, 2)
        max_connections = trial.suggest_int('max_connections', 5, 15)
        
        # Suggest CfC architecture parameters
        proj_size = trial.suggest_categorical('proj_size', [None, 2])
        mode = trial.suggest_categorical('mode', ['default', 'pure', 'no_gate'])
        mixed_memory = trial.suggest_categorical('mixed_memory', [True, False])
        return_sequences = trial.suggest_categorical('return_sequences', [False])
        
        # Use historical data to bias suggestions if available
        if self.historical_fitness_data:
            self._apply_historical_bias(trial, hidden_nodes, max_connections)
        
        return {
            'hidden_nodes': hidden_nodes,
            'output_nodes': output_nodes,
            'max_connections': max_connections,
            'proj_size': proj_size,
            'mode': mode,
            'mixed_memory': mixed_memory,
            'return_sequences': return_sequences
        }
    
    def _generate_random_cppn_structure(self, genome: HyperNEATGenome, bo_params: Dict[str, Any]):
        """
        Generate a random CPPN structure for the genome.
        
        Args:
            genome: Genome to populate with CPPN structure
            bo_params: BO-suggested parameters
        """
        # Clear existing structure
        genome.nodes = []
        genome.connections = []
        
        # Create input nodes
        for i in range(genome.input_nodes):
            genome.nodes.append(CPPNNode(
                node_id=i,
                node_type='input',
                activation='linear',
                bias=0.0
            ))
        
        # Create hidden nodes
        hidden_start = genome.input_nodes
        for i in range(bo_params['hidden_nodes']):
            genome.nodes.append(CPPNNode(
                node_id=hidden_start + i,
                node_type='hidden',
                activation=random.choice(['tanh', 'sigmoid', 'relu', 'sin', 'cos']),
                bias=random.uniform(-1.0, 1.0)
            ))
        
        # Create output nodes
        output_start = hidden_start + bo_params['hidden_nodes']
        for i in range(bo_params['output_nodes']):
            genome.nodes.append(CPPNNode(
                node_id=output_start + i,
                node_type='output',
                activation='tanh',
                bias=0.0
            ))
        
        # Create random connections
        max_connections = min(bo_params['max_connections'], len(genome.nodes) * 2)
        connections_created = 0
        attempts = 0
        
        while connections_created < max_connections and attempts < 100:
            from_node = random.choice(genome.nodes).node_id
            to_node = random.choice(genome.nodes).node_id
            
            # Avoid self-connections and duplicates
            if (from_node != to_node and 
                not any(c.from_node == from_node and c.to_node == to_node 
                       for c in genome.connections)):
                
                genome.connections.append(CPPNConnection(
                    from_node=from_node,
                    to_node=to_node,
                    weight=random.uniform(-3.0, 3.0),
                    enabled=True
                ))
                connections_created += 1
            
            attempts += 1
    
    def _apply_historical_bias(self, trial: optuna.Trial, hidden_nodes: int, max_connections: int):
        """
        Apply bias based on historical fitness data.
        
        Args:
            trial: Optuna trial
            hidden_nodes: Suggested hidden nodes
            max_connections: Suggested max connections
        """
        if not self.historical_fitness_data:
            return
        
        # Analyze historical performance
        high_fitness_genomes = [data for data in self.historical_fitness_data 
                               if data.get('fitness', 0) > 0.7]  # Threshold for "good" fitness
        
        if high_fitness_genomes:
            # Extract patterns from high-fitness genomes
            avg_hidden = np.mean([g.get('hidden_nodes', 6) for g in high_fitness_genomes])
            avg_connections = np.mean([g.get('max_connections', 10) for g in high_fitness_genomes])
            
            # Bias towards historically successful values
            if abs(hidden_nodes - avg_hidden) > 2:
                # Suggest closer to historical average
                trial.suggest_int('hidden_nodes', max(2, int(avg_hidden - 1)), 
                                min(8, int(avg_hidden + 1)))
            
            if abs(max_connections - avg_connections) > 3:
                # Suggest closer to historical average
                trial.suggest_int('max_connections', max(5, int(avg_connections - 2)), 
                                min(15, int(avg_connections + 2)))
    
    def update_historical_data(self, genome: HyperNEATGenome, fitness: Dict[str, float]):
        """
        Update historical data with new genome evaluation.
        
        Args:
            genome: Evaluated genome
            fitness: Fitness scores
        """
        historical_entry = {
            'hidden_nodes': genome.hidden_nodes,
            'output_nodes': genome.output_nodes,
            'max_connections': genome.max_connections,
            'proj_size': genome.proj_size,
            'mode': genome.mode,
            'mixed_memory': genome.mixed_memory,
            'return_sequences': genome.return_sequences,
            'fitness': fitness.get('overall_fitness', 0.0),
            'accuracy': fitness.get('clean_accuracy', 0.0),
            'noise_resilience': fitness.get('noise_resilience', 0.0),
            'complexity_score': fitness.get('complexity_score', 0.0)
        }
        
        self.historical_fitness_data.append(historical_entry)
        
        # Keep only recent data to avoid memory bloat
        if len(self.historical_fitness_data) > 100:
            self.historical_fitness_data = self.historical_fitness_data[-50:]
    
    def get_bo_suggestions_summary(self) -> Dict[str, Any]:
        """
        Get summary of BO suggestions and historical performance.
        
        Returns:
            Dictionary with BO statistics
        """
        if not self.historical_fitness_data:
            return {'message': 'No historical data available'}
        
        recent_data = self.historical_fitness_data[-20:]  # Last 20 evaluations
        
        return {
            'total_evaluations': len(self.historical_fitness_data),
            'recent_avg_fitness': np.mean([d['fitness'] for d in recent_data]),
            'recent_avg_accuracy': np.mean([d['accuracy'] for d in recent_data]),
            'recent_avg_noise_resilience': np.mean([d['noise_resilience'] for d in recent_data]),
            'best_fitness': max([d['fitness'] for d in self.historical_fitness_data]),
            'best_accuracy': max([d['accuracy'] for d in self.historical_fitness_data])
        } 