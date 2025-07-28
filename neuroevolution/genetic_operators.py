import random
import numpy as np
from typing import List, Tuple, Optional
from copy import deepcopy
from .architecture_genome import ArchitectureGenome, LayerConfig


class GeneticOperators:
    """Genetic operators for architecture evolution"""
    
    def __init__(self, mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def mutate(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Apply mutations to a genome"""
        mutated = deepcopy(genome)
        
        # Reset fitness scores
        mutated.accuracy = 0.0
        mutated.noise_resilience = 0.0
        mutated.complexity_score = 0.0
        mutated.overfitting_score = 0.0
        mutated.overall_fitness = 0.0
        
        # Mutation types and their probabilities
        mutation_types = [
            self._mutate_layer_parameters,
            self._mutate_architecture_structure,
            self._mutate_training_parameters,
            self._mutate_regularization
        ]
        
        # Apply mutations based on mutation rate
        for mutation_func in mutation_types:
            if random.random() < self.mutation_rate:
                mutated = mutation_func(mutated)
        
        return mutated
    
    def _mutate_layer_parameters(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Mutate individual layer parameters"""
        if not genome.layers:
            return genome
        
        # Select random layer to mutate
        layer_idx = random.randint(0, len(genome.layers) - 1)
        layer = genome.layers[layer_idx]
        
        # Mutation operations
        mutations = [
            lambda: self._mutate_channels(layer),
            lambda: self._mutate_kernel_size(layer),
            lambda: self._mutate_activation(layer),
            lambda: self._mutate_dropout(layer),
            lambda: self._mutate_batch_norm(layer)
        ]
        
        # Apply random mutation
        mutation = random.choice(mutations)
        mutation()
        
        return genome
    
    def _mutate_channels(self, layer: LayerConfig):
        """Mutate channel dimensions"""
        if layer.layer_type in ['conv1d', 'conv2d', 'fc']:
            # Mutate out_channels
            current = layer.out_channels
            if random.random() < 0.5:
                # Increase
                layer.out_channels = min(current * 2, 512)
            else:
                # Decrease
                layer.out_channels = max(current // 2, 16)
        
        elif layer.layer_type in ['lstm', 'gru', 'cfc', 'ncp']:
            # Mutate hidden size
            current = layer.hidden_size or layer.out_channels
            if random.random() < 0.5:
                layer.hidden_size = min(current * 2, 512)
            else:
                layer.hidden_size = max(current // 2, 16)
            
            # For NCP/CfC, ensure output size is less than units-2
            if layer.layer_type in ['cfc', 'ncp']:
                layer.out_channels = max(2, layer.hidden_size - 3)
            else:
                layer.out_channels = layer.hidden_size
    
    def _mutate_kernel_size(self, layer: LayerConfig):
        """Mutate kernel size for convolutional layers"""
        if layer.layer_type == 'conv1d':
            current = layer.kernel_size[0] if layer.kernel_size else 3
            new_size = random.choice([3, 5, 7, 9, 11])
            layer.kernel_size = (new_size,)
            layer.padding = (new_size // 2,)
        
        elif layer.layer_type == 'conv2d':
            current = layer.kernel_size
            new_sizes = [(1, 3), (1, 5), (1, 7), (3, 3), (5, 5)]
            layer.kernel_size = random.choice(new_sizes)
            layer.padding = (0, layer.kernel_size[1] // 2)
    
    def _mutate_activation(self, layer: LayerConfig):
        """Mutate activation function"""
        activations = ['relu', 'elu', 'leaky_relu', 'tanh', 'sigmoid']
        layer.activation = random.choice(activations)
    
    def _mutate_dropout(self, layer: LayerConfig):
        """Mutate dropout rate"""
        if layer.layer_type != 'dropout':
            layer.dropout_rate = random.uniform(0.0, 0.5)
        else:
            layer.dropout_rate = random.uniform(0.1, 0.7)
    
    def _mutate_batch_norm(self, layer: LayerConfig):
        """Toggle batch normalization"""
        if layer.layer_type in ['conv1d', 'conv2d']:
            layer.batch_norm = not layer.batch_norm
    
    def _mutate_architecture_structure(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Mutate the overall architecture structure"""
        operations = [
            lambda: self._add_layer(genome),
            lambda: self._remove_layer(genome),
            lambda: self._swap_layers(genome),
            lambda: self._change_layer_type(genome)
        ]
        
        operation = random.choice(operations)
        operation()
        
        return genome
    
    def _add_layer(self, genome: ArchitectureGenome):
        """Add a new layer to the architecture"""
        if len(genome.layers) >= 10:  # Max layers
            return
        
        # Don't add layer before the final classification layer
        insert_idx = random.randint(0, len(genome.layers) - 2)
        
        # Generate a new layer based on the current architecture
        prev_layer = genome.layers[insert_idx]
        next_layer = genome.layers[insert_idx + 1]
        
        # Create intermediate layer
        new_layer = self._create_intermediate_layer(prev_layer, next_layer)
        
        genome.layers.insert(insert_idx + 1, new_layer)
    
    def _create_intermediate_layer(self, prev_layer: LayerConfig, next_layer: LayerConfig) -> LayerConfig:
        """Create an intermediate layer between two existing layers"""
        in_channels = prev_layer.out_channels
        
        # Choose layer type that makes sense
        if prev_layer.layer_type in ['conv1d', 'conv2d']:
            layer_type = random.choice(['conv1d', 'conv2d', 'dropout', 'bn'])
        elif prev_layer.layer_type in ['lstm', 'gru', 'cfc', 'ncp']:
            layer_type = random.choice(['fc', 'dropout', 'bn'])
        else:
            layer_type = random.choice(['fc', 'dropout', 'bn'])
        
        if layer_type == 'conv1d':
            return LayerConfig(
                layer_type='conv1d',
                in_channels=in_channels,
                out_channels=random.choice([16, 32, 64, 128]),
                kernel_size=(random.choice([3, 5, 7]),),
                stride=(1,),
                padding=(3 // 2,),
                activation=random.choice(['relu', 'elu']),
                batch_norm=random.choice([True, False])
            )
        elif layer_type == 'conv2d':
            return LayerConfig(
                layer_type='conv2d',
                in_channels=in_channels,
                out_channels=random.choice([16, 32, 64, 128]),
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
                activation=random.choice(['relu', 'elu']),
                batch_norm=random.choice([True, False])
            )
        elif layer_type == 'dropout':
            return LayerConfig(
                layer_type='dropout',
                in_channels=in_channels,
                out_channels=in_channels,
                dropout_rate=random.uniform(0.1, 0.3),
                activation='linear'
            )
        elif layer_type == 'bn':
            return LayerConfig(
                layer_type='batch_norm',
                in_channels=in_channels,
                out_channels=in_channels,
                activation='linear'
            )
        else:  # fc
            return LayerConfig(
                layer_type='fc',
                in_channels=in_channels,
                out_channels=random.choice([64, 128, 256]),
                activation=random.choice(['relu', 'elu']),
                dropout_rate=random.uniform(0.0, 0.3)
            )
    
    def _remove_layer(self, genome: ArchitectureGenome):
        """Remove a layer from the architecture"""
        if len(genome.layers) <= 3:  # Keep minimum layers
            return
        
        # Don't remove the final classification layer
        remove_idx = random.randint(0, len(genome.layers) - 2)
        genome.layers.pop(remove_idx)
    
    def _swap_layers(self, genome: ArchitectureGenome):
        """Swap two layers in the architecture"""
        if len(genome.layers) < 3:
            return
        
        # Don't swap the final classification layer
        idx1 = random.randint(0, len(genome.layers) - 2)
        idx2 = random.randint(0, len(genome.layers) - 2)
        
        if idx1 != idx2:
            genome.layers[idx1], genome.layers[idx2] = genome.layers[idx2], genome.layers[idx1]
    
    def _change_layer_type(self, genome: ArchitectureGenome):
        """Change the type of a layer"""
        if len(genome.layers) < 2:
            return
        
        # Don't change the final classification layer
        layer_idx = random.randint(0, len(genome.layers) - 2)
        layer = genome.layers[layer_idx]
        
        # Change to compatible layer type
        if layer.layer_type in ['conv1d', 'conv2d']:
            new_type = random.choice(['conv1d', 'conv2d'])
        elif layer.layer_type in ['lstm', 'gru']:
            new_type = random.choice(['lstm', 'gru', 'cfc', 'ncp'])
        else:
            new_type = random.choice(['fc', 'dropout', 'bn'])
        
        layer.layer_type = new_type
        
        # For NCP/CfC, ensure proper sizing
        if new_type in ['cfc', 'ncp']:
            units = random.choice([16, 32, 64, 128])
            layer.hidden_size = units
            layer.out_channels = max(2, units - 3)
    
    def _mutate_training_parameters(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Mutate training hyperparameters"""
        if random.random() < 0.5:
            genome.learning_rate *= random.uniform(0.5, 2.0)
            genome.learning_rate = max(0.0001, min(0.01, genome.learning_rate))
        
        if random.random() < 0.5:
            genome.weight_decay *= random.uniform(0.5, 2.0)
            genome.weight_decay = max(0.0001, min(0.01, genome.weight_decay))
        
        if random.random() < 0.3:
            genome.batch_size = random.choice([32, 64, 128])
        
        return genome
    
    def _mutate_regularization(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Mutate regularization parameters"""
        if random.random() < 0.5:
            genome.dropout_rate = random.uniform(0.1, 0.4)
        
        return genome
    
    def crossover(self, parent1: ArchitectureGenome, parent2: ArchitectureGenome) -> Tuple[ArchitectureGenome, ArchitectureGenome]:
        """Perform crossover between two parent genomes"""
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        
        # Create offspring
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)
        
        # Reset fitness scores
        for child in [child1, child2]:
            child.accuracy = 0.0
            child.noise_resilience = 0.0
            child.complexity_score = 0.0
            child.overfitting_score = 0.0
            child.overall_fitness = 0.0
        
        # Crossover operations
        crossover_types = [
            self._crossover_layers,
            self._crossover_training_params,
            self._crossover_regularization
        ]
        
        for crossover_func in crossover_types:
            if random.random() < 0.5:
                crossover_func(child1, child2)
        
        return child1, child2
    
    def _crossover_layers(self, child1: ArchitectureGenome, child2: ArchitectureGenome):
        """Crossover layer configurations"""
        if len(child1.layers) < 2 or len(child2.layers) < 2:
            return
        
        # Single-point crossover for layers
        min_len = min(len(child1.layers), len(child2.layers))
        crossover_point = random.randint(1, min_len - 1)
        
        # Swap layer configurations
        child1.layers[crossover_point:], child2.layers[crossover_point:] = \
            child2.layers[crossover_point:], child1.layers[crossover_point:]
    
    def _crossover_training_params(self, child1: ArchitectureGenome, child2: ArchitectureGenome):
        """Crossover training parameters"""
        # Swap learning rate
        if random.random() < 0.5:
            child1.learning_rate, child2.learning_rate = child2.learning_rate, child1.learning_rate
        
        # Swap weight decay
        if random.random() < 0.5:
            child1.weight_decay, child2.weight_decay = child2.weight_decay, child1.weight_decay
        
        # Swap batch size
        if random.random() < 0.5:
            child1.batch_size, child2.batch_size = child2.batch_size, child1.batch_size
    
    def _crossover_regularization(self, child1: ArchitectureGenome, child2: ArchitectureGenome):
        """Crossover regularization parameters"""
        if random.random() < 0.5:
            child1.dropout_rate, child2.dropout_rate = child2.dropout_rate, child1.dropout_rate


class TournamentSelection:
    """Tournament selection for genetic algorithm"""
    
    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size
    
    def select(self, population: List[ArchitectureGenome]) -> ArchitectureGenome:
        """Select an individual using tournament selection"""
        if not population:
            raise ValueError("Population cannot be empty")
        
        # Randomly select tournament participants
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        
        # Return the best individual from the tournament
        return max(tournament, key=lambda x: x.overall_fitness)
    
    def select_parents(self, population: List[ArchitectureGenome], num_parents: int = 2) -> List[ArchitectureGenome]:
        """Select multiple parents using tournament selection"""
        return [self.select(population) for _ in range(num_parents)] 