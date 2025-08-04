"""
HyperNEAT Genome for CfC Evolution

This module defines the genome representation for HyperNEAT evolution,
which encodes a CPPN (Compositional Pattern Producing Network) that
generates connection patterns for CfC networks.
"""

import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import json


@dataclass
class CPPNNode:
    """Represents a node in the CPPN"""
    node_id: int
    node_type: str  # 'input', 'hidden', 'output'
    activation: str = 'tanh'  # 'tanh', 'sigmoid', 'relu', 'sin', 'cos', 'abs'
    bias: float = 0.0
    
    def __post_init__(self):
        """Validate node after initialization"""
        valid_types = ['input', 'hidden', 'output']
        if self.node_type not in valid_types:
            raise ValueError(f"Invalid node type: {self.node_type}")
        
        valid_activations = ['tanh', 'sigmoid', 'relu', 'sin', 'cos', 'abs', 'linear']
        if self.activation not in valid_activations:
            raise ValueError(f"Invalid activation: {self.activation}")


@dataclass
class CPPNConnection:
    """Represents a connection in the CPPN"""
    from_node: int
    to_node: int
    weight: float
    enabled: bool = True
    
    def __post_init__(self):
        """Validate connection after initialization"""
        if self.from_node == self.to_node:
            raise ValueError("Self-connections not allowed")


class HyperNEATGenome:
    """
    Genome representation for HyperNEAT evolution.
    
    This genome encodes a CPPN that generates connection patterns
    for CfC networks based on geometric relationships between cells.
    """
    
    def __init__(
        self,
        input_nodes: int = 4,  # x1, y1, x2, y2 coordinates
        hidden_nodes: int = 8,
        output_nodes: int = 1,  # connection weight
        max_connections: int = 20,
        weight_range: Tuple[float, float] = (-3.0, 3.0),
        bias_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.max_connections = max_connections
        self.weight_range = weight_range
        self.bias_range = bias_range
        
        # Genome components
        self.nodes: List[CPPNNode] = []
        self.connections: List[CPPNConnection] = []
        
        # Fitness and metadata
        self.fitness: float = 0.0
        self.generation: int = 0
        self.innovation_number: int = 0
        
        # Initialize genome
        self._initialize_genome()
    
    def _initialize_genome(self):
        """Initialize the genome with basic structure"""
        # Create input nodes (x1, y1, x2, y2)
        for i in range(self.input_nodes):
            self.nodes.append(CPPNNode(
                node_id=i,
                node_type='input',
                activation='linear',
                bias=0.0
            ))
        
        # Create hidden nodes
        for i in range(self.hidden_nodes):
            self.nodes.append(CPPNNode(
                node_id=self.input_nodes + i,
                node_type='hidden',
                activation=random.choice(['tanh', 'sigmoid', 'relu', 'sin', 'cos']),
                bias=random.uniform(*self.bias_range)
            ))
        
        # Create output node (connection weight)
        for i in range(self.output_nodes):
            self.nodes.append(CPPNNode(
                node_id=self.input_nodes + self.hidden_nodes + i,
                node_type='output',
                activation='tanh',  # Output weight should be bounded
                bias=0.0
            ))
        
        # Create initial connections
        self._create_initial_connections()
    
    def _create_initial_connections(self):
        """Create initial connections for the CPPN"""
        # Connect inputs to some hidden nodes
        for i in range(min(self.input_nodes, self.hidden_nodes)):
            self.connections.append(CPPNConnection(
                from_node=i,
                to_node=self.input_nodes + i,
                weight=random.uniform(*self.weight_range),
                enabled=True
            ))
        
        # Connect some hidden nodes to output
        for i in range(min(self.hidden_nodes, self.output_nodes)):
            self.connections.append(CPPNConnection(
                from_node=self.input_nodes + i,
                to_node=self.input_nodes + self.hidden_nodes + i,
                weight=random.uniform(*self.weight_range),
                enabled=True
            ))
        
        # Add some random hidden-to-hidden connections
        num_hidden_connections = min(3, self.hidden_nodes // 2)
        for _ in range(num_hidden_connections):
            from_node = random.randint(self.input_nodes, self.input_nodes + self.hidden_nodes - 1)
            to_node = random.randint(self.input_nodes, self.input_nodes + self.hidden_nodes - 1)
            if from_node != to_node:
                self.connections.append(CPPNConnection(
                    from_node=from_node,
                    to_node=to_node,
                    weight=random.uniform(*self.weight_range),
                    enabled=True
                ))
    
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate the genome"""
        # Mutate connection weights
        for connection in self.connections:
            if random.random() < mutation_rate:
                connection.weight += random.gauss(0, 0.1)
                connection.weight = np.clip(connection.weight, *self.weight_range)
        
        # Mutate node biases
        for node in self.nodes:
            if node.node_type != 'input' and random.random() < mutation_rate:
                node.bias += random.gauss(0, 0.1)
                node.bias = np.clip(node.bias, *self.bias_range)
        
        # Mutate node activations
        for node in self.nodes:
            if node.node_type == 'hidden' and random.random() < mutation_rate * 0.5:
                node.activation = random.choice(['tanh', 'sigmoid', 'relu', 'sin', 'cos'])
        
        # Add new connections
        if random.random() < mutation_rate * 0.3:
            self._add_random_connection()
        
        # Remove connections
        if random.random() < mutation_rate * 0.2:
            self._remove_random_connection()
    
    def _add_random_connection(self):
        """Add a random connection to the genome"""
        if len(self.connections) >= self.max_connections:
            return
        
        # Find valid connection
        attempts = 0
        while attempts < 10:
            from_node = random.choice(self.nodes).node_id
            to_node = random.choice(self.nodes).node_id
            
            # Check if connection already exists
            existing = any(c.from_node == from_node and c.to_node == to_node 
                          for c in self.connections)
            
            if not existing and from_node != to_node:
                self.connections.append(CPPNConnection(
                    from_node=from_node,
                    to_node=to_node,
                    weight=random.uniform(*self.weight_range),
                    enabled=True
                ))
                break
            attempts += 1
    
    def _remove_random_connection(self):
        """Remove a random connection from the genome"""
        if len(self.connections) > 1:
            connection = random.choice(self.connections)
            self.connections.remove(connection)
    
    def crossover(self, other: 'HyperNEATGenome') -> 'HyperNEATGenome':
        """Perform crossover with another genome"""
        child = HyperNEATGenome(
            input_nodes=self.input_nodes,
            hidden_nodes=self.hidden_nodes,
            output_nodes=self.output_nodes,
            max_connections=self.max_connections,
            weight_range=self.weight_range,
            bias_range=self.bias_range
        )
        
        # Clear child's genome
        child.nodes = []
        child.connections = []
        
        # Inherit nodes from both parents
        for node in self.nodes:
            if random.random() < 0.5:
                child.nodes.append(CPPNNode(
                    node_id=node.node_id,
                    node_type=node.node_type,
                    activation=node.activation,
                    bias=node.bias
                ))
            else:
                # Find corresponding node in other parent
                other_node = next((n for n in other.nodes if n.node_id == node.node_id), None)
                if other_node:
                    child.nodes.append(CPPNNode(
                        node_id=node.node_id,
                        node_type=node.node_type,
                        activation=other_node.activation,
                        bias=other_node.bias
                    ))
                else:
                    child.nodes.append(CPPNNode(
                        node_id=node.node_id,
                        node_type=node.node_type,
                        activation=node.activation,
                        bias=node.bias
                    ))
        
        # Inherit connections
        for connection in self.connections:
            if random.random() < 0.5:
                child.connections.append(CPPNConnection(
                    from_node=connection.from_node,
                    to_node=connection.to_node,
                    weight=connection.weight,
                    enabled=connection.enabled
                ))
            else:
                # Find corresponding connection in other parent
                other_connection = next((c for c in other.connections 
                                       if c.from_node == connection.from_node and c.to_node == connection.to_node), None)
                if other_connection:
                    child.connections.append(CPPNConnection(
                        from_node=connection.from_node,
                        to_node=connection.to_node,
                        weight=other_connection.weight,
                        enabled=other_connection.enabled
                    ))
                else:
                    child.connections.append(CPPNConnection(
                        from_node=connection.from_node,
                        to_node=connection.to_node,
                        weight=connection.weight,
                        enabled=connection.enabled
                    ))
        
        return child
    
    def evaluate_cppn(self, inputs: List[float]) -> float:
        """
        Evaluate the CPPN with given inputs.
        
        Args:
            inputs: List of input values [x1, y1, x2, y2]
        
        Returns:
            Connection weight between the two points
        """
        if len(inputs) != self.input_nodes:
            raise ValueError(f"Expected {self.input_nodes} inputs, got {len(inputs)}")
        
        # Initialize node values
        node_values = {}
        for node in self.nodes:
            if node.node_type == 'input':
                node_values[node.node_id] = inputs[node.node_id]
            else:
                node_values[node.node_id] = 0.0
        
        # Forward pass through CPPN
        for node in self.nodes:
            if node.node_type == 'input':
                continue
            
            # Collect input to this node
            input_sum = node.bias
            for connection in self.connections:
                if connection.to_node == node.node_id and connection.enabled:
                    input_sum += node_values[connection.from_node] * connection.weight
            
            # Apply activation function
            if node.activation == 'tanh':
                node_values[node.node_id] = np.tanh(input_sum)
            elif node.activation == 'sigmoid':
                node_values[node.node_id] = 1.0 / (1.0 + np.exp(-input_sum))
            elif node.activation == 'relu':
                node_values[node.node_id] = max(0, input_sum)
            elif node.activation == 'sin':
                node_values[node.node_id] = np.sin(input_sum)
            elif node.activation == 'cos':
                node_values[node.node_id] = np.cos(input_sum)
            elif node.activation == 'abs':
                node_values[node.node_id] = abs(input_sum)
            elif node.activation == 'linear':
                node_values[node.node_id] = input_sum
        
        # Return output value
        output_node = next(node for node in self.nodes if node.node_type == 'output')
        return node_values[output_node.node_id]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary for serialization"""
        return {
            'nodes': [{'node_id': n.node_id, 'node_type': n.node_type, 
                      'activation': n.activation, 'bias': n.bias} for n in self.nodes],
            'connections': [{'from_node': c.from_node, 'to_node': c.to_node,
                           'weight': c.weight, 'enabled': c.enabled} for c in self.connections],
            'fitness': self.fitness,
            'generation': self.generation,
            'innovation_number': self.innovation_number
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HyperNEATGenome':
        """Create genome from dictionary"""
        genome = cls()
        genome.nodes = [CPPNNode(**node_data) for node_data in data['nodes']]
        genome.connections = [CPPNConnection(**conn_data) for conn_data in data['connections']]
        genome.fitness = data.get('fitness', 0.0)
        genome.generation = data.get('generation', 0)
        genome.innovation_number = data.get('innovation_number', 0)
        return genome
    
    def save(self, filepath: str):
        """Save genome to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'HyperNEATGenome':
        """Load genome from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data) 