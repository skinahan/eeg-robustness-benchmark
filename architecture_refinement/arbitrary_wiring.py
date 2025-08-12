"""
Arbitrary wiring module for NCP compatibility with WS-flex graphs.

This module provides a wiring class that can ingest arbitrary WS-flex graphs
from the architecture search outputs and convert them to NCP-compatible wiring.
"""

import numpy as np
import networkx as nx
from ncps.wirings import Wiring
from typing import Dict, List, Optional, Any, Union
import json
from pathlib import Path
import logging


class ArbitraryWiring(Wiring):
    """
    Wiring class that can ingest arbitrary WS-flex graphs from architecture search.
    
    This class takes a wiring matrix and neuron configuration from the architecture
    search outputs and creates a compatible wiring structure for NCP models.
    """
    
    def __init__(self, 
                 wiring_matrix: np.ndarray,
                 input_size: int,
                 hidden_size: int, 
                 output_size: int,
                 neuron_types: Optional[List[str]] = None,
                 connection_weights: Optional[np.ndarray] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the arbitrary wiring.
        
        Args:
            wiring_matrix: Connection matrix from architecture search
            input_size: Number of input features
            hidden_size: Number of hidden neurons
            output_size: Number of output classes
            neuron_types: List of neuron types ('sensory', 'inter', 'motor')
            connection_weights: Optional connection weights
            metadata: Additional metadata about the architecture
            logger: Optional logger for output
        """
        # Total units is the sum of all layers
        total_units = input_size + hidden_size + output_size
        super().__init__(units=total_units)
        
        # Set the output dimension (number of motor neurons)
        self.set_output_dim(output_size)
        
        # Store architecture parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.total_units = total_units
        
        # Store the wiring matrix and metadata
        self.wiring_matrix = wiring_matrix
        self.connection_weights = connection_weights if connection_weights is not None else wiring_matrix
        self.metadata = metadata or {}
        
        # Set up logger
        self.logger = logger or logging.getLogger(__name__)
        
        # Determine neuron types if not provided
        if neuron_types is None:
            self.neuron_types = self._create_default_neuron_types()
        else:
            self.neuron_types = neuron_types
            
        # Validate the wiring matrix
        self._validate_wiring_matrix()
        
        # Build the wiring structure
        self._build_wiring_from_matrix()
        
        self.logger.info(f"ArbitraryWiring initialized: {input_size}->{hidden_size}->{output_size}")
    
    def _create_default_neuron_types(self) -> List[str]:
        """Create default neuron types based on layer structure."""
        neuron_types = []
        neuron_types.extend(['sensory'] * self.input_size)
        neuron_types.extend(['inter'] * self.hidden_size)
        neuron_types.extend(['motor'] * self.output_size)
        return neuron_types
    
    def _validate_wiring_matrix(self):
        """Validate the wiring matrix dimensions and structure."""
        expected_shape = (self.total_units, self.total_units)
        if self.wiring_matrix.shape != expected_shape:
            raise ValueError(f"Wiring matrix shape mismatch: expected {expected_shape}, got {self.wiring_matrix.shape}")
        
        # Check for any negative weights (which shouldn't exist in our architecture search)
        if np.any(self.wiring_matrix < 0):
            self.logger.warning("Found negative weights in wiring matrix - converting to absolute values")
            self.wiring_matrix = np.abs(self.wiring_matrix)
    
    def _build_wiring_from_matrix(self):
        """Build the wiring structure from the wiring matrix."""
        # Clear any existing synapses
        self._synapses = []
        
        # Get the indices for each layer
        input_start = 0
        input_end = self.input_size
        hidden_start = self.input_size
        hidden_end = self.input_size + self.hidden_size
        output_start = self.input_size + self.hidden_size
        output_end = self.total_units
        
        # Build synapses based on the wiring matrix
        for i in range(self.total_units):
            for j in range(self.total_units):
                weight = self.wiring_matrix[i, j]
                if weight > 0:  # Connection exists
                    # Determine connection type and add synapse
                    self._add_synapse_from_matrix(i, j, weight)
        
        self.logger.info(f"Built wiring with {len(self._synapses)} synapses")
    
    def _add_synapse_from_matrix(self, src_idx: int, dest_idx: int, weight: float):
        """Add a synapse based on the wiring matrix."""
        # Determine the type of connection
        if src_idx < self.input_size:
            src_type = 'sensory'
        elif src_idx < self.input_size + self.hidden_size:
            src_type = 'inter'
        else:
            src_type = 'motor'
            
        if dest_idx < self.input_size:
            dest_type = 'sensory'
        elif dest_idx < self.input_size + self.hidden_size:
            dest_type = 'inter'
        else:
            dest_type = 'motor'
        
        # Add the synapse with appropriate polarity
        # For now, we'll use positive polarity for excitatory connections
        # In practice, you might want to determine this based on the weight sign or other criteria
        polarity = 1 if weight > 0 else -1
        
        # Map the matrix indices to the wiring indices
        # The wiring expects indices relative to the total units
        self.add_synapse(src_idx, dest_idx, polarity)
        
        # Log some key connections for debugging
        if (src_type == 'sensory' and dest_type == 'inter') or \
           (src_type == 'inter' and dest_type == 'motor') or \
           (src_type == 'inter' and dest_type == 'inter'):
            self.logger.debug(f"Added {src_type}->{dest_type} connection: {src_idx}->{dest_idx} (weight: {weight:.3f})")
    
    def build(self, input_shape):
        """Build the wiring with input shape - required by NCPs."""
        super().build(input_shape)
        
        # Connect sensory inputs to the appropriate neurons
        # This maps the input features to the sensory neurons in our wiring
        for src in range(self.input_dim):
            # Connect each input to the corresponding sensory neuron
            if src < self.input_size:
                # Add sensory synapse from input to sensory neuron
                self.add_sensory_synapse(src, src, 1.0)
            else:
                # If we have more inputs than sensory neurons, distribute them
                target_neuron = src % self.input_size
                self.add_sensory_synapse(src, target_neuron, 1.0)
        
        self.logger.info(f"Connected {self.input_dim} inputs to sensory neurons")
    
    @property
    def num_layers(self):
        """Return the number of layers - required by WiredCfCCell."""
        return 3  # Sensory -> Inter -> Motor
    
    def get_neurons_of_layer(self, layer_id):
        """Return neurons for each layer - required by WiredCfCCell."""
        if layer_id == 0:
            return list(range(self.input_size))  # Sensory neurons
        elif layer_id == 1:
            return list(range(self.input_size, self.input_size + self.hidden_size))  # Inter neurons
        elif layer_id == 2:
            return list(range(self.input_size + self.hidden_size, self.total_units))  # Motor neurons (outputs)
        else:
            raise ValueError(f"Unknown layer {layer_id}")
    
    def get_type_of_neuron(self, neuron_id):
        """Return the type of neuron as expected by NCPs."""
        if neuron_id < self.input_size:
            return "sensory"
        elif neuron_id < self.input_size + self.hidden_size:
            return "inter"
        else:
            return "motor"
    
    def get_wiring_summary(self) -> Dict[str, Any]:
        """Get a summary of the wiring structure."""
        # Count connections by type
        input_start = 0
        input_end = self.input_size
        hidden_start = self.input_size
        hidden_end = self.input_size + self.hidden_size
        output_start = self.input_size + self.hidden_size
        output_end = self.total_units
        
        # Sensory layer connections
        sensory_to_sensory = np.sum(self.wiring_matrix[input_start:input_end, input_start:input_end])
        sensory_to_inter = np.sum(self.wiring_matrix[input_start:input_end, hidden_start:hidden_end])
        sensory_to_motor = np.sum(self.wiring_matrix[input_start:input_end, output_start:output_end])
        
        # Inter layer connections
        inter_to_sensory = np.sum(self.wiring_matrix[hidden_start:hidden_end, input_start:input_end])
        inter_to_inter = np.sum(self.wiring_matrix[hidden_start:hidden_end, hidden_start:hidden_end])
        inter_to_motor = np.sum(self.wiring_matrix[hidden_start:hidden_end, output_start:output_end])
        
        # Motor layer connections
        motor_to_sensory = np.sum(self.wiring_matrix[output_start:output_end, input_start:input_end])
        motor_to_inter = np.sum(self.wiring_matrix[output_start:output_end, hidden_start:hidden_end])
        motor_to_motor = np.sum(self.wiring_matrix[output_start:output_end, output_start:output_end])
        
        total_connections = np.sum(self.wiring_matrix > 0)
        total_possible = self.total_units * self.total_units
        connection_density = total_connections / total_possible
        
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'total_units': self.total_units,
            'total_connections': int(total_connections),
            'connection_density': float(connection_density),
            'connectivity_breakdown': {
                'sensory_to_sensory': float(sensory_to_sensory),
                'sensory_to_inter': float(sensory_to_inter),
                'sensory_to_motor': float(sensory_to_motor),
                'inter_to_sensory': float(inter_to_sensory),
                'inter_to_inter': float(inter_to_inter),
                'inter_to_motor': float(inter_to_motor),
                'motor_to_sensory': float(motor_to_sensory),
                'motor_to_inter': float(motor_to_inter),
                'motor_to_motor': float(motor_to_motor)
            },
            'metadata': self.metadata
        }


def load_architecture_from_file(filepath: str, logger: Optional[logging.Logger] = None) -> ArbitraryWiring:
    """
    Load an architecture from a JSON file and create an ArbitraryWiring instance.
    
    Args:
        filepath: Path to the architecture JSON file
        logger: Optional logger for output
        
    Returns:
        ArbitraryWiring instance
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract the required fields
        input_size = data['input_size']
        hidden_size = data['hidden_size']
        output_size = data['output_size']
        wiring_matrix = np.array(data['wiring_matrix'])
        
        # Optional fields
        neuron_types = data.get('neuron_types')
        connection_weights = data.get('connection_weights')
        metadata = data.get('metadata', {})
        
        # Create the wiring instance
        wiring = ArbitraryWiring(
            wiring_matrix=wiring_matrix,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            neuron_types=neuron_types,
            connection_weights=connection_weights,
            metadata=metadata,
            logger=logger
        )
        
        if logger:
            logger.info(f"Successfully loaded architecture from {filepath}")
        
        return wiring
        
    except Exception as e:
        if logger:
            logger.error(f"Error loading architecture from {filepath}: {e}")
        raise


def create_wiring_from_architecture_data(architecture_data: Dict[str, Any], 
                                       logger: Optional[logging.Logger] = None) -> ArbitraryWiring:
    """
    Create an ArbitraryWiring instance from architecture data dictionary.
    
    Args:
        architecture_data: Dictionary containing architecture parameters
        logger: Optional logger for output
        
    Returns:
        ArbitraryWiring instance
    """
    # Extract the required fields
    input_size = architecture_data['input_size']
    hidden_size = architecture_data['hidden_size']
    output_size = architecture_data['output_size']
    wiring_matrix = np.array(architecture_data['wiring_matrix'])
    
    # Optional fields
    neuron_types = architecture_data.get('neuron_types')
    connection_weights = architecture_data.get('connection_weights')
    metadata = architecture_data.get('metadata', {})
    
    # Create the wiring instance
    wiring = ArbitraryWiring(
        wiring_matrix=wiring_matrix,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        neuron_types=neuron_types,
        connection_weights=connection_weights,
        metadata=metadata,
        logger=logger
    )
    
    return wiring
