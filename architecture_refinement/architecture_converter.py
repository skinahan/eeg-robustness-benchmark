"""
Architecture conversion module for WiredCfC compatibility.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
import json
from pathlib import Path
import torch
import torch.nn as nn
from .config import ArchitectureConfig

@dataclass
class WiredCfCArchitecture:
    """WiredCfC architecture specification."""
    input_size: int
    hidden_size: int
    output_size: int
    wiring_matrix: np.ndarray
    neuron_types: List[str]
    layer_sizes: List[int]
    connection_weights: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

class WiredCfCConverter:
    """
    Converts optimized graph architectures to WiredCfC-compatible formats.
    
    This class handles the conversion from arbitrary NetworkX graphs (like WS-flex)
    to the specific wiring patterns and architectures required by WiredCfC models.
    """
    
    def __init__(self, config: ArchitectureConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the architecture converter.
        
        Args:
            config: Architecture configuration
            logger: Optional logger for output
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir = Path("outputs/architectures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_graph_to_wiredcfc(
        self, 
        graph: nx.Graph, 
        input_size: Optional[int] = None,
        output_size: Optional[int] = None
    ) -> WiredCfCArchitecture:
        """
        Convert an arbitrary NetworkX graph to WiredCfC architecture.
        
        Args:
            graph: NetworkX graph to convert (e.g., WS-flex graph)
            input_size: Number of input features
            output_size: Number of output classes
            
        Returns:
            WiredCfC architecture specification
        """
        self.logger.info(f"Converting WS-flex graph with {graph.number_of_nodes()} nodes to WiredCfC architecture")
        
        # Determine sizes
        input_size = input_size or self.config.input_size
        output_size = output_size or self.config.output_size
        
        # Analyze graph structure and create proper layer mapping
        node_analysis = self._analyze_ws_flex_structure(graph, input_size, output_size)
        
        # Create wiring matrix with proper layer structure
        wiring_matrix = self._create_wiring_matrix_from_ws_flex(graph, node_analysis)
        
        # Determine neuron types and layer structure
        neuron_types, layer_sizes = self._determine_ws_flex_architecture_structure(node_analysis)
        
        # Create connection weights
        connection_weights = self._extract_connection_weights_from_ws_flex(graph, wiring_matrix, node_analysis)
        
        # Create architecture specification
        architecture = WiredCfCArchitecture(
            input_size=input_size,
            hidden_size=node_analysis['hidden_size'],
            output_size=output_size,
            wiring_matrix=wiring_matrix,
            neuron_types=neuron_types,
            layer_sizes=layer_sizes,
            connection_weights=connection_weights,
            metadata={
                'graph_nodes': graph.number_of_nodes(),
                'graph_edges': graph.number_of_edges(),
                'density': nx.density(graph),
                'is_connected': nx.is_connected(graph),
                'num_components': nx.number_connected_components(graph),
                'ws_flex_parameters': node_analysis.get('ws_flex_parameters', {}),
                'original_graph_stats': {
                    'nodes': graph.number_of_nodes(),
                    'edges': graph.number_of_edges(),
                    'density': nx.density(graph),
                    'clustering': nx.average_clustering(graph) if graph.number_of_nodes() > 1 else 0.0,
                    'path_length': nx.average_shortest_path_length(graph) if nx.is_connected(graph) else float('inf')
                },
                'architecture_stats': {
                    'total_neurons': input_size + node_analysis['hidden_size'] + output_size,
                    'total_connections': int(np.sum(wiring_matrix > 0)),
                    'connection_density': float(np.sum(wiring_matrix > 0) / (wiring_matrix.shape[0] * wiring_matrix.shape[1])),
                    'connectivity_enforcement_added': self._count_enforcement_connections(wiring_matrix, input_size, node_analysis['hidden_size'], output_size)
                }
            }
        )
        
        self.logger.info(f"Successfully converted WS-flex graph to WiredCfC architecture")
        return architecture
    
    def _analyze_ws_flex_structure(
        self, 
        graph: nx.Graph, 
        input_size: int, 
        output_size: int
    ) -> Dict[str, Any]:
        """
        Analyze the structure of a WS-flex graph to determine architecture parameters.
        
        For WS-flex graphs, we need to:
        1. Ensure we have exactly input_size sensory neurons
        2. Ensure we have exactly output_size motor neurons  
        3. All other neurons become inter neurons
        4. Maintain the graph's connectivity structure
        """
        # Get all nodes from the graph
        all_nodes = sorted(list(graph.nodes()))
        total_nodes = len(all_nodes)
        
        # Calculate required hidden size
        required_total = input_size + output_size
        if total_nodes < required_total:
            self.logger.warning(f"Graph has {total_nodes} nodes but needs at least {required_total} for input+output")
            # We'll need to handle this case
        
        # Determine how to map the graph nodes to our architecture
        # Strategy: Use the first 'input_size' nodes as sensory, last 'output_size' as motor, rest as inter
        
        # Map nodes to layers
        sensory_nodes = all_nodes[:input_size]  # First nodes become sensory
        motor_nodes = all_nodes[-output_size:]  # Last nodes become motor
        inter_nodes = all_nodes[input_size:-output_size] if len(all_nodes) > input_size + output_size else []
        
        # Ensure we have enough nodes
        if len(inter_nodes) < 8:  # Minimum hidden size
            # If we don't have enough inter nodes, we need to adjust
            if total_nodes >= input_size + output_size + 8:
                # We can fit the minimum
                inter_nodes = all_nodes[input_size:input_size + 8]
                motor_nodes = all_nodes[input_size + 8:input_size + 8 + output_size]
            else:
                # We need to create a minimal architecture
                self.logger.warning(f"Insufficient nodes for proper architecture, creating minimal structure")
                inter_nodes = all_nodes[input_size:min(input_size + 8, total_nodes - output_size)]
        
        # Validate that we have the minimum required structure
        if len(sensory_nodes) < input_size:
            self.logger.error(f"Cannot create architecture: need {input_size} sensory nodes but only have {len(sensory_nodes)} total nodes")
            raise ValueError(f"Insufficient nodes for input size {input_size}")
        
        if len(motor_nodes) < output_size:
            self.logger.error(f"Cannot create architecture: need {output_size} motor nodes but only have {len(motor_nodes)} nodes available")
            raise ValueError(f"Insufficient nodes for output size {output_size}")
        
        if len(inter_nodes) < 1:  # Need at least 1 inter neuron
            self.logger.error(f"Cannot create architecture: need at least 1 inter neuron but have none")
            raise ValueError("No inter neurons available")
        
        hidden_size = len(inter_nodes)
        
        # Log the node mapping for debugging
        self.logger.info(f"WS-flex graph node mapping:")
        self.logger.info(f"  Total nodes: {total_nodes}")
        self.logger.info(f"  Sensory nodes: {sensory_nodes} (count: {len(sensory_nodes)})")
        self.logger.info(f"  Inter nodes: {inter_nodes} (count: {len(inter_nodes)})")
        self.logger.info(f"  Motor nodes: {motor_nodes} (count: {len(motor_nodes)})")
        self.logger.info(f"  Architecture: {input_size} -> {hidden_size} -> {output_size}")
        
        # Analyze connectivity patterns
        connectivity_analysis = {
            'sensory_connections': 0,
            'inter_connections': 0,
            'motor_connections': 0,
            'sensory_to_inter': 0,
            'inter_to_motor': 0,
            'inter_to_inter': 0
        }
        
        # Count connections by type
        for edge in graph.edges():
            u, v = edge
            if u in sensory_nodes and v in inter_nodes:
                connectivity_analysis['sensory_to_inter'] += 1
            elif u in inter_nodes and v in motor_nodes:
                connectivity_analysis['inter_to_motor'] += 1
            elif u in inter_nodes and v in inter_nodes:
                connectivity_analysis['inter_to_inter'] += 1
            elif u in sensory_nodes and v in motor_nodes:
                connectivity_analysis['sensory_to_inter'] += 1  # Count as sensory->inter
            elif u in motor_nodes and v in sensory_nodes:
                connectivity_analysis['inter_to_motor'] += 1   # Count as inter->motor
        
        return {
            'sensory_nodes': sensory_nodes,
            'inter_nodes': inter_nodes,
            'motor_nodes': motor_nodes,
            'input_size': input_size,
            'output_size': output_size,
            'hidden_size': hidden_size,
            'total_nodes': total_nodes,
            'connectivity_analysis': connectivity_analysis,
            'all_nodes': all_nodes
        }
    
    def _create_wiring_matrix_from_ws_flex(
        self, 
        graph: nx.Graph, 
        node_analysis: Dict[str, Any]
    ) -> np.ndarray:
        """
        Create the wiring matrix for WiredCfC from a WS-flex graph.
        
        The wiring matrix should have the structure:
        [Sensory x Sensory] [Sensory x Inter] [Sensory x Motor]
        [Inter x Sensory]   [Inter x Inter]   [Inter x Motor]  
        [Motor x Sensory]   [Motor x Inter]   [Motor x Motor]
        """
        input_size = node_analysis['input_size']
        hidden_size = node_analysis['hidden_size']
        output_size = node_analysis['output_size']
        total_size = input_size + hidden_size + output_size
        
        # Initialize wiring matrix
        wiring_matrix = np.zeros((total_size, total_size), dtype=np.float32)
        
        # Create node mapping from graph nodes to matrix indices
        node_mapping = {}
        current_idx = 0
        
        # Map sensory nodes (first layer)
        for node in node_analysis['sensory_nodes']:
            node_mapping[node] = current_idx
            current_idx += 1
        
        # Map inter nodes (hidden layer)
        for node in node_analysis['inter_nodes']:
            node_mapping[node] = current_idx
            current_idx += 1
        
        # Map motor nodes (output layer)
        for node in node_analysis['motor_nodes']:
            node_mapping[node] = current_idx
            current_idx += 1
        
        # Fill wiring matrix based on graph edges
        original_connections = 0
        mapped_connections = 0
        
        for edge in graph.edges():
            u, v = edge
            weight = graph[u][v].get('weight', 1.0)
            original_connections += 1
            
            # Get mapped indices
            if u in node_mapping and v in node_mapping:
                u_idx = node_mapping[u]
                v_idx = node_mapping[v]
                
                # Set connection in wiring matrix (directed)
                wiring_matrix[u_idx, v_idx] = weight
                # For undirected graphs, also set the reverse
                if not graph.is_directed():
                    wiring_matrix[v_idx, u_idx] = weight
                
                mapped_connections += 1
            else:
                if self.logger:
                    self.logger.debug(f"Edge {u}->{v} not mapped (nodes not in architecture)")
        
        if self.logger:
            self.logger.info(f"Edge mapping: {mapped_connections}/{original_connections} edges mapped to wiring matrix")
            if mapped_connections < original_connections:
                self.logger.warning(f"Lost {original_connections - mapped_connections} edges during mapping")
        
        # Ensure proper layer connectivity
        self._enforce_layer_connectivity(wiring_matrix, input_size, hidden_size, output_size)
        
        # Log wiring matrix statistics
        total_connections = np.sum(wiring_matrix > 0)
        self.logger.info(f"Wiring matrix created:")
        self.logger.info(f"  Matrix shape: {wiring_matrix.shape}")
        self.logger.info(f"  Total connections: {total_connections}")
        self.logger.info(f"  Connection density: {total_connections / (total_size * total_size):.4f}")
        
        # Show detailed connectivity breakdown
        self._log_connectivity_breakdown(wiring_matrix, input_size, hidden_size, output_size)
        
        # Validate wiring matrix
        if total_connections == 0:
            self.logger.warning("Warning: Wiring matrix has no connections!")
        
        return wiring_matrix
    
    def _enforce_layer_connectivity(
        self, 
        wiring_matrix: np.ndarray, 
        input_size: int, 
        hidden_size: int, 
        output_size: int
    ):
        """
        Ensure minimal layer connectivity for WiredCfC compatibility.
        
        We want to ensure:
        - Sensory neurons can reach inter neurons (at least one connection)
        - Inter neurons can reach motor neurons (at least one connection)
        - But we add as few connections as possible to preserve the optimized topology
        
        Strategy: Only add connections if there are NO connections between layers
        """
        # Check if sensory neurons can reach inter neurons
        sensory_to_inter = np.sum(wiring_matrix[:input_size, input_size:input_size + hidden_size])
        if sensory_to_inter == 0:
            self.logger.warning("No connections from sensory to inter neurons - adding minimal connectivity")
            # Add just one connection from first sensory to first inter
            wiring_matrix[0, input_size] = 0.1
            self.logger.info("Added minimal sensory->inter connection")
        
        # Check if inter neurons can reach motor neurons
        inter_to_motor = np.sum(wiring_matrix[input_size:input_size + hidden_size, input_size + hidden_size:])
        if inter_to_motor == 0:
            self.logger.warning("No connections from inter to motor neurons - adding minimal connectivity")
            # Add just one connection from first inter to first motor
            wiring_matrix[input_size, input_size + hidden_size] = 0.1
            self.logger.info("Added minimal inter->motor connection")
        
        # Check if inter neurons have any internal connectivity (for recurrent dynamics)
        inter_to_inter = np.sum(wiring_matrix[input_size:input_size + hidden_size, input_size:input_size + hidden_size])
        if inter_to_inter == 0:
            self.logger.warning("No inter-to-inter connections - adding minimal recurrent connectivity")
            # Add just one self-connection to the first inter neuron
            wiring_matrix[input_size, input_size] = 0.1
            self.logger.info("Added minimal inter->inter connection")
        
        # Log connectivity enforcement results
        sensory_to_inter_final = np.sum(wiring_matrix[:input_size, input_size:input_size + hidden_size])
        inter_to_motor_final = np.sum(wiring_matrix[input_size:input_size + hidden_size, input_size + hidden_size:])
        inter_to_inter_final = np.sum(wiring_matrix[input_size:input_size + hidden_size, input_size:input_size + hidden_size])
        
        self.logger.info(f"Layer connectivity check completed:")
        self.logger.info(f"  Sensory->Inter: {sensory_to_inter_final:.1f} connections")
        self.logger.info(f"  Inter->Motor: {inter_to_motor_final:.1f} connections")
        self.logger.info(f"  Inter->Inter: {inter_to_inter_final:.1f} connections")
        
        # Warn if we had to add connections
        connections_added = 0
        if sensory_to_inter == 0:
            connections_added += 1
        if inter_to_motor == 0:
            connections_added += 1
        if inter_to_inter == 0:
            connections_added += 1
            
        if connections_added > 0:
            self.logger.info(f"Added {connections_added} minimal connections to ensure basic layer connectivity")
        else:
            self.logger.info("No additional connections needed - graph already has proper layer connectivity")
    
    def _log_connectivity_breakdown(
        self, 
        wiring_matrix: np.ndarray, 
        input_size: int, 
        hidden_size: int, 
        output_size: int
    ):
        """Log detailed connectivity breakdown for the wiring matrix."""
        # Sensory layer connections
        sensory_to_sensory = np.sum(wiring_matrix[:input_size, :input_size])
        sensory_to_inter = np.sum(wiring_matrix[:input_size, input_size:input_size + hidden_size])
        sensory_to_motor = np.sum(wiring_matrix[:input_size, input_size + hidden_size:])
        
        # Inter layer connections
        inter_to_sensory = np.sum(wiring_matrix[input_size:input_size + hidden_size, :input_size])
        inter_to_inter = np.sum(wiring_matrix[input_size:input_size + hidden_size, input_size:input_size + hidden_size])
        inter_to_motor = np.sum(wiring_matrix[input_size:input_size + hidden_size, input_size + hidden_size:])
        
        # Motor layer connections
        motor_to_sensory = np.sum(wiring_matrix[input_size + hidden_size:, :input_size])
        motor_to_inter = np.sum(wiring_matrix[input_size + hidden_size:, input_size:input_size + hidden_size])
        motor_to_motor = np.sum(wiring_matrix[input_size + hidden_size:, input_size + hidden_size:])
        
        self.logger.info(f"Connectivity breakdown:")
        self.logger.info(f"  Sensory layer:")
        self.logger.info(f"    -> Sensory: {sensory_to_sensory:.1f}")
        self.logger.info(f"    -> Inter: {sensory_to_inter:.1f}")
        self.logger.info(f"    -> Motor: {sensory_to_motor:.1f}")
        self.logger.info(f"  Inter layer:")
        self.logger.info(f"    -> Sensory: {inter_to_sensory:.1f}")
        self.logger.info(f"    -> Inter: {inter_to_inter:.1f}")
        self.logger.info(f"    -> Motor: {inter_to_motor:.1f}")
        self.logger.info(f"  Motor layer:")
        self.logger.info(f"    -> Sensory: {motor_to_sensory:.1f}")
        self.logger.info(f"    -> Inter: {motor_to_inter:.1f}")
        self.logger.info(f"    -> Motor: {motor_to_motor:.1f}")
        
        # Check for direct sensory->motor connections (which we want to avoid)
        if sensory_to_motor > 0:
            self.logger.warning(f"Found {sensory_to_motor:.1f} direct sensory->motor connections (should go through inter layer)")
        
        # Check for motor->sensory connections (which we want to avoid)
        if motor_to_sensory > 0:
            self.logger.warning(f"Found {motor_to_sensory:.1f} motor->sensory connections (feedback should go through inter layer)")
    
    def _count_enforcement_connections(
        self, 
        wiring_matrix: np.ndarray, 
        input_size: int, 
        hidden_size: int, 
        output_size: int
    ) -> Dict[str, int]:
        """Count connections that were added by connectivity enforcement."""
        # These are the specific connections we add in _enforce_layer_connectivity
        enforcement_connections = {
            'sensory_to_inter': 0,
            'inter_to_motor': 0,
            'inter_to_inter': 0
        }
        
        # Check if we added sensory->inter connection
        if wiring_matrix[0, input_size] == 0.1:  # Our enforcement weight
            enforcement_connections['sensory_to_inter'] = 1
        
        # Check if we added inter->motor connection
        if wiring_matrix[input_size, input_size + hidden_size] == 0.1:  # Our enforcement weight
            enforcement_connections['inter_to_motor'] = 1
        
        # Check if we added inter->inter connection
        if wiring_matrix[input_size, input_size] == 0.1:  # Our enforcement weight
            enforcement_connections['inter_to_inter'] = 1
        
        return enforcement_connections
    
    def _determine_ws_flex_architecture_structure(
        self, 
        node_analysis: Dict[str, Any]
    ) -> Tuple[List[str], List[int]]:
        """Determine the layer structure and neuron types for WiredCfC from WS-flex graph."""
        # Define layer structure
        layer_sizes = [
            node_analysis['input_size'],
            node_analysis['hidden_size'],
            node_analysis['output_size']
        ]
        
        # Define neuron types
        neuron_types = []
        
        # Sensory layer neurons
        neuron_types.extend(['sensory'] * node_analysis['input_size'])
        
        # Hidden layer neurons (inter)
        neuron_types.extend(['inter'] * node_analysis['hidden_size'])
        
        # Output layer neurons (motor)
        neuron_types.extend(['motor'] * node_analysis['output_size'])
        
        # Validate neuron type assignment
        expected_total = node_analysis['input_size'] + node_analysis['hidden_size'] + node_analysis['output_size']
        if len(neuron_types) != expected_total:
            self.logger.error(f"Neuron type count mismatch: expected {expected_total}, got {len(neuron_types)}")
            raise ValueError(f"Neuron type count mismatch: {len(neuron_types)} != {expected_total}")
        
        self.logger.info(f"Neuron types assigned:")
        self.logger.info(f"  Sensory: {neuron_types.count('sensory')}")
        self.logger.info(f"  Inter: {neuron_types.count('inter')}")
        self.logger.info(f"  Motor: {neuron_types.count('motor')}")
        self.logger.info(f"  Total: {len(neuron_types)}")
        
        return neuron_types, layer_sizes
    
    def _extract_connection_weights_from_ws_flex(
        self, 
        graph: nx.Graph, 
        wiring_matrix: np.ndarray,
        node_analysis: Dict[str, Any]
    ) -> np.ndarray:
        """Extract connection weights from the WS-flex graph."""
        # For now, use the wiring matrix as connection weights
        # In practice, this could be enhanced with learned weights or edge attributes
        return wiring_matrix.copy()
    
    def create_wiredcfc_model(
        self, 
        architecture: WiredCfCArchitecture
    ) -> nn.Module:
        """
        Create a PyTorch WiredCfC model from the architecture specification.
        
        Args:
            architecture: WiredCfC architecture specification
            
        Returns:
            PyTorch module implementing the WiredCfC architecture
        """
        self.logger.info("Creating PyTorch WiredCfC model from WS-flex architecture")
        
        # Create the model
        model = WiredCfCModule(architecture)
        
        self.logger.info("Successfully created WiredCfC model from WS-flex architecture")
        return model
    
    def save_architecture(
        self, 
        architecture: WiredCfCArchitecture, 
        filename: str
    ) -> str:
        """
        Save architecture specification to file.
        
        Args:
            architecture: Architecture to save
            filename: Base filename
            
        Returns:
            Path to saved file
        """
        # Convert numpy arrays to lists for JSON serialization
        save_data = {
            'input_size': architecture.input_size,
            'hidden_size': architecture.hidden_size,
            'output_size': architecture.output_size,
            'wiring_matrix': architecture.wiring_matrix.tolist(),
            'neuron_types': architecture.neuron_types,
            'layer_sizes': architecture.layer_sizes,
            'connection_weights': architecture.connection_weights.tolist() if architecture.connection_weights is not None else None,
            'metadata': architecture.metadata
        }
        
        # Save as JSON
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        self.logger.info(f"WS-flex architecture saved to {json_path}")
        return str(json_path)
    
    def load_architecture(self, filepath: str) -> WiredCfCArchitecture:
        """
        Load architecture specification from file.
        
        Args:
            filepath: Path to architecture file
            
        Returns:
            Loaded architecture specification
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert lists back to numpy arrays
            architecture = WiredCfCArchitecture(
                input_size=data['input_size'],
                hidden_size=data['hidden_size'],
                output_size=data['output_size'],
                wiring_matrix=np.array(data['wiring_matrix']),
                neuron_types=data['neuron_types'],
                layer_sizes=data['layer_sizes'],
                connection_weights=np.array(data['connection_weights']) if data['connection_weights'] else None,
                metadata=data['metadata']
            )
            
            self.logger.info(f"WS-flex architecture loaded from {filepath}")
            return architecture
            
        except Exception as e:
            self.logger.error(f"Error loading WS-flex architecture: {e}")
            raise
    
    def convert_batch_architectures(
        self, 
        graphs: List[nx.Graph], 
        input_size: Optional[int] = None,
        output_size: Optional[int] = None
    ) -> List[WiredCfCArchitecture]:
        """
        Convert a batch of WS-flex graphs to WiredCfC architectures.
        
        Args:
            graphs: List of NetworkX graphs (WS-flex graphs)
            input_size: Number of input features
            output_size: Number of output classes
            
        Returns:
            List of WiredCfC architectures
        """
        self.logger.info(f"Converting batch of {len(graphs)} WS-flex graphs to WiredCfC architectures")
        
        architectures = []
        for i, graph in enumerate(graphs):
            try:
                architecture = self.convert_graph_to_wiredcfc(graph, input_size, output_size)
                architectures.append(architecture)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Converted {i + 1}/{len(graphs)} WS-flex graphs")
                    
            except Exception as e:
                self.logger.error(f"Error converting WS-flex graph {i}: {e}")
                # Create a default architecture for failed conversions
                default_arch = self._create_default_ws_flex_architecture(input_size, output_size)
                architectures.append(default_arch)
        
        self.logger.info(f"Successfully converted {len(architectures)} WS-flex graphs to architectures")
        return architectures
    
    def _create_default_ws_flex_architecture(
        self, 
        input_size: Optional[int] = None, 
        output_size: Optional[int] = None
    ) -> WiredCfCArchitecture:
        """Create a default WS-flex architecture when conversion fails."""
        input_size = input_size or self.config.input_size
        output_size = output_size or self.config.output_size
        hidden_size = max(32, (input_size + output_size) // 2)
        
        # Create simple fully connected architecture
        total_size = input_size + hidden_size + output_size
        wiring_matrix = np.zeros((total_size, total_size), dtype=np.float32)
        
        # Connect input to hidden
        for i in range(input_size):
            for j in range(input_size, input_size + hidden_size):
                wiring_matrix[i, j] = 1.0
        
        # Connect hidden to output
        for i in range(input_size, input_size + hidden_size):
            for j in range(input_size + hidden_size, total_size):
                wiring_matrix[i, j] = 1.0
        
        # Connect hidden to hidden
        for i in range(input_size, input_size + hidden_size):
            for j in range(input_size, input_size + hidden_size):
                if i != j:
                    wiring_matrix[i, j] = 0.5
        
        neuron_types = ['sensory'] * input_size + ['inter'] * hidden_size + ['motor'] * output_size
        layer_sizes = [input_size, hidden_size, output_size]
        
        return WiredCfCArchitecture(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            wiring_matrix=wiring_matrix,
            neuron_types=neuron_types,
            layer_sizes=layer_sizes,
            connection_weights=wiring_matrix.copy(),
            metadata={'is_default': True, 'ws_flex_fallback': True}
        )
    
    def validate_architecture(self, architecture: WiredCfCArchitecture) -> List[str]:
        """
        Validate a WiredCfC architecture specification from WS-flex graph.
        
        Args:
            architecture: Architecture to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check dimensions
        expected_size = architecture.input_size + architecture.hidden_size + architecture.output_size
        if architecture.wiring_matrix.shape != (expected_size, expected_size):
            errors.append(f"Wiring matrix shape mismatch: expected ({expected_size}, {expected_size}), got {architecture.wiring_matrix.shape}")
        
        # Check neuron types
        expected_types = ['sensory'] * architecture.input_size + ['inter'] * architecture.hidden_size + ['motor'] * architecture.output_size
        if architecture.neuron_types != expected_types:
            errors.append(f"Neuron types mismatch: expected {expected_types}, got {architecture.neuron_types}")
        
        # Check layer sizes
        expected_layers = [architecture.input_size, architecture.hidden_size, architecture.output_size]
        if architecture.layer_sizes != expected_layers:
            errors.append(f"Layer sizes mismatch: expected {expected_layers}, got {architecture.layer_sizes}")
        
        # Check for isolated neurons (but be more lenient for WS-flex graphs)
        row_sums = np.sum(architecture.wiring_matrix, axis=1)
        col_sums = np.sum(architecture.wiring_matrix, axis=0)
        total_connections = row_sums + col_sums
        
        isolated_neurons = np.where(total_connections == 0)[0]
        if len(isolated_neurons) > 0:
            # Check if isolated neurons are only in the input layer (which is acceptable)
            isolated_in_input = [n for n in isolated_neurons if n < architecture.input_size]
            isolated_not_in_input = [n for n in isolated_neurons if n >= architecture.input_size]
            
            if len(isolated_not_in_input) > 0:
                errors.append(f"Found {len(isolated_not_in_input)} isolated neurons outside input layer: {isolated_not_in_input}")
            else:
                self.logger.info(f"Found {len(isolated_in_input)} isolated input neurons (this is acceptable)")
        
        # Check layer connectivity (but be more lenient)
        input_size = architecture.input_size
        hidden_size = architecture.hidden_size
        output_size = architecture.output_size
        
        # Check sensory to inter connections
        sensory_to_inter = np.sum(architecture.wiring_matrix[:input_size, input_size:input_size + hidden_size])
        if sensory_to_inter == 0:
            errors.append("No connections from sensory to inter neurons")
        elif sensory_to_inter < 0.1:  # Very weak connections
            self.logger.warning(f"Very weak sensory->inter connectivity: {sensory_to_inter:.3f}")
        
        # Check inter to motor connections
        inter_to_motor = np.sum(architecture.wiring_matrix[input_size:input_size + hidden_size, input_size + hidden_size:])
        if inter_to_motor == 0:
            errors.append("No connections from inter to motor neurons")
        elif inter_to_motor < 0.1:  # Very weak connections
            self.logger.warning(f"Very weak inter->motor connectivity: {inter_to_motor:.3f}")
        
        # Check inter-to-inter connectivity
        inter_to_inter = np.sum(architecture.wiring_matrix[input_size:input_size + hidden_size, input_size:input_size + hidden_size])
        if inter_to_inter == 0:
            self.logger.warning("No inter-to-inter connections (recurrent dynamics may be limited)")
        else:
            self.logger.info(f"Inter-to-inter connectivity: {inter_to_inter:.3f}")
        
        # Check overall connectivity density
        total_connections = np.sum(architecture.wiring_matrix > 0)
        total_possible = architecture.wiring_matrix.shape[0] * architecture.wiring_matrix.shape[1]
        density = total_connections / total_possible
        
        if density < 0.01:  # Very sparse
            self.logger.warning(f"Very sparse connectivity: {density:.4f} (may limit information flow)")
        elif density > 0.8:  # Very dense
            self.logger.warning(f"Very dense connectivity: {density:.4f} (may not preserve optimized topology)")
        else:
            self.logger.info(f"Connectivity density: {density:.4f} (within reasonable range)")
        
        return errors
    
    def get_architecture_summary(self, architecture: WiredCfCArchitecture) -> Dict[str, Any]:
        """
        Get a summary of the WS-flex architecture properties.
        
        Args:
            architecture: Architecture to summarize
            
        Returns:
            Dictionary containing architecture summary
        """
        # Count connections by type
        wiring_matrix = architecture.wiring_matrix
        
        # Input connections
        input_connections = np.sum(wiring_matrix[:architecture.input_size, :])
        
        # Hidden connections
        hidden_start = architecture.input_size
        hidden_end = architecture.input_size + architecture.hidden_size
        hidden_connections = np.sum(wiring_matrix[hidden_start:hidden_end, :])
        
        # Output connections
        output_start = architecture.input_size + architecture.hidden_size
        output_connections = np.sum(wiring_matrix[output_start:, :])
        
        # Total connections
        total_connections = np.sum(wiring_matrix > 0)
        
        # Connection density
        total_possible = wiring_matrix.shape[0] * wiring_matrix.shape[1]
        connection_density = total_connections / total_possible
        
        return {
            'input_size': architecture.input_size,
            'hidden_size': architecture.hidden_size,
            'output_size': architecture.output_size,
            'total_neurons': wiring_matrix.shape[0],
            'total_connections': int(total_connections),
            'connection_density': float(connection_density),
            'input_connections': int(input_connections),
            'hidden_connections': int(hidden_connections),
            'output_connections': int(output_connections),
            'layer_sizes': architecture.layer_sizes,
            'neuron_types': architecture.neuron_types,
            'metadata': architecture.metadata,
            'connectivity_enforcement': architecture.metadata.get('architecture_stats', {}).get('connectivity_enforcement_added', {}),
            'original_graph_stats': architecture.metadata.get('original_graph_stats', {}),
            'architecture_stats': architecture.metadata.get('architecture_stats', {})
        }


class WiredCfCModule(nn.Module):
    """
    PyTorch module implementing a WiredCfC architecture from WS-flex graphs.
    
    This is a simplified implementation that demonstrates the concept.
    In practice, you would integrate with the actual WiredCfC library.
    """
    
    def __init__(self, architecture: WiredCfCArchitecture):
        """
        Initialize the WiredCfC module.
        
        Args:
            architecture: WiredCfC architecture specification from WS-flex graph
        """
        super().__init__()
        
        self.architecture = architecture
        self.input_size = architecture.input_size
        self.hidden_size = architecture.hidden_size
        self.output_size = architecture.output_size
        
        # Create layers
        self.input_layer = nn.Linear(architecture.input_size, architecture.hidden_size)
        self.hidden_layer = nn.Linear(architecture.hidden_size, architecture.hidden_size)
        self.output_layer = nn.Linear(architecture.hidden_size, architecture.output_size)
        
        # Apply wiring constraints
        self._apply_wiring_constraints()
        
        # Activation functions
        self.activation = nn.ReLU()
        self.output_activation = nn.Softmax(dim=-1)
    
    def _apply_wiring_constraints(self):
        """Apply the wiring matrix constraints to the layers."""
        # This is a simplified implementation
        # In practice, you would use the wiring matrix to constrain connections
        
        # For now, we'll just use the standard fully connected layers
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Input to hidden
        x = self.input_layer(x)
        x = self.activation(x)
        
        # Hidden to hidden (with residual connection)
        hidden_out = self.hidden_layer(x)
        x = self.activation(x + hidden_out)
        
        # Hidden to output
        x = self.output_layer(x)
        x = self.output_activation(x)
        
        return x
    
    def get_wiring_info(self) -> Dict[str, Any]:
        """Get information about the wiring structure."""
        return {
            'wiring_matrix': self.architecture.wiring_matrix,
            'neuron_types': self.architecture.neuron_types,
            'layer_sizes': self.architecture.layer_sizes,
            'connection_weights': self.architecture.connection_weights
        }
