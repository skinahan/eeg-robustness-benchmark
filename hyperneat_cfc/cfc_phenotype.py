"""
CfC Phenotype for HyperNEAT Evolution

This module implements the phenotype development process for HyperNEAT,
converting a CPPN genome into a functional CfC neural network.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional
from ncps.torch import CfC
from ncps.wirings import AutoNCP, Wiring
from ncps.torch.wired_cfc_cell import WiredCfCCell

from .cfc_substrate import CfCSubstrate, CfCCell
from .hyperneat_genome import HyperNEATGenome


class HyperNEATWiring(Wiring):
    """
    Custom wiring class that uses HyperNEAT evolved connection patterns.
    
    This wiring class creates a sparse connection matrix based on the
    CPPN-evolved connections from HyperNEAT, rather than using random
    or predefined patterns.
    """
    
    def __init__(self, connections: Dict[Tuple[int, int], float], substrate: CfCSubstrate):
        # Get cell information from substrate
        input_cells = substrate.get_input_cells()
        hidden_cells = substrate.get_hidden_cells()
        output_cells = substrate.get_output_cells()
        
        # Call parent constructor with required units parameter
        super().__init__(units=len(hidden_cells))
        
        self.connections = connections
        self.substrate = substrate
        
        self.input_size = len(input_cells)
        self.hidden_size = len(hidden_cells)
        self.output_size = len(output_cells)
        self.units = len(hidden_cells)
        self.input_dim = len(input_cells)
        self.output_dim = len(output_cells)
        
        # Create cell ID to cell object mapping
        self.cell_mapping = {cell.cell_id: cell for cell in substrate.cells}
        
        # Build the adjacency matrices
        self._build_adjacency_matrices()
    
    def _build_adjacency_matrices(self):
        """Build adjacency matrices from evolved connections"""
        # Initialize adjacency matrices
        self.adjacency_matrix = np.zeros((self.hidden_size, self.hidden_size))
        self.sensory_adjacency_matrix = np.zeros((self.input_size, self.hidden_size))
        
        # Process evolved connections
        for (from_cell_id, to_cell_id), weight in self.connections.items():
            from_cell_obj = self.cell_mapping.get(from_cell_id)
            to_cell_obj = self.cell_mapping.get(to_cell_id)
            
            if from_cell_obj and to_cell_obj:
                if from_cell_obj.cell_type == 'input' and to_cell_obj.cell_type == 'hidden':
                    # Input to hidden connection
                    channel_idx = from_cell_obj.features.get('channel_idx', 0)
                    neuron_idx = to_cell_obj.features.get('neuron_idx', 0)
                    self.sensory_adjacency_matrix[channel_idx, neuron_idx] = weight
                elif from_cell_obj.cell_type == 'hidden' and to_cell_obj.cell_type == 'hidden':
                    # Hidden to hidden connection
                    from_neuron_idx = from_cell_obj.features.get('neuron_idx', 0)
                    to_neuron_idx = to_cell_obj.features.get('neuron_idx', 0)
                    self.adjacency_matrix[from_neuron_idx, to_neuron_idx] = weight
                elif from_cell_obj.cell_type == 'hidden' and to_cell_obj.cell_type == 'output':
                    # Hidden to output connection (handled by proj_size in CfC)
                    pass
    
    def get_neurons_of_layer(self, layer_idx: int):
        """Get neurons in a specific layer"""
        if layer_idx == 0:
            return list(range(self.hidden_size))
        return []
    
    def build(self, input_size: int):
        """Build the wiring with the given input size"""
        self.input_dim = input_size
        # Adjacency matrices are already built in __init__
    
    def is_built(self) -> bool:
        """Check if wiring is properly built"""
        return True


class CfCPhenotype:
    """
    Converts HyperNEAT genomes into CfC networks.
    
    This class takes a HyperNEAT genome (CPPN) and a substrate, then
    generates connection patterns to create a CfC network using the ncps framework.
    """
    
    def __init__(self, substrate: CfCSubstrate, connection_threshold: float = 0.1):
        self.substrate = substrate
        self.connection_threshold = connection_threshold
    
    def develop(self, genome: HyperNEATGenome) -> 'HyperNEATCfC':
        """
        Develop a genome into a CfC network.
        
        Args:
            genome: HyperNEAT genome encoding the CPPN
            
        Returns:
            HyperNEATCfC: The developed CfC network
        """
        # Generate connection matrix using CPPN
        connections = self._generate_connections(genome)
        
        # Create custom wiring using evolved connections
        wiring = self._create_hyperneat_wiring(connections)
        
        # Create the CfC network
        return HyperNEATCfC(
            input_size=self.substrate.input_size,
            wiring=wiring,
            substrate=self.substrate,
            connections=connections
        )
    
    def _generate_connections(self, genome: HyperNEATGenome) -> Dict[Tuple[int, int], float]:
        """
        Generate connection weights using the CPPN.
        
        Args:
            genome: HyperNEAT genome
            
        Returns:
            Dictionary mapping (from_cell, to_cell) -> weight
        """
        connections = {}
        cell_coords = self.substrate.get_cell_coordinates()
        
        # Get all cells
        all_cells = self.substrate.cells
        
        # Generate connections between all pairs of cells
        for from_cell in all_cells:
            for to_cell in all_cells:
                if from_cell.cell_id == to_cell.cell_id:
                    continue  # Skip self-connections
                
                # Create CPPN inputs: [x1, y1, x2, y2]
                cppn_inputs = [
                    from_cell.x, from_cell.y,
                    to_cell.x, to_cell.y
                ]
                
                # Evaluate CPPN to get connection weight
                weight = genome.evaluate_cppn(cppn_inputs)
                
                # Apply threshold to create sparse connections
                if abs(weight) > self.connection_threshold:
                    connections[(from_cell.cell_id, to_cell.cell_id)] = weight
        
        return connections
    
    def _create_hyperneat_wiring(self, connections: Dict[Tuple[int, int], float]) -> HyperNEATWiring:
        """
        Create a custom Wiring object using evolved HyperNEAT connections.
        
        Args:
            connections: Dictionary of (from_cell, to_cell) -> weight
            
        Returns:
            HyperNEATWiring: Custom wiring using evolved connections
        """
        # Create custom wiring with evolved connections
        wiring = HyperNEATWiring(
            connections=connections,
            substrate=self.substrate
        )
        
        return wiring
    
    def _create_wiring(self, connections: Dict[Tuple[int, int], float]) -> AutoNCP:
        """
        Create a Wiring object for ncps using AutoNCP (legacy method).
        
        Args:
            connections: Dictionary of (from_cell, to_cell) -> weight
            
        Returns:
            AutoNCP: ncps AutoNCP wiring object
        """
        # Get cell information
        input_cells = self.substrate.get_input_cells()
        hidden_cells = self.substrate.get_hidden_cells()
        output_cells = self.substrate.get_output_cells()
        
        # Calculate sparsity based on actual connections
        total_possible_connections = len(input_cells) * len(hidden_cells) + len(hidden_cells) * len(hidden_cells) + len(hidden_cells) * len(output_cells)
        actual_connections = len(connections)
        sparsity = 1.0 - (actual_connections / total_possible_connections) if total_possible_connections > 0 else 0.8
        
        # Use AutoNCP which is proven to work with ncps
        wiring = AutoNCP(
            units=len(hidden_cells),
            output_size=len(output_cells),
            sparsity_level=max(0.1, min(0.9, sparsity))  # Ensure reasonable sparsity
        )
        wiring.input_dim = len(input_cells)
        wiring.output_dim = len(output_cells)  # Set output_dim properly
        
        return wiring


class HyperNEATCfC(nn.Module):
    """
    CfC network developed from a HyperNEAT genome.
    
    This class implements a CfC network using the ncps framework with
    connection patterns generated by HyperNEAT.
    """
    
    def __init__(
        self,
        input_size: int,
        wiring: Wiring,
        substrate: CfCSubstrate,
        connections: Optional[Dict[Tuple[int, int], float]] = None,
        batch_first: bool = True,
        return_sequences: bool = False,
        use_evolved_architecture: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.wiring = wiring
        self.substrate = substrate
        self.connections = connections
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.use_evolved_architecture = use_evolved_architecture
        
        if use_evolved_architecture and isinstance(wiring, HyperNEATWiring):
            # Use WiredCfCCell with evolved architecture
            self.rnn_cell = WiredCfCCell(
                input_size=input_size,
                wiring=wiring,
                mode="default"
            )
            self.state_size = wiring.units
            self.output_size = wiring.output_dim
        else:
            # Fallback to standard CfC (current implementation)
            self.cfc = CfC(
                input_size=input_size,
                units=wiring.units,
                proj_size=wiring.output_dim,
                return_sequences=return_sequences,
                batch_first=batch_first,
                mixed_memory=True,
                mode='default'
            )
        
    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the CfC network.
        
        Args:
            x: Input tensor [batch_size, channels, time_steps] (EEG format)
            hx: Initial hidden state [batch_size, hidden_size] or None
            
        Returns:
            output: Output tensor [batch_size, output_size] or [batch_size, seq_len, output_size]
            hidden: Final hidden state [batch_size, hidden_size]
        """
        # Transform input from EEG format (batch_size, channels, time_steps) 
        # to CfC format (batch_size, time_steps, features)
        if x.dim() == 3 and x.size(1) == self.input_size:
            # Input is (batch_size, channels, time_steps) - transform to (batch_size, time_steps, channels)
            x = x.transpose(1, 2)
        
        if self.use_evolved_architecture and isinstance(self.wiring, HyperNEATWiring):
            # Use WiredCfCCell forward pass - following base CfC implementation pattern
            device = x.device
            batch_size, seq_len = x.size(0), x.size(1)
            
            # Initialize hidden state
            if hx is None:
                h_state = torch.zeros((batch_size, self.state_size), device=device)
            else:
                h_state = hx
            
            output_sequence = []
            for t in range(seq_len):
                inputs = x[:, t]  # [batch_size, features]
                ts = 1.0  # Default timespan
                
                # Forward pass through WiredCfCCell
                h_out, h_state = self.rnn_cell.forward(inputs, h_state, ts)
                
                if self.return_sequences:
                    output_sequence.append(h_out)
            
            if self.return_sequences:
                output = torch.stack(output_sequence, dim=1)  # [batch_size, seq_len, output_size]
            else:
                output = h_out  # [batch_size, output_size]
            
            hidden = h_state
        else:
            # Standard CfC forward pass
            output, hidden = self.cfc(x, hx)
        
        return output, hidden
    
    def get_parameter_count(self) -> int:
        """Get the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def visualize_connections(self, save_path: Optional[str] = None):
        """
        Visualize the connection patterns of the network.
        
        Args:
            save_path: Optional path to save the visualization
        """
        import matplotlib.pyplot as plt
        
        if isinstance(self.wiring, HyperNEATWiring):
            # Create connection matrix from evolved connections
            input_cells = self.substrate.get_input_cells()
            hidden_cells = self.substrate.get_hidden_cells()
            output_cells = self.substrate.get_output_cells()
            
            # Create full adjacency matrix
            total_size = len(input_cells) + len(hidden_cells) + len(output_cells)
            adjacency_matrix = np.zeros((total_size, total_size))
            
            # Fill in connections
            for (from_cell, to_cell), weight in self.connections.items():
                adjacency_matrix[from_cell, to_cell] = weight
            
            title = 'CfC Network Connections (HyperNEAT Evolved)'
        else:
            # Get adjacency matrix from wiring
            adjacency_matrix = self.wiring.adjacency_matrix
            title = 'CfC Network Connections (AutoNCP)'
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Plot adjacency matrix
        im = ax.imshow(adjacency_matrix, cmap='RdBu', aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Neuron Index')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Connection visualization saved to: {save_path}")
        
        plt.show() 