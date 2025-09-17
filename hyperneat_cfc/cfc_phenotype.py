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
    
    def __init__(self, substrate: CfCSubstrate, connection_threshold: float = 0.15, batch_first: bool = True):
        self.substrate = substrate
        self.connection_threshold = connection_threshold
        self.batch_first = batch_first
    
    def develop(self, genome: HyperNEATGenome, n_chans: int = 22, n_times: int = 1000) -> 'HyperNEATCfC':
        """
        Develop a genome into a CfC network.
        
        Args:
            genome: HyperNEAT genome encoding the CPPN
            n_chans: Number of EEG channels (default: 22 for BNCI2014_001)
            n_times: Number of time points (default: 1000)
            
        Returns:
            HyperNEATCfC: The developed CfC network
        """
        # Validate proj_size against substrate output size
        if genome.proj_size is not None and genome.proj_size != self.substrate.output_size:
            print(f"Warning: Genome proj_size ({genome.proj_size}) doesn't match substrate output_size ({self.substrate.output_size}). "
                  f"Setting proj_size to {self.substrate.output_size}.")
            genome.proj_size = self.substrate.output_size
        
        # Generate connection matrix using CPPN
        connections = self._generate_connections(genome)
        
        # Create custom wiring using evolved connections
        wiring = self._create_hyperneat_wiring(connections)
        
        # Create the CfC network with convolutional head
        return HyperNEATCfC(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=self.substrate.output_size,
            input_size=self.substrate.input_size,
            wiring=wiring,
            substrate=self.substrate,
            connections=connections,
            proj_size=genome.proj_size,
            mode=genome.mode,
            mixed_memory=genome.mixed_memory,
            return_sequences=genome.return_sequences,
            batch_first=self.batch_first
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
        
        # Get all cells
        all_cells = self.substrate.cells
        
        # Generate connections between all pairs of cells
        for from_cell in all_cells:
            for to_cell in all_cells:
                if from_cell.cell_id == to_cell.cell_id:
                    continue  # Skip self-connections
                
                # Create CPPN inputs based on substrate dimensions
                # For 2D substrate: [x1, y1, x2, y2]
                # For 1D substrate: [x1, x2]
                # For 3D substrate: [x1, y1, z1, x2, y2, z2]
                if hasattr(from_cell, 'z') and hasattr(to_cell, 'z'):
                    # 3D substrate
                    cppn_inputs = [from_cell.x, from_cell.y, from_cell.z, to_cell.x, to_cell.y, to_cell.z]
                elif hasattr(from_cell, 'y') and hasattr(to_cell, 'y'):
                    # 2D substrate
                    cppn_inputs = [from_cell.x, from_cell.y, to_cell.x, to_cell.y]
                else:
                    # 1D substrate
                    cppn_inputs = [from_cell.x, to_cell.x]
                
                # Ensure CPPN has correct number of inputs
                if len(cppn_inputs) != genome.input_nodes:
                    raise ValueError(f"CPPN expects {genome.input_nodes} inputs but substrate provides {len(cppn_inputs)}")
                
                # Evaluate CPPN to get connection weight
                weight = genome.evaluate_cppn(cppn_inputs)
                
                # Apply threshold to create sparse connections
                if abs(weight) > self.connection_threshold:
                    connections[(from_cell.cell_id, to_cell.cell_id)] = weight
        
        # Fallback: if no connections generated, create very sparse basic connections
        if len(connections) == 0:
            print(f"Warning: No connections generated with threshold {self.connection_threshold}. Creating sparse fallback connections.")
            # Create very sparse basic connections
            input_cells = self.substrate.get_input_cells()
            hidden_cells = self.substrate.get_hidden_cells()
            
            # Add only a few input-to-hidden connections (very sparse)
            num_input_connections = min(2, len(input_cells), len(hidden_cells))
            for i in range(num_input_connections):
                connections[(input_cells[i].cell_id, hidden_cells[i].cell_id)] = 0.5
            
            # Add only 1-2 hidden-to-hidden connections (very sparse)
            if len(hidden_cells) >= 2:
                connections[(hidden_cells[0].cell_id, hidden_cells[1].cell_id)] = 0.3
                if len(hidden_cells) >= 3:
                    connections[(hidden_cells[1].cell_id, hidden_cells[2].cell_id)] = 0.3
        
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
    CfC network developed from a HyperNEAT genome with fixed convolutional head.
    
    This class implements a CfC network using the ncps framework with
    connection patterns generated by HyperNEAT, incorporating the proven
    convolutional head architecture from CNNCfCv2.
    """
    
    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        input_size: int,
        wiring: Wiring,
        substrate: CfCSubstrate,
        connections: Optional[Dict[Tuple[int, int], float]] = None,
        proj_size: Optional[int] = None,
        mode: str = "default",
        mixed_memory: bool = False,
        return_sequences: bool = False,
        batch_first: bool = True,
        use_evolved_architecture: bool = True,
        # CNNCfCv2 parameters
        drop_prob: float = 0.15,
        F1: int = 8,
        D: int = 2,
        kernel_length: int = 128,
        temporal_kernel_size: int = 3,
        temporal_stride: int = 4,
        max_seq_length: int = 250
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.n_outputs = n_outputs
        self.input_size = input_size
        self.wiring = wiring
        self.substrate = substrate
        self.connections = connections
        self.proj_size = proj_size
        self.mode = mode
        self.mixed_memory = mixed_memory
        self.return_sequences = return_sequences
        self.batch_first = batch_first
        self.use_evolved_architecture = use_evolved_architecture
        self.max_seq_length = max_seq_length
        self.temporal_stride = temporal_stride
        
        # CNNCfCv2 convolutional head parameters
        batch_norm_momentum = 0.01
        batch_norm_eps = 1e-3
        self.F1 = F1
        F2 = F1 * D
        self.kernel_length = kernel_length
        
        # 1. Fixed Convolutional Head (from CNNCfCv2)
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=self.F1,
            kernel_size=(1, self.kernel_length),
            stride=(1, 1),
            padding=(0, self.kernel_length // 2),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.F1, momentum=batch_norm_momentum, eps=batch_norm_eps)
        self.elu = nn.ELU()

        # 2. Depthwise Conv2D
        self.depthwise_conv = nn.Conv2d(
            in_channels=self.F1, out_channels=F2,
            kernel_size=(n_chans, 1),
            groups=self.F1,
            stride=(1, 1), 
            padding=(0, 0), 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F2, momentum=batch_norm_momentum, eps=batch_norm_eps)

        # 3. Average Pooling
        self.avgpool = nn.AvgPool2d(kernel_size=(1, F2), stride=(1, self.F1))

        # 4. Dropout
        self.dropout = nn.Dropout(p=drop_prob)

        # 5. Temporal downsampler
        self.temporal_downsampler = nn.Conv1d(
            in_channels=F2,
            out_channels=F2,
            kernel_size=temporal_kernel_size,
            stride=temporal_stride,
            padding=temporal_kernel_size // 2
        )
        
        # 6. HyperNEAT-evolved CfC component
        ncp_input_size = F2  # Output from convolutional head
        ncp_output_size = F2  # Keep consistent
        
        if use_evolved_architecture and isinstance(wiring, HyperNEATWiring):
            # Use WiredCfCCell with evolved architecture
            self.rnn_cell = WiredCfCCell(
                input_size=ncp_input_size,
                wiring=wiring,
                mode=mode
            )
            self.state_size = wiring.units
            self.cfc_output_size = wiring.output_dim
            
            # Handle mixed memory
            self.use_mixed = mixed_memory
            if self.use_mixed:
                from ncps.torch.lstm import LSTMCell
                self.lstm = LSTMCell(ncp_input_size, self.state_size)
        else:
            # Fallback to standard CfC
            self.cfc = CfC(
                input_size=ncp_input_size,
                units=wiring.units,
                proj_size=ncp_output_size,
                return_sequences=return_sequences,
                batch_first=batch_first,
                mixed_memory=mixed_memory,
                mode=mode
            )
            self.use_mixed = mixed_memory

        # 7. Separable Conv2D (from CNNCfCv2)
        self.sep_depthwise = nn.Conv2d(
            in_channels=ncp_output_size, 
            out_channels=ncp_output_size, 
            kernel_size=(3, 1),
            stride=(1, 1), 
            padding=(1, 0), 
            groups=ncp_output_size, 
            bias=False
        )
        self.sep_pointwise = nn.Conv2d(
            in_channels=ncp_output_size, 
            out_channels=F2,
            kernel_size=(1, 1), 
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2, momentum=batch_norm_momentum, eps=batch_norm_eps)

        # 8. Final layers
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 9. Classification layer
        self.fc = nn.Linear(F2, n_outputs)
        
        # Initialize weights
        self._glorot_weight_zero_bias()
        
    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the CfC network with convolutional head.
        
        Args:
            x: Input tensor [batch_size, channels, time_steps] (EEG format)
            hx: Initial hidden state [batch_size, hidden_size] or None
            
        Returns:
            output: Output tensor [batch_size, n_outputs]
            hidden: Final hidden state [batch_size, hidden_size]
        """
        # 1. Convolutional Head Processing (similar to CNNCfCv2)
        # Input: [batch_size, channels, time_steps] -> [batch_size, 1, channels, time_steps]
        x = x.unsqueeze(1)
        
        # Input Conv2D
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)

        # Depthwise Conv2D
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu(x)

        # Average pooling for temporal reduction
        x = self.avgpool(x)
        x = self.dropout(x)
        
        # 2. Reshape for temporal processing
        # x shape: [B, F2, 1, T'] -> [B, T', F2]
        x = x.permute(0, 3, 2, 1)  # [B, T', 1, F2]
        num_features = x.shape[3]  # This will be F2
        x = x.contiguous().view(x.shape[0], x.shape[1], num_features)  # [B, T', F2]
        
        # 3. Apply temporal downsampling
        x = self.temporal_downsampler(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T'', F2]
        
        # 4. Limit sequence length for CfC
        if x.shape[1] > self.max_seq_length:
            # Take the middle portion to maintain temporal context
            start_idx = (x.shape[1] - self.max_seq_length) // 2
            x = x[:, start_idx:start_idx + self.max_seq_length, :]
        
        # 5. HyperNEAT-evolved CfC processing
        if self.use_evolved_architecture and isinstance(self.wiring, HyperNEATWiring):
            # Use WiredCfCCell forward pass
            device = x.device
            batch_size, seq_len = x.size(0), x.size(1)
            
            # Initialize hidden state
            if hx is None:
                h_state = torch.zeros((batch_size, self.state_size), device=device)
                c_state = torch.zeros((batch_size, self.state_size), device=device) if self.use_mixed else None
            else:
                if self.use_mixed and isinstance(hx, torch.Tensor):
                    raise RuntimeError("Running a CfC with mixed_memory=True, requires a tuple (h0,c0) to be passed as state")
                h_state, c_state = hx if self.use_mixed else (hx, None)
            
            output_sequence = []
            for t in range(seq_len):
                inputs = x[:, t]  # [batch_size, features]
                ts = 1.0  # Default timespan
                
                # Handle mixed memory if enabled
                if self.use_mixed:
                    h_state, c_state = self.lstm(inputs, (h_state, c_state))
                
                # Forward pass through WiredCfCCell
                h_out, h_state = self.rnn_cell.forward(inputs, h_state, ts)
                
                if self.return_sequences:
                    output_sequence.append(h_out)
            
            if self.return_sequences:
                x = torch.stack(output_sequence, dim=1)  # [batch_size, seq_len, output_size]
            else:
                x = h_out  # [batch_size, output_size]
            
            hidden = (h_state, c_state) if self.use_mixed else h_state
        else:
            # Standard CfC forward pass
            x, hidden = self.cfc(x, hx)
        
        # 6. Reshape for separable conv (take last timestep if sequences)
        if self.return_sequences or len(x.shape) == 3:
            # Take mean across sequence or last timestep
            x = x.mean(dim=1) if self.return_sequences else x[:, -1, :]
        
        # Reshape to [B, F2, 1, 1] for separable conv
        x = x.unsqueeze(-1).unsqueeze(-1).permute(0, 1, 2, 3)  # [B, F2, 1, 1]
        
        # 7. Separable Conv2D processing
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.dropout2(x)

        # 8. Global pooling and classification
        x = self.global_pool(x)  # [B, F2, 1, 1]
        x = x.view(x.shape[0], -1)  # [B, F2]
        output = self.fc(x)  # [B, n_outputs]
        
        return output, hidden
    
    def _glorot_weight_zero_bias(self):
        """Initialize weights using Glorot (Xavier) initialization."""
        for module in self.modules():
            if hasattr(module, "weight"):
                if "BatchNorm" not in module.__class__.__name__:
                    nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    nn.init.constant_(module.weight, 1)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
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