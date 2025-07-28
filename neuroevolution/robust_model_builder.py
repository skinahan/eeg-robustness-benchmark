import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np
from ncps.torch import LTC, CfC
from ncps.wirings import AutoNCP
from braindecode import EEGClassifier
from braindecode.models.base import EEGModuleMixin
from braindecode.models.modules import Ensure4d
from einops.layers.torch import Rearrange

from .architecture_genome import ArchitectureGenome, LayerConfig


class RobustNeuroevolutionModel(EEGModuleMixin, nn.Module):
    """Robust model built from architecture genome with careful shape handling"""
    
    def __init__(self, genome: ArchitectureGenome):
        super().__init__(
            n_outputs=genome.num_classes,
            n_chans=genome.input_channels,
            n_times=genome.input_times,
            input_window_seconds=None,
            sfreq=None,
        )
        
        self.genome = genome
        self.layers = nn.ModuleList()
        self._build_architecture()
        self._glorot_weight_zero_bias()
    
    def _build_architecture(self):
        """Build the neural network architecture from genome"""
        current_channels = self.genome.input_channels
        
        for layer_config in self.genome.layers:
            layer = self._create_layer(layer_config, current_channels)
            self.layers.append(layer)
            
            # Update dimensions for next layer
            if layer_config.layer_type in ['conv1d', 'conv2d']:
                current_channels = layer_config.out_channels
            elif layer_config.layer_type in ['lstm', 'gru', 'cfc', 'ncp']:
                current_channels = layer_config.out_channels
            elif layer_config.layer_type == 'fc':
                current_channels = layer_config.out_channels
    
    def _create_layer(self, layer_config: LayerConfig, in_channels: int) -> nn.Module:
        """Create a layer based on configuration"""
        if layer_config.layer_type == 'conv1d':
            return self._create_conv1d_layer(layer_config, in_channels)
        elif layer_config.layer_type == 'conv2d':
            return self._create_conv2d_layer(layer_config, in_channels)
        elif layer_config.layer_type == 'lstm':
            return self._create_lstm_layer(layer_config, in_channels)
        elif layer_config.layer_type == 'gru':
            return self._create_gru_layer(layer_config, in_channels)
        elif layer_config.layer_type == 'cfc':
            return self._create_cfc_layer(layer_config, in_channels)
        elif layer_config.layer_type == 'ncp':
            return self._create_ncp_layer(layer_config, in_channels)
        elif layer_config.layer_type == 'fc':
            return self._create_fc_layer(layer_config, in_channels)
        elif layer_config.layer_type == 'temporal_downsample':
            return self._create_temporal_downsample_layer(layer_config, in_channels)
        else:
            raise ValueError(f"Unknown layer type: {layer_config.layer_type}")
    
    def _create_temporal_downsample_layer(self, layer_config: LayerConfig, in_channels: int) -> nn.Module:
        """Create a temporal downsampling layer for speed optimization"""
        kernel_size = layer_config.temporal_kernel_size or 3
        stride = layer_config.temporal_stride or 4
        
        return nn.Conv1d(
            in_channels=in_channels,
            out_channels=layer_config.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2
        )
    
    def _create_conv1d_layer(self, layer_config: LayerConfig, in_channels: int) -> nn.Module:
        """Create a 1D convolutional layer"""
        kernel_size = layer_config.kernel_size[0] if isinstance(layer_config.kernel_size, tuple) else layer_config.kernel_size
        stride = layer_config.stride[0] if layer_config.stride and isinstance(layer_config.stride, tuple) else (layer_config.stride or 1)
        padding = layer_config.padding[0] if layer_config.padding and isinstance(layer_config.padding, tuple) else (layer_config.padding or 0)
        
        conv = nn.Conv1d(in_channels, layer_config.out_channels, kernel_size, stride=stride, padding=padding)
        
        layers = [conv]
        
        if layer_config.batch_norm:
            layers.append(nn.BatchNorm1d(layer_config.out_channels))
        
        if layer_config.activation != 'linear':
            layers.append(self._get_activation(layer_config.activation))
        
        if layer_config.dropout_rate > 0:
            layers.append(nn.Dropout(layer_config.dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _create_conv2d_layer(self, layer_config: LayerConfig, in_channels: int) -> nn.Module:
        """Create a 2D convolutional layer"""
        kernel_size = layer_config.kernel_size
        stride = layer_config.stride or (1, 1)
        padding = layer_config.padding or (0, 0)
        
        conv = nn.Conv2d(in_channels, layer_config.out_channels, kernel_size, stride=stride, padding=padding)
        
        layers = [conv]
        
        if layer_config.batch_norm:
            layers.append(nn.BatchNorm2d(layer_config.out_channels))
        
        if layer_config.activation != 'linear':
            layers.append(self._get_activation(layer_config.activation))
        
        if layer_config.dropout_rate > 0:
            layers.append(nn.Dropout(layer_config.dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _create_lstm_layer(self, layer_config: LayerConfig, in_channels: int) -> nn.Module:
        """Create an LSTM layer"""
        hidden_size = layer_config.hidden_size or layer_config.out_channels
        bidirectional = layer_config.bidirectional
        
        lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Don't wrap in Sequential for recurrent layers to avoid tuple issues
        return lstm
    
    def _create_gru_layer(self, layer_config: LayerConfig, in_channels: int) -> nn.Module:
        """Create a GRU layer"""
        hidden_size = layer_config.hidden_size or layer_config.out_channels
        bidirectional = layer_config.bidirectional
        
        gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Don't wrap in Sequential for recurrent layers to avoid tuple issues
        return gru
    
    def _create_cfc_layer(self, layer_config: LayerConfig, in_channels: int) -> nn.Module:
        """Create a CfC layer with configurable activation and mixed memory"""
        hidden_size = layer_config.hidden_size or layer_config.out_channels
        sparsity = layer_config.sparsity or 0.75
        
        # Ensure output size is less than units-2 for NCP library
        output_size = min(layer_config.out_channels, hidden_size - 2)
        
        wiring = AutoNCP(hidden_size, output_size, sparsity_level=sparsity)
        
        # Get CfC-specific parameters
        cfc_activation = layer_config.cfc_activation or 'tanh'  # Default to tanh
        mixed_memory = layer_config.mixed_memory if layer_config.mixed_memory is not None else False  # Default to False
        
        # Create CfC with specified parameters
        cfc = CfC(in_channels, wiring, activation=cfc_activation, mixed_memory=mixed_memory)
        
        return cfc
    
    def _create_ncp_layer(self, layer_config: LayerConfig, in_channels: int) -> nn.Module:
        """Create an NCP layer"""
        hidden_size = layer_config.hidden_size or layer_config.out_channels
        sparsity = layer_config.sparsity or 0.75
        
        # Ensure output size is less than units-2 for NCP library
        output_size = min(layer_config.out_channels, hidden_size - 2)
        
        wiring = AutoNCP(hidden_size, output_size, sparsity_level=sparsity)
        ncp = LTC(in_channels, wiring)
        
        return ncp
    
    def _create_fc_layer(self, layer_config: LayerConfig, in_channels: int) -> nn.Module:
        """Create a fully connected layer"""
        fc = nn.Linear(in_channels, layer_config.out_channels)
        
        layers = [fc]
        
        if layer_config.activation != 'linear':
            layers.append(self._get_activation(layer_config.activation))
        
        if layer_config.dropout_rate > 0:
            layers.append(nn.Dropout(layer_config.dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name"""
        if activation_name == 'relu':
            return nn.ReLU()
        elif activation_name == 'elu':
            return nn.ELU()
        elif activation_name == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation_name == 'tanh':
            return nn.Tanh()
        elif activation_name == 'sigmoid':
            return nn.Sigmoid()
        else:
            return nn.Identity()
    
    def forward(self, x):
        """Forward pass through the network with robust shape handling and temporal optimization"""
        # Ensure input is 4D for EEG data: (batch, channels, time, 1)
        if x.dim() == 2:
            # If input is (batch, features), reshape to (batch, channels, time, 1)
            # This assumes the input is flattened EEG data
            batch_size = x.size(0)
            # For EEG data, we expect 22 channels and some time dimension
            # Try to reshape to reasonable dimensions
            if x.size(1) == 22:  # Single time point
                x = x.unsqueeze(-1).unsqueeze(-1)  # (batch, 22, 1, 1)
            elif x.size(1) % 22 == 0:  # Multiple time points
                time_steps = x.size(1) // 22
                x = x.view(batch_size, 22, time_steps, 1)
            else:
                # If we can't determine the shape, assume it's (batch, features)
                # and reshape to (batch, 1, features, 1) for Conv2D
                x = x.unsqueeze(1).unsqueeze(-1)
        elif x.dim() == 3:
            # If input is (batch, channels, time), add the last dimension
            x = x.unsqueeze(-1)
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            layer_config = self.genome.layers[i]
            
            try:
                if layer_config.layer_type in ['lstm', 'gru']:
                    # Reshape for recurrent layers: (batch, time, channels)
                    if x.dim() == 4:
                        x = x.squeeze(-1).transpose(1, 2)
                    output, _ = layer(x)
                    # Take the last output
                    x = output[:, -1, :]
                
                elif layer_config.layer_type in ['cfc', 'ncp']:
                    # Reshape for CfC/NCP layers: (batch, time, channels)
                    if x.dim() == 4:
                        x = x.squeeze(-1).transpose(1, 2)
                    
                    # Apply temporal windowing if max_seq_length is specified
                    if layer_config.max_seq_length and x.shape[1] > layer_config.max_seq_length:
                        # Take the middle portion to maintain temporal context
                        start_idx = (x.shape[1] - layer_config.max_seq_length) // 2
                        x = x[:, start_idx:start_idx + layer_config.max_seq_length, :]
                    
                    output = layer(x)
                    
                    # Handle different output types from CfC/NCP layers
                    if isinstance(output, tuple):
                        # If it returns a tuple, take the first element (usually the output)
                        x = output[0]
                    else:
                        x = output
                    
                    # Take the last output if it's a sequence
                    if x.dim() == 3:  # (batch, time, features)
                        x = x[:, -1, :]
                    elif x.dim() == 2:  # (batch, features) - already in correct shape
                        pass
                    else:
                        # If it's not in expected shape, flatten
                        x = x.view(x.size(0), -1)
                
                elif layer_config.layer_type == 'temporal_downsample':
                    # Temporal downsampling for speed optimization
                    if x.dim() == 4:
                        x = x.squeeze(-1).transpose(1, 2)  # (batch, time, channels)
                    x = layer(x.transpose(1, 2)).transpose(1, 2)  # Apply Conv1D and transpose back
                
                elif layer_config.layer_type == 'fc':
                    # For FC layers, we need to handle the transition from conv layers
                    if x.dim() > 2:
                        # If coming from conv layers, apply global average pooling
                        if x.dim() == 4:  # Conv2D output
                            # Global average pooling: (B, C, H, W) -> (B, C)
                            x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
                        elif x.dim() == 3:  # Conv1D output
                            # Global average pooling: (B, C, T) -> (B, C)
                            x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
                        else:
                            # Flatten for other cases
                            x = x.view(x.size(0), -1)
                    x = layer(x)
                
                else:
                    # Standard convolutional layers
                    x = layer(x)
                    
            except Exception as e:
                print(f"Error in layer {i} ({layer_config.layer_type}): {e}")
                print(f"Input shape: {x.shape}")
                raise e
        
        return x
    
    def _glorot_weight_zero_bias(self):
        """Initialize weights using Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class RobustNeuroevolutionModelBuilder:
    """Robust builder class for creating models from genomes"""
    
    @staticmethod
    def create_model(genome: ArchitectureGenome) -> RobustNeuroevolutionModel:
        """Create a model from genome"""
        return RobustNeuroevolutionModel(genome)
    
    @staticmethod
    def create_classifier(genome: ArchitectureGenome) -> EEGClassifier:
        """Create an EEGClassifier from genome"""
        model = RobustNeuroevolutionModelBuilder.create_model(genome)
        
        # Create classifier with genome's training parameters
        classifier = EEGClassifier(
            model,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            train_split=None,
            optimizer__lr=genome.learning_rate,
            optimizer__weight_decay=genome.weight_decay,
            batch_size=genome.batch_size,
            max_epochs=genome.max_epochs,
            callbacks=[]
        )
        
        return classifier
    
    @staticmethod
    def validate_genome(genome: ArchitectureGenome) -> bool:
        """Validate genome for common issues."""
        try:
            # Check basic structure
            if not genome.layers:
                return False
            
            # Check parameter count
            param_count = genome.get_parameter_count()
            if param_count > 15000:  # Strict limit
                return False
            
            # Check layer transitions
            for i in range(len(genome.layers) - 1):
                current_layer = genome.layers[i]
                next_layer = genome.layers[i + 1]
                
                # Ensure output channels match input channels
                if current_layer.out_channels != next_layer.in_channels:
                    return False
                
                # Check for problematic layer combinations
                if (current_layer.layer_type in ['cfc', 'ncp'] and 
                    next_layer.layer_type in ['conv1d', 'conv2d']):
                    return False  # Can't go from recurrent to conv directly
                
                if (current_layer.layer_type in ['fc'] and 
                    next_layer.layer_type in ['conv1d', 'conv2d']):
                    return False  # Can't go from FC to conv directly
            
            # Ensure final layer is FC with correct output size
            final_layer = genome.layers[-1]
            if final_layer.layer_type != 'fc' or final_layer.out_channels != 2:
                return False
            
            # Check for CfC/NCP specific constraints
            for layer in genome.layers:
                if layer.layer_type in ['cfc', 'ncp']:
                    if layer.hidden_size is None:
                        return False
                    if layer.out_channels >= layer.hidden_size - 1:
                        return False  # CfC/NCP constraint
            
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def test_model_forward(genome: ArchitectureGenome, input_shape: tuple = (32, 22, 1000, 1)) -> bool:
        """Test if the model can perform a forward pass without errors."""
        try:
            model = RobustNeuroevolutionModelBuilder.create_model(genome)
            
            # Create test input
            test_input = torch.randn(input_shape)
            
            # Test forward pass
            with torch.no_grad():
                output = model(test_input)
            
            # Check output shape
            if output.shape[0] != input_shape[0] or output.shape[1] != 2:
                return False
            
            return True
            
        except Exception as e:
            print(f"Model forward test failed: {e}")
            return False 