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


class NeuroevolutionModel(EEGModuleMixin, nn.Module):
    """Dynamic model built from architecture genome"""
    
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
        current_times = self.genome.input_times
        
        for layer_config in self.genome.layers:
            layer = self._create_layer(layer_config, current_channels, current_times)
            self.layers.append(layer)
            
            # Update dimensions for next layer
            if layer_config.layer_type in ['conv1d', 'conv2d']:
                current_channels = layer_config.out_channels
                if layer_config.layer_type == 'conv1d':
                    # Estimate temporal dimension change
                    kernel_size = layer_config.kernel_size[0] if isinstance(layer_config.kernel_size, tuple) else layer_config.kernel_size
                    stride = layer_config.stride[0] if layer_config.stride and isinstance(layer_config.stride, tuple) else (layer_config.stride or 1)
                    current_times = max(1, (current_times - kernel_size) // stride + 1)
            elif layer_config.layer_type in ['lstm', 'gru', 'cfc', 'ncp']:
                current_channels = layer_config.out_channels
            elif layer_config.layer_type == 'fc':
                current_channels = layer_config.out_channels
    
    def _create_layer(self, layer_config: LayerConfig, in_channels: int, in_times: int) -> nn.Module:
        """Create a PyTorch layer from layer configuration"""
        
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
        
        elif layer_config.layer_type == 'attention':
            return self._create_attention_layer(layer_config, in_channels)
        
        elif layer_config.layer_type in ['max_pool', 'avg_pool']:
            return self._create_pool_layer(layer_config, in_channels)
        
        elif layer_config.layer_type == 'dropout':
            return self._create_dropout_layer(layer_config)
        
        elif layer_config.layer_type == 'batch_norm':
            return self._create_batch_norm_layer(layer_config, in_channels)
        
        elif layer_config.layer_type == 'fc':
            return self._create_fc_layer(layer_config, in_channels)
        
        else:
            raise ValueError(f"Unknown layer type: {layer_config.layer_type}")
    
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
        
        layers = [lstm]
        
        if layer_config.dropout_rate > 0:
            layers.append(nn.Dropout(layer_config.dropout_rate))
        
        return nn.Sequential(*layers)
    
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
        
        layers = [gru]
        
        if layer_config.dropout_rate > 0:
            layers.append(nn.Dropout(layer_config.dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _create_cfc_layer(self, layer_config: LayerConfig, in_channels: int) -> nn.Module:
        """Create a CfC layer"""
        hidden_size = layer_config.hidden_size or layer_config.out_channels
        sparsity = layer_config.sparsity or 0.75
        
        # Ensure output size is less than units-2 for NCP library
        output_size = min(layer_config.out_channels, hidden_size - 2)
        
        wiring = AutoNCP(hidden_size, output_size, sparsity_level=sparsity)
        cfc = CfC(in_channels, wiring)
        
        layers = [cfc]
        
        if layer_config.dropout_rate > 0:
            layers.append(nn.Dropout(layer_config.dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _create_ncp_layer(self, layer_config: LayerConfig, in_channels: int) -> nn.Module:
        """Create an NCP layer"""
        hidden_size = layer_config.hidden_size or layer_config.out_channels
        sparsity = layer_config.sparsity or 0.75
        
        # Ensure output size is less than units-2 for NCP library
        output_size = min(layer_config.out_channels, hidden_size - 2)
        
        wiring = AutoNCP(hidden_size, output_size, sparsity_level=sparsity)
        ncp = LTC(in_channels, wiring)
        
        layers = [ncp]
        
        if layer_config.dropout_rate > 0:
            layers.append(nn.Dropout(layer_config.dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _create_attention_layer(self, layer_config: LayerConfig, in_channels: int) -> nn.Module:
        """Create a multi-head attention layer"""
        num_heads = layer_config.num_heads or 4
        
        attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=True
        )
        
        layers = [attention]
        
        if layer_config.dropout_rate > 0:
            layers.append(nn.Dropout(layer_config.dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _create_pool_layer(self, layer_config: LayerConfig, in_channels: int) -> nn.Module:
        """Create a pooling layer"""
        kernel_size = layer_config.kernel_size[0] if isinstance(layer_config.kernel_size, tuple) else layer_config.kernel_size
        stride = layer_config.stride[0] if layer_config.stride and isinstance(layer_config.stride, tuple) else (layer_config.stride or kernel_size)
        
        if layer_config.layer_type == 'max_pool':
            return nn.MaxPool1d(kernel_size, stride=stride)
        else:  # avg_pool
            return nn.AvgPool1d(kernel_size, stride=stride)
    
    def _create_dropout_layer(self, layer_config: LayerConfig) -> nn.Module:
        """Create a dropout layer"""
        return nn.Dropout(layer_config.dropout_rate)
    
    def _create_batch_norm_layer(self, layer_config: LayerConfig, in_channels: int) -> nn.Module:
        """Create a batch normalization layer"""
        return nn.BatchNorm1d(in_channels)
    
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
        """Forward pass through the network"""
        # Ensure input is 4D for EEG data: (batch, channels, time, 1)
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            layer_config = self.genome.layers[i]
            
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
                x = layer(x)
                # Take the last output
                x = x[:, -1, :]
            
            elif layer_config.layer_type == 'attention':
                # Reshape for attention: (batch, time, channels)
                if x.dim() == 4:
                    x = x.squeeze(-1).transpose(1, 2)
                x, _ = layer(x, x, x)
                # Global average pooling
                x = x.mean(dim=1)
            
            elif layer_config.layer_type == 'fc':
                # Flatten for fully connected layers
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                x = layer(x)
            
            elif layer_config.layer_type == 'dropout':
                # Handle dropout layers - they expect tensor input
                if isinstance(x, tuple):
                    x = x[0]  # Take first element if tuple
                x = layer(x)
            
            elif layer_config.layer_type == 'batch_norm':
                # Handle batch norm layers
                if x.dim() == 4:
                    x = x.squeeze(-1)  # Remove last dimension for 1D batch norm
                x = layer(x)
            
            else:
                # Standard convolutional/pooling layers
                x = layer(x)
        
        return x
    
    def _glorot_weight_zero_bias(self):
        """Initialize weights using Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class NeuroevolutionModelBuilder:
    """Builder class for creating models from genomes"""
    
    @staticmethod
    def create_model(genome: ArchitectureGenome) -> NeuroevolutionModel:
        """Create a model from genome"""
        return NeuroevolutionModel(genome)
    
    @staticmethod
    def create_classifier(genome: ArchitectureGenome) -> EEGClassifier:
        """Create an EEGClassifier from genome"""
        model = NeuroevolutionModelBuilder.create_model(genome)
        
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
        """Validate if a genome can create a valid model"""
        try:
            # Try to create a model
            model = NeuroevolutionModelBuilder.create_model(genome)
            
            # Test with dummy input
            dummy_input = torch.randn(1, genome.input_channels, genome.input_times)
            output = model(dummy_input)
            
            # Check output shape
            expected_output_size = genome.num_classes
            if output.shape[-1] != expected_output_size:
                return False
            
            return True
        
        except Exception as e:
            print(f"Genome validation failed: {e}")
            return False 