import random
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict
from copy import deepcopy


@dataclass
class LayerConfig:
    """Configuration for a single layer in the architecture"""
    layer_type: str  # 'conv1d', 'conv2d', 'lstm', 'gru', 'cfc', 'ncp', 'attention', 'pool', 'dropout', 'bn', 'fc', 'temporal_downsample'
    in_channels: int
    out_channels: int
    kernel_size: Optional[Tuple[int, ...]] = None
    stride: Optional[Tuple[int, ...]] = None
    padding: Optional[Tuple[int, ...]] = None
    activation: str = 'elu'  # 'relu', 'elu', 'leaky_relu', 'tanh', 'sigmoid'
    dropout_rate: float = 0.0
    batch_norm: bool = False
    # For recurrent layers
    hidden_size: Optional[int] = None
    bidirectional: bool = False
    # For attention
    num_heads: Optional[int] = None
    # For NCP/CfC specific
    sparsity: Optional[float] = None
    wiring_type: Optional[str] = None  # 'auto', 'manual', 'random'
    # For CfC specific
    cfc_activation: Optional[str] = None  # 'silu', 'relu', 'tanh', 'gelu', 'lecun_tanh'
    mixed_memory: Optional[bool] = None  # True/False for CfC mixed memory mode
    # For temporal downsampling and windowing
    temporal_kernel_size: Optional[int] = None  # Kernel size for temporal downsampling
    temporal_stride: Optional[int] = None  # Stride for temporal downsampling
    max_seq_length: Optional[int] = None  # Maximum sequence length for CfC/NCP


@dataclass
class ArchitectureGenome:
    """Complete architecture genome for EEG models"""
    # Architecture parameters
    layers: List[LayerConfig]
    
    # Global parameters
    input_channels: int = 22
    input_times: int = 1001
    num_classes: int = 2
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    batch_size: int = 64
    max_epochs: int = 100
    
    # Regularization
    dropout_rate: float = 0.15
    batch_norm_momentum: float = 0.01
    
    # Fitness scores (will be populated during evolution)
    accuracy: float = 0.0
    noise_resilience: float = 0.0
    complexity_score: float = 0.0
    overfitting_score: float = 0.0
    overall_fitness: float = 0.0
    
    def __post_init__(self):
        """Validate genome after initialization"""
        if not self.layers:
            raise ValueError("Architecture must have at least one layer")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArchitectureGenome':
        """Create genome from dictionary"""
        # Convert layer dicts back to LayerConfig objects
        layers = []
        for layer_data in data['layers']:
            layers.append(LayerConfig(**layer_data))
        data['layers'] = layers
        return cls(**data)
    
    def get_parameter_count(self) -> int:
        """Estimate total number of parameters"""
        total_params = 0
        prev_channels = self.input_channels
        
        for layer in self.layers:
            if layer.layer_type == 'conv1d':
                # Conv1D: in_channels * out_channels * kernel_size + out_channels
                kernel_size = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
                total_params += prev_channels * layer.out_channels * kernel_size + layer.out_channels
                prev_channels = layer.out_channels
                
            elif layer.layer_type == 'conv2d':
                # Conv2D: in_channels * out_channels * kernel_size[0] * kernel_size[1] + out_channels
                if layer.kernel_size:
                    k1, k2 = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
                    total_params += prev_channels * layer.out_channels * k1 * k2 + layer.out_channels
                prev_channels = layer.out_channels
                
            elif layer.layer_type in ['lstm', 'gru']:
                # LSTM/GRU: 4 * (input_size + hidden_size) * hidden_size + hidden_size
                input_size = prev_channels
                hidden_size = layer.hidden_size or layer.out_channels
                total_params += 4 * (input_size + hidden_size) * hidden_size + hidden_size
                prev_channels = hidden_size
                
            elif layer.layer_type in ['cfc', 'ncp']:
                # CfC/NCP: simplified estimation
                total_params += prev_channels * layer.out_channels * 2 + layer.out_channels
                prev_channels = layer.out_channels
                
            elif layer.layer_type == 'fc':
                # Fully connected: in_features * out_features + out_features
                total_params += prev_channels * layer.out_channels + layer.out_channels
                prev_channels = layer.out_channels
        
        return total_params
    
    def get_complexity_score(self) -> float:
        """Calculate complexity score based on parameters and layers"""
        param_count = self.get_parameter_count()
        layer_count = len(self.layers)
        
        # Normalize by typical ranges
        param_score = min(param_count / 100000, 1.0)  # Cap at 100k params
        layer_score = min(layer_count / 20, 1.0)  # Cap at 20 layers
        
        return (param_score + layer_score) / 2


class GenomeGenerator:
    """Generates random architecture genomes"""
    
    def __init__(self, input_channels: int = 22, input_times: int = 1001, num_classes: int = 2):
        self.input_channels = input_channels
        self.input_times = input_times
        self.num_classes = num_classes
    
    def generate_random_genome(self, min_layers: int = 3, max_layers: int = 8) -> ArchitectureGenome:
        """Generate a random architecture genome"""
        num_layers = random.randint(min_layers, max_layers)
        layers = []
        
        current_channels = self.input_channels
        current_times = self.input_times
        
        for i in range(num_layers):
            layer = self._generate_random_layer(i, current_channels, current_times, is_last=(i == num_layers - 1))
            layers.append(layer)
            
            # Update dimensions for next layer
            if layer.layer_type in ['conv1d', 'conv2d']:
                current_channels = layer.out_channels
                if layer.layer_type == 'conv1d':
                    # Estimate temporal dimension change
                    kernel_size = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
                    stride = layer.stride[0] if layer.stride and isinstance(layer.stride, tuple) else (layer.stride or 1)
                    current_times = max(1, (current_times - kernel_size) // stride + 1)
            elif layer.layer_type in ['lstm', 'gru', 'cfc', 'ncp']:
                current_channels = layer.out_channels
            elif layer.layer_type == 'fc':
                current_channels = layer.out_channels
        
        return ArchitectureGenome(
            layers=layers,
            input_channels=self.input_channels,
            input_times=self.input_times,
            num_classes=self.num_classes,
            learning_rate=random.uniform(0.0001, 0.01),
            weight_decay=random.uniform(0.0001, 0.01),
            batch_size=random.choice([32, 64, 128]),
            dropout_rate=random.uniform(0.1, 0.3)
        )
    
    def _generate_random_layer(self, layer_idx: int, in_channels: int, in_times: int, is_last: bool) -> LayerConfig:
        """Generate a random layer configuration"""
        
        if is_last:
            # Final layer should be fully connected to output classes
            return LayerConfig(
                layer_type='fc',
                in_channels=in_channels,
                out_channels=self.num_classes,
                activation='linear'  # No activation for final layer
            )
        
        # Choose layer type based on position and current dimensions
        layer_types = ['conv1d', 'conv2d', 'lstm', 'gru', 'cfc', 'ncp', 'attention', 'dropout', 'bn', 'fc']
        
        # Adjust probabilities based on current state
        if in_times < 10:  # Too small for temporal layers
            layer_types = [lt for lt in layer_types if lt not in ['conv1d', 'lstm', 'gru', 'cfc', 'ncp']]
        
        if layer_idx < 2:  # Early layers: prefer conv layers
            layer_types.extend(['conv1d', 'conv2d'] * 2)
        
        layer_type = random.choice(layer_types)
        
        if layer_type == 'conv1d':
            out_channels = random.choice([16, 32, 64, 128])
            kernel_size = random.choice([3, 5, 7, 9, 11])
            stride = random.choice([1, 2])
            return LayerConfig(
                layer_type='conv1d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size,),
                stride=(stride,),
                padding=(kernel_size // 2,),
                activation=random.choice(['relu', 'elu', 'leaky_relu']),
                batch_norm=random.choice([True, False]),
                dropout_rate=random.uniform(0.0, 0.3)
            )
        
        elif layer_type == 'conv2d':
            out_channels = random.choice([16, 32, 64, 128])
            kernel_size = random.choice([(1, 3), (1, 5), (1, 7), (3, 3), (5, 5)])
            return LayerConfig(
                layer_type='conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=(0, kernel_size[1] // 2),
                activation=random.choice(['relu', 'elu', 'leaky_relu']),
                batch_norm=random.choice([True, False]),
                dropout_rate=random.uniform(0.0, 0.3)
            )
        
        elif layer_type in ['lstm', 'gru']:
            hidden_size = random.choice([32, 64, 128, 256])
            return LayerConfig(
                layer_type=layer_type,
                in_channels=in_channels,
                out_channels=hidden_size,
                hidden_size=hidden_size,
                bidirectional=random.choice([True, False]),
                activation=random.choice(['tanh', 'relu']),
                dropout_rate=random.uniform(0.0, 0.3)
            )
        
        elif layer_type in ['cfc', 'ncp']:
            # For NCP/CfC, output size must be less than units-2
            # So we need to ensure proper sizing
            units = random.choice([16, 32, 64, 128])
            output_size = max(2, units - 3)  # Ensure output_size < units-2
            
            return LayerConfig(
                layer_type=layer_type,
                in_channels=in_channels,
                out_channels=output_size,
                hidden_size=units,
                sparsity=random.uniform(0.5, 0.9),
                wiring_type='auto',
                activation=random.choice(['tanh', 'relu']),
                dropout_rate=random.uniform(0.0, 0.3)
            )
        
        elif layer_type == 'attention':
            num_heads = random.choice([2, 4, 8])
            return LayerConfig(
                layer_type='attention',
                in_channels=in_channels,
                out_channels=in_channels,  # Self-attention preserves dimensions
                num_heads=num_heads,
                activation=random.choice(['relu', 'elu']),
                dropout_rate=random.uniform(0.0, 0.3)
            )
        
        elif layer_type == 'dropout':
            return LayerConfig(
                layer_type='dropout',
                in_channels=in_channels,
                out_channels=in_channels,
                dropout_rate=random.uniform(0.1, 0.5),
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
            out_channels = random.choice([64, 128, 256, 512])
            return LayerConfig(
                layer_type='fc',
                in_channels=in_channels,
                out_channels=out_channels,
                activation=random.choice(['relu', 'elu', 'leaky_relu']),
                dropout_rate=random.uniform(0.0, 0.3)
            ) 