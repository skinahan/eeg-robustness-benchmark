import random
from typing import List, Optional
from neuroevolution.architecture_genome import ArchitectureGenome, LayerConfig


class SimpleGenomeGenerator:
    """Simplified genome generator that produces more reliable architectures."""
    
    def __init__(self, max_layers: int = 6, max_params: int = 15000):
        self.max_layers = max_layers
        self.max_params = max_params  # Strict parameter limit
    
    def generate_random_genome(self) -> ArchitectureGenome:
        """Generate a random genome with strict parameter limits."""
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                genome = self._generate_genome_attempt()
                if self._validate_genome(genome):
                    return genome
            except Exception as e:
                print(f"Genome generation attempt {attempt + 1} failed: {e}")
                continue
        
        # If all attempts failed, return a simple known-good architecture
        return self._generate_fallback_genome()
    
    def _generate_genome_attempt(self) -> ArchitectureGenome:
        """Generate a single genome attempt."""
        layers = []
        current_channels = 1  # Start with 1 channel (will be expanded to 22)
        
        # Add input layer
        layers.append(LayerConfig(
            layer_type='conv2d',
            in_channels=1,
            out_channels=8,
            kernel_size=(1, 15),
            stride=(1, 1),
            padding=(0, 7),
            activation='elu',
            dropout_rate=0.1,
            batch_norm=True
        ))
        current_channels = 8
        
        num_layers = random.randint(3, self.max_layers)
        
        for i in range(1, num_layers):
            layer_type = random.choice(['conv2d', 'fc'])
            
            if layer_type == 'conv2d':
                # Conv2D layer with strict channel limits
                out_channels = random.choice([8, 16, 32])
                kernel_size = random.choice([(1, 3), (1, 5), (1, 7)])
                
                layers.append(LayerConfig(
                    layer_type='conv2d',
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=(1, 1),
                    padding=(0, kernel_size[1] // 2),
                    activation='elu',
                    dropout_rate=0.1,
                    batch_norm=True
                ))
                current_channels = out_channels
                
            elif layer_type == 'fc':
                # FC layer with strict size limits
                hidden_size = random.choice([32, 64, 128])
                
                layers.append(LayerConfig(
                    layer_type='fc',
                    in_channels=current_channels,
                    out_channels=hidden_size,
                    activation='elu',
                    dropout_rate=0.2,
                    batch_norm=False
                ))
                current_channels = hidden_size
        
        # Add final output layer
        layers.append(LayerConfig(
            layer_type='fc',
            in_channels=current_channels,
            out_channels=2,  # Binary classification
            activation='linear',
            dropout_rate=0.0,
            batch_norm=False
        ))
        
        # Global parameters
        genome = ArchitectureGenome(
            layers=layers,
            learning_rate=random.uniform(1e-4, 1e-2),
            batch_size=random.choice([32, 64, 128]),
            weight_decay=random.uniform(1e-5, 1e-3)
        )
        
        return genome
    
    def _validate_genome(self, genome: ArchitectureGenome) -> bool:
        """Validate genome for common issues."""
        try:
            # Check parameter count
            if genome.get_parameter_count() > self.max_params:
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
            
            return True
            
        except Exception:
            return False
    
    def _generate_fallback_genome(self) -> ArchitectureGenome:
        """Generate a simple fallback genome that's guaranteed to work."""
        layers = [
            LayerConfig(layer_type='conv2d', in_channels=1, out_channels=8, 
                      kernel_size=(1, 15), stride=(1, 1), padding=(0, 7),
                      activation='elu', dropout_rate=0.1, batch_norm=True),
            LayerConfig(layer_type='conv2d', in_channels=8, out_channels=16, 
                      kernel_size=(1, 5), stride=(1, 1), padding=(0, 2),
                      activation='elu', dropout_rate=0.1, batch_norm=True),
            LayerConfig(layer_type='fc', in_channels=16, out_channels=64,
                      activation='elu', dropout_rate=0.2, batch_norm=False),
            LayerConfig(layer_type='fc', in_channels=64, out_channels=2,
                      activation='linear', dropout_rate=0.0, batch_norm=False)
        ]
        
        return ArchitectureGenome(
            layers=layers,
            learning_rate=1e-3,
            batch_size=64,
            weight_decay=1e-4
        )
    
    def generate_known_good_architectures(self) -> List[ArchitectureGenome]:
        """Generate some known good architectures for testing."""
        architectures = []
        
        # Simple CNN architecture
        layers = [
            LayerConfig(layer_type='conv2d', in_channels=1, out_channels=8, 
                      kernel_size=(1, 15), stride=(1, 1), padding=(0, 7),
                      activation='elu', dropout_rate=0.1, batch_norm=True),
            LayerConfig(layer_type='conv2d', in_channels=8, out_channels=16, 
                      kernel_size=(1, 5), stride=(1, 1), padding=(0, 2),
                      activation='elu', dropout_rate=0.1, batch_norm=True),
            LayerConfig(layer_type='fc', in_channels=16, out_channels=64,
                      activation='elu', dropout_rate=0.2, batch_norm=False),
            LayerConfig(layer_type='fc', in_channels=64, out_channels=2,
                      activation='linear', dropout_rate=0.0, batch_norm=False)
        ]
        
        architectures.append(ArchitectureGenome(
            layers=layers,
            learning_rate=1e-3,
            batch_size=64,
            weight_decay=1e-4
        ))
        
        return architectures 