from ncps.wirings import Wiring
import numpy as np
import random

class ModularSmallWorldWiring(Wiring):
    def __init__(self, units=32, output_size=8, n_modules=4, rewiring_prob=0.2, seed=None):
        super().__init__(units=units)
        
        # Set the output dimension (number of motor neurons)
        self.set_output_dim(output_size)
        
        self.total_neurons = units
        self.n_modules = n_modules
        self.module_size = self.total_neurons // n_modules
        self.rewiring_prob = rewiring_prob
        self.output_size = output_size
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Define neuron types and layers
        self._motor_neurons = list(range(0, self.output_size))
        self._inter_neurons = list(range(self.output_size, self.total_neurons))
        
        # Build the wiring structure
        self._build_small_world_wiring()

    def _build_small_world_wiring(self):
        """Build a true Small World network structure."""
        # 1. Create local modular connections (high clustering)
        self._build_modular_connections()
        
        # 2. Add long-range shortcuts (short average path length)
        self._build_small_world_shortcuts()
        
        # 3. Ensure motor neurons receive input from inter neurons
        self._build_motor_connections()

    def _build_modular_connections(self):
        """Build dense local connections within modules for high clustering."""
        for m in range(self.n_modules):
            module_start = m * self.module_size
            module_end = min(module_start + self.module_size, self.total_neurons)
            
            # Get neurons in this module
            module_neurons = [n for n in range(module_start, module_end) if n in self._inter_neurons]
            
            # Create dense local connections within the module
            for i, neuron1 in enumerate(module_neurons):
                for j, neuron2 in enumerate(module_neurons):
                    if i != j:  # Avoid self-connections
                        # Higher probability of connection for nearby neurons
                        distance = abs(i - j)
                        if distance <= 2 or random.random() < 0.7:  # 70% chance for nearby, 100% for very close
                            polarity = random.choice([-1, 1])
                            self.add_synapse(neuron1, neuron2, polarity)

    def _build_small_world_shortcuts(self):
        """Add long-range connections to create small-world properties."""
        # Add some random long-range connections between modules
        num_shortcuts = int(self.rewiring_prob * self.total_neurons)
        
        for _ in range(num_shortcuts):
            # Pick two neurons from different modules
            module1 = random.randint(0, self.n_modules - 1)
            module2 = random.randint(0, self.n_modules - 1)
            
            if module1 != module2:
                # Get neurons from each module
                start1 = module1 * self.module_size
                end1 = min(start1 + self.module_size, self.total_neurons)
                start2 = module2 * self.module_size
                end2 = min(start2 + self.module_size, self.total_neurons)
                
                # Get inter neurons from each module
                module1_inter = [n for n in range(start1, end1) if n in self._inter_neurons]
                module2_inter = [n for n in range(start2, end2) if n in self._inter_neurons]
                
                # Only proceed if both modules have inter neurons
                if module1_inter and module2_inter:
                    neuron1 = random.choice(module1_inter)
                    neuron2 = random.choice(module2_inter)
                    
                    if neuron1 != neuron2:
                        polarity = random.choice([-1, 1])
                        self.add_synapse(neuron1, neuron2, polarity)

    def _build_motor_connections(self):
        """Ensure motor neurons receive input from inter neurons."""
        for motor_neuron in self._motor_neurons:
            # Connect each motor neuron to several inter neurons
            num_inputs = min(8, len(self._inter_neurons))  # Each motor neuron gets up to 8 inputs
            selected_inter = random.sample(self._inter_neurons, num_inputs)
            
            for inter_neuron in selected_inter:
                polarity = random.choice([-1, 1])
                self.add_synapse(inter_neuron, motor_neuron, polarity)

    def build(self, input_shape):
        """Build the wiring with input shape - required by NCPs."""
        super().build(input_shape)
        
        # Connect sensory inputs to inter neurons
        for src in range(self.input_dim):
            # Connect each input to multiple inter neurons for redundancy
            num_targets = min(12, len(self._inter_neurons))  # Each input connects to up to 12 inter neurons
            selected_targets = random.sample(self._inter_neurons, num_targets)
            
            for dest in selected_targets:
                polarity = random.choice([-1, 1])
                self.add_sensory_synapse(src, dest, polarity)

    @property
    def num_layers(self):
        """Return the number of layers - required by WiredCfCCell."""
        return 2  # Sensory -> Inter, Inter -> Motor

    def get_neurons_of_layer(self, layer_id):
        """Return neurons for each layer - required by WiredCfCCell."""
        if layer_id == 0:
            return self._inter_neurons  # First layer: inter neurons
        elif layer_id == 1:
            return self._motor_neurons  # Second layer: motor neurons (outputs)
        else:
            raise ValueError(f"Unknown layer {layer_id}")

    def get_type_of_neuron(self, neuron_id):
        """Return the type of neuron as expected by NCPs."""
        if neuron_id < self.output_dim:
            return "motor"
        else:
            return "inter"
