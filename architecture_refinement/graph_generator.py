"""
Graph generation module for creating candidate wiring architectures.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
import random
from dataclasses import dataclass
import logging
from .config import GraphGenerationConfig

@dataclass
class GraphParameters:
    """Parameters for a single graph generation."""
    units: int
    output_size: int
    # n_modules: int
    rewiring_prob: float
    # connection_density: float
    # module_connectivity: float
    # inter_module_connectivity: float
    # Watts-Strogatz specific parameters
    k_degree: int  # Number of neighbors each node connects to in initial ring lattice
    p_rewiring: float  # Probability of rewiring each edge
    # Target properties for flex graphs
    target_clustering: Optional[float] = None  # For Watts-Strogatz flex graphs
    target_path_length: Optional[float] = None  # For Watts-Strogatz flex graphs
    seed: Optional[int] = None

class ModularSmallWorldGraphGenerator:
    """
    Generates candidate wiring graphs using modular small-world strategies.
    
    This class implements various strategies for creating biologically-inspired
    network architectures that balance local clustering with global connectivity.
    """
    
    def __init__(self, config: GraphGenerationConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the graph generator.
        
        Args:
            config: Configuration for graph generation
            logger: Optional logger for output
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Set random seed if specified
        if config.seed is not None:
            np.random.seed(config.seed)
            random.seed(config.seed)
    
    def generate_candidate_graphs(self, num_candidates: Optional[int] = None) -> List[Tuple[nx.Graph, GraphParameters]]:
        """
        Generate a set of candidate graphs with varying parameters.
        
        Args:
            num_candidates: Number of candidates to generate (uses config default if None)
            
        Returns:
            List of tuples containing (graph, parameters)
        """
        num_candidates = num_candidates or self.config.num_candidates
        
        self.logger.info(f"Generating {num_candidates} candidate graphs...")
        
        candidates = []
        
        for i in range(num_candidates):
            # Generate random parameters within bounds
            params = self._generate_random_parameters()
            
            # Generate graph with these parameters
            graph = self._create_modular_small_world_graph(params)
            
            candidates.append((graph, params))
            
            if (i + 1) % 100 == 0:
                self.logger.info(f"Generated {i + 1}/{num_candidates} graphs")
        
        self.logger.info(f"Successfully generated {len(candidates)} candidate graphs")
        return candidates
    
    def _generate_random_parameters(self) -> GraphParameters:
        """Generate random parameters within the configured bounds."""
        units = np.random.randint(
            self.config.min_units, 
            self.config.max_units + 1
        )
        
        output_size = np.random.randint(
            self.config.min_output_size, 
            self.config.max_output_size + 1
        )
        
        n_modules = np.random.randint(
            self.config.min_modules, 
            self.config.max_modules + 1
        )
        
        rewiring_prob = np.random.uniform(
            self.config.min_rewiring_prob, 
            self.config.max_rewiring_prob
        )
        
        connection_density = np.random.uniform(
            self.config.min_connection_density, 
            self.config.max_connection_density
        )
        
        # Derived parameters
        module_connectivity = np.random.uniform(0.3, 0.9)
        inter_module_connectivity = np.random.uniform(0.1, 0.5)
        
        return GraphParameters(
            units=units,
            output_size=output_size,
            # n_modules=n_modules,
            rewiring_prob=rewiring_prob,
            connection_density=connection_density,
            # module_connectivity=module_connectivity,
            # inter_module_connectivity=inter_module_connectivity,
            seed=np.random.randint(0, 10000)
        )
    
    def _create_modular_small_world_graph(self, params: GraphParameters) -> nx.Graph:
        """
        Create a modular small-world graph with the given parameters.
        
        Args:
            params: Graph generation parameters
            
        Returns:
            NetworkX graph object
        """
        # Set seed for this specific graph
        if params.seed is not None:
            np.random.seed(params.seed)
            random.seed(params.seed)
        
        # Create empty graph
        G = nx.Graph()
        
        # Add nodes
        G.add_nodes_from(range(params.units))
        
        # Define neuron types
        motor_neurons = list(range(0, params.output_size))
        inter_neurons = list(range(params.output_size, params.units))
        
        # Calculate module sizes
        module_size = params.units // params.n_modules
        
        # Build modular structure
        self._build_modular_connections(G, params, motor_neurons, inter_neurons, module_size)
        
        # Add small-world shortcuts
        self._add_small_world_shortcuts(G, params, inter_neurons, module_size)
        
        # Ensure motor neurons receive input
        self._connect_motor_neurons(G, params, motor_neurons, inter_neurons)
        
        # Add sensory input connections
        self._add_sensory_connections(G, params, inter_neurons)
        
        # Validate graph properties
        self._validate_graph(G, params)
        
        return G
    
    def _build_modular_connections(
        self, 
        G: nx.Graph, 
        params: GraphParameters, 
        motor_neurons: List[int], 
        inter_neurons: List[int], 
        module_size: int
    ):
        """Build dense local connections within modules."""
        for m in range(params.n_modules):
            module_start = m * module_size
            module_end = min(module_start + module_size, params.units)
            
            # Get inter neurons in this module
            module_inter = [n for n in range(module_start, module_end) if n in inter_neurons]
            
            if len(module_inter) < 2:
                continue
            
            # Create dense local connections within the module
            for i, neuron1 in enumerate(module_inter):
                for j, neuron2 in enumerate(module_inter):
                    if i != j:  # Avoid self-connections
                        # Higher probability for nearby neurons
                        distance = abs(i - j)
                        if distance <= 2 or random.random() < params.module_connectivity:
                            # Add edge with random weight
                            weight = random.choice([-1, 1])
                            G.add_edge(neuron1, neuron2, weight=weight, type="intra_module")
    
    def _add_small_world_shortcuts(
        self, 
        G: nx.Graph, 
        params: GraphParameters, 
        inter_neurons: List[int], 
        module_size: int
    ):
        """Add long-range connections to create small-world properties."""
        num_shortcuts = int(params.rewiring_prob * len(inter_neurons))
        
        shortcuts_added = 0
        max_attempts = num_shortcuts * 10  # Prevent infinite loops
        
        for attempt in range(max_attempts):
            if shortcuts_added >= num_shortcuts:
                break
            
            # Pick two neurons from different modules
            module1 = random.randint(0, params.n_modules - 1)
            module2 = random.randint(0, params.n_modules - 1)
            
            if module1 != module2:
                # Get neurons from each module
                start1 = module1 * module_size
                end1 = min(start1 + module_size, params.units)
                start2 = module2 * module_size
                end2 = min(start2 + module_size, params.units)
                
                # Get inter neurons from each module
                module1_inter = [n for n in range(start1, end1) if n in inter_neurons]
                module2_inter = [n for n in range(start2, end2) if n in inter_neurons]
                
                if module1_inter and module2_inter:
                    neuron1 = random.choice(module1_inter)
                    neuron2 = random.choice(module2_inter)
                    
                    if neuron1 != neuron2 and not G.has_edge(neuron1, neuron2):
                        weight = random.choice([-1, 1])
                        G.add_edge(neuron1, neuron2, weight=weight, type="inter_module")
                        shortcuts_added += 1
    
    def _connect_motor_neurons(
        self, 
        G: nx.Graph, 
        params: GraphParameters, 
        motor_neurons: List[int], 
        inter_neurons: List[int]
    ):
        """Ensure motor neurons receive input from inter neurons."""
        for motor_neuron in motor_neurons:
            # Connect each motor neuron to several inter neurons
            num_inputs = min(8, len(inter_neurons))
            selected_inter = random.sample(inter_neurons, num_inputs)
            
            for inter_neuron in selected_inter:
                if not G.has_edge(inter_neuron, motor_neuron):
                    weight = random.choice([-1, 1])
                    G.add_edge(inter_neuron, motor_neuron, weight=weight, type="motor_input")
    
    def _add_sensory_connections(
        self, 
        G: nx.Graph, 
        params: GraphParameters, 
        inter_neurons: List[int]
    ):
        """Add sensory input connections to inter neurons."""
        # For now, we'll add virtual sensory nodes
        # In practice, these would be connected to actual input data
        num_sensory_inputs = min(16, len(inter_neurons) // 2)
        
        for i in range(num_sensory_inputs):
            # Connect each sensory input to multiple inter neurons
            num_targets = min(6, len(inter_neurons))
            selected_targets = random.sample(inter_neurons, num_targets)
            
            for target in selected_targets:
                weight = random.choice([-1, 1])
                G.add_edge(f"sensory_{i}", target, weight=weight, type="sensory")
    
    def _validate_graph(self, G: nx.Graph, params: GraphParameters):
        """Validate that the generated graph meets basic requirements."""
        # Check connectivity
        if not nx.is_connected(G):
            # Remove isolated nodes and try to connect components
            isolated_nodes = list(nx.isolates(G))
            G.remove_nodes_from(isolated_nodes)
            
            components = list(nx.connected_components(G))
            if len(components) > 1:
                # Connect components with minimal edges
                for i in range(len(components) - 1):
                    node1 = list(components[i])[0]
                    node2 = list(components[i + 1])[0]
                    G.add_edge(node1, node2, weight=1, type="bridge")
        
        # Check minimum degree
        min_degree = min(dict(G.degree()).values())
        if min_degree < 2:
            self.logger.warning(f"Graph has nodes with degree < 2: {min_degree}")
    
    def generate_parameterized_graphs(self, parameter_ranges: Dict[str, List[Any]]) -> List[Tuple[nx.Graph, GraphParameters]]:
        """
        Generate graphs by systematically varying specific parameters.
        
        Args:
            parameter_ranges: Dictionary mapping parameter names to lists of values
            
        Returns:
            List of tuples containing (graph, parameters)
        """
        self.logger.info("Generating parameterized graphs...")
        
        # Generate all parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_ranges)
        
        graphs = []
        for params in param_combinations:
            graph = self._create_modular_small_world_graph(params)
            graphs.append((graph, params))
        
        self.logger.info(f"Generated {len(graphs)} parameterized graphs")
        return graphs
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List[Any]]) -> List[GraphParameters]:
        """Generate all combinations of the given parameter ranges."""
        import itertools
        
        # Get parameter names and their possible values
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        combinations = []
        for values in itertools.product(*param_values):
            # Create parameter dict
            param_dict = dict(zip(param_names, values))
            
            # Create GraphParameters object with defaults for unspecified parameters
            params = GraphParameters(
                units=param_dict.get('units', self.config.min_units),
                output_size=param_dict.get('output_size', self.config.min_output_size),
                n_modules=param_dict.get('n_modules', self.config.min_modules),
                rewiring_prob=param_dict.get('rewiring_prob', self.config.min_rewiring_prob),
                connection_density=param_dict.get('connection_density', self.config.min_connection_density),
                module_connectivity=param_dict.get('module_connectivity', 0.5),
                inter_module_connectivity=param_dict.get('inter_module_connectivity', 0.3)
            )
            
            combinations.append(params)
        
        return combinations
    
    def analyze_graph_properties(self, graph: nx.Graph) -> Dict[str, float]:
        """
        Analyze basic properties of a generated graph.
        
        Args:
            graph: NetworkX graph to analyze
            
        Returns:
            Dictionary of graph properties
        """
        properties = {}
        
        # Basic properties
        properties['num_nodes'] = graph.number_of_nodes()
        properties['num_edges'] = graph.number_of_edges()
        properties['density'] = nx.density(graph)
        
        # Connectivity
        if nx.is_connected(graph):
            properties['diameter'] = nx.diameter(graph)
            properties['avg_path_length'] = nx.average_shortest_path_length(graph)
        else:
            properties['diameter'] = float('inf')
            properties['avg_path_length'] = float('inf')
        
        # Clustering
        properties['clustering_coefficient'] = nx.average_clustering(graph)
        
        # Degree distribution
        degrees = [d for n, d in graph.degree()]
        properties['avg_degree'] = np.mean(degrees)
        properties['std_degree'] = np.std(degrees)
        properties['min_degree'] = np.min(degrees)
        properties['max_degree'] = np.max(degrees)
        
        return properties
    
    def save_graph(self, graph: nx.Graph, filepath: str, format: str = "graphml"):
        """
        Save a graph to file.
        
        Args:
            graph: NetworkX graph to save
            filepath: Path to save the graph
            format: File format ('graphml', 'gml', 'pajek', 'edgelist')
        """
        if format == "graphml":
            nx.write_graphml(graph, filepath)
        elif format == "gml":
            nx.write_gml(graph, filepath)
        elif format == "pajek":
            nx.write_pajek(graph, filepath)
        elif format == "edgelist":
            nx.write_edgelist(graph, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Graph saved to {filepath}")
    
    def load_graph(self, filepath: str, format: str = "auto") -> nx.Graph:
        """
        Load a graph from file.
        
        Args:
            filepath: Path to the graph file
            format: File format ('auto', 'graphml', 'gml', 'pajek', 'edgelist')
            
        Returns:
            Loaded NetworkX graph
        """
        import os
        
        if format == "auto":
            ext = os.path.splitext(filepath)[1].lower()
            if ext == ".graphml":
                format = "graphml"
            elif ext == ".gml":
                format = "gml"
            elif ext == ".net":
                format = "pajek"
            elif ext == ".txt":
                format = "edgelist"
            else:
                format = "graphml"  # Default
        
        if format == "graphml":
            return nx.read_graphml(filepath)
        elif format == "gml":
            return nx.read_gml(filepath)
        elif format == "pajek":
            return nx.read_pajek(filepath)
        elif format == "edgelist":
            return nx.read_edgelist(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def generate_watts_strogatz_flex_graphs(self, num_candidates: int = 100) -> List[Tuple[nx.Graph, GraphParameters]]:
        """
        Generate candidate graphs using Watts-Strogatz flex approach with varying parameters.
        
        This expands the search space significantly by exploring different clustering coefficients
        and path lengths as suggested by the original study authors.
        
        Args:
            num_candidates: Number of candidate graphs to generate
            
        Returns:
            List of tuples containing (graph, parameters)
        """
        if self.logger:
            self.logger.info(f"Generating {num_candidates} Watts-Strogatz flex candidate graphs")
        
        # Use a consistent base seed for reproducibility
        base_seed = self.config.seed if self.config.seed is not None else 42
        
        if self.logger:
            self.logger.info(f"Using base seed: {base_seed} for reproducible graph generation")
        
        candidate_graphs = []
        
        for i in range(num_candidates):
            try:
                # Set seed for this specific graph generation (deterministic)
                graph_seed = base_seed + i
                np.random.seed(graph_seed)
                random.seed(graph_seed)
                
                if self.logger and i % 20 == 0:
                    self.logger.info(f"Generating graph {i+1} with seed: {graph_seed}")
                
                # Vary network size more broadly
                units = np.random.randint(self.config.min_units, self.config.max_units + 1)
                
                # Vary clustering coefficient from very low to very high
                target_clustering = np.random.uniform(0.1, 0.9)
                
                # Vary average path length (shorter = more random, longer = more regular)
                target_path_length = np.random.uniform(2.0, min(8.0, units / 4))
                
                # Generate Watts-Strogatz graph with target properties
                graph = self._create_watts_strogatz_flex_graph(
                    units, 
                    self.config.min_k_degree,  # Use config k_degree
                    self.config.min_p_rewiring,  # Use config p_rewiring
                    target_clustering, 
                    target_path_length,
                    graph_seed  # Pass the seed to the graph creation
                )
                
                if graph and graph.number_of_nodes() > 0:
                    # Create parameters for this graph
                    params = GraphParameters(
                        units=units,
                        output_size=max(2, units // 8),
                        n_modules=max(2, units // 16),
                        rewiring_prob=np.random.uniform(0.05, 0.8),  # Wider range
                        connection_density=np.random.uniform(0.2, 0.9),  # Wider range
                        module_connectivity=np.random.uniform(0.3, 0.95),
                        inter_module_connectivity=np.random.uniform(0.05, 0.7),
                        seed=graph_seed  # Use the consistent seed
                    )
                    
                    candidate_graphs.append((graph, params))
                    
                    if self.logger and (i + 1) % 20 == 0:
                        self.logger.info(f"Generated {i + 1}/{num_candidates} flex graphs")
                        
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to generate flex graph {i}: {e}")
                continue
        
        if self.logger:
            self.logger.info(f"Successfully generated {len(candidate_graphs)} flex candidate graphs")
        
        return candidate_graphs
    
    def _create_watts_strogatz_flex_graph(self, n: int, k_degree: int, p_rewiring: float, target_clustering: float, target_path_length: float, seed: Optional[int] = None) -> nx.Graph:
        """
        Create a Watts-Strogatz flex graph with target clustering and path length.
        
        Args:
            n: Number of nodes
            k_degree: Number of neighbors each node connects to in initial ring lattice
            p_rewiring: Probability of rewiring each edge
            target_clustering: Target clustering coefficient
            target_path_length: Target average path length
            seed: Random seed for reproducibility
            
        Returns:
            NetworkX graph with desired properties
        """
        # Set seed for this graph generation if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            if self.logger:
                self.logger.debug(f"Creating WS graph with seed: {seed}")
        
        # Start with a regular ring lattice using the specified k_degree
        k = max(2, min(k_degree, n // 2))  # Ensure k is valid for the graph size
        G = nx.watts_strogatz_graph(n, k, 0.0, seed=seed)  # Start with regular graph
        
        # If p_rewiring is specified, use it directly; otherwise optimize for target clustering
        if p_rewiring is not None and p_rewiring >= 0:
            # Use the specified rewiring probability
            G = nx.watts_strogatz_graph(n, k, p_rewiring, seed=seed)
        else:
            # Iteratively adjust rewiring probability to achieve target clustering
            best_clustering = nx.average_clustering(G)
            best_rewiring = 0.0
            best_graph = G.copy()
            
            # Try different rewiring probabilities
            for rewiring in np.linspace(0.0, 1.0, 20):
                test_graph = nx.watts_strogatz_graph(n, k, rewiring, seed=seed + int(rewiring * 1000))
                clustering = nx.average_clustering(test_graph)
                
                # Check if this rewiring gives us closer to target clustering
                if abs(clustering - target_clustering) < abs(best_clustering - target_clustering):
                    best_clustering = clustering
                    best_rewiring = rewiring
                    best_graph = test_graph.copy()
            
            G = best_graph.copy()
        
        # Now try to fine-tune the path length by selective edge rewiring
        current_path_length = nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf')
        
        # If path length is too long, add some random shortcuts
        if current_path_length > target_path_length:
            num_shortcuts = int(n * 0.1)  # Add 10% of nodes as shortcuts
            for _ in range(num_shortcuts):
                u = np.random.randint(0, n)
                v = np.random.randint(0, n)
                if u != v and not G.has_edge(u, v):
                    G.add_edge(u, v)
        
        # If path length is too short, remove some edges to make it more regular
        elif current_path_length < target_path_length * 0.8:
            edges_to_remove = int(G.number_of_edges() * 0.05)  # Remove 5% of edges
            edges = list(G.edges())
            for _ in range(min(edges_to_remove, len(edges))):
                if edges:
                    edge = edges.pop(np.random.randint(0, len(edges)))
                    G.remove_edge(*edge)
        
        # Ensure connectivity
        if not nx.is_connected(G):
            # Connect components with minimal edges
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                node1 = list(components[i])[0]
                node2 = list(components[i + 1])[0]
                G.add_edge(node1, node2)
        
        # Add edge weights and types for analysis
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(0.5, 2.0)
            G[u][v]['type'] = 'flex'
        
        return G
