# Architecture Refinement for Robustness-Aware CfC Networks

A comprehensive system for optimizing CfC (Closed-form Continuous-time) network architectures using graph-theoretic metrics and multi-objective optimization. This project implements a complete pipeline from graph generation to WiredCfC architecture conversion, designed for creating robust and efficient neural network architectures.

## 🎯 Project Overview

The Architecture Refinement project addresses the challenge of designing robust neural network architectures by:

1. **Generating candidate wiring graphs** using modular small-world strategies
2. **Evaluating architectures** using pre-training graph-theoretic metrics
3. **Optimizing designs** through multi-objective optimization with Optuna
4. **Converting to WiredCfC** for downstream training and deployment

## 🏗️ Architecture Components

### Core Modules

- **`graph_generator.py`** - Modular small-world graph generation strategies
- **`topology_analyzer.py`** - Graph-theoretic metric computation (entropy, curvature, connectivity)
- **`optimizer.py`** - Multi-objective optimization using Optuna
- **`architecture_converter.py`** - WiredCfC architecture conversion
- **`config.py`** - Comprehensive configuration management
- **`utils.py`** - Logging, visualization, and utility functions

### Key Features

- **Modular Small-World Generation**: Creates biologically-inspired network architectures
- **Topological Analysis**: Computes entropy, Ollivier-Ricci curvature, algebraic connectivity
- **Multi-Objective Optimization**: Balances multiple architectural objectives
- **WiredCfC Integration**: Seamless conversion to trainable architectures
- **Comprehensive Logging**: Rich logging and visualization capabilities
- **Reproducible Research**: Deterministic execution and result caching

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd architecture_refinement
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the demo**:
   ```bash
   # Full demo
   python demo.py
   
   # Quick demo (faster, fewer trials)
   python demo.py --quick
   ```

### Basic Usage

```python
from architecture_refinement import (
    Config, 
    ModularSmallWorldGraphGenerator,
    TopologyAnalyzer,
    MultiObjectiveOptimizer,
    WiredCfCConverter
)

# Setup configuration
config = Config()

# Generate candidate graphs
graph_generator = ModularSmallWorldGraphGenerator(config.graph_generation)
candidate_graphs = graph_generator.generate_candidate_graphs(num_candidates=100)

# Analyze topology
topology_analyzer = TopologyAnalyzer(config)
topology_metrics = topology_analyzer.analyze_graph_batch([g for g, p in candidate_graphs])

# Run optimization
optimizer = MultiObjectiveOptimizer(config.optimization, graph_generator, topology_analyzer)
results = optimizer.optimize(n_trials=50)

# Convert to WiredCfC
converter = WiredCfCConverter(config.architecture)
best_graphs = [g for g, p in candidate_graphs[:5]]  # Top 5 graphs
architectures = converter.convert_batch_architectures(best_graphs)
```

## 📊 Graph-Theoretic Metrics

### Entropy Measures
- **Degree Entropy**: Measures the diversity of node degrees
- **Weight Entropy**: Quantifies connection weight distribution
- **Path Length Entropy**: Captures path length distribution complexity

### Curvature Measures
- **Ollivier-Ricci Curvature**: Geometric curvature for network robustness
- **Forman-Ricci Curvature**: Alternative curvature measure for edges

### Connectivity Measures
- **Algebraic Connectivity**: Second smallest eigenvalue of Laplacian
- **Edge/Node Connectivity**: Minimum cuts for network resilience
- **Expansion Ratio**: Network expansion properties

### Efficiency Measures
- **Global Efficiency**: Average inverse shortest path length
- **Local Efficiency**: Local clustering efficiency
- **Cost Efficiency**: Efficiency per connection

## 🔧 Configuration

The system uses a hierarchical configuration structure:

```python
from architecture_refinement.config import Config

config = Config()

# Graph generation parameters
config.graph_generation.min_units = 32
config.graph_generation.max_units = 128
config.graph_generation.n_modules = 4

# Topology analysis parameters
config.topology_metrics.entropy_bins = 20
config.topology_metrics.ricci_curvature_method = "ollivier"

# Optimization parameters
config.optimization.n_trials = 100
config.optimization.entropy_weight = 0.3
config.optimization.curvature_weight = 0.3

# Architecture parameters
config.architecture.input_size = 64
config.architecture.hidden_size = 64
config.architecture.output_size = 8
```

## 📈 Optimization Process

### Multi-Objective Optimization

The system optimizes four key objectives:

1. **Entropy Maximization**: Higher information content
2. **Curvature Optimization**: Better geometric properties
3. **Connectivity Enhancement**: Improved network resilience
4. **Efficiency Maximization**: Better information flow

### Optimization Strategy

- **Algorithm**: Tree-structured Parzen Estimator (TPE)
- **Sampling**: Multi-objective with Pareto front identification
- **Constraints**: Configurable minimum thresholds for each metric
- **Validation**: Automatic constraint checking and solution filtering

## 🎨 Visualization and Analysis

### Generated Plots

- **Optimization History**: Trial progress and parameter importance
- **Pareto Front**: Multi-objective solution space
- **Topology Metrics**: Distribution of graph properties
- **Parameter Relationships**: Architecture parameter effects
- **Architecture Summary**: Network structure visualization

### Output Structure

```
outputs/
├── plots/                    # Generated visualizations
├── models/                   # Saved model architectures
├── logs/                     # Experiment logs
├── optimization/             # Optimization results
├── best_graphs/             # Top-ranked graph files
└── architectures/            # WiredCfC specifications
```

## 🔬 Research Applications

### EEG Signal Processing
- **Robust Architecture Design**: Noise-resistant network topologies
- **Cross-Session Generalization**: Improved transfer learning
- **Real-time Processing**: Efficient inference architectures

### Neuroscience Research
- **Brain-Inspired Networks**: Biologically plausible connectivity
- **Modular Organization**: Functional specialization modeling
- **Small-World Properties**: Optimal information flow patterns

### Machine Learning
- **Architecture Search**: Automated neural network design
- **Robustness Engineering**: Adversarial attack resistance
- **Efficiency Optimization**: Performance vs. complexity trade-offs

## 🧪 Experiment Management

### Reproducibility Features

- **Deterministic Execution**: Fixed random seeds and deterministic operations
- **Result Caching**: Automatic saving of intermediate results
- **Configuration Versioning**: Complete experiment parameter tracking
- **Logging**: Comprehensive experiment logging and debugging

### Experiment Tracking

```python
# Create experiment logger
logger = create_experiment_logger("my_experiment", "outputs")

# Track experiment parameters
logger.info(f"Starting experiment with config: {config}")

# Save results
save_results(results, "experiment_results", "outputs")

# Load previous results
previous_results = load_results("outputs/experiment_results.json")
```

## 🚧 Advanced Usage

### Custom Graph Generation

```python
# Parameterized graph generation
parameter_ranges = {
    'units': [32, 64, 128],
    'n_modules': [2, 4, 8],
    'rewiring_prob': [0.1, 0.3, 0.5]
}

graphs = graph_generator.generate_parameterized_graphs(parameter_ranges)
```

### Custom Topology Metrics

```python
# Extend topology analyzer
class CustomTopologyAnalyzer(TopologyAnalyzer):
    def _compute_custom_metric(self, graph):
        # Implement custom metric
        return custom_value
```

### Integration with External Tools

```python
# Export to NetworkX format
nx.write_graphml(graph, "architecture.graphml")

# Import from external sources
external_graph = nx.read_graphml("external_architecture.graphml")
architecture = converter.convert_graph_to_wiredcfc(external_graph)
```

## 📚 API Reference

### Core Classes

#### `ModularSmallWorldGraphGenerator`
- `generate_candidate_graphs(num_candidates)`: Generate random candidate graphs
- `generate_parameterized_graphs(parameter_ranges)`: Systematic parameter variation
- `analyze_graph_properties(graph)`: Basic graph property analysis

#### `TopologyAnalyzer`
- `analyze_graph(graph)`: Comprehensive topological analysis
- `analyze_graph_batch(graphs)`: Batch analysis for efficiency
- `compute_robustness_score(metrics)`: Composite robustness scoring

#### `MultiObjectiveOptimizer`
- `optimize(n_trials, timeout, n_jobs)`: Run optimization process
- `get_best_solutions(n_solutions)`: Retrieve top-ranked solutions
- `plot_optimization_history(save_path)`: Generate optimization plots

#### `WiredCfCConverter`
- `convert_graph_to_wiredcfc(graph, input_size, output_size)`: Convert to WiredCfC
- `create_wiredcfc_model(architecture)`: Create PyTorch model
- `validate_architecture(architecture)`: Architecture validation

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and paths are correct
2. **Memory Issues**: Reduce `num_candidates` or `n_trials` for large-scale experiments
3. **Optimization Failures**: Check constraint parameters and metric computation
4. **Visualization Errors**: Ensure matplotlib and seaborn are properly installed

### Debug Mode

```python
# Enable debug logging
logger = setup_logging(level="DEBUG")

# Validate configuration
errors = validate_config(config)
if errors:
    print(f"Configuration errors: {errors}")
```

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**
3. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8
   ```
4. **Run tests and linting**:
   ```bash
   pytest tests/
   black .
   flake8 .
   ```

### Code Style

- **Python**: PEP 8 compliance with Black formatting
- **Documentation**: Comprehensive docstrings and type hints
- **Testing**: Unit tests for all major components
- **Logging**: Structured logging throughout the system

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **NCPs Library**: For the CfC implementation foundation
- **Optuna**: For the optimization framework
- **NetworkX**: For graph analysis capabilities
- **Research Community**: For inspiration and feedback

## 📞 Support

For questions, issues, or contributions:

1. **GitHub Issues**: Report bugs and request features
2. **Discussions**: Join community discussions
3. **Documentation**: Check the comprehensive documentation
4. **Examples**: Review the demo scripts and examples

---

**Note**: This is a research-oriented project. For production use, additional testing, validation, and performance optimization may be required.
