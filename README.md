# Architecture Refinement for Robustness-Aware CfC Networks

A comprehensive system for optimizing neural network architectures using graph-theoretic metrics and multi-objective optimization. This system generates candidate wiring graphs using modular small-world strategies, evaluates them using advanced topological analysis, and optimizes them for multiple objectives including robustness, modularity, redundancy, and interpretability.

## Key Features

- **Modular Small-World Graph Generation**: Creates diverse network architectures with balanced local clustering and global connectivity
- **7-Objective Optimization**: Optimizes for entropy, curvature, connectivity, efficiency, modularity, redundant pathways, and interpretability
- **Advanced Topology Analysis**: Computes 20+ graph-theoretic metrics including Ricci curvature, algebraic connectivity, and community structure
- **Multi-Objective Optimization**: Uses Optuna for efficient Pareto-optimal solution discovery
- **WiredCfC Integration**: Converts optimized graphs to WiredCfC-compatible architectures
- **Comprehensive Visualization**: Generates detailed plots for optimization history, Pareto fronts, and metric distributions
- **Reproducible Research**: Built-in logging, configuration management, and result caching

## Extended Optimization Objectives

Beyond traditional robustness metrics, this system now optimizes for:

### 1. **Modularity** (Regional/Functional Subnetworks)
- **Newman-Girvan Modularity**: Measures the strength of community structure
- **Community Detection**: Identifies functional subnetworks using Louvain method
- **Intra/Inter-community Density**: Balances local vs. global connectivity
- **Community Size Distribution**: Ensures balanced module sizes

### 2. **Redundant-but-Diverse Pathways**
- **Pathway Coverage**: Measures how many sensory-motor pairs have multiple routes
- **Path Redundancy**: Counts alternative pathways between nodes
- **Path Diversity**: Quantifies how different alternative routes are
- **Sensory-Motor Connectivity**: Ensures robust information flow from input to output

### 3. **Interpretability & Structural Clarity**
- **Degree Regularity**: Measures structural consistency
- **Hierarchical Structure**: Identifies organizational patterns
- **Symmetry Analysis**: Detects structural balance
- **Clustering Hierarchy**: Analyzes multi-scale organization

## System Architecture

```
 
 Graph Generator Topology Analyzer Multi-Objective 
 Optimizer 
 
 
 
 
 Metric Summary Pareto Front 
 & Statistics & Best Solutions
 
 
 
 
 WiredCfC 
 Converter 
 
```

## Graph-Theoretic Metrics

### Core Robustness Metrics
- **Topological Entropy**: Degree, weight, and path length distribution diversity
- **Ricci Curvature**: Ollivier-Ricci and Forman-Ricci curvature for edge robustness
- **Algebraic Connectivity**: Second smallest Laplacian eigenvalue for network robustness
- **Global/Local Efficiency**: Information flow efficiency measures

### New Extended Metrics
- **Modularity**: Newman-Girvan modularity, community structure analysis
- **Redundancy**: Pathway coverage, alternative route analysis
- **Interpretability**: Structural regularity, hierarchical organization

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd eeg-noise-robustness/architecture_refinement

# Install dependencies
pip install -r requirements.txt
```

## Measuring Robustness (test_perturb)

The `test_perturb` evaluation mode provides a standardized approach to measuring neural network robustness to EEG corruptions:

### Key Principles

- **Clean Training**: Train and validate on clean data only
- **Test-Time Corruption**: Apply corruptions only to the test split
- **Deterministic Evaluation**: Use deterministic seeding for reproducible results
- **Comprehensive Metrics**: Compute AURC, worst-case performance, and relative drops

### Usage Examples

**Basic robustness evaluation:**
```bash
python evaluation/run_experiments.py \
 --model cnncfc_v2 \
 --mode test_perturb \
 --subjects 1 2 3 \
 --seed 42
```

**Custom corruption families:**
```bash
python evaluation/run_experiments.py \
 --model cnncfc_v2 \
 --mode test_perturb \
 --subjects 1 \
 --seed 42 \
 --families gaussian,eog \
 --intensities gaussian:20.0,50.0,eog:30.0,60.0
```

**Cross-session evaluation:**
```bash
python evaluation/run_cross_session_experiments.py \
 --model cnncfc_v2 \
 --mode test_perturb \
 --subjects 1 \
 --seed 42
```

### Output Structure

Each evaluation run produces:
- **Per-example CSV**: Individual corruption results with metadata
- **Summary CSV**: Per-family AURC and robustness metrics 
- **Metadata files**: Corruption plan, grid configuration, and seed mappings
- **Human-readable summary**: Overview of the evaluation setup

### Robustness Metrics

- **AURC (Area Under Robustness Curve)**: Overall robustness across intensity levels
- **Worst-case Performance**: Minimum performance under any corruption
- **Relative Performance Drop**: `(clean - corrupted) / clean`
- **Per-family Analysis**: Separate metrics for each corruption type

### Corruption Families

The system supports multiple corruption types:
- **Gaussian**: Additive noise with varying standard deviations
- **Dropout**: Random channel dropout with percentage scaling
- **EOG/EMG**: Simulated eye and muscle artifacts
- **Line Noise**: Power line interference (50/60 Hz)
- **Drift**: Baseline drift and slow variations

### Important Notes

- **Robustness Claims**: Only results from `test_perturb` mode should be used for architectural robustness claims
- **Training Regimes**: Other modes (augment, perturb) are for ablation studies, not robustness measurement
- **Reproducibility**: All corruptions use deterministic seeding for consistent evaluation
- **Standardization**: The same corruption grid is applied across all models for fair comparison

### 2. Basic Usage

```python
from architecture_refinement import (
 Config, ModularSmallWorldGraphGenerator, 
 TopologyAnalyzer, MultiObjectiveOptimizer
)

# Initialize configuration
config = Config()

# Generate candidate graphs
generator = ModularSmallWorldGraphGenerator(config.graph_generation)
candidate_graphs = generator.generate_candidate_graphs(num_candidates=100)

# Analyze topology
analyzer = TopologyAnalyzer(config)
topology_metrics = analyzer.analyze_graph_batch([g for g, _ in candidate_graphs])

# Run optimization
optimizer = MultiObjectiveOptimizer(config.optimization, generator, analyzer)
results = optimizer.optimize(n_trials=50)
```

### 3. Run Demo

```bash
# Full demo with 7-objective optimization
python demo.py

# Quick demo for testing
python demo.py --quick
```

### 4. Run Tests

```bash
# Test basic functionality
python test_basic.py
```

## Configuration

The system uses a hierarchical configuration system with the following components:

### Graph Generation
```python
@dataclass
class GraphGenerationConfig:
 min_units: int = 32
 max_units: int = 128
 n_modules: int = 4
 rewiring_prob: float = 0.2
 connection_density: float = 0.5
```

### Optimization
```python
@dataclass
class OptimizationConfig:
 n_trials: int = 100
 timeout: int = 3600
 
 # 7-objective weights
 entropy_weight: float = 0.15
 curvature_weight: float = 0.15
 connectivity_weight: float = 0.15
 efficiency_weight: float = 0.15
 modularity_weight: float = 0.15
 redundancy_weight: float = 0.15
 interpretability_weight: float = 0.10
```

## Optimization Process

### 1. **Graph Generation Phase**
- Generates diverse candidate architectures using modular small-world principles
- Varies parameters: units, modules, connectivity, rewiring probability
- Ensures structural diversity while maintaining biological plausibility

### 2. **Topology Analysis Phase**
- Computes comprehensive set of graph-theoretic metrics
- Includes new metrics for modularity, redundancy, and interpretability
- Provides normalized scores for optimization objectives

### 3. **Multi-Objective Optimization Phase**
- Uses Optuna's TPE sampler for efficient parameter exploration
- Optimizes 7 objectives simultaneously:
 - **Entropy**: Structural diversity
 - **Curvature**: Edge robustness
 - **Connectivity**: Network resilience
 - **Efficiency**: Information flow
 - **Modularity**: Community structure
 - **Redundancy**: Alternative pathways
 - **Interpretability**: Structural clarity

### 4. **Solution Selection Phase**
- Identifies Pareto-optimal solutions
- Ranks solutions by composite robustness score
- Exports best architectures for downstream use

## Research Applications

### Neuroscience & Brain-Inspired Computing
- **Modular Organization**: Design networks with functional subnetworks
- **Redundant Pathways**: Ensure robust information flow
- **Interpretable Structure**: Create understandable architectures

### Robust Machine Learning
- **Noise Resilience**: Optimize for perturbation resistance
- **Structural Stability**: Ensure consistent performance
- **Adaptive Capacity**: Enable graceful degradation

### Network Science
- **Community Detection**: Analyze modular structure
- **Path Analysis**: Study information flow patterns
- **Structural Balance**: Understand organizational principles

## Output Structure

```
outputs/
 logs/ # Experiment logs
 plots/ # Visualization plots
 optimization_history.png # Objective convergence
 pareto_front.png # Pareto optimal solutions
 new_metrics_distributions.png # Extended metrics
 objective_correlations.png # Objective relationships
 parameter_objective_relationships.png
 optimization/ # Optimization results
 extended_demo_optimization_results.json
 architectures/ # Converted architectures
 best_architecture_1.json
 best_architecture_2.json
 best_graphs/ # Top-ranked graphs
 best_graph_1_trial_X.graphml
 best_graph_1_trial_X_params.json
```

## Experiment Management

### Logging & Monitoring
- Rich console output with progress bars
- Structured logging with timestamps
- Performance metrics tracking

### Result Caching
- Automatic result saving/loading
- Configurable output formats
- Reproducible experiment tracking

### Visualization Suite
- Optimization convergence plots
- Pareto front analysis
- Metric distribution histograms
- Parameter-objective relationships

## Advanced Usage

### Custom Metric Development
```python
class CustomTopologyAnalyzer(TopologyAnalyzer):
 def _compute_custom_metric(self, graph):
 # Implement custom metric
 return custom_value
```

### Parameter Space Exploration
```python
# Custom parameter ranges
config.optimization.min_units = 64
config.optimization.max_units = 256
config.optimization.n_modules = 8
```

### Multi-Objective Weighting
```python
# Adjust objective importance
config.optimization.modularity_weight = 0.25
config.optimization.redundancy_weight = 0.25
config.optimization.interpretability_weight = 0.20
```

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce `num_candidates` or `n_trials`
3. **Timeout Errors**: Increase `timeout` parameter
4. **Plotting Errors**: Check matplotlib/seaborn installation

### Performance Optimization
- Use `n_jobs > 1` for parallel optimization
- Reduce `ricci_curvature_samples` for faster analysis
- Enable result caching for repeated experiments

## Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black architecture_refinement/
```

### Adding New Metrics
1. Extend `TopologyAnalyzer` class
2. Add configuration parameters
3. Update optimization objectives
4. Include in visualization suite

## API Reference

### Core Classes

#### `ModularSmallWorldGraphGenerator`
- `generate_candidate_graphs(num_candidates)`: Generate diverse architectures
- `_create_modular_small_world_graph(params)`: Create single architecture

#### `TopologyAnalyzer`
- `analyze_graph(graph)`: Compute all metrics
- `analyze_graph_batch(graphs)`: Batch analysis
- `compute_robustness_score(metrics)`: Composite score

#### `MultiObjectiveOptimizer`
- `optimize(n_trials, timeout, n_jobs)`: Run optimization
- `get_best_solutions(n_solutions)`: Extract top solutions
- `plot_optimization_history()`: Visualization

#### `WiredCfCConverter`
- `convert_graph_to_wiredcfc(graph)`: Convert to WiredCfC format
- `create_wiredcfc_model(architecture)`: Create PyTorch model

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **NetworkX**: Graph analysis and manipulation
- **Optuna**: Hyperparameter optimization framework
- **PyTorch**: Deep learning framework
- **Research Community**: For inspiration and feedback

---

**Note**: This system represents a significant advancement in architecture optimization, combining traditional robustness metrics with novel objectives for modularity, redundancy, and interpretability. The 7-objective optimization framework enables the discovery of architectures that are not only robust but also well-structured and understandable.
