# Neuroevolution Framework for EEG Architecture Search

This module provides a complete neuroevolution framework for automatically designing EEG classification architectures that balance model complexity, overfitting, and noise resilience.

## Overview

The neuroevolution framework uses genetic algorithms to evolve neural network architectures specifically designed for EEG signal classification. It optimizes across multiple objectives:

- **Accuracy**: Classification performance on clean data
- **Noise Resilience**: Performance under various noise conditions (dropout, Gaussian, EOG artifacts)
- **Model Complexity**: Parameter count and architectural complexity
- **Overfitting Prevention**: Balance between model capacity and generalization

## Key Features

- **Flexible Architecture Representation**: Supports various layer types (Conv1D, Conv2D, LSTM, GRU, CfC, NCP, Attention, etc.)
- **Multi-Objective Optimization**: Balances accuracy, noise resilience, complexity, and overfitting
- **Multiple Evaluation Strategies**: Fast screening, full evaluation, and Pareto-optimal search
- **Integration with Existing Framework**: Works with your current MOABB evaluation pipeline
- **Comprehensive Analysis**: Evolution plots, population statistics, and detailed results

## Quick Start

### Basic Usage

```python
from neuroevolution import NeuroevolutionEngine

# Create evolution engine
engine = NeuroevolutionEngine(
    population_size=20,
    generations=10,
    evaluator_type='fast',  # Use fast evaluator for quick testing
    output_dir='results'
)

# Run evolution
best_genome, history = engine.evolve()

# Create model from best genome
from neuroevolution import NeuroevolutionModelBuilder
classifier = NeuroevolutionModelBuilder.create_classifier(best_genome)
```

### Demo Script

Run the demonstration script to see the framework in action:

```bash
python run_neuroevolution_demo.py --mode demo
```

This will:
- Run a quick evolution (15 individuals, 8 generations)
- Use the fast evaluator for quick results
- Save results to `neuroevolution_demo_results/`
- Display the best architecture found

## Architecture Genome

The framework uses a flexible genome representation that encodes:

### Layer Types Supported
- **Convolutional**: Conv1D, Conv2D
- **Recurrent**: LSTM, GRU, CfC, NCP
- **Attention**: Multi-head attention
- **Pooling**: Max/Average pooling
- **Regularization**: Dropout, BatchNorm
- **Fully Connected**: Linear layers

### Genome Structure

```python
from neuroevolution import ArchitectureGenome, LayerConfig

# Example genome
genome = ArchitectureGenome(
    layers=[
        LayerConfig(
            layer_type='conv1d',
            in_channels=22,
            out_channels=64,
            kernel_size=(7,),
            activation='elu',
            dropout_rate=0.2,
            batch_norm=True
        ),
        LayerConfig(
            layer_type='lstm',
            in_channels=64,
            out_channels=128,
            hidden_size=128,
            dropout_rate=0.3
        ),
        LayerConfig(
            layer_type='fc',
            in_channels=128,
            out_channels=2,
            activation='linear'
        )
    ],
    learning_rate=0.001,
    weight_decay=0.001,
    batch_size=64,
    dropout_rate=0.15
)
```

## Evolution Strategies

### 1. Standard Evolution
Uses weighted multi-objective fitness function:

```python
from neuroevolution import NeuroevolutionEngine

engine = NeuroevolutionEngine(
    population_size=20,
    generations=15,
    evaluator_type='full',  # Full evaluation
    mutation_rate=0.1,
    crossover_rate=0.8
)
```

### 2. Multi-Objective Evolution
Uses Pareto-optimal selection:

```python
from neuroevolution import MultiObjectiveEvolutionEngine

engine = MultiObjectiveEvolutionEngine(
    population_size=30,
    generations=20,
    evaluator_type='pareto'
)
```

### 3. Fast Screening
For initial exploration and testing:

```python
from neuroevolution import NeuroevolutionEngine

engine = NeuroevolutionEngine(
    population_size=15,
    generations=8,
    evaluator_type='fast'  # Fast heuristic evaluation
)
```

## Evaluation Types

### Fast Evaluator
- Uses heuristics based on architecture properties
- No actual training required
- Good for initial screening and testing
- Evaluates in seconds rather than minutes

### Full Evaluator
- Trains models on actual data
- Tests noise resilience with multiple noise types
- Most accurate but slowest
- Best for final optimization

### Pareto Evaluator
- Supports true multi-objective optimization
- Maintains Pareto front of non-dominated solutions
- Good for exploring trade-offs between objectives

## Genetic Operators

### Mutation Operations
- **Layer Parameters**: Channel dimensions, kernel sizes, activation functions
- **Architecture Structure**: Add/remove layers, swap layers, change layer types
- **Training Parameters**: Learning rate, weight decay, batch size
- **Regularization**: Dropout rates, batch normalization

### Crossover Operations
- **Layer Crossover**: Exchange layer configurations between parents
- **Parameter Crossover**: Exchange training hyperparameters
- **Regularization Crossover**: Exchange regularization settings

## Integration with Existing Framework

The neuroevolution framework integrates seamlessly with your existing evaluation pipeline:

```python
from neuroevolution import NeuroevolutionModelBuilder
from evaluation.run_experiment import run_evaluation

# Create model from evolved genome
genome = load_best_genome('results/best_genome_final.json')
classifier = NeuroevolutionModelBuilder.create_classifier(genome)

# Use with existing evaluation
results = run_evaluation(
    model_fn=lambda: classifier,
    model_name='neuroevolution_model',
    noise_type='dropout',
    intensity=25.0
)
```

## Output and Analysis

### Generated Files
- `best_genome_final.json`: Best architecture genome
- `evolution_plots.png`: Evolution progress visualization
- `population_summary.csv`: Final population statistics
- `evolution_history.json`: Complete evolution history

### Analysis Tools

```python
# Analyze a saved genome
python run_neuroevolution_demo.py --mode analyze --genome results/best_genome_final.json

# Compare different evolution strategies
python run_neuroevolution_demo.py --mode comparison
```

## Advanced Usage

### Custom Fitness Function

```python
from neuroevolution import FitnessEvaluator

class CustomFitnessEvaluator(FitnessEvaluator):
    def _calculate_overall_fitness(self, accuracy, noise_resilience, complexity_score, overfitting_score):
        # Custom weighting scheme
        weights = {
            'accuracy': 0.5,
            'noise_resilience': 0.3,
            'complexity': 0.15,
            'overfitting': 0.05
        }
        
        complexity_penalty = 1.0 - complexity_score
        
        fitness = (
            weights['accuracy'] * accuracy +
            weights['noise_resilience'] * noise_resilience +
            weights['complexity'] * complexity_penalty +
            weights['overfitting'] * overfitting_score
        )
        
        return fitness
```

### Custom Genetic Operators

```python
from neuroevolution import GeneticOperators

class CustomGeneticOperators(GeneticOperators):
    def _mutate_architecture_structure(self, genome):
        # Custom mutation logic
        # Add your specific mutation operations here
        pass
```

## Performance Tips

1. **Start with Fast Evaluator**: Use `evaluator_type='fast'` for initial exploration
2. **Small Population**: Start with 10-15 individuals for quick testing
3. **Fewer Generations**: 5-10 generations often sufficient for initial results
4. **Parallel Evaluation**: The framework supports parallel evaluation (set `n_jobs` in evaluator)
5. **Save Intermediate Results**: Checkpoints every 5 generations

## Troubleshooting

### Common Issues

1. **Invalid Genomes**: Some randomly generated architectures may be invalid
   - Solution: The framework automatically filters invalid genomes
   - Check genome validation in `NeuroevolutionModelBuilder.validate_genome()`

2. **Memory Issues**: Large populations or complex models may cause memory problems
   - Solution: Reduce population size or use fast evaluator
   - Monitor parameter count in genome.get_parameter_count()

3. **Slow Evaluation**: Full evaluation can be very slow
   - Solution: Use fast evaluator for initial screening
   - Reduce max_epochs in evaluator configuration

### Debugging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test genome validation
from neuroevolution import GenomeGenerator, NeuroevolutionModelBuilder
generator = GenomeGenerator()
genome = generator.generate_random_genome()
is_valid = NeuroevolutionModelBuilder.validate_genome(genome)
print(f"Genome valid: {is_valid}")
```

## Examples

See `run_neuroevolution_demo.py` for complete examples of:
- Basic evolution
- Comparison experiments
- Genome analysis
- Integration with existing framework

## Citation

If you use this neuroevolution framework in your research, please cite:

```bibtex
@article{eeg_neuroevolution_2024,
  title={Neuroevolution for EEG Architecture Search: Balancing Complexity, Overfitting, and Noise Resilience},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
``` 