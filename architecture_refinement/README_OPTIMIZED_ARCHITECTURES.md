# Optimized Architecture Integration

This directory contains components for integrating optimized architectures from the architecture search process with the existing EEG evaluation framework.

## Overview

The architecture search process in `demo.py` generates optimized WS-flex graph architectures that are saved as JSON files in `outputs/architectures/`. This system allows you to:

1. **Load optimized architectures** from the search outputs
2. **Create CNNWiredCfC models** that use these architectures
3. **Test the models** on EEG data
4. **Integrate with the existing evaluation framework**

## Components

### 1. ArbitraryWiring (`arbitrary_wiring.py`)

A wiring class that can ingest arbitrary WS-flex graphs from architecture search and convert them to NCP-compatible wiring.

**Key Features:**
- Loads wiring matrices from JSON files
- Validates architecture specifications
- Creates NCP-compatible wiring structures
- Provides wiring summaries and statistics

**Usage:**
```python
from architecture_refinement.arbitrary_wiring import load_architecture_from_file

# Load an architecture file
wiring = load_architecture_from_file("path/to/architecture.json")

# Get wiring summary
summary = wiring.get_wiring_summary()
print(f"Connections: {summary['total_connections']}")
```

### 2. CNNWiredCfC Model (`models/cnnncp.py`)

A new model class that uses arbitrary wiring configurations. Similar to `CNNSmallWorld` but uses `ArbitraryWiring` instead of `ModularSmallWorldWiring`.

**Key Features:**
- Uses optimized architectures from search
- Compatible with existing evaluation framework
- Configurable CNN and CfC parameters
- Provides wiring information access

**Usage:**
```python
from models.cnnncp import create_cnnwiredcfc_classifier
from architecture_refinement.arbitrary_wiring import load_architecture_from_file

# Load architecture and create model
wiring = load_architecture_from_file("architecture.json")
model = create_cnnwiredcfc_classifier(
    n_chans=22, n_times=1000, n_outputs=2, wiring=wiring
)
```

### 3. Testing Script (`test_optimized_architectures.py`)

A comprehensive testing script that loads all architectures and tests them on EEG data.

**Features:**
- Loads all architecture files from `outputs/architectures/`
- Validates architecture specifications
- Tests each architecture on EEG data
- Provides detailed results and statistics

**Usage:**
```bash
# Test all architectures on subject 1
python test_optimized_architectures.py --subject-id 1

# Test with verbose logging
python test_optimized_architectures.py --verbose

# Test with custom test split
python test_optimized_architectures.py --test-size 0.2
```

### 4. Integration Script (`integrate_with_evaluation.py`)

Utilities for integrating the new models with the existing evaluation framework.

**Features:**
- Creates model factories from architecture files
- Lists available architectures
- Provides integration examples
- Creates architecture registries

**Usage:**
```python
from architecture_refinement.integrate_with_evaluation import (
    create_model_factory_from_architecture,
    list_available_architectures
)

# List available architectures
architectures = list_available_architectures()

# Create a model factory
factory = create_model_factory_from_architecture("architecture.json")
```

## Quick Start

### 1. Test a Single Architecture

```python
from architecture_refinement.arbitrary_wiring import load_architecture_from_file
from models.cnnncp import create_cnnwiredcfc_classifier

# Load architecture
wiring = load_architecture_from_file("outputs/architectures/best_architecture_1_trial_1.json")

# Create model
model = create_cnnwiredcfc_classifier(
    n_chans=22, n_times=1000, n_outputs=2, wiring=wiring
)

# Use the model
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
```

### 2. Test All Architectures

```bash
cd architecture_refinement
python test_optimized_architectures.py --verbose
```

### 3. Explore Available Architectures

```bash
cd architecture_refinement
python integrate_with_evaluation.py
```

## Integration with Evaluation Framework

### Option 1: Direct Model Creation

Use the models directly in your evaluation scripts:

```python
from architecture_refinement.arbitrary_wiring import load_architecture_from_file
from models.cnnncp import create_cnnwiredcfc_classifier

def evaluate_architecture(architecture_file, X_train, y_train, X_test, y_test):
    wiring = load_architecture_from_file(architecture_file)
    model = create_cnnwiredcfc_classifier(
        n_chans=X_train.shape[1], 
        n_times=X_train.shape[2], 
        n_outputs=len(np.unique(y_train)), 
        wiring=wiring
    )
    
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)
```

### Option 2: Model Registry Integration

Add the models to the existing model registry in `config.py`:

```python
# In config.py
from architecture_refinement.integrate_with_evaluation import create_model_factory_from_architecture

def get_model_registry():
    return {
        # ... existing models ...
        "wiredcfc_arch1": create_model_factory_from_architecture("outputs/architectures/best_architecture_1_trial_1.json"),
        "wiredcfc_arch2": create_model_factory_from_architecture("outputs/architectures/best_architecture_2_trial_6.json"),
    }
```

Then use them like any other model:

```bash
python evaluation/run_cross_session_experiments.py --model wiredcfc_arch1
```

## Architecture File Format

The architecture JSON files contain:

```json
{
  "input_size": 8,
  "hidden_size": 4,
  "output_size": 5,
  "wiring_matrix": [[...], [...], ...],
  "neuron_types": ["sensory", "sensory", ..., "motor", "motor"],
  "layer_sizes": [8, 4, 5],
  "connection_weights": [[...], [...], ...],
  "metadata": {...}
}
```

**Required Fields:**
- `input_size`: Number of input features
- `hidden_size`: Number of hidden neurons
- `output_size`: Number of output classes
- `wiring_matrix`: Connection matrix (N×N where N = input + hidden + output)

**Optional Fields:**
- `neuron_types`: List of neuron types
- `connection_weights`: Connection weight matrix
- `metadata`: Additional architecture information

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the project root is in your Python path
2. **Architecture Validation**: Check that architecture files have the correct format
3. **Memory Issues**: Some architectures may be large; consider using smaller batch sizes
4. **CUDA Issues**: Models will fall back to CPU if CUDA is not available

### Debugging

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check architecture validity:

```python
from architecture_refinement.arbitrary_wiring import load_architecture_from_file

try:
    wiring = load_architecture_from_file("architecture.json")
    print("Architecture loaded successfully")
    print(wiring.get_wiring_summary())
except Exception as e:
    print(f"Error loading architecture: {e}")
```

## Performance Considerations

- **Large Architectures**: Very dense wiring matrices may impact training speed
- **Memory Usage**: Monitor memory usage when loading multiple architectures
- **Training Time**: Complex architectures may require longer training times
- **Batch Size**: Adjust batch size based on architecture complexity

## Future Enhancements

- **Architecture Selection**: Automated selection of best architectures
- **Hyperparameter Tuning**: Architecture-specific hyperparameter optimization
- **Ensemble Methods**: Combining multiple architectures
- **Architecture Evolution**: Continuous improvement of architectures

## Examples

See the `examples/` directory for additional usage examples and case studies.
