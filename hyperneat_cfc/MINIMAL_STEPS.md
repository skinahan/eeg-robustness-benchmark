# Minimal Steps for HyperNEAT CfC Proof-of-Concept

This document outlines the minimal steps required to implement a proof-of-concept for using HyperNEAT with CfC networks for EEG classification.

## Overview

The goal is to use HyperNEAT's indirect encoding approach to evolve CfC (Closed-form Continuous) networks that can classify EEG signals while being resilient to noise. The key insight is that HyperNEAT can leverage geometric relationships between CfC cells to create sophisticated connection patterns.

## Core Components (Minimal Implementation)

### 1. CfC Substrate (`cfc_substrate.py`)
**Purpose**: Define the geometric layout of CfC cells in 2D space

**Minimal Requirements**:
- Define cell positions in 2D coordinates
- Support different layout types (grid, circular, hierarchical)
- Map cell IDs to coordinates for CPPN evaluation

**Key Methods**:
```python
class CfCSubstrate:
    def __init__(self, input_size, hidden_size, output_size, layout_type)
    def get_cell_coordinates(self) -> Dict[int, Tuple[float, float]]
    def visualize_layout(self, save_path=None)
```

### 2. HyperNEAT Genome (`hyperneat_genome.py`)
**Purpose**: Encode CPPNs that generate connection patterns

**Minimal Requirements**:
- Represent CPPN as nodes and connections
- Support basic genetic operations (mutation, crossover)
- Evaluate CPPN with coordinate inputs to output connection weights

**Key Methods**:
```python
class HyperNEATGenome:
    def evaluate_cppn(self, inputs: List[float]) -> float
    def mutate(self, mutation_rate: float)
    def crossover(self, other: HyperNEATGenome) -> HyperNEATGenome
```

### 3. CfC Phenotype (`cfc_phenotype.py`)
**Purpose**: Convert genomes into actual CfC networks

**Minimal Requirements**:
- Generate connection matrix using CPPN
- Create WiredCfC network using ncps framework
- Handle connection thresholding for sparse networks

**Key Methods**:
```python
class CfCPhenotype:
    def develop(self, genome: HyperNEATGenome) -> HyperNEATCfC
    def _generate_connections(self, genome) -> Dict[Tuple[int, int], float]
```

### 4. Fitness Evaluator (`fitness_evaluator.py`)
**Purpose**: Evaluate CfC networks on EEG classification tasks

**Minimal Requirements**:
- Train CfC networks on EEG data
- Test noise resilience (dropout, Gaussian, EOG artifacts)
- Calculate multi-objective fitness (accuracy, resilience, complexity)

**Key Methods**:
```python
class HyperNEATFitnessEvaluator:
    def evaluate_genome(self, genome: HyperNEATGenome) -> Dict[str, float]
    def _evaluate_noise_resilience(self, model) -> float
```

### 5. Evolution Engine (`hyperneat_engine.py`)
**Purpose**: Orchestrate the complete evolution process

**Minimal Requirements**:
- Population management
- Selection and reproduction
- Evolution statistics tracking
- Checkpointing and visualization

**Key Methods**:
```python
class HyperNEATEvolutionEngine:
    def evolve(self) -> Tuple[HyperNEATGenome, List[Dict]]
    def _evaluate_population(self)
    def _create_next_generation(self)
```

## Minimal Working Example

```python
# 1. Create substrate
substrate = CfCSubstrate(input_size=22, hidden_size=32, output_size=2)

# 2. Create fitness evaluator with EEG data
evaluator = HyperNEATFitnessEvaluator(substrate, train_data, val_data, test_data)

# 3. Create evolution engine
engine = HyperNEATEvolutionEngine(substrate, evaluator, population_size=20, generations=50)

# 4. Run evolution
best_genome, history = engine.evolve()

# 5. Develop best genome into network
phenotype = CfCPhenotype(substrate)
best_model = phenotype.develop(best_genome)
```

## Key Innovations

### 1. Geometric Connection Patterns
- CPPN takes cell coordinates as input
- Generates connection weights based on spatial relationships
- Enables discovery of geometrically-aware patterns

### 2. CfC Integration
- Uses ncps framework for WiredCfC networks
- Leverages CfC's closed-form solution for efficiency
- Maintains temporal processing capabilities

### 3. Noise Resilience Focus
- Evaluates networks under multiple noise conditions
- Balances accuracy with robustness
- Targets real-world EEG challenges

## Minimal Steps for Proof-of-Concept

### Step 1: Basic Substrate (1-2 days)
- [x] Implement CfCSubstrate with grid layout
- [x] Add cell coordinate mapping
- [x] Create visualization method

### Step 2: CPPN Genome (2-3 days)
- [x] Implement HyperNEATGenome with basic CPPN structure
- [x] Add mutation and crossover operations
- [x] Implement CPPN evaluation with coordinate inputs

### Step 3: Phenotype Development (2-3 days)
- [x] Create CfCPhenotype to convert genomes to networks
- [x] Generate connection matrices using CPPN
- [x] Integrate with ncps WiredCfC framework

### Step 4: Fitness Evaluation (3-4 days)
- [x] Implement HyperNEATFitnessEvaluator
- [x] Add EEG training and evaluation
- [x] Implement noise resilience testing
- [x] Create multi-objective fitness function

### Step 5: Evolution Engine (2-3 days)
- [x] Implement HyperNEATEvolutionEngine
- [x] Add population management and selection
- [x] Create evolution statistics tracking
- [x] Add checkpointing and visualization

### Step 6: Integration and Testing (2-3 days)
- [x] Create demo script with synthetic data
- [x] Test complete pipeline
- [x] Add visualization tools
- [x] Create comprehensive documentation

## Expected Outcomes

### 1. Functional System
- Complete HyperNEAT CfC evolution pipeline
- Working integration with ncps framework
- Multi-objective fitness evaluation

### 2. Proof-of-Concept Results
- Evolution of CfC networks for EEG classification
- Demonstration of geometric connection patterns
- Noise resilience improvements over baseline

### 3. Research Foundation
- Framework for exploring CfC architectures
- Platform for noise resilience research
- Foundation for more advanced HyperNEAT applications

## Technical Challenges and Solutions

### Challenge 1: CPPN Design
**Issue**: Designing effective CPPN architecture for CfC connection patterns
**Solution**: Start with simple architectures (4 inputs, 8 hidden, 1 output) and evolve complexity

### Challenge 2: Connection Thresholding
**Issue**: CPPN may generate too many connections
**Solution**: Apply thresholding to create sparse, efficient networks

### Challenge 3: Evaluation Speed
**Issue**: Training CfC networks is computationally expensive
**Solution**: Use early stopping, smaller datasets for initial exploration

### Challenge 4: ncps Integration
**Issue**: Integrating with ncps framework for custom wirings
**Solution**: Create CustomWiring class that implements ncps interface

## Future Extensions

### 1. Advanced Substrates
- 3D geometric layouts
- Hierarchical multi-scale patterns
- Task-specific substrate designs

### 2. Enhanced CPPNs
- Modular CPPN architectures
- Multi-output CPPNs for different connection types
- Adaptive CPPN complexity

### 3. Real-world Integration
- Integration with MOABB datasets
- Cross-subject and cross-session evaluation
- Real-time adaptation capabilities

## Conclusion

This proof-of-concept demonstrates that HyperNEAT can be effectively applied to CfC networks for EEG classification. The key insight is leveraging geometric relationships to create sophisticated connection patterns that are both accurate and noise-resilient.

The minimal implementation provides a solid foundation for exploring more advanced applications, including 3D substrates, modular evolution, and real-time adaptation. 