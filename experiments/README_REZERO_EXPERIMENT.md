# ReZero Initialization Strategy Comparison Experiment

## Purpose

This experiment compares two ReZero initialization strategies for the BranchedWiredCfC Architecture 4 model:

1. **backwards_rezero**: Current (accidental) implementation where recurrent compartment starts at full strength
2. **correct_rezero**: Correct ReZero implementation where residual/identity passes through at initialization

The goal is to determine if the promising robustness results are connected to the accidentally backwards initialization strategy.

## Background

See `REZERO_INITIALIZATION_ANALYSIS.md` for detailed analysis of the initialization issue.

## Usage

### Basic Usage

```bash
python experiments/compare_rezero_init_strategies.py --dataset BNCI2014_001 --train
```

### With Custom Architecture File

```bash
python experiments/compare_rezero_init_strategies.py \
    --dataset BNCI2014_001 \
    --architecture outputs/architectures/best_architecture_4_trial_178.json \
    --train
```

### Multiple Seeds for Robustness

```bash
python experiments/compare_rezero_init_strategies.py \
    --dataset BNCI2014_001 \
    --seeds 42 123 456 789 1011 \
    --train
```

### Load Pre-trained Models

If you've already trained models and saved them:

```bash
python experiments/compare_rezero_init_strategies.py \
    --dataset BNCI2014_001 \
    --model-dir models/rezero_comparison \
    --output rezero_comparison_report.md
```

## Command-Line Arguments

- `--dataset`: Dataset to use (default: `BNCI2014_001`)
- `--architecture`: Path to Architecture 4 JSON file (default: auto-detect)
- `--seeds`: Space-separated list of random seeds (default: `42 123 456`)
- `--train`: Train new models (default: False, will try to load from `--model-dir`)
- `--model-dir`: Directory to save/load model checkpoints
- `--output`: Output report file path (default: `rezero_comparison_report.md`)

## What the Experiment Does

1. **Loads Architecture 4**: Loads the wiring configuration from the specified JSON file
2. **Creates Two Model Variants**: 
   - One with `residual_init_strategy="backwards_rezero"`
   - One with `residual_init_strategy="correct_rezero"`
3. **Trains Both Models**: On the same training data for each seed
4. **Evaluates Robustness**: Tests both models under three noise types:
   - EOG artifacts
   - Gaussian noise
   - Channel dropout
5. **Compares Results**: Generates a detailed comparison report

## Output Files

The experiment generates several output files:

1. **Markdown Report** (`rezero_comparison_report.md`): Human-readable comparison report
2. **CSV Files** (in `rezero_comparison_data/`):
   - `backwards_rezero_results.csv`: Detailed results for backwards ReZero
   - `correct_rezero_results.csv`: Detailed results for correct ReZero

## Interpreting Results

The report includes:

- **Summary Statistics**: Mean performance and retention for each strategy
- **Performance by Noise Type**: Breakdown for EOG, Gaussian, and Dropout noise
- **Direct Comparison**: Side-by-side comparison at key intensity levels
- **Overall Robustness Comparison**: Summary of which strategy is more robust

Key metrics:
- **Retention**: Percentage of clean performance maintained under noise
- **ROC-AUC**: Area under ROC curve (higher is better)
- **Performance Drop**: Absolute difference between clean and corrupted performance

## Expected Duration

The experiment can take several hours depending on:
- Number of seeds (default: 3)
- Dataset size
- Hardware (CPU/GPU)
- Training time per model

Each model needs to:
1. Train on clean data (~30-60 minutes per model)
2. Evaluate on 60 noise conditions (3 noise types × 20 intensities) (~10-20 minutes)

**Total estimate**: ~3-6 hours for default configuration (3 seeds × 2 strategies)

## Troubleshooting

### Architecture File Not Found

If you see an error about the architecture file, specify it explicitly:

```bash
--architecture outputs/architectures/best_architecture_4_trial_178.json
```

### Dimension Mismatches

If you get dimension errors, check that:
- Dataset dimensions match expected values
- Architecture wiring matrix matches model input/output sizes
- Resampling rate is consistent (default: 512 Hz)

### Out of Memory

If you run out of memory:
- Reduce batch size in model creation (modify script or model defaults)
- Test with fewer seeds
- Use a smaller dataset subset

## Next Steps

After reviewing the results:

1. **If backwards ReZero is more robust**: Consider keeping the current implementation or further investigating why
2. **If correct ReZero is more robust**: Proceed with implementing Fix Option 2 (swap formula)
3. **If both are similar**: Either strategy works, but correct ReZero aligns with literature

## Implementation Details

The experiment uses:
- `branched_diva_base.py`: Base class with `residual_init_strategy` parameter
- `branched_wiredcfc.py`: Model class that accepts and passes through the strategy
- `augmentation/noise.py`: For applying noise perturbations
- `utils.py`: For noise intensity bounds and metrics

## Related Files

- `REZERO_INITIALIZATION_ANALYSIS.md`: Detailed analysis of the issue
- `models/branched_diva_base.py`: Implementation with both strategies
- `models/branched_wiredcfc.py`: Model using the base class
