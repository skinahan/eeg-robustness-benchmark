# EEG Experiment Automation System

This automation system provides a comprehensive solution for managing and running EEG experiments at scale. It automatically identifies missing experiments and generates shell scripts to complete your benchmark.

## Features

- **Configuration-driven**: Define all experimental combinations in a YAML configuration file
- **Smart aggregation**: Automatically aggregates results from all datasets and paradigms
- **Missing detection**: Identifies which experimental combinations are still needed
- **Shell script generation**: Creates executable scripts to run missing experiments
- **Progress tracking**: Monitors experiment completion and handles failures
- **Multi-dataset support**: Works with both MotorImagery and SSVEP datasets

## Quick Start

### 1. Configure Your Experiments

Edit `experiment_config.yaml` to define your experimental combinations:

```yaml
# Example configuration
datasets:
  BNCI2014_001:
    name: "BNCI2014_001"
    paradigm: "MotorImagery"
    subjects: [1, 2, 3, 4, 5]  # Adjust as needed

models:
  - name: "eegnet"
  - name: "reegnet"
  - name: "cnn_ncp"

eval_modes:
  - "WithinSession"
  - "CrossSession"

experiment_modes:
  - name: "baseline"
    requires_noise: false
    supports_tuning: true
  - name: "augment"
    requires_noise: true
    supports_tuning: true

noise_types: ["gaussian", "dropout", "eog"]
noise_intensities: [10.0, 20.0, 30.0]
seeds: [100, 200, 300]
```

### 2. Run the Automation

```bash
# Full automation (recommended) - generates sbatch script for cluster
python experiment_automation.py

# Or with custom output directory
python experiment_automation.py --output-dir my_experiments

# For local execution - generates Python script instead of sbatch script
python experiment_automation.py --local

# Or only aggregate existing results
python experiment_automation.py --aggregate-only
```

### 3. Execute Missing Experiments

#### Option A: Cluster Execution (Default)

The automation will generate a shell script with all missing experiments:

```bash
# Make executable and run
chmod +x generated_scripts/run_missing_multirun_jobs.sh
./generated_scripts/run_missing_multirun_jobs.sh
```

#### Option B: Local Execution (with --local flag)

The automation will generate a Python script for local execution:

```bash
# Run directly with Python
python generated_scripts/run_missing_multirun_jobs.py
```

The local execution script includes:
- Progress monitoring with tqdm
- Automatic garbage collection between experiments
- Time tracking for each experiment
- Detailed error reporting
- Memory cleanup after each job

## Configuration Options

### Dataset Configuration
- **datasets**: Define which datasets to use (BNCI2014_001, Lee2019_SSVEP)
- **subjects**: Specify which subjects to include for each dataset

### Model Configuration
- **models**: List of models to test (eegnet, reegnet, cnn_ncp, etc.)
- Models are automatically loaded from your `config.py`

### Experiment Modes
- **baseline**: Standard baseline experiments
- **tune**: Hyperparameter optimization
- **augment**: Data augmentation experiments
- **perturb**: Data perturbation experiments
- **augment_notune**: Augmentation without tuning
- **perturb_notune**: Perturbation without tuning
- **test_perturb**: Test perturbation robustness

### Evaluation Modes
- **WithinSession**: Within-session cross-validation
- **CrossSession**: Cross-session evaluation
- **CrossSubject**: Cross-subject evaluation

### Noise Configuration
- **noise_types**: Types of noise to apply (gaussian, dropout, eog)
- **noise_intensities**: Intensity levels as percentages

## Output Files

The automation system generates several output files:

1. **Shell script** (`run_missing_experiments.sh`): Executable script with missing experiment commands
2. **Summary report** (`missing_experiments_report.csv`): CSV file listing all missing experiments
3. **Aggregated results** (`unified_all_results.csv`): Combined results from all datasets

## Advanced Usage

### Custom Configuration File
```bash
python experiment_automation.py --config my_custom_config.yaml
```

### Only Aggregate Results
```bash
python experiment_automation.py --aggregate-only
```

### Only Identify Missing Experiments
```bash
# For cluster execution (generates bash script)
python experiment_automation.py --missing-only

# For local execution (generates Python script)
python experiment_automation.py --missing-only --local
```

### Using Pre-aggregated Results
If you have already aggregated results and want to save time, you can skip the aggregation step:
```bash
python experiment_automation.py --preaggregated-results path/to/results.csv --local
```

### Local vs Cluster Execution

**Use `--local` when:**
- Running on your local machine or a single server
- You want built-in progress monitoring with tqdm
- You prefer Python-based execution with better error handling
- You want automatic garbage collection between experiments

**Use default (sbatch) when:**
- Running on an HPC cluster with SLURM
- You need parallel job submission
- You want to leverage cluster scheduling and resource allocation

## File Structure

```
├── experiment_config.yaml          # Main configuration file
├── experiment_automation.py        # Main automation script
├── evaluation/
│   ├── experiment_utils.py         # Updated with unified aggregation
│   └── results/                    # Results directory
├── generated_scripts/              # Generated shell scripts
└── README_automation.md           # This file
```

## Troubleshooting

### Common Issues

1. **Configuration file not found**
   - Ensure `experiment_config.yaml` exists in the project root
   - Check file permissions

2. **Python executable path incorrect**
   - Update `python_executable` in the configuration file
   - Use absolute path to your Python interpreter

3. **No results found**
   - Check that your results are in the expected directory structure
   - Ensure CSV files follow the naming convention

4. **Shell script permissions**
   - Run `chmod +x run_missing_experiments.sh` to make executable

### Getting Help

- Check the generated summary report for detailed missing experiment information
- Review the shell script to understand the exact commands being generated
- Use `--aggregate-only` to test result aggregation without generating scripts

## Example Workflow

1. **Initial setup**: Configure your experiments in `experiment_config.yaml`
2. **First run**: `python experiment_automation.py` to see what's missing
3. **Execute experiments**: Run the generated shell script
4. **Re-run automation**: Run automation again to identify remaining missing experiments
5. **Repeat**: Continue until all experiments are complete

This system ensures comprehensive coverage of your experimental design while automatically managing the complexity of running hundreds or thousands of individual experiments.


