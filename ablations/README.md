# HYDRA Model Ablation Studies

This directory contains the experiment design and implementation for ablation studies on the HYDRA model (branched_wiredcfc architecture #4).

## Overview

The ablation studies evaluate the contribution of individual HYDRA model mechanisms by disabling one feature at a time:

1. **No Carry Gate**: Disables the weighted residual connection
2. **No Branching**: Processes the entire sequence in a single temporal bin
3. **Replace CfC with LSTM**: Uses traditional LSTM instead of CfC recurrent cells

## Files

- `experiment_specification.txt`: Detailed experiment specification
- `run_ablations.py`: Main Python script to run all ablation experiments
- `ablation_models.py`: Model variant definitions (No Carry Gate, No Branching)
- `README.md`: This file

## Usage

### Prerequisites

- Ensure you have collected baseline data for `branched_wiredcfc_arch4` on BNCI2014_001 dataset
- The architecture file should exist at: `outputs/architectures/best_architecture_4_trial_178.json`

### Running the Experiments

To run all ablation experiments:

```bash
python ablations/run_ablations.py
```

Or using the preferred Python interpreter:

```bash
C:\Users\Sean\anaconda3\envs\ncp_robustness_proj\python.exe ablations/run_ablations.py
```

### What the Script Does

The script runs in two phases:

**Phase 1: Run All Experiments**
1. **Loads Architecture 4**: Loads the wiring configuration from the architecture file
2. **Registers Model Variants**: Creates and registers three ablation variants plus baseline
3. **Runs Experiments**: For each variant, runs 5 experiments with different seeds (100, 200, 300, 400, 500)
   - Each ablation completes all 5 runs before moving to the next
   - Partial results are saved after each run to prevent data loss
4. **Saves Results**: Saves partial results after each run and combined results per ablation

**Phase 2: Statistical Analysis and Plotting** (only after all experiments complete)
5. **Statistical Analysis**: Performs paired t-tests and Wilcoxon signed-rank tests
6. **Creates Plots**: Generates comparison plots and statistical summary tables
7. **Combines Results**: Creates a combined results file with all ablations

### Output Structure

All results are isolated in the `ablations/` directory to avoid mixing with primary benchmark results in `results/` or `sol_results/` directories.

```
ablations/
├── results/
│   ├── baseline_results.csv
│   ├── baseline_partial_run1.csv, ..., baseline_partial_run5.csv
│   ├── ablation1_no_carry_gate_results.csv
│   ├── ablation1_no_carry_gate_partial_run1.csv, ..., partial_run5.csv
│   ├── ablation2_no_branching_results.csv
│   ├── ablation2_no_branching_partial_run1.csv, ..., partial_run5.csv
│   ├── ablation3_lstm_replacement_results.csv
│   ├── ablation3_lstm_replacement_partial_run1.csv, ..., partial_run5.csv
│   ├── combined_results.csv
│   └── statistical_tests.csv
├── plots/
│   ├── ablation_comparison_boxplot.png
│   ├── ablation_comparison_violin.png
│   └── statistical_summary.png
└── models/ (cached models if needed)
```

**Note**: Intermediate files from UnifiedExperimentRunner may be written to the base `results/` directory, but all final ablation results CSV files are saved exclusively to `ablations/results/` to maintain isolation.

## Model Variants

### Baseline: Full HYDRA Model
- Model: `branched_wiredcfc_arch4`
- All features enabled

### Ablation 1: No Carry Gate
- Model: `branched_wiredcfc_arch4_no_carry_gate`
- Disables weighted residual connection
- Hypothesis: Less stable model, decreased robustness

### Ablation 2: No Branching
- Model: `branched_wiredcfc_arch4_no_branching`
- Processes entire sequence in single temporal bin
- Hypothesis: Negative impact on robustness and inference speed

### Ablation 3: LSTM Replacement
- Model: `branched_lstm_arch4_equivalent`
- Uses BranchedLSTM instead of BranchedWiredCfC
- Hypothesis: Negative impact on model robustness

## Statistical Tests

The script performs:
- **Paired t-tests**: For within-subject comparisons
- **Wilcoxon signed-rank tests**: Non-parametric alternative
- **Effect size calculations**: Cohen's d
- **Multiple comparison correction**: Bonferroni correction (3 ablations)

## Notes

- **Experimental Runs**: Each ablation runs 5 complete experimental runs with seeds [100, 200, 300, 400, 500]
- **Two-Phase Execution**: 
  - Phase 1: All experiments run first (5 runs per ablation)
  - Phase 2: Statistical tests and plotting only happen after all experiments complete
- **Result Isolation**: All final results are saved to `ablations/results/` to avoid mixing with primary benchmark results
- **Partial Results**: Partial results are saved after each run to prevent data loss
- **Baseline Data**: The script assumes baseline results may already exist. If not found, it will run the baseline experiment first.
- **Model Caching**: Models are cached to avoid retraining when possible
- **Noise Evaluation**: The script uses `test_perturb` mode which evaluates on multiple noise types and intensities automatically

