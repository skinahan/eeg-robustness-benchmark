# Noteworthy Backup Results - Analysis Tools

This directory contains noteworthy experimental results and tools for analyzing model performance under different noise conditions.

## Directory Structure

```
noteworthy_backup_results/
├── analyze_models.py          # Main analysis script
├── README.md                   # This file
├── branched_diva_ncp/         # Branched DIVA-NCP results
├── cnn_ncp_branch/            # CNN-NCP Branch results
├── diva_ncp_no_branch/        # DIVA-NCP (no branch) results
└── plots/                      # Generated visualization plots (if --plot used)
```

## Quick Start

### Basic Analysis

Run the analysis script from this directory:

```bash
python analyze_models.py
```

This will:
- Automatically find all result CSV files
- Calculate performance statistics
- Generate `analysis_summary.md` with detailed comparison

### With Visualizations

To generate plots (requires matplotlib):

```bash
python analyze_models.py --plot
```

Plots will be saved in the `plots/` subdirectory.

### Filter by Seed

To analyze only results from a specific seed:

```bash
python analyze_models.py --seed 42
```

### Custom Output File

To specify a different output file:

```bash
python analyze_models.py --output my_analysis.md
```

## Command-Line Options

```
usage: analyze_models.py [-h] [--plot] [--output OUTPUT] [--seed SEED] [--dir DIR]

Options:
  --plot              Generate visualization plots (requires matplotlib)
  --output FILE       Output markdown file path (default: analysis_summary.md)
  --seed SEED         Filter results by specific seed
  --dir DIR           Base directory to search (default: current directory)
```

## What the Script Does

1. **Finds Result Files**: Automatically discovers all test_perturb CSV files in the directory structure
2. **Loads Data**: Reads and combines data from all models
3. **Calculates Statistics**: Computes performance metrics including:
   - Clean performance
   - Performance at key noise intensities (25%, 50%, 75%, 100%)
   - Retention percentages (% of clean performance maintained)
   - Average performance across intensities
4. **Generates Report**: Creates a markdown file with:
   - Model comparisons by noise type
   - Performance tables
   - Rankings by clean performance and noise robustness
   - Recommendations
5. **Creates Plots** (optional): Generates line plots showing performance degradation

## Output Files

- `analysis_summary.md`: Detailed markdown report
- `plots/model_comparison.png`: Visual comparison plot (if --plot used)

## Requirements

- Python 3.6+
- pandas
- numpy
- matplotlib (optional, for plotting)

## Example Output

The script generates console output like:

```
================================================================================
MODEL PERFORMANCE ANALYSIS
================================================================================

Searching for result files...
Found 3 model(s)

Loading data...
✓ Loaded: branched_diva_ncp (61 rows)
✓ Loaded: cnn_ncp_branch (61 rows)
✓ Loaded: diva_ncp_no_branch (61 rows)

Calculating statistics...

================================================================================
SUMMARY
================================================================================

branched_diva_ncp:
  Clean ROC-AUC: 0.8667
  EOG:
    Mean: 0.8545
    Range: 0.8150 - 0.8983
  ...

✓ Report saved to: analysis_summary.md

================================================================================
Analysis complete!
================================================================================
```

## Interpreting Results

### Performance Metrics

- **ROC-AUC**: Area Under the Receiver Operating Characteristic curve (higher is better)
- **Retention**: Percentage of clean performance maintained under noise
- **Clean Performance**: Model performance on uncorrupted data (baseline)

### Noise Types

- **EOG**: Eye movement artifacts (spatial-temporal noise)
- **Gaussian**: Random additive noise (tests SNR tolerance)
- **Dropout**: Random channel dropout (tests information loss tolerance)

### Key Comparisons

1. **By Clean Performance**: Which model performs best on clean data
2. **By Noise Robustness**: Which model maintains performance under noise
3. **By Noise Type**: Which model excels at specific noise conditions

## Customization

The script can be easily modified:

- Edit `find_result_files()` to change file discovery logic
- Edit `calculate_statistics()` to add new metrics
- Edit `generate_markdown_report()` to customize report format
- Edit `plot_results()` to create different visualizations

## Troubleshooting

### "No result files found"

Check that:
- You're running from the correct directory
- CSV files exist in the expected structure: `model_name/.../test_perturb/*.csv`
- File names contain "test_perturb" and "seed"

### "matplotlib not available"

Either:
- Install matplotlib: `pip install matplotlib`
- Or run without `--plot` flag

### Different seed values

If models use different seeds, you can:
- Analyze all together: `python analyze_models.py`
- Filter by specific seed: `python analyze_models.py --seed 42`

## Adding New Models

To analyze additional models:

1. Place results in a new subdirectory: `new_model_name/`
2. Ensure CSV files follow the same structure
3. Run `python analyze_models.py` - new model will be automatically detected

## Notes

- The script assumes test session results (`1test`) are preferred over train (`0train`)
- All models are compared fairly using the same metrics
- Results are reproducible - running multiple times produces identical output

