# Custom Plotting and Data Extraction

This document describes the flexible plotting and data extraction methods added to `analyze_results.py`.

## Overview

Two new methods have been added to provide flexible data filtering, extraction, and plotting:

1. `plot_custom_comparison()` - Create custom plots with flexible filtering
2. `extract_custom_data()` - Extract filtered data without plotting

## Quick Start

```python
import pandas as pd
from analyze_results import plot_custom_comparison, extract_custom_data

# Load your results
df = pd.read_csv('sol_results/all_results.csv')

# Create a custom comparison plot
plot_custom_comparison(
    df,
    filters={
        'model': ['eegnet', 'cnn_ncp'],
        'noise_type': 'eog',
        'eval_mode': 'CrossSession',
        'mode': 'test_perturb',
        'tune': True
    },
    hue_var='model',
    style_var='session',
    output_filename='my_custom_plot.png'
)
```

## plot_custom_comparison()

### Key Features

- **Flexible Filtering**: Filter by any combination of columns
- **Multiple Plot Types**: Line, bar, scatter, box plots
- **Faceting**: Create multi-panel plots with `col_var` and `row_var`
- **Multiple Metrics**: ROC AUC, accuracy, precision, recall, F1
- **Auto-labeling**: Automatic titles and labels, or custom ones
- **Clean Baseline**: Automatically includes intensity=0 baseline for test_perturb mode

### Basic Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | required | The results dataframe |
| `filters` | dict | None | Column filters (see Filter Syntax below) |
| `x_var` | str | 'intensity' | X-axis variable |
| `y_var` | str | 'corrupted_score' | Y-axis variable |
| `hue_var` | str | 'model' | Color differentiation |
| `style_var` | str | None | Style differentiation (line plots) |
| `col_var` | str | None | Column faceting |
| `row_var` | str | None | Row faceting |
| `plot_type` | str | 'line' | 'line', 'bar', 'scatter', 'box' |
| `metric` | str | 'roc_auc' | Metric to plot |
| `output_filename` | str | None | Custom filename (auto-generated if None) |
| `title` | str | None | Plot title (auto-generated if None) |

### Filter Syntax

The `filters` parameter accepts a dictionary where:

1. **Single value**: Exact match
   ```python
   filters={'noise_type': 'eog', 'tune': True}
   ```

2. **List of values**: Match any value in list
   ```python
   filters={'model': ['eegnet', 'cnn_ncp', 'reegnet']}
   ```

3. **Lambda function**: Custom filtering logic
   ```python
   filters={'intensity': lambda x: x <= 20.0}
   ```

4. **Combined filters**: All conditions must be met
   ```python
   filters={
       'model': ['eegnet', 'cnn_ncp'],
       'noise_type': 'eog',
       'eval_mode': 'CrossSession',
       'intensity': lambda x: x in [1.0, 10.0, 20.0, 30.0]
   }
   ```

## extract_custom_data()

Extract and optionally aggregate data without plotting.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | DataFrame | The results dataframe |
| `filters` | dict | Same syntax as plot_custom_comparison |
| `columns` | list | Specific columns to return |
| `aggregate` | str/dict | Aggregation function ('mean', 'median', 'std', etc.) |
| `group_by` | str/list | Columns to group by before aggregation |

### Examples

```python
# Extract raw data
data = extract_custom_data(
    df,
    filters={'model': 'eegnet', 'noise_type': 'eog'},
    columns=['model', 'intensity', 'corrupted_score', 'session']
)

# Extract aggregated statistics
mean_scores = extract_custom_data(
    df,
    filters={'model': ['eegnet', 'cnn_ncp'], 'noise_type': 'eog'},
    aggregate='mean',
    group_by=['model', 'intensity']
)
```

## Common Use Cases

### 1. Compare Two Models Under Specific Noise

```python
plot_custom_comparison(
    df,
    filters={
        'model': ['eegnet', 'cnn_ncp'],
        'noise_type': 'eog',
        'eval_mode': 'CrossSession',
        'mode': 'test_perturb',
        'tune': True
    },
    hue_var='model',
    style_var='session'
)
```

### 2. Compare All Noise Types for One Model

```python
plot_custom_comparison(
    df,
    filters={
        'model': 'eegnet',
        'eval_mode': 'CrossSession',
        'mode': 'test_perturb',
        'tune': True
    },
    hue_var='noise_type'
)
```

### 3. Tuned vs Baseline Comparison

```python
plot_custom_comparison(
    df,
    filters={
        'model': 'cnn_ncp',
        'noise_type': 'gaussian',
        'eval_mode': 'CrossSession',
        'mode': 'test_perturb'
    },
    hue_var='tune'
)
```

### 4. Low Intensity Analysis Only

```python
plot_custom_comparison(
    df,
    filters={
        'model': ['eegnet', 'reegnet', 'cnn_ncp'],
        'noise_type': 'dropout',
        'intensity': lambda x: x <= 10.0
    },
    hue_var='model'
)
```

### 5. Faceted Multi-Panel Comparison

```python
plot_custom_comparison(
    df,
    filters={
        'model': ['eegnet', 'cnn_ncp'],
        'eval_mode': 'CrossSession',
        'tune': True
    },
    hue_var='model',
    col_var='noise_type',  # Separate panel for each noise type
    row_var='session'      # Separate row for train/test
)
```

### 6. Specific Subjects Only

```python
plot_custom_comparison(
    df,
    filters={
        'model': 'eegnet',
        'subject': [1, 2, 3, 4, 5]  # First 5 subjects
    },
    hue_var='subject'
)
```

### 7. Box Plot for Variability Analysis

```python
plot_custom_comparison(
    df,
    filters={
        'model': ['eegnet', 'cnn_ncp'],
        'noise_type': 'eog',
        'intensity': lambda x: x in [10.0, 20.0, 30.0]
    },
    plot_type='box',
    hue_var='model'
)
```

### 8. Multiple Metrics

```python
for metric in ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']:
    plot_custom_comparison(
        df,
        filters={'model': 'eegnet', 'noise_type': 'gaussian'},
        metric=metric,
        output_filename=f'eegnet_gaussian_{metric}.png'
    )
```

## Available Columns for Filtering

Common columns in the results dataframe:

- `model`: Model name (e.g., 'eegnet', 'cnn_ncp', 'reegnet')
- `mode`: Experiment mode (e.g., 'test_perturb')
- `eval_mode`: Evaluation type (e.g., 'CrossSession', 'WithinSession')
- `noise_type`: Type of noise (e.g., 'gaussian', 'dropout', 'eog')
- `intensity`: Noise intensity level
- `session`: Session type (e.g., '0train', '1test')
- `subject`: Subject ID
- `tune`: Whether hyperparameters were tuned (True/False)
- `seed`: Random seed used
- `dataset`: Dataset name

## Output

Both functions provide informative output:

```python
filtered_data, plot_path = plot_custom_comparison(df, filters={...})
# Prints:
# Plot saved to: plots/custom_plot_model_eegnet_cnn_ncp_noise_type_eog.png
# Filtered data shape: (1024, 50)

# Returns:
# - filtered_data: The filtered DataFrame used for plotting
# - plot_path: Path to saved plot file
```

## Examples

See `example_custom_plots.py` for 10 comprehensive examples demonstrating various use cases.

To run the examples:

```bash
cd analysis
python example_custom_plots.py
```

## Integration with Existing Code

The new methods are fully compatible with existing plotting functions in `analyze_results.py`. The main script includes an example usage in the `if __name__ == '__main__'` block.


