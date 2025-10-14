# Quick Start Guide

## TL;DR

```bash
# Linux/Mac
./run_analysis.sh

# Windows
run_analysis.bat

# Or directly with Python
python analyze_models.py --plot
```

## What You Get

- **analysis_summary.md**: Detailed comparison report
- **plots/model_comparison.png**: Visual performance comparison (if matplotlib installed)

## Common Use Cases

### 1. Quick Analysis (No Plots)
```bash
python analyze_models.py
```

### 2. Full Analysis with Visualizations
```bash
python analyze_models.py --plot
```

### 3. Compare Only Seed 42 Results
```bash
python analyze_models.py --seed 42 --plot
```

### 4. Save to Different File
```bash
python analyze_models.py --output my_comparison.md
```

### 5. Programmatic Use
```python
from analyze_models import find_result_files, load_and_prepare_data, calculate_statistics

# Find and load data
files = find_result_files('.', seed=42)
df = load_and_prepare_data(files)
stats = calculate_statistics(df)

# Access results
for model, model_stats in stats.items():
    print(f"{model}: {model_stats['clean_performance']:.4f}")
```

## Understanding the Output

### Key Metrics

- **ROC-AUC**: Classification performance (0.5 = random, 1.0 = perfect)
- **Retention**: % of clean performance maintained under noise
- **Avg Retention**: Average across all intensities

### Performance Levels

| Retention | Interpretation |
|-----------|----------------|
| >95% | Excellent noise tolerance |
| 85-95% | Good noise tolerance |
| 70-85% | Moderate noise tolerance |
| <70% | Poor noise tolerance |

### Noise Types

- **EOG**: Eye movement artifacts (most common in real EEG)
- **Gaussian**: Random additive noise (tests SNR)
- **Dropout**: Channel/data dropout (tests redundancy)

## Troubleshooting

### No results found
- Check you're in `noteworthy_backup_results/` directory
- Verify CSV files exist in subdirectories

### Module errors
```bash
pip install pandas numpy matplotlib
```

### Permission denied (Linux/Mac)
```bash
chmod +x run_analysis.sh
./run_analysis.sh
```

## File Organization

```
noteworthy_backup_results/
├── analyze_models.py       # Main script
├── run_analysis.sh         # Linux/Mac launcher
├── run_analysis.bat        # Windows launcher
├── README.md               # Full documentation
├── QUICK_START.md          # This file
├── example_usage.py        # Programmatic examples
│
├── model_name_1/           # Results for model 1
│   └── CrossSessionEvaluation/.../*.csv
├── model_name_2/           # Results for model 2
│   └── CrossSessionEvaluation/.../*.csv
└── plots/                  # Generated plots (if --plot used)
```

## Next Steps

- Read **README.md** for detailed documentation
- Check **example_usage.py** for programmatic examples
- Modify **analyze_models.py** for custom analyses

## Tips

1. Always use the same seed across models for fair comparison
2. Use `--plot` for presentations and reports
3. Check `analysis_summary.md` for rankings and recommendations
4. Run analysis after any model changes to track improvements

