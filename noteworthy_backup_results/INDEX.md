# Noteworthy Backup Results - Index

This directory contains experimental results comparing EEG classification models under various noise conditions, along with automated analysis tools.

## 📁 Files and Directories

### 📊 Result Directories
- **branched_diva_ncp/** - Branched DIVA-NCP model results
- **cnn_ncp_branch/** - CNN-NCP Branch model results  
- **diva_ncp_no_branch/** - DIVA-NCP (no branch) model results

### 🔧 Analysis Tools
- **analyze_models.py** - Main analysis script (see [README.md](README.md))
- **run_analysis.sh** - Unix/Linux/Mac launcher script
- **run_analysis.bat** - Windows launcher script
- **validate_setup.py** - Setup validation script

### 📖 Documentation
- **README.md** - Complete documentation for analysis tools
- **QUICK_START.md** - Quick reference for common tasks
- **INDEX.md** - This file (overview and links)

### 💡 Examples
- **example_usage.py** - Programmatic usage examples

### 📈 Generated Files (after running analysis)
- **analysis_summary.md** - Comparative analysis report
- **plots/** - Visualization plots (if --plot flag used)

## 🚀 Getting Started

### Option 1: Quick Start (Recommended)
```bash
# On Windows
run_analysis.bat

# On Linux/Mac
chmod +x run_analysis.sh
./run_analysis.sh
```

### Option 2: Direct Python
```bash
# Basic analysis
python analyze_models.py

# With plots
python analyze_models.py --plot

# Specific seed only
python analyze_models.py --seed 42 --plot
```

### Option 3: Validate First
```bash
# Check if everything is set up correctly
python validate_setup.py

# Then run analysis
python analyze_models.py --plot
```

## 📋 What the Analysis Provides

1. **Performance Comparison**: ROC-AUC scores across models and noise types
2. **Noise Robustness**: How well models maintain performance under noise
3. **Rankings**: By clean performance and noise tolerance
4. **Recommendations**: Which model to use for different scenarios
5. **Visualizations**: Performance degradation plots (optional)

## 🎯 Use Cases

| Task | Command | Output |
|------|---------|--------|
| Quick comparison | `python analyze_models.py` | analysis_summary.md |
| Full analysis with plots | `python analyze_models.py --plot` | analysis_summary.md + plots/ |
| Compare seed 42 only | `python analyze_models.py --seed 42` | analysis_summary.md |
| Validate setup | `python validate_setup.py` | Console output |
| Programmatic use | See example_usage.py | Custom analysis |

## 📊 Results Format

Each model directory contains:
```
model_name/
└── CrossSessionEvaluation/
    └── {seed}/
        └── sub-{subject}/
            └── {session}/
                └── test_perturb/
                    └── model_test_perturb_*.csv
```

CSV files contain:
- `noise_type`: eog, gaussian, or dropout
- `intensity`: 0-100%
- `clean_roc_auc`: Performance on clean data
- `corrupted_roc_auc`: Performance on corrupted data
- Plus many other metrics...

## 🔍 Understanding Results

### Noise Types
- **EOG**: Eye movement artifacts (most clinically relevant)
- **Gaussian**: Random additive noise (tests SNR tolerance)
- **Dropout**: Channel/sample dropout (tests information loss tolerance)

### Key Metrics
- **ROC-AUC**: Classification performance (0.5-1.0)
- **Retention**: % of clean performance maintained
- **Relative Drop**: Performance decrease from baseline

### Performance Guidelines
- **>95% retention**: Excellent noise tolerance
- **85-95% retention**: Good noise tolerance
- **70-85% retention**: Moderate noise tolerance
- **<70% retention**: Poor noise tolerance

## 🔄 Workflow

```
1. Run Experiment → 2. Save Results → 3. Run Analysis → 4. Review Report
                ↓                                           ↑
                └──────────── Make Changes ←────────────────┘
```

## 📝 Notes

- All models should use the same seed for fair comparison
- Analysis automatically discovers all CSV files
- Results are read-only; analysis never modifies source data
- Generated files (analysis_summary.md, plots/) are safe to delete and regenerate

## 🆘 Help and Troubleshooting

| Issue | Solution |
|-------|----------|
| "No result files found" | Check directory structure and CSV files exist |
| "Module not found" | Install: `pip install pandas numpy matplotlib` |
| "Permission denied" | Run: `chmod +x run_analysis.sh` (Unix/Linux/Mac) |
| Different seeds | Use `--seed` flag or analyze all together |
| Custom analysis | See example_usage.py for programmatic use |

## 📚 Documentation Hierarchy

```
QUICK_START.md          ← Start here for TL;DR
    ↓
INDEX.md (this file)    ← Overview and navigation
    ↓
README.md               ← Complete documentation
    ↓
example_usage.py        ← Advanced usage examples
```

## 🎓 Learn More

- **Basic Usage**: See [QUICK_START.md](QUICK_START.md)
- **Complete Guide**: See [README.md](README.md)
- **Code Examples**: See [example_usage.py](example_usage.py)
- **Validation**: Run `validate_setup.py`

## ✅ Quick Checklist

Before running analysis, ensure:
- [ ] Python 3.6+ installed
- [ ] pandas and numpy installed (`pip install pandas numpy`)
- [ ] Result CSV files present in model directories
- [ ] Running from noteworthy_backup_results directory
- [ ] (Optional) matplotlib installed for plots

After analysis:
- [ ] Check analysis_summary.md for detailed results
- [ ] Review model rankings
- [ ] Check plots/ for visualizations (if --plot used)
- [ ] Apply insights to model development

## 🔗 Quick Links

- [Main Analysis Script](analyze_models.py)
- [Documentation](README.md)
- [Quick Start Guide](QUICK_START.md)
- [Usage Examples](example_usage.py)
- [Validation Tool](validate_setup.py)

---

*Last Updated: Created with automated analysis tools*
*For questions or issues, check README.md or validate_setup.py*

