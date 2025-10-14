#!/bin/bash
# Quick analysis script with sensible defaults

set -e

echo "========================================"
echo "  Model Performance Analysis"
echo "========================================"
echo

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.6 or later."
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD=$(command -v python3 || command -v python)

# Check for required packages
echo "Checking dependencies..."
$PYTHON_CMD -c "import pandas; import numpy" 2>/dev/null || {
    echo "Error: Required packages not found."
    echo "Please install: pip install pandas numpy"
    exit 1
}

# Check for optional matplotlib
if $PYTHON_CMD -c "import matplotlib" 2>/dev/null; then
    echo "✓ matplotlib available - plots will be generated"
    PLOT_FLAG="--plot"
else
    echo "ℹ matplotlib not available - skipping plots"
    PLOT_FLAG=""
fi

echo
echo "Running analysis..."
echo

# Run the analysis
$PYTHON_CMD analyze_models.py $PLOT_FLAG --output analysis_summary.md

echo
echo "========================================"
echo "Analysis complete!"
echo "Check 'analysis_summary.md' for results"
if [ -n "$PLOT_FLAG" ]; then
    echo "Check 'plots/' directory for visualizations"
fi
echo "========================================"

