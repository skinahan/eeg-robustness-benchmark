@echo off
REM Quick analysis script for Windows

echo ========================================
echo   Model Performance Analysis
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.6 or later.
    exit /b 1
)

REM Check for required packages
echo Checking dependencies...
python -c "import pandas; import numpy" >nul 2>&1
if errorlevel 1 (
    echo Error: Required packages not found.
    echo Please install: pip install pandas numpy
    exit /b 1
)

REM Check for optional matplotlib
python -c "import matplotlib" >nul 2>&1
if errorlevel 0 (
    echo [OK] matplotlib available - plots will be generated
    set PLOT_FLAG=--plot
) else (
    echo [INFO] matplotlib not available - skipping plots
    set PLOT_FLAG=
)

echo.
echo Running analysis...
echo.

REM Run the analysis
python analyze_models.py %PLOT_FLAG% --output analysis_summary.md

echo.
echo ========================================
echo Analysis complete!
echo Check 'analysis_summary.md' for results
if defined PLOT_FLAG (
    echo Check 'plots\' directory for visualizations
)
echo ========================================

pause

