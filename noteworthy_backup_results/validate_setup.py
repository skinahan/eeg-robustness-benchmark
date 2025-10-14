#!/usr/bin/env python3
"""
Validation script to check if analysis can be run.

This script checks:
- Required Python packages are installed
- Result files are present and readable
- Data structure is correct
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 6):
        print("✗ Python 3.6+ required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_packages():
    """Check required packages."""
    required = ['pandas', 'numpy']
    optional = ['matplotlib']
    
    all_ok = True
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} NOT installed (REQUIRED)")
            all_ok = False
    
    for package in optional:
        try:
            __import__(package)
            print(f"✓ {package} installed (optional)")
        except ImportError:
            print(f"ℹ {package} not installed (optional - for plots)")
    
    return all_ok


def check_result_files():
    """Check if result files exist."""
    base_path = Path('.')
    
    # Look for model directories
    model_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name not in 
                  ['plots', 'example_plots', '__pycache__', '.git']]
    
    if not model_dirs:
        print("✗ No model directories found")
        return False
    
    print(f"\nFound {len(model_dirs)} model directory(ies):")
    
    total_files = 0
    for model_dir in model_dirs:
        csv_files = list(model_dir.rglob("*.csv"))
        if csv_files:
            print(f"  ✓ {model_dir.name}: {len(csv_files)} CSV file(s)")
            total_files += len(csv_files)
        else:
            print(f"  ⚠ {model_dir.name}: No CSV files found")
    
    if total_files == 0:
        print("✗ No CSV files found in any model directory")
        return False
    
    return True


def check_data_structure():
    """Check if CSV files have expected structure."""
    from analyze_models import find_result_files, load_and_prepare_data
    
    try:
        result_files = find_result_files('.')
        if not result_files:
            print("✗ No valid result files found")
            return False
        
        print(f"\nValidating data structure...")
        df = load_and_prepare_data(result_files)
        
        # Check required columns
        required_cols = ['noise_type', 'intensity', 'clean_roc_auc', 
                        'corrupted_roc_auc', 'model_name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"✗ Missing columns: {missing_cols}")
            return False
        
        print(f"✓ Data structure valid")
        print(f"  - Total rows: {len(df)}")
        print(f"  - Models: {df['model_name'].nunique()}")
        print(f"  - Noise types: {df['noise_type'].nunique()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error validating data: {e}")
        return False


def main():
    print("=" * 60)
    print("VALIDATION: Analysis Setup")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_packages),
        ("Result Files", check_result_files),
        ("Data Structure", check_data_structure),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 60)
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = all(result for _, result in results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}: {status}")
    
    print()
    
    if all_passed:
        print("✓ All checks passed! Ready to run analysis.")
        print("\nRun: python analyze_models.py --plot")
        return 0
    else:
        print("✗ Some checks failed. Please fix issues above.")
        print("\nCommon fixes:")
        print("  - Install packages: pip install pandas numpy matplotlib")
        print("  - Check directory structure and CSV files")
        return 1


if __name__ == '__main__':
    sys.exit(main())

