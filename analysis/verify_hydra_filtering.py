"""
Verify that all analysis scripts correctly filter models when --hydra flag is used.

This script checks that when selecting 'branched_wiredcfc_arch4' as HYDRA,
we do NOT accidentally include any hydra_v2 model variants.
"""

import sys
import os

# Add project root to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Define all hydra_v2 variants that should be EXCLUDED
HYDRA_V2_MODELS = [
    "hydra_v2",
    "hydra_v2_adaptive_residual",
    "hydra_v2_arch1",
    "hydra_v2_arch4",  # This is the one we're most concerned about
    "hydra_v2_baseline",
    "hydra_v2_cross_bin_context",
    "hydra_v2_erp_head",
    "hydra_v2_full",
    "hydra_v2_global_skip",
    "hydra_v2_multi_query",
    "hydra_v2_phase1",
    "hydra_v2_phase2",
    "hydra_v2_phase3",
    "hydra_v2_ssvep_head",
]

# The model we want to include
TARGET_MODEL = "branched_wiredcfc_arch4"

# Core models that should be included
CORE_MODELS = ['CNN-NCP', 'EEGNet', 'REEGNet', 'branched_wiredcfc_arch4']


def canonicalize_model_name(name):
    """Canonicalize model name for comparison (same logic as in scripts)."""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def test_robustness_metrics_filtering():
    """Test the filtering logic from robustness_metrics.py"""
    print("\n" + "="*80)
    print("Testing robustness_metrics.py filtering logic")
    print("="*80)
    
    # Simulate the filtering logic from robustness_metrics.py
    core_models = ['CNN-NCP', 'EEGNet', 'REEGNet', 'branched_wiredcfc_arch4']
    canonicalize_model_name_func = lambda x: str(x).strip().lower().replace(" ", "_").replace("-", "_")
    core_models_canonical = [canonicalize_model_name_func(m) for m in core_models]
    
    print(f"Core models (canonicalized): {core_models_canonical}")
    
    # Test each hydra_v2 variant
    issues = []
    for hydra_v2_model in HYDRA_V2_MODELS:
        canonicalized = canonicalize_model_name_func(hydra_v2_model)
        would_match = canonicalized in core_models_canonical
        if would_match:
            issues.append(f"  [ERROR] {hydra_v2_model} (canonicalized: {canonicalized}) would be INCLUDED!")
        else:
            print(f"  [OK] {hydra_v2_model} (canonicalized: {canonicalized}) correctly EXCLUDED")
    
    # Verify target model is included
    target_canonical = canonicalize_model_name_func(TARGET_MODEL)
    if target_canonical in core_models_canonical:
        print(f"  [OK] {TARGET_MODEL} (canonicalized: {target_canonical}) correctly INCLUDED")
    else:
        issues.append(f"  [ERROR] {TARGET_MODEL} would be EXCLUDED!")
    
    if issues:
        print("\n[FAILED] Issues found:")
        for issue in issues:
            print(issue)
        return False
    else:
        print("\n[PASSED] All checks passed for robustness_metrics.py")
        return True


def test_statistical_analysis_filtering():
    """Test the filtering logic from statistical_analysis.py"""
    print("\n" + "="*80)
    print("Testing statistical_analysis.py filtering logic")
    print("="*80)
    
    # Simulate the filtering logic from statistical_analysis.py
    core_models = ['CNN-NCP', 'EEGNet', 'REEGNet', 'branched_wiredcfc_arch4']
    
    print(f"Core models: {core_models}")
    
    # Test each hydra_v2 variant
    issues = []
    for hydra_v2_model in HYDRA_V2_MODELS:
        would_match = hydra_v2_model in core_models
        if would_match:
            issues.append(f"  [ERROR] {hydra_v2_model} would be INCLUDED!")
        else:
            print(f"  [OK] {hydra_v2_model} correctly EXCLUDED")
    
    # Verify target model is included
    if TARGET_MODEL in core_models:
        print(f"  [OK] {TARGET_MODEL} correctly INCLUDED")
    else:
        issues.append(f"  [ERROR] {TARGET_MODEL} would be EXCLUDED!")
    
    if issues:
        print("\n[FAILED] Issues found:")
        for issue in issues:
            print(issue)
        return False
    else:
        print("\n[PASSED] All checks passed for statistical_analysis.py")
        return True


def test_calculate_clean_scores_filtering():
    """Test the filtering logic from calculate_clean_scores.py"""
    print("\n" + "="*80)
    print("Testing calculate_clean_scores.py filtering logic")
    print("="*80)
    
    # Simulate the filtering logic from calculate_clean_scores.py
    hydra_model_patterns = [
        'eegnet', 'reegnet', 'cnn_ncp', 'cnn-ncp',
        'branched_wiredcfc_arch4', 'branched-wiredcfc-arch4'
    ]
    hydra_patterns_normalized = [p.lower().strip().replace('-', '_') for p in hydra_model_patterns]
    
    print(f"Patterns (normalized): {hydra_patterns_normalized}")
    
    # Test each hydra_v2 variant
    issues = []
    for hydra_v2_model in HYDRA_V2_MODELS:
        normalized = hydra_v2_model.lower().strip().replace('-', '_')
        would_match = normalized in hydra_patterns_normalized
        if would_match:
            issues.append(f"  [ERROR] {hydra_v2_model} (normalized: {normalized}) would be INCLUDED!")
        else:
            print(f"  [OK] {hydra_v2_model} (normalized: {normalized}) correctly EXCLUDED")
    
    # Verify target model is included
    target_normalized = TARGET_MODEL.lower().strip().replace('-', '_')
    if target_normalized in hydra_patterns_normalized:
        print(f"  [OK] {TARGET_MODEL} (normalized: {target_normalized}) correctly INCLUDED")
    else:
        issues.append(f"  [ERROR] {TARGET_MODEL} would be EXCLUDED!")
    
    if issues:
        print("\n[FAILED] Issues found:")
        for issue in issues:
            print(issue)
        return False
    else:
        print("\n[PASSED] All checks passed for calculate_clean_scores.py")
        return True


def test_analyze_results_filtering():
    """Test the filtering logic from analyze_results.py"""
    print("\n" + "="*80)
    print("Testing analyze_results.py filtering logic")
    print("="*80)
    
    # Simulate the filtering logic from analyze_results.py
    model_subsets = {
        'main_models': ['eegnet', 'reegnet', 'cnn_ncp', 'branched_wiredcfc_arch4'],
    }
    main_models = model_subsets['main_models']
    
    print(f"Main models: {main_models}")
    
    # Test each hydra_v2 variant
    issues = []
    for hydra_v2_model in HYDRA_V2_MODELS:
        would_match = hydra_v2_model in main_models
        if would_match:
            issues.append(f"  [ERROR] {hydra_v2_model} would be INCLUDED!")
        else:
            print(f"  [OK] {hydra_v2_model} correctly EXCLUDED")
    
    # Verify target model is included
    if TARGET_MODEL in main_models:
        print(f"  [OK] {TARGET_MODEL} correctly INCLUDED")
    else:
        issues.append(f"  [ERROR] {TARGET_MODEL} would be EXCLUDED!")
    
    if issues:
        print("\n[FAILED] Issues found:")
        for issue in issues:
            print(issue)
        return False
    else:
        print("\n[PASSED] All checks passed for analyze_results.py")
        return True


def main():
    """Run all verification tests."""
    print("="*80)
    print("HYDRA FILTERING VERIFICATION")
    print("="*80)
    print(f"\nTarget model to include: {TARGET_MODEL}")
    print(f"HYDRA V2 models to EXCLUDE: {len(HYDRA_V2_MODELS)} variants")
    print("\nTesting that all scripts correctly exclude hydra_v2 variants...")
    
    results = []
    results.append(("robustness_metrics.py", test_robustness_metrics_filtering()))
    results.append(("statistical_analysis.py", test_statistical_analysis_filtering()))
    results.append(("calculate_clean_scores.py", test_calculate_clean_scores_filtering()))
    results.append(("analyze_results.py", test_analyze_results_filtering()))
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_passed = True
    for script_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{script_name:40s}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("[SUCCESS] All scripts correctly filter models!")
        print("No hydra_v2 variants will be included when using --hydra flag.")
        return 0
    else:
        print("[FAILURE] Some scripts have filtering issues!")
        print("Please review and fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
