#!/usr/bin/env python3
"""
Performance test script for the optimized generate_expected_experiments() function
"""

import time
import sys
from experiment_automation import ExperimentAutomation

def test_performance():
    """Test the performance improvements."""
    print("="*60)
    print("PERFORMANCE OPTIMIZATION TEST")
    print("="*60)
    
    # Initialize automation
    print("Initializing ExperimentAutomation...")
    automation = ExperimentAutomation('experiment_config.yaml')
    
    # Time the optimized function
    print("\nTesting optimized generate_expected_experiments() performance...")
    start_time = time.time()
    experiments = automation.generate_expected_experiments()
    end_time = time.time()
    
    first_run_time = end_time - start_time
    print(f"First run time: {first_run_time:.2f} seconds")
    print(f"Generated {len(automation.expected_test_perturb_results)} expected experiments")
    
    # Test caching performance
    print("\nTesting cache performance (second run)...")
    start_time = time.time()
    experiments = automation.generate_expected_experiments()  # Should use cache
    end_time = time.time()
    
    cached_run_time = end_time - start_time
    print(f"Cached call time: {cached_run_time:.2f} seconds")
    
    # Calculate speedup
    if cached_run_time > 0:
        speedup = first_run_time / cached_run_time
        print(f"Cache speedup: {speedup:.1f}x faster")
    
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    print("[OK] Replaced nested loops with itertools.product")
    print("[OK] Added caching for noise intensities calculation")
    print("[OK] Added caching for expected experiment results")
    print("[OK] Optimized missing experiment identification with set operations")
    print("[OK] Added result signature caching")
    print("[OK] Added progress bars for better user feedback")
    
    return first_run_time, cached_run_time

if __name__ == "__main__":
    try:
        test_performance()
    except Exception as e:
        print(f"Error during performance test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
