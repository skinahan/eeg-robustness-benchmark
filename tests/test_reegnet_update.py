#!/usr/bin/env python3
"""
Simple test script to validate the updated REEGNet implementation.

This script runs a CrossSession evaluation on subject 1 from the BNCI2014_001
dataset with an experimental seed of 42.
"""

import subprocess
import sys
import os

def main():
    """Run the test experiment."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    unified_runner = os.path.join(script_dir, "evaluation", "unified_experiment_runner.py")
    
    # Parameters for the test
    model = "reegnet"
    dataset = "BNCI2014_001"
    subjects = [1]
    mode = "test_perturb"
    eval_mode = "CrossSession"
    seed = 42
    
    # Build the command
    cmd = [
        sys.executable,
        unified_runner,
        "--model", model,
        "--dataset", dataset,
        "--subjects"] + [str(s) for s in subjects] + [
        "--mode", mode,
        "--eval_mode", eval_mode,
        "--seed", str(seed),
        "--tune"
    ]
    
    print("=" * 80)
    print("Testing Updated REEGNet Implementation")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"Dataset: {dataset}")
    print(f"Subjects: {subjects}")
    print(f"Mode: {mode}")
    print(f"Eval Mode: {eval_mode}")
    print(f"Seed: {seed}")
    print(f"Tune: True")
    print("=" * 80)
    print("\nRunning experiment...\n")
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True, cwd=script_dir)
        print("\n" + "=" * 80)
        print("Test completed successfully!")
        print("=" * 80)
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 80)
        print(f"Test failed with error code {e.returncode}")
        print("=" * 80)
        return e.returncode
    except Exception as e:
        print(f"\nError running test: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
