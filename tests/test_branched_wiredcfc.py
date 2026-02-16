#!/usr/bin/env python3
"""
Simple test script to validate the BranchedWiredCfC implementation after wiring fixes.

This script runs a CrossSession evaluation on subject 1 from the BNCI2014_001
dataset with an experimental seed of 42, no tuning, and overwrite flag.
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
    model = "branched_wiredcfc_arch4"  # Architecture 4 with wiring fixes
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
        "--overwrite",  # Overwrite existing results
        # Note: No --tune flag, so no tuning will be performed
    ]
    
    # Output file for subprocess output
    output_file = os.path.join(script_dir, "test_branched_wiredcfc_output.txt")
    
    print("=" * 80)
    print("Testing BranchedWiredCfC Implementation (After Wiring Fixes)")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"Dataset: {dataset}")
    print(f"Subjects: {subjects}")
    print(f"Mode: {mode}")
    print(f"Eval Mode: {eval_mode}")
    print(f"Seed: {seed}")
    print(f"Tune: False (no tuning)")
    print(f"Overwrite: True")
    print(f"Output file: {output_file}")
    print("=" * 80)
    print("\nRunning experiment...\n")
    
    # Run the command with output redirected to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            result = subprocess.run(
                cmd, 
                check=True, 
                cwd=script_dir,
                stdout=f,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True
            )
        print("\n" + "=" * 80)
        print("Test completed successfully!")
        print(f"Full output saved to: {output_file}")
        print("=" * 80)
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 80)
        print(f"Test failed with error code {e.returncode}")
        print(f"Full output saved to: {output_file}")
        print("=" * 80)
        # Print last 50 lines of output for quick reference
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) > 50:
                    print("\nLast 50 lines of output:")
                    print("".join(lines[-50:]))
                else:
                    print("\nFull output:")
                    print("".join(lines))
        except Exception:
            pass
        return e.returncode
    except Exception as e:
        print(f"\nError running test: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
