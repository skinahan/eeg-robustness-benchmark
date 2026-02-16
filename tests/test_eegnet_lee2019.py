#!/usr/bin/env python3
"""
Test script to validate EEGNet performance on Lee2019_SSVEP dataset.

This script runs a CrossSession evaluation on multiple subjects from the 
Lee2019_SSVEP dataset with an experimental seed of 42 to check for 
suspiciously high and consistent reported accuracy.
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
    model = "eegnet"
    dataset = "Lee2019_SSVEP"
    # Test with a few subjects to check for suspiciously consistent results
    subjects = [1, 2, 3, 4, 5]
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
        # "--tune"  # Uncomment if you want to test with hyperparameter tuning
    ]
    
    # Output file for subprocess output
    output_file = os.path.join(script_dir, "test_eegnet_lee2019_output.txt")
    
    print("=" * 80)
    print("Testing EEGNet on Lee2019_SSVEP Dataset")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"Dataset: {dataset}")
    print(f"Subjects: {subjects}")
    print(f"Mode: {mode}")
    print(f"Eval Mode: {eval_mode}")
    print(f"Seed: {seed}")
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
        
        # Print a summary of results from the output file
        print("\nExtracting accuracy results from output...")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Look for accuracy-related lines
                accuracy_lines = [line for line in lines if 'accuracy' in line.lower() or 'score' in line.lower() or 'result' in line.lower()]
                if accuracy_lines:
                    print("\nRelevant accuracy/score lines found:")
                    for line in accuracy_lines[-20:]:  # Show last 20 relevant lines
                        print(line.rstrip())
        except Exception as e:
            print(f"Could not extract accuracy summary: {e}")
        
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
