#!/bin/bash
# Test script for unified_eval_script.sh
# This shows example usage without actually submitting to SLURM

echo "Testing unified_eval_script.sh with example parameters..."

# Example 1: CrossSession with single subject, no tuning
echo ""
echo "Example 1: CrossSession, Subject 1, No Tuning"
echo "Command: ./unified_eval_script.sh 1 BNCI2014_001 CrossSession false"
echo ""

# Example 2: CrossSession with single subject, with tuning  
echo "Example 2: CrossSession, Subject 1, With Tuning"
echo "Command: ./unified_eval_script.sh 1 BNCI2014_001 CrossSession true"
echo ""

# Example 3: WithinSession with multiple subjects, no tuning
echo "Example 3: WithinSession, All Subjects, No Tuning"
echo "Command: ./unified_eval_script.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 WithinSession false"
echo ""

# Example 4: WithinSession with multiple subjects, with tuning
echo "Example 4: WithinSession, All Subjects, With Tuning"
echo "Command: ./unified_eval_script.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 WithinSession true"
echo ""

echo "Note: These are example commands. To actually run experiments:"
echo "1. Use 'sbatch unified_eval_script.sh ...' to submit to SLURM"
echo "2. Or use the generated 'run_missing_multirun_jobs.sh' script"
