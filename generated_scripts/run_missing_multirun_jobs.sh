#!/bin/bash
# Generated multirun sbatch automation script
# Generated on: 2025-10-09 15:44:43
# Total missing multirun jobs: 540

set -e  # Exit on any error

echo "Starting multirun experiment automation..."
echo "Total multirun jobs to submit: 540"

# Multirun Job 1/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [6] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 1/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 1 submitted successfully"
else
    echo "[ERROR] Multirun job 1 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 2/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [6] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 2/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 2 submitted successfully"
else
    echo "[ERROR] Multirun job 2 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 3/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [3] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 3/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 3 submitted successfully"
else
    echo "[ERROR] Multirun job 3 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 4/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [6] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 4/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 4 submitted successfully"
else
    echo "[ERROR] Multirun job 4 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 5/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [7] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 5/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 5 submitted successfully"
else
    echo "[ERROR] Multirun job 5 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 6/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [6] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 6/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 6 submitted successfully"
else
    echo "[ERROR] Multirun job 6 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 7/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [1] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 7/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 7 submitted successfully"
else
    echo "[ERROR] Multirun job 7 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 8/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [5] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 8/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 8 submitted successfully"
else
    echo "[ERROR] Multirun job 8 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 9/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [4] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 9/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 9 submitted successfully"
else
    echo "[ERROR] Multirun job 9 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 10/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [7] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 10/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 10 submitted successfully"
else
    echo "[ERROR] Multirun job 10 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 11/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [4] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 11/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 11 submitted successfully"
else
    echo "[ERROR] Multirun job 11 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 12/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [1] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 12/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 12 submitted successfully"
else
    echo "[ERROR] Multirun job 12 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 13/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [7] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 13/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 13 submitted successfully"
else
    echo "[ERROR] Multirun job 13 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 14/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [5] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 14/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 14 submitted successfully"
else
    echo "[ERROR] Multirun job 14 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 15/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [4] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 15/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 15 submitted successfully"
else
    echo "[ERROR] Multirun job 15 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 16/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [5] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 16/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 16 submitted successfully"
else
    echo "[ERROR] Multirun job 16 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 17/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [1] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 17/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 17 submitted successfully"
else
    echo "[ERROR] Multirun job 17 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 18/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [4] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 18/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 18 submitted successfully"
else
    echo "[ERROR] Multirun job 18 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 19/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [6] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 19/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 19 submitted successfully"
else
    echo "[ERROR] Multirun job 19 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 20/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [4] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 20/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 20 submitted successfully"
else
    echo "[ERROR] Multirun job 20 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 21/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [7] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 21/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 21 submitted successfully"
else
    echo "[ERROR] Multirun job 21 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 22/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [7] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 22/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 22 submitted successfully"
else
    echo "[ERROR] Multirun job 22 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 23/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [1] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 23/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 23 submitted successfully"
else
    echo "[ERROR] Multirun job 23 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 24/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [9] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 24/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 24 submitted successfully"
else
    echo "[ERROR] Multirun job 24 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 25/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [9] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 25/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 25 submitted successfully"
else
    echo "[ERROR] Multirun job 25 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 26/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [3] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 26/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 26 submitted successfully"
else
    echo "[ERROR] Multirun job 26 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 27/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [8] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 27/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 27 submitted successfully"
else
    echo "[ERROR] Multirun job 27 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 28/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [3] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 28/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 28 submitted successfully"
else
    echo "[ERROR] Multirun job 28 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 29/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [3] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 29/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 29 submitted successfully"
else
    echo "[ERROR] Multirun job 29 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 30/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [5] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 30/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 30 submitted successfully"
else
    echo "[ERROR] Multirun job 30 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 31/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [2] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 31/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 31 submitted successfully"
else
    echo "[ERROR] Multirun job 31 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 32/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [1] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 32/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 32 submitted successfully"
else
    echo "[ERROR] Multirun job 32 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 33/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [3] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 33/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 33 submitted successfully"
else
    echo "[ERROR] Multirun job 33 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 34/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [8] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 34/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 34 submitted successfully"
else
    echo "[ERROR] Multirun job 34 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 35/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [9] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 35/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 35 submitted successfully"
else
    echo "[ERROR] Multirun job 35 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 36/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [2] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 36/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 36 submitted successfully"
else
    echo "[ERROR] Multirun job 36 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 37/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [8] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 37/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 37 submitted successfully"
else
    echo "[ERROR] Multirun job 37 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 38/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [5] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 38/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 38 submitted successfully"
else
    echo "[ERROR] Multirun job 38 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 39/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [4] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 39/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 39 submitted successfully"
else
    echo "[ERROR] Multirun job 39 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 40/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [7] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 40/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 40 submitted successfully"
else
    echo "[ERROR] Multirun job 40 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 41/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [4] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 41/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 41 submitted successfully"
else
    echo "[ERROR] Multirun job 41 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 42/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [3] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 42/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 42 submitted successfully"
else
    echo "[ERROR] Multirun job 42 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 43/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [4] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 43/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 43 submitted successfully"
else
    echo "[ERROR] Multirun job 43 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 44/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [1] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 44/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 44 submitted successfully"
else
    echo "[ERROR] Multirun job 44 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 45/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [7] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 45/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 45 submitted successfully"
else
    echo "[ERROR] Multirun job 45 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 46/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [2] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 46/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 46 submitted successfully"
else
    echo "[ERROR] Multirun job 46 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 47/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [3] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 47/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 47 submitted successfully"
else
    echo "[ERROR] Multirun job 47 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 48/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [6] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 48/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 48 submitted successfully"
else
    echo "[ERROR] Multirun job 48 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 49/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [9] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 49/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 49 submitted successfully"
else
    echo "[ERROR] Multirun job 49 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 50/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [1] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 50/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 50 submitted successfully"
else
    echo "[ERROR] Multirun job 50 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 51/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [7] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 51/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 51 submitted successfully"
else
    echo "[ERROR] Multirun job 51 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 52/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [2] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 52/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 52 submitted successfully"
else
    echo "[ERROR] Multirun job 52 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 53/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [1] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 53/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 53 submitted successfully"
else
    echo "[ERROR] Multirun job 53 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 54/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [6] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 54/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 54 submitted successfully"
else
    echo "[ERROR] Multirun job 54 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 55/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [5] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 55/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 55 submitted successfully"
else
    echo "[ERROR] Multirun job 55 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 56/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [5] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 56/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 56 submitted successfully"
else
    echo "[ERROR] Multirun job 56 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 57/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [1] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 57/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 57 submitted successfully"
else
    echo "[ERROR] Multirun job 57 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 58/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [4] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 58/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 58 submitted successfully"
else
    echo "[ERROR] Multirun job 58 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 59/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [2] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 59/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 59 submitted successfully"
else
    echo "[ERROR] Multirun job 59 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 60/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [5] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 60/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 60 submitted successfully"
else
    echo "[ERROR] Multirun job 60 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 61/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [4] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 61/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 61 submitted successfully"
else
    echo "[ERROR] Multirun job 61 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 62/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [7] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 62/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 62 submitted successfully"
else
    echo "[ERROR] Multirun job 62 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 63/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [2] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 63/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 63 submitted successfully"
else
    echo "[ERROR] Multirun job 63 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 64/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [6] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 64/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 64 submitted successfully"
else
    echo "[ERROR] Multirun job 64 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 65/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [8] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 65/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 65 submitted successfully"
else
    echo "[ERROR] Multirun job 65 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 66/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [8] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 66/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 66 submitted successfully"
else
    echo "[ERROR] Multirun job 66 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 67/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [2] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 67/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 67 submitted successfully"
else
    echo "[ERROR] Multirun job 67 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 68/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [6] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 68/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 68 submitted successfully"
else
    echo "[ERROR] Multirun job 68 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 69/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [8] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 69/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 69 submitted successfully"
else
    echo "[ERROR] Multirun job 69 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 70/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [7] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 70/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 70 submitted successfully"
else
    echo "[ERROR] Multirun job 70 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 71/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [4] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 71/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 71 submitted successfully"
else
    echo "[ERROR] Multirun job 71 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 72/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [1] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 72/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 72 submitted successfully"
else
    echo "[ERROR] Multirun job 72 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 73/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [4] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 73/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 73 submitted successfully"
else
    echo "[ERROR] Multirun job 73 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 74/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [7] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 74/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 74 submitted successfully"
else
    echo "[ERROR] Multirun job 74 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 75/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [5] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 75/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 75 submitted successfully"
else
    echo "[ERROR] Multirun job 75 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 76/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [3] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 76/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 76 submitted successfully"
else
    echo "[ERROR] Multirun job 76 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 77/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [7] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 77/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 77 submitted successfully"
else
    echo "[ERROR] Multirun job 77 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 78/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [4] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 78/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 78 submitted successfully"
else
    echo "[ERROR] Multirun job 78 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 79/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [9] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 79/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 79 submitted successfully"
else
    echo "[ERROR] Multirun job 79 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 80/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [8] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 80/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 80 submitted successfully"
else
    echo "[ERROR] Multirun job 80 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 81/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [2] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 81/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 81 submitted successfully"
else
    echo "[ERROR] Multirun job 81 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 82/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [9] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 82/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 82 submitted successfully"
else
    echo "[ERROR] Multirun job 82 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 83/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [2] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 83/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 83 submitted successfully"
else
    echo "[ERROR] Multirun job 83 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 84/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [5] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 84/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 84 submitted successfully"
else
    echo "[ERROR] Multirun job 84 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 85/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [9] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 85/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 85 submitted successfully"
else
    echo "[ERROR] Multirun job 85 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 86/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [8] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 86/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 86 submitted successfully"
else
    echo "[ERROR] Multirun job 86 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 87/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [5] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 87/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 87 submitted successfully"
else
    echo "[ERROR] Multirun job 87 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 88/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [4] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 88/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 88 submitted successfully"
else
    echo "[ERROR] Multirun job 88 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 89/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [5] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 89/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 89 submitted successfully"
else
    echo "[ERROR] Multirun job 89 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 90/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [6] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 90/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 90 submitted successfully"
else
    echo "[ERROR] Multirun job 90 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 91/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [3] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 91/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 91 submitted successfully"
else
    echo "[ERROR] Multirun job 91 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 92/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [3] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 92/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 92 submitted successfully"
else
    echo "[ERROR] Multirun job 92 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 93/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [6] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 93/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 93 submitted successfully"
else
    echo "[ERROR] Multirun job 93 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 94/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [6] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 94/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 94 submitted successfully"
else
    echo "[ERROR] Multirun job 94 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 95/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [2] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 95/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 95 submitted successfully"
else
    echo "[ERROR] Multirun job 95 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 96/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [1] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 96/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 96 submitted successfully"
else
    echo "[ERROR] Multirun job 96 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 97/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [6] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 97/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 97 submitted successfully"
else
    echo "[ERROR] Multirun job 97 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 98/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [6] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 98/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 98 submitted successfully"
else
    echo "[ERROR] Multirun job 98 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 99/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [5] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 99/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 99 submitted successfully"
else
    echo "[ERROR] Multirun job 99 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 100/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [1] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 100/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 100 submitted successfully"
else
    echo "[ERROR] Multirun job 100 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 101/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [5] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 101/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 101 submitted successfully"
else
    echo "[ERROR] Multirun job 101 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 102/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [1] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 102/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 102 submitted successfully"
else
    echo "[ERROR] Multirun job 102 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 103/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [4] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 103/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 103 submitted successfully"
else
    echo "[ERROR] Multirun job 103 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 104/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [1] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 104/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 104 submitted successfully"
else
    echo "[ERROR] Multirun job 104 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 105/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [3] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 105/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 105 submitted successfully"
else
    echo "[ERROR] Multirun job 105 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 106/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [5] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 106/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 106 submitted successfully"
else
    echo "[ERROR] Multirun job 106 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 107/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [7] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 107/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 107 submitted successfully"
else
    echo "[ERROR] Multirun job 107 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 108/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [2] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 108/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 108 submitted successfully"
else
    echo "[ERROR] Multirun job 108 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 109/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [9] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 109/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 109 submitted successfully"
else
    echo "[ERROR] Multirun job 109 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 110/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [9] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 110/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 110 submitted successfully"
else
    echo "[ERROR] Multirun job 110 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 111/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [8] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 111/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 111 submitted successfully"
else
    echo "[ERROR] Multirun job 111 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 112/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [8] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 112/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 112 submitted successfully"
else
    echo "[ERROR] Multirun job 112 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 113/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [8] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 113/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 113 submitted successfully"
else
    echo "[ERROR] Multirun job 113 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 114/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [2] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 114/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 114 submitted successfully"
else
    echo "[ERROR] Multirun job 114 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 115/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [9] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 115/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 115 submitted successfully"
else
    echo "[ERROR] Multirun job 115 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 116/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [9] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 116/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 116 submitted successfully"
else
    echo "[ERROR] Multirun job 116 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 117/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [4] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 117/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 117 submitted successfully"
else
    echo "[ERROR] Multirun job 117 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 118/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [8] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 118/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 118 submitted successfully"
else
    echo "[ERROR] Multirun job 118 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 119/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [1] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 119/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 119 submitted successfully"
else
    echo "[ERROR] Multirun job 119 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 120/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [3] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 120/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 120 submitted successfully"
else
    echo "[ERROR] Multirun job 120 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 121/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [6] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 121/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 121 submitted successfully"
else
    echo "[ERROR] Multirun job 121 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 122/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [6] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 122/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 122 submitted successfully"
else
    echo "[ERROR] Multirun job 122 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 123/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [3] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 123/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 123 submitted successfully"
else
    echo "[ERROR] Multirun job 123 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 124/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [7] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 124/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 124 submitted successfully"
else
    echo "[ERROR] Multirun job 124 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 125/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [8] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 125/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 125 submitted successfully"
else
    echo "[ERROR] Multirun job 125 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 126/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [1] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 126/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 126 submitted successfully"
else
    echo "[ERROR] Multirun job 126 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 127/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [4] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 127/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 127 submitted successfully"
else
    echo "[ERROR] Multirun job 127 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 128/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [2] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 128/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 128 submitted successfully"
else
    echo "[ERROR] Multirun job 128 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 129/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [5] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 129/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 129 submitted successfully"
else
    echo "[ERROR] Multirun job 129 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 130/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [9] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 130/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 130 submitted successfully"
else
    echo "[ERROR] Multirun job 130 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 131/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [2] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 131/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 131 submitted successfully"
else
    echo "[ERROR] Multirun job 131 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 132/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [4] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 132/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 132 submitted successfully"
else
    echo "[ERROR] Multirun job 132 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 133/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [1] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 133/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 133 submitted successfully"
else
    echo "[ERROR] Multirun job 133 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 134/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [8] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 134/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 134 submitted successfully"
else
    echo "[ERROR] Multirun job 134 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 135/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [5] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 135/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 135 submitted successfully"
else
    echo "[ERROR] Multirun job 135 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 136/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [4] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 136/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 136 submitted successfully"
else
    echo "[ERROR] Multirun job 136 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 137/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [7] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 137/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 137 submitted successfully"
else
    echo "[ERROR] Multirun job 137 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 138/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [2] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 138/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 138 submitted successfully"
else
    echo "[ERROR] Multirun job 138 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 139/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [6] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 139/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 139 submitted successfully"
else
    echo "[ERROR] Multirun job 139 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 140/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [7] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 140/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 140 submitted successfully"
else
    echo "[ERROR] Multirun job 140 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 141/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [6] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 141/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 141 submitted successfully"
else
    echo "[ERROR] Multirun job 141 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 142/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [8] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 142/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 142 submitted successfully"
else
    echo "[ERROR] Multirun job 142 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 143/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [7] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 143/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 143 submitted successfully"
else
    echo "[ERROR] Multirun job 143 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 144/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [1] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 144/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 144 submitted successfully"
else
    echo "[ERROR] Multirun job 144 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 145/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [9] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 145/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 145 submitted successfully"
else
    echo "[ERROR] Multirun job 145 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 146/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [2] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 146/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 146 submitted successfully"
else
    echo "[ERROR] Multirun job 146 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 147/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [6] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 147/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 147 submitted successfully"
else
    echo "[ERROR] Multirun job 147 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 148/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [5] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 148/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 148 submitted successfully"
else
    echo "[ERROR] Multirun job 148 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 149/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [6] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 149/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 149 submitted successfully"
else
    echo "[ERROR] Multirun job 149 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 150/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [2] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 150/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 150 submitted successfully"
else
    echo "[ERROR] Multirun job 150 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 151/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [9] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 151/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 151 submitted successfully"
else
    echo "[ERROR] Multirun job 151 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 152/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [6] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 152/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 152 submitted successfully"
else
    echo "[ERROR] Multirun job 152 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 153/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [8] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 153/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 153 submitted successfully"
else
    echo "[ERROR] Multirun job 153 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 154/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [8] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 154/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 154 submitted successfully"
else
    echo "[ERROR] Multirun job 154 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 155/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [7] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 155/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 155 submitted successfully"
else
    echo "[ERROR] Multirun job 155 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 156/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [7] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 156/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 156 submitted successfully"
else
    echo "[ERROR] Multirun job 156 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 157/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [5] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 157/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 157 submitted successfully"
else
    echo "[ERROR] Multirun job 157 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 158/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [4] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 158/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 158 submitted successfully"
else
    echo "[ERROR] Multirun job 158 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 159/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [7] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 159/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 159 submitted successfully"
else
    echo "[ERROR] Multirun job 159 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 160/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [7] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 160/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 160 submitted successfully"
else
    echo "[ERROR] Multirun job 160 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 161/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [4] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 161/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 161 submitted successfully"
else
    echo "[ERROR] Multirun job 161 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 162/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [7] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 162/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 162 submitted successfully"
else
    echo "[ERROR] Multirun job 162 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 163/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [6] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 163/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 163 submitted successfully"
else
    echo "[ERROR] Multirun job 163 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 164/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [4] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 164/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 164 submitted successfully"
else
    echo "[ERROR] Multirun job 164 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 165/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [7] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 165/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 165 submitted successfully"
else
    echo "[ERROR] Multirun job 165 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 166/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [1] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 166/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 166 submitted successfully"
else
    echo "[ERROR] Multirun job 166 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 167/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [9] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 167/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 167 submitted successfully"
else
    echo "[ERROR] Multirun job 167 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 168/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [9] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 168/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 168 submitted successfully"
else
    echo "[ERROR] Multirun job 168 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 169/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [9] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 169/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 169 submitted successfully"
else
    echo "[ERROR] Multirun job 169 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 170/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [8] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 170/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 170 submitted successfully"
else
    echo "[ERROR] Multirun job 170 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 171/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [1] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 171/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 171 submitted successfully"
else
    echo "[ERROR] Multirun job 171 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 172/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [5] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 172/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 172 submitted successfully"
else
    echo "[ERROR] Multirun job 172 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 173/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [3] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 173/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 173 submitted successfully"
else
    echo "[ERROR] Multirun job 173 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 174/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [5] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 174/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 174 submitted successfully"
else
    echo "[ERROR] Multirun job 174 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 175/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [1] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 175/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 175 submitted successfully"
else
    echo "[ERROR] Multirun job 175 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 176/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [5] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 176/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 176 submitted successfully"
else
    echo "[ERROR] Multirun job 176 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 177/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [3] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 177/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 177 submitted successfully"
else
    echo "[ERROR] Multirun job 177 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 178/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [3] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 178/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 178 submitted successfully"
else
    echo "[ERROR] Multirun job 178 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 179/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [8] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 179/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 179 submitted successfully"
else
    echo "[ERROR] Multirun job 179 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 180/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [8] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 180/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 180 submitted successfully"
else
    echo "[ERROR] Multirun job 180 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 181/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [6] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 181/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 181 submitted successfully"
else
    echo "[ERROR] Multirun job 181 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 182/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [2] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 182/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 182 submitted successfully"
else
    echo "[ERROR] Multirun job 182 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 183/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [8] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 183/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 183 submitted successfully"
else
    echo "[ERROR] Multirun job 183 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 184/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [1] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 184/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 184 submitted successfully"
else
    echo "[ERROR] Multirun job 184 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 185/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [7] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 185/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 185 submitted successfully"
else
    echo "[ERROR] Multirun job 185 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 186/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [4] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 186/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 186 submitted successfully"
else
    echo "[ERROR] Multirun job 186 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 187/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [3] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 187/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 187 submitted successfully"
else
    echo "[ERROR] Multirun job 187 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 188/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [7] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 188/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 188 submitted successfully"
else
    echo "[ERROR] Multirun job 188 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 189/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [4] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 189/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 189 submitted successfully"
else
    echo "[ERROR] Multirun job 189 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 190/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [3] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 190/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 190 submitted successfully"
else
    echo "[ERROR] Multirun job 190 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 191/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [3] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 191/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 191 submitted successfully"
else
    echo "[ERROR] Multirun job 191 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 192/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [9] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 192/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 192 submitted successfully"
else
    echo "[ERROR] Multirun job 192 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 193/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [2] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 193/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 193 submitted successfully"
else
    echo "[ERROR] Multirun job 193 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 194/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [8] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 194/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 194 submitted successfully"
else
    echo "[ERROR] Multirun job 194 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 195/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [2] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 195/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 195 submitted successfully"
else
    echo "[ERROR] Multirun job 195 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 196/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [9] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 196/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 196 submitted successfully"
else
    echo "[ERROR] Multirun job 196 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 197/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [9] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 197/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 197 submitted successfully"
else
    echo "[ERROR] Multirun job 197 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 198/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [9] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 198/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 198 submitted successfully"
else
    echo "[ERROR] Multirun job 198 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 199/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [8] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 199/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 199 submitted successfully"
else
    echo "[ERROR] Multirun job 199 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 200/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [4] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 200/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 200 submitted successfully"
else
    echo "[ERROR] Multirun job 200 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 201/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [8] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 201/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 201 submitted successfully"
else
    echo "[ERROR] Multirun job 201 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 202/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [2] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 202/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 202 submitted successfully"
else
    echo "[ERROR] Multirun job 202 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 203/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [5] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 203/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 203 submitted successfully"
else
    echo "[ERROR] Multirun job 203 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 204/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [5] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 204/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 204 submitted successfully"
else
    echo "[ERROR] Multirun job 204 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 205/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [4] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 205/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 205 submitted successfully"
else
    echo "[ERROR] Multirun job 205 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 206/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [3] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 206/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 206 submitted successfully"
else
    echo "[ERROR] Multirun job 206 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 207/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [3] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 207/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 207 submitted successfully"
else
    echo "[ERROR] Multirun job 207 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 208/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [6] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 208/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 208 submitted successfully"
else
    echo "[ERROR] Multirun job 208 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 209/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [3] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 209/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 209 submitted successfully"
else
    echo "[ERROR] Multirun job 209 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 210/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [2] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 210/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 210 submitted successfully"
else
    echo "[ERROR] Multirun job 210 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 211/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [3] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 211/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 211 submitted successfully"
else
    echo "[ERROR] Multirun job 211 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 212/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [7] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 212/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 212 submitted successfully"
else
    echo "[ERROR] Multirun job 212 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 213/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [1] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 213/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 213 submitted successfully"
else
    echo "[ERROR] Multirun job 213 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 214/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [6] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 214/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 214 submitted successfully"
else
    echo "[ERROR] Multirun job 214 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 215/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [2] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 215/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 215 submitted successfully"
else
    echo "[ERROR] Multirun job 215 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 216/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [2] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 216/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 216 submitted successfully"
else
    echo "[ERROR] Multirun job 216 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 217/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [9] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 217/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 217 submitted successfully"
else
    echo "[ERROR] Multirun job 217 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 218/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [4] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 218/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 218 submitted successfully"
else
    echo "[ERROR] Multirun job 218 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 219/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [1] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 219/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 219 submitted successfully"
else
    echo "[ERROR] Multirun job 219 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 220/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [5] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 220/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 220 submitted successfully"
else
    echo "[ERROR] Multirun job 220 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 221/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [6] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 221/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 221 submitted successfully"
else
    echo "[ERROR] Multirun job 221 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 222/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [5] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 222/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 222 submitted successfully"
else
    echo "[ERROR] Multirun job 222 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 223/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [4] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 223/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 223 submitted successfully"
else
    echo "[ERROR] Multirun job 223 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 224/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [1] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 224/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 224 submitted successfully"
else
    echo "[ERROR] Multirun job 224 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 225/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [9] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 225/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 225 submitted successfully"
else
    echo "[ERROR] Multirun job 225 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 226/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [8] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 226/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 226 submitted successfully"
else
    echo "[ERROR] Multirun job 226 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 227/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [7] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 227/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 227 submitted successfully"
else
    echo "[ERROR] Multirun job 227 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 228/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [9] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 228/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 228 submitted successfully"
else
    echo "[ERROR] Multirun job 228 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 229/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [8] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 229/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 229 submitted successfully"
else
    echo "[ERROR] Multirun job 229 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 230/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [9] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 230/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 230 submitted successfully"
else
    echo "[ERROR] Multirun job 230 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 231/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [3] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 231/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 231 submitted successfully"
else
    echo "[ERROR] Multirun job 231 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 232/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [7] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 232/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 232 submitted successfully"
else
    echo "[ERROR] Multirun job 232 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 233/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [6] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 233/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 233 submitted successfully"
else
    echo "[ERROR] Multirun job 233 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 234/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [9] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 234/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 234 submitted successfully"
else
    echo "[ERROR] Multirun job 234 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 235/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [6] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 235/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 235 submitted successfully"
else
    echo "[ERROR] Multirun job 235 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 236/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [1] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 236/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 236 submitted successfully"
else
    echo "[ERROR] Multirun job 236 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 237/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [7] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 237/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 237 submitted successfully"
else
    echo "[ERROR] Multirun job 237 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 238/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [6] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 238/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 238 submitted successfully"
else
    echo "[ERROR] Multirun job 238 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 239/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [9] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 239/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 239 submitted successfully"
else
    echo "[ERROR] Multirun job 239 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 240/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [6] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 240/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 240 submitted successfully"
else
    echo "[ERROR] Multirun job 240 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 241/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [1] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 241/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 241 submitted successfully"
else
    echo "[ERROR] Multirun job 241 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 242/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [5] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 242/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 242 submitted successfully"
else
    echo "[ERROR] Multirun job 242 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 243/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [4] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 243/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 243 submitted successfully"
else
    echo "[ERROR] Multirun job 243 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 244/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [7] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 244/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 244 submitted successfully"
else
    echo "[ERROR] Multirun job 244 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 245/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [8] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 245/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 245 submitted successfully"
else
    echo "[ERROR] Multirun job 245 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 246/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [4] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 246/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 246 submitted successfully"
else
    echo "[ERROR] Multirun job 246 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 247/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [1] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 247/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 247 submitted successfully"
else
    echo "[ERROR] Multirun job 247 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 248/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [5] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 248/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 248 submitted successfully"
else
    echo "[ERROR] Multirun job 248 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 249/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [5] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 249/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 249 submitted successfully"
else
    echo "[ERROR] Multirun job 249 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 250/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [7] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 250/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 250 submitted successfully"
else
    echo "[ERROR] Multirun job 250 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 251/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [6] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 251/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 251 submitted successfully"
else
    echo "[ERROR] Multirun job 251 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 252/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [4] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 252/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 252 submitted successfully"
else
    echo "[ERROR] Multirun job 252 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 253/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [7] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 253/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 253 submitted successfully"
else
    echo "[ERROR] Multirun job 253 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 254/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [8] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 254/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 254 submitted successfully"
else
    echo "[ERROR] Multirun job 254 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 255/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [1] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 255/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 255 submitted successfully"
else
    echo "[ERROR] Multirun job 255 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 256/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [9] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 256/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 256 submitted successfully"
else
    echo "[ERROR] Multirun job 256 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 257/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [9] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 257/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 257 submitted successfully"
else
    echo "[ERROR] Multirun job 257 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 258/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [3] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 258/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 258 submitted successfully"
else
    echo "[ERROR] Multirun job 258 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 259/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [8] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 259/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 259 submitted successfully"
else
    echo "[ERROR] Multirun job 259 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 260/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [1] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 260/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 260 submitted successfully"
else
    echo "[ERROR] Multirun job 260 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 261/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [1] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 261/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 261 submitted successfully"
else
    echo "[ERROR] Multirun job 261 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 262/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [5] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 262/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 262 submitted successfully"
else
    echo "[ERROR] Multirun job 262 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 263/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [2] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 263/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 263 submitted successfully"
else
    echo "[ERROR] Multirun job 263 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 264/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [1] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 264/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 264 submitted successfully"
else
    echo "[ERROR] Multirun job 264 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 265/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [3] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 265/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 265 submitted successfully"
else
    echo "[ERROR] Multirun job 265 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 266/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [8] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 266/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 266 submitted successfully"
else
    echo "[ERROR] Multirun job 266 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 267/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [8] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 267/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 267 submitted successfully"
else
    echo "[ERROR] Multirun job 267 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 268/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [9] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 268/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 268 submitted successfully"
else
    echo "[ERROR] Multirun job 268 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 269/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [2] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 269/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 269 submitted successfully"
else
    echo "[ERROR] Multirun job 269 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 270/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [8] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 270/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 270 submitted successfully"
else
    echo "[ERROR] Multirun job 270 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 271/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [8] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 271/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 271 submitted successfully"
else
    echo "[ERROR] Multirun job 271 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 272/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [5] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 272/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 272 submitted successfully"
else
    echo "[ERROR] Multirun job 272 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 273/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [7] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 273/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 273 submitted successfully"
else
    echo "[ERROR] Multirun job 273 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 274/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [4] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 274/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 274 submitted successfully"
else
    echo "[ERROR] Multirun job 274 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 275/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [3] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 275/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 275 submitted successfully"
else
    echo "[ERROR] Multirun job 275 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 276/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [2] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 276/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 276 submitted successfully"
else
    echo "[ERROR] Multirun job 276 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 277/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [7] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 277/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 277 submitted successfully"
else
    echo "[ERROR] Multirun job 277 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 278/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [2] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 278/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 278 submitted successfully"
else
    echo "[ERROR] Multirun job 278 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 279/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [9] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 279/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 279 submitted successfully"
else
    echo "[ERROR] Multirun job 279 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 280/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [1] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 280/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 280 submitted successfully"
else
    echo "[ERROR] Multirun job 280 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 281/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [7] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 281/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 281 submitted successfully"
else
    echo "[ERROR] Multirun job 281 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 282/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [2] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 282/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 282 submitted successfully"
else
    echo "[ERROR] Multirun job 282 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 283/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [5] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 283/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 283 submitted successfully"
else
    echo "[ERROR] Multirun job 283 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 284/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [5] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 284/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 284 submitted successfully"
else
    echo "[ERROR] Multirun job 284 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 285/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [1] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 285/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 285 submitted successfully"
else
    echo "[ERROR] Multirun job 285 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 286/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [4] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 286/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 286 submitted successfully"
else
    echo "[ERROR] Multirun job 286 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 287/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [7] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 287/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 287 submitted successfully"
else
    echo "[ERROR] Multirun job 287 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 288/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [3] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 288/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 288 submitted successfully"
else
    echo "[ERROR] Multirun job 288 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 289/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [3] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 289/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 289 submitted successfully"
else
    echo "[ERROR] Multirun job 289 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 290/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [5] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 290/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 290 submitted successfully"
else
    echo "[ERROR] Multirun job 290 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 291/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [7] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 291/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 291 submitted successfully"
else
    echo "[ERROR] Multirun job 291 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 292/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [3] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 292/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 292 submitted successfully"
else
    echo "[ERROR] Multirun job 292 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 293/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [2] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 293/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 293 submitted successfully"
else
    echo "[ERROR] Multirun job 293 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 294/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [6] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 294/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 294 submitted successfully"
else
    echo "[ERROR] Multirun job 294 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 295/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [6] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 295/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 295 submitted successfully"
else
    echo "[ERROR] Multirun job 295 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 296/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [2] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 296/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 296 submitted successfully"
else
    echo "[ERROR] Multirun job 296 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 297/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [2] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 297/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 297 submitted successfully"
else
    echo "[ERROR] Multirun job 297 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 298/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [8] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 298/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 298 submitted successfully"
else
    echo "[ERROR] Multirun job 298 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 299/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [6] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 299/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 299 submitted successfully"
else
    echo "[ERROR] Multirun job 299 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 300/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [5] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 300/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 300 submitted successfully"
else
    echo "[ERROR] Multirun job 300 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 301/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [3] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 301/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 301 submitted successfully"
else
    echo "[ERROR] Multirun job 301 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 302/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [7] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 302/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 302 submitted successfully"
else
    echo "[ERROR] Multirun job 302 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 303/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [4] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 303/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 303 submitted successfully"
else
    echo "[ERROR] Multirun job 303 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 304/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [1] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 304/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 304 submitted successfully"
else
    echo "[ERROR] Multirun job 304 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 305/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [4] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 305/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 305 submitted successfully"
else
    echo "[ERROR] Multirun job 305 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 306/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [7] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 306/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 306 submitted successfully"
else
    echo "[ERROR] Multirun job 306 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 307/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [5] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 307/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 307 submitted successfully"
else
    echo "[ERROR] Multirun job 307 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 308/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [3] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 308/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 308 submitted successfully"
else
    echo "[ERROR] Multirun job 308 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 309/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [9] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 309/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 309 submitted successfully"
else
    echo "[ERROR] Multirun job 309 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 310/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [9] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 310/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 310 submitted successfully"
else
    echo "[ERROR] Multirun job 310 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 311/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [2] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 311/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 311 submitted successfully"
else
    echo "[ERROR] Multirun job 311 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 312/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [8] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 312/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 312 submitted successfully"
else
    echo "[ERROR] Multirun job 312 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 313/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [9] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 313/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 313 submitted successfully"
else
    echo "[ERROR] Multirun job 313 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 314/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [4] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 314/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 314 submitted successfully"
else
    echo "[ERROR] Multirun job 314 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 315/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [9] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 315/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 315 submitted successfully"
else
    echo "[ERROR] Multirun job 315 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 316/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [3] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 316/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 316 submitted successfully"
else
    echo "[ERROR] Multirun job 316 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 317/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [8] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 317/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 317 submitted successfully"
else
    echo "[ERROR] Multirun job 317 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 318/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [5] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 318/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 318 submitted successfully"
else
    echo "[ERROR] Multirun job 318 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 319/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [4] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 319/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 319 submitted successfully"
else
    echo "[ERROR] Multirun job 319 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 320/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [3] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 320/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 320 submitted successfully"
else
    echo "[ERROR] Multirun job 320 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 321/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [6] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 321/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 321 submitted successfully"
else
    echo "[ERROR] Multirun job 321 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 322/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [3] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 322/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 322 submitted successfully"
else
    echo "[ERROR] Multirun job 322 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 323/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [6] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 323/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 323 submitted successfully"
else
    echo "[ERROR] Multirun job 323 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 324/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [3] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 324/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 324 submitted successfully"
else
    echo "[ERROR] Multirun job 324 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 325/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [7] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 325/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 325 submitted successfully"
else
    echo "[ERROR] Multirun job 325 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 326/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [3] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 326/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 326 submitted successfully"
else
    echo "[ERROR] Multirun job 326 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 327/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [6] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 327/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 327 submitted successfully"
else
    echo "[ERROR] Multirun job 327 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 328/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [6] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 328/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 328 submitted successfully"
else
    echo "[ERROR] Multirun job 328 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 329/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [2] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 329/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 329 submitted successfully"
else
    echo "[ERROR] Multirun job 329 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 330/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [1] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 330/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 330 submitted successfully"
else
    echo "[ERROR] Multirun job 330 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 331/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [1] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 331/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 331 submitted successfully"
else
    echo "[ERROR] Multirun job 331 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 332/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [4] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 332/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 332 submitted successfully"
else
    echo "[ERROR] Multirun job 332 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 333/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [1] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 333/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 333 submitted successfully"
else
    echo "[ERROR] Multirun job 333 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 334/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [5] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 334/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 334 submitted successfully"
else
    echo "[ERROR] Multirun job 334 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 335/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [5] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 335/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 335 submitted successfully"
else
    echo "[ERROR] Multirun job 335 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 336/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [1] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 336/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 336 submitted successfully"
else
    echo "[ERROR] Multirun job 336 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 337/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [4] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 337/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 337 submitted successfully"
else
    echo "[ERROR] Multirun job 337 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 338/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [1] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 338/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 338 submitted successfully"
else
    echo "[ERROR] Multirun job 338 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 339/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [4] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 339/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 339 submitted successfully"
else
    echo "[ERROR] Multirun job 339 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 340/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [3] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 340/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 340 submitted successfully"
else
    echo "[ERROR] Multirun job 340 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 341/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [7] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 341/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 341 submitted successfully"
else
    echo "[ERROR] Multirun job 341 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 342/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [9] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 342/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 342 submitted successfully"
else
    echo "[ERROR] Multirun job 342 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 343/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [9] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 343/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 343 submitted successfully"
else
    echo "[ERROR] Multirun job 343 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 344/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [8] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 344/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 344 submitted successfully"
else
    echo "[ERROR] Multirun job 344 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 345/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [2] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 345/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 345 submitted successfully"
else
    echo "[ERROR] Multirun job 345 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 346/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [5] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 346/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 346 submitted successfully"
else
    echo "[ERROR] Multirun job 346 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 347/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [2] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 347/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 347 submitted successfully"
else
    echo "[ERROR] Multirun job 347 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 348/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [1] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 348/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 348 submitted successfully"
else
    echo "[ERROR] Multirun job 348 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 349/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [3] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 349/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 349 submitted successfully"
else
    echo "[ERROR] Multirun job 349 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 350/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [7] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 350/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 350 submitted successfully"
else
    echo "[ERROR] Multirun job 350 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 351/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [9] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 351/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 351 submitted successfully"
else
    echo "[ERROR] Multirun job 351 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 352/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [9] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 352/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 352 submitted successfully"
else
    echo "[ERROR] Multirun job 352 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 353/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [2] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 353/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 353 submitted successfully"
else
    echo "[ERROR] Multirun job 353 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 354/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [8] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 354/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 354 submitted successfully"
else
    echo "[ERROR] Multirun job 354 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 355/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [1] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 355/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 355 submitted successfully"
else
    echo "[ERROR] Multirun job 355 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 356/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [5] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 356/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 356 submitted successfully"
else
    echo "[ERROR] Multirun job 356 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 357/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [4] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 357/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 357 submitted successfully"
else
    echo "[ERROR] Multirun job 357 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 358/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [5] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 358/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 358 submitted successfully"
else
    echo "[ERROR] Multirun job 358 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 359/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [9] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 359/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 359 submitted successfully"
else
    echo "[ERROR] Multirun job 359 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 360/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [2] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 360/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 360 submitted successfully"
else
    echo "[ERROR] Multirun job 360 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 361/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [4] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 361/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 361 submitted successfully"
else
    echo "[ERROR] Multirun job 361 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 362/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [1] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 362/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 362 submitted successfully"
else
    echo "[ERROR] Multirun job 362 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 363/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [5] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 363/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 363 submitted successfully"
else
    echo "[ERROR] Multirun job 363 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 364/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [7] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 364/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 364 submitted successfully"
else
    echo "[ERROR] Multirun job 364 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 365/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [2] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 365/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 365 submitted successfully"
else
    echo "[ERROR] Multirun job 365 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 366/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [6] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 366/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 366 submitted successfully"
else
    echo "[ERROR] Multirun job 366 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 367/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [7] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 367/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 367 submitted successfully"
else
    echo "[ERROR] Multirun job 367 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 368/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [2] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 368/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 368 submitted successfully"
else
    echo "[ERROR] Multirun job 368 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 369/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [1] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 369/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 369 submitted successfully"
else
    echo "[ERROR] Multirun job 369 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 370/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [6] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 370/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 370 submitted successfully"
else
    echo "[ERROR] Multirun job 370 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 371/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [8] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 371/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 371 submitted successfully"
else
    echo "[ERROR] Multirun job 371 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 372/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [5] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 372/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 372 submitted successfully"
else
    echo "[ERROR] Multirun job 372 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 373/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [7] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 373/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 373 submitted successfully"
else
    echo "[ERROR] Multirun job 373 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 374/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [1] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 374/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 374 submitted successfully"
else
    echo "[ERROR] Multirun job 374 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 375/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [2] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 375/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 375 submitted successfully"
else
    echo "[ERROR] Multirun job 375 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 376/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [5] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 376/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 376 submitted successfully"
else
    echo "[ERROR] Multirun job 376 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 377/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [4] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 377/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 377 submitted successfully"
else
    echo "[ERROR] Multirun job 377 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 378/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [7] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 378/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 378 submitted successfully"
else
    echo "[ERROR] Multirun job 378 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 379/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [6] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 379/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 379 submitted successfully"
else
    echo "[ERROR] Multirun job 379 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 380/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [8] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 380/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 380 submitted successfully"
else
    echo "[ERROR] Multirun job 380 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 381/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [2] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 381/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 381 submitted successfully"
else
    echo "[ERROR] Multirun job 381 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 382/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [9] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 382/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 382 submitted successfully"
else
    echo "[ERROR] Multirun job 382 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 383/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [6] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 383/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 383 submitted successfully"
else
    echo "[ERROR] Multirun job 383 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 384/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [8] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 384/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 384 submitted successfully"
else
    echo "[ERROR] Multirun job 384 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 385/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [8] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 385/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 385 submitted successfully"
else
    echo "[ERROR] Multirun job 385 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 386/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [8] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 386/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 386 submitted successfully"
else
    echo "[ERROR] Multirun job 386 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 387/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [7] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 387/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 387 submitted successfully"
else
    echo "[ERROR] Multirun job 387 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 388/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [5] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 388/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 388 submitted successfully"
else
    echo "[ERROR] Multirun job 388 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 389/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [3] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 389/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 389 submitted successfully"
else
    echo "[ERROR] Multirun job 389 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 390/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [7] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 390/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 390 submitted successfully"
else
    echo "[ERROR] Multirun job 390 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 391/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [4] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 391/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 391 submitted successfully"
else
    echo "[ERROR] Multirun job 391 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 392/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [7] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 392/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 392 submitted successfully"
else
    echo "[ERROR] Multirun job 392 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 393/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [9] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 393/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 393 submitted successfully"
else
    echo "[ERROR] Multirun job 393 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 394/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [2] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 394/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 394 submitted successfully"
else
    echo "[ERROR] Multirun job 394 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 395/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [9] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 395/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 395 submitted successfully"
else
    echo "[ERROR] Multirun job 395 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 396/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [1] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 396/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 396 submitted successfully"
else
    echo "[ERROR] Multirun job 396 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 397/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [8] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 397/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 397 submitted successfully"
else
    echo "[ERROR] Multirun job 397 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 398/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [5] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 398/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 398 submitted successfully"
else
    echo "[ERROR] Multirun job 398 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 399/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [4] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 399/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 399 submitted successfully"
else
    echo "[ERROR] Multirun job 399 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 400/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [5] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 400/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 400 submitted successfully"
else
    echo "[ERROR] Multirun job 400 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 401/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [1] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 401/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 401 submitted successfully"
else
    echo "[ERROR] Multirun job 401 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 402/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [6] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 402/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 402 submitted successfully"
else
    echo "[ERROR] Multirun job 402 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 403/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [3] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 403/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 403 submitted successfully"
else
    echo "[ERROR] Multirun job 403 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 404/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [3] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 404/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 404 submitted successfully"
else
    echo "[ERROR] Multirun job 404 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 405/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [6] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 405/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 405 submitted successfully"
else
    echo "[ERROR] Multirun job 405 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 406/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [3] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 406/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 406 submitted successfully"
else
    echo "[ERROR] Multirun job 406 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 407/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [6] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 407/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 407 submitted successfully"
else
    echo "[ERROR] Multirun job 407 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 408/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [3] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 408/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 408 submitted successfully"
else
    echo "[ERROR] Multirun job 408 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 409/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [2] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 409/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 409 submitted successfully"
else
    echo "[ERROR] Multirun job 409 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 410/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [8] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 410/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 410 submitted successfully"
else
    echo "[ERROR] Multirun job 410 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 411/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [6] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 411/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 411 submitted successfully"
else
    echo "[ERROR] Multirun job 411 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 412/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [2] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 412/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 412 submitted successfully"
else
    echo "[ERROR] Multirun job 412 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 413/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [5] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 413/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 413 submitted successfully"
else
    echo "[ERROR] Multirun job 413 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 414/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [1] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 414/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 414 submitted successfully"
else
    echo "[ERROR] Multirun job 414 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 415/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [4] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 415/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 415 submitted successfully"
else
    echo "[ERROR] Multirun job 415 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 416/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [1] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 416/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 416 submitted successfully"
else
    echo "[ERROR] Multirun job 416 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 417/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [3] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 417/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 417 submitted successfully"
else
    echo "[ERROR] Multirun job 417 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 418/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [4] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 418/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 418 submitted successfully"
else
    echo "[ERROR] Multirun job 418 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 419/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [1] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 419/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 419 submitted successfully"
else
    echo "[ERROR] Multirun job 419 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 420/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [2] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 420/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 420 submitted successfully"
else
    echo "[ERROR] Multirun job 420 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 421/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [9] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 421/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 421 submitted successfully"
else
    echo "[ERROR] Multirun job 421 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 422/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [8] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 422/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 422 submitted successfully"
else
    echo "[ERROR] Multirun job 422 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 423/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [2] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 423/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 423 submitted successfully"
else
    echo "[ERROR] Multirun job 423 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 424/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [9] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 424/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 424 submitted successfully"
else
    echo "[ERROR] Multirun job 424 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 425/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [9] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 425/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 425 submitted successfully"
else
    echo "[ERROR] Multirun job 425 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 426/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [4] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 426/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 426 submitted successfully"
else
    echo "[ERROR] Multirun job 426 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 427/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [8] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 427/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 427 submitted successfully"
else
    echo "[ERROR] Multirun job 427 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 428/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [3] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 428/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 428 submitted successfully"
else
    echo "[ERROR] Multirun job 428 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 429/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [3] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 429/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 429 submitted successfully"
else
    echo "[ERROR] Multirun job 429 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 430/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [5] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 430/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 430 submitted successfully"
else
    echo "[ERROR] Multirun job 430 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 431/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [3] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 431/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 431 submitted successfully"
else
    echo "[ERROR] Multirun job 431 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 432/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [3] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 432/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 432 submitted successfully"
else
    echo "[ERROR] Multirun job 432 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 433/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [6] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 433/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 433 submitted successfully"
else
    echo "[ERROR] Multirun job 433 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 434/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [3] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 434/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 434 submitted successfully"
else
    echo "[ERROR] Multirun job 434 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 435/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [7] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 435/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 435 submitted successfully"
else
    echo "[ERROR] Multirun job 435 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 436/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [2] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 436/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 436 submitted successfully"
else
    echo "[ERROR] Multirun job 436 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 437/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [7] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 437/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 437 submitted successfully"
else
    echo "[ERROR] Multirun job 437 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 438/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [1] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 438/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 438 submitted successfully"
else
    echo "[ERROR] Multirun job 438 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 439/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [2] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 439/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 439 submitted successfully"
else
    echo "[ERROR] Multirun job 439 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 440/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [5] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 440/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 440 submitted successfully"
else
    echo "[ERROR] Multirun job 440 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 441/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [9] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 441/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 441 submitted successfully"
else
    echo "[ERROR] Multirun job 441 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 442/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [4] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 442/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 442 submitted successfully"
else
    echo "[ERROR] Multirun job 442 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 443/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [1] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 443/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 443 submitted successfully"
else
    echo "[ERROR] Multirun job 443 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 444/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [4] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 444/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 444 submitted successfully"
else
    echo "[ERROR] Multirun job 444 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 445/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [5] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 445/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 445 submitted successfully"
else
    echo "[ERROR] Multirun job 445 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 446/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [7] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 446/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 446 submitted successfully"
else
    echo "[ERROR] Multirun job 446 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 447/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [6] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 447/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 447 submitted successfully"
else
    echo "[ERROR] Multirun job 447 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 448/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [4] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 448/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 448 submitted successfully"
else
    echo "[ERROR] Multirun job 448 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 449/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [1] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 449/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 449 submitted successfully"
else
    echo "[ERROR] Multirun job 449 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 450/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [9] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 450/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 450 submitted successfully"
else
    echo "[ERROR] Multirun job 450 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 451/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [8] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 451/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 451 submitted successfully"
else
    echo "[ERROR] Multirun job 451 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 452/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [7] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 452/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 452 submitted successfully"
else
    echo "[ERROR] Multirun job 452 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 453/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [9] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 453/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 453 submitted successfully"
else
    echo "[ERROR] Multirun job 453 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 454/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [8] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 454/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 454 submitted successfully"
else
    echo "[ERROR] Multirun job 454 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 455/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [2] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 455/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 455 submitted successfully"
else
    echo "[ERROR] Multirun job 455 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 456/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [6] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 456/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 456 submitted successfully"
else
    echo "[ERROR] Multirun job 456 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 457/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [5] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 457/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 457 submitted successfully"
else
    echo "[ERROR] Multirun job 457 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 458/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [2] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 458/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 458 submitted successfully"
else
    echo "[ERROR] Multirun job 458 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 459/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [3] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 459/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 459 submitted successfully"
else
    echo "[ERROR] Multirun job 459 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 460/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [6] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 460/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 460 submitted successfully"
else
    echo "[ERROR] Multirun job 460 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 461/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [7] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 461/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 461 submitted successfully"
else
    echo "[ERROR] Multirun job 461 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 462/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [6] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 462/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 462 submitted successfully"
else
    echo "[ERROR] Multirun job 462 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 463/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [9] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 463/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 463 submitted successfully"
else
    echo "[ERROR] Multirun job 463 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 464/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [6] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 464/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 464 submitted successfully"
else
    echo "[ERROR] Multirun job 464 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 465/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [1] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 465/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 465 submitted successfully"
else
    echo "[ERROR] Multirun job 465 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 466/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [8] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 466/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 466 submitted successfully"
else
    echo "[ERROR] Multirun job 466 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 467/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [8] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 467/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 467 submitted successfully"
else
    echo "[ERROR] Multirun job 467 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 468/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [7] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 468/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 468 submitted successfully"
else
    echo "[ERROR] Multirun job 468 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 469/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [5] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 469/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 469 submitted successfully"
else
    echo "[ERROR] Multirun job 469 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 470/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [4] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 470/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 470 submitted successfully"
else
    echo "[ERROR] Multirun job 470 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 471/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [8] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 471/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 471 submitted successfully"
else
    echo "[ERROR] Multirun job 471 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 472/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [4] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 472/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 472 submitted successfully"
else
    echo "[ERROR] Multirun job 472 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 473/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [6] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 473/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 473 submitted successfully"
else
    echo "[ERROR] Multirun job 473 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 474/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [4] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 474/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 474 submitted successfully"
else
    echo "[ERROR] Multirun job 474 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 475/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [7] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 475/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 475 submitted successfully"
else
    echo "[ERROR] Multirun job 475 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 476/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [6] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 476/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 476 submitted successfully"
else
    echo "[ERROR] Multirun job 476 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 477/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [4] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 477/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 477 submitted successfully"
else
    echo "[ERROR] Multirun job 477 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 478/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [9] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 478/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 478 submitted successfully"
else
    echo "[ERROR] Multirun job 478 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 479/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [8] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 479/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 479 submitted successfully"
else
    echo "[ERROR] Multirun job 479 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 480/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [1] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 480/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 480 submitted successfully"
else
    echo "[ERROR] Multirun job 480 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 481/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [9] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 481/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 481 submitted successfully"
else
    echo "[ERROR] Multirun job 481 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 482/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [3] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 482/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 482 submitted successfully"
else
    echo "[ERROR] Multirun job 482 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 483/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [5] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 483/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 483 submitted successfully"
else
    echo "[ERROR] Multirun job 483 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 484/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [1] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 484/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 484 submitted successfully"
else
    echo "[ERROR] Multirun job 484 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 485/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [5] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 485/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 485 submitted successfully"
else
    echo "[ERROR] Multirun job 485 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 486/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [2] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 486/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 486 submitted successfully"
else
    echo "[ERROR] Multirun job 486 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 487/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [5] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 487/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 487 submitted successfully"
else
    echo "[ERROR] Multirun job 487 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 488/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [2] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 488/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 488 submitted successfully"
else
    echo "[ERROR] Multirun job 488 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 489/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [3] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 489/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 489 submitted successfully"
else
    echo "[ERROR] Multirun job 489 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 490/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [9] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 490/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 490 submitted successfully"
else
    echo "[ERROR] Multirun job 490 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 491/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [8] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 491/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 491 submitted successfully"
else
    echo "[ERROR] Multirun job 491 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 492/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [6] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 492/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 492 submitted successfully"
else
    echo "[ERROR] Multirun job 492 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 493/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [8] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 493/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 493 submitted successfully"
else
    echo "[ERROR] Multirun job 493 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 494/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [9] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 494/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 494 submitted successfully"
else
    echo "[ERROR] Multirun job 494 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 495/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [6] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 495/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 495 submitted successfully"
else
    echo "[ERROR] Multirun job 495 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 496/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [7] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 496/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 496 submitted successfully"
else
    echo "[ERROR] Multirun job 496 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 497/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [4] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 497/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 497 submitted successfully"
else
    echo "[ERROR] Multirun job 497 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 498/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [3] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 498/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 498 submitted successfully"
else
    echo "[ERROR] Multirun job 498 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 499/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [2] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 499/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 499 submitted successfully"
else
    echo "[ERROR] Multirun job 499 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 500/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [2] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 500/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 500 submitted successfully"
else
    echo "[ERROR] Multirun job 500 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 501/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [2] | Seed: 500
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 501/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 501 submitted successfully"
else
    echo "[ERROR] Multirun job 501 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 502/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [9] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 502/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 502 submitted successfully"
else
    echo "[ERROR] Multirun job 502 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 503/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [1] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 503/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 503 submitted successfully"
else
    echo "[ERROR] Multirun job 503 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 504/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [9] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 504/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 504 submitted successfully"
else
    echo "[ERROR] Multirun job 504 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 505/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [7] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 505/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 505 submitted successfully"
else
    echo "[ERROR] Multirun job 505 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 506/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [8] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 506/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 506 submitted successfully"
else
    echo "[ERROR] Multirun job 506 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 507/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [8] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 507/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 507 submitted successfully"
else
    echo "[ERROR] Multirun job 507 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 508/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [2] | Seed: 200
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 508/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 508 submitted successfully"
else
    echo "[ERROR] Multirun job 508 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 509/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [5] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 509/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 509 submitted successfully"
else
    echo "[ERROR] Multirun job 509 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 510/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [9] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 510/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 510 submitted successfully"
else
    echo "[ERROR] Multirun job 510 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 511/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [5] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 511/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 511 submitted successfully"
else
    echo "[ERROR] Multirun job 511 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 512/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [4] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 512/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 512 submitted successfully"
else
    echo "[ERROR] Multirun job 512 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 513/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [1] | Seed: 100
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 513/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 513 submitted successfully"
else
    echo "[ERROR] Multirun job 513 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 514/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [4] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 514/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 514 submitted successfully"
else
    echo "[ERROR] Multirun job 514 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 515/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [7] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 515/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 515 submitted successfully"
else
    echo "[ERROR] Multirun job 515 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 516/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [3] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 516/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 516 submitted successfully"
else
    echo "[ERROR] Multirun job 516 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 517/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [6] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 517/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 517 submitted successfully"
else
    echo "[ERROR] Multirun job 517 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 518/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [6] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 518/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 518 submitted successfully"
else
    echo "[ERROR] Multirun job 518 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 519/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [3] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 519/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 519 submitted successfully"
else
    echo "[ERROR] Multirun job 519 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 520/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [3] | Seed: 200
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 520/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 520 submitted successfully"
else
    echo "[ERROR] Multirun job 520 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 521/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [6] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 521/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 521 submitted successfully"
else
    echo "[ERROR] Multirun job 521 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 522/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [1] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 522/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 522 submitted successfully"
else
    echo "[ERROR] Multirun job 522 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 523/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [6] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 523/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 523 submitted successfully"
else
    echo "[ERROR] Multirun job 523 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 524/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [2] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 524/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 524 submitted successfully"
else
    echo "[ERROR] Multirun job 524 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 525/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [2] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 525/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 525 submitted successfully"
else
    echo "[ERROR] Multirun job 525 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 526/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [8] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 526/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 526 submitted successfully"
else
    echo "[ERROR] Multirun job 526 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 527/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [5] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 527/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 527 submitted successfully"
else
    echo "[ERROR] Multirun job 527 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 528/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [3] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 528/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 528 submitted successfully"
else
    echo "[ERROR] Multirun job 528 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 529/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [4] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 529/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 529 submitted successfully"
else
    echo "[ERROR] Multirun job 529 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 530/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [1] | Seed: 300
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 530/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 530 submitted successfully"
else
    echo "[ERROR] Multirun job 530 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 531/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [8] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 531/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 531 submitted successfully"
else
    echo "[ERROR] Multirun job 531 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 532/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [7] | Seed: 500
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 532/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 532 submitted successfully"
else
    echo "[ERROR] Multirun job 532 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 533/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [9] | Seed: 400
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 533/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 533 submitted successfully"
else
    echo "[ERROR] Multirun job 533 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 534/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [2] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 534/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 534 submitted successfully"
else
    echo "[ERROR] Multirun job 534 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 535/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [8] | Seed: 300
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 535/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 535 submitted successfully"
else
    echo "[ERROR] Multirun job 535 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 536/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [9] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 536/540..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 536 submitted successfully"
else
    echo "[ERROR] Multirun job 536 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 537/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [4] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 537/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 537 submitted successfully"
else
    echo "[ERROR] Multirun job 537 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 538/540
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [9] | Seed: 100
# Timeout: --time=0-01:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 538/540..."
sbatch --time=0-01:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 538 submitted successfully"
else
    echo "[ERROR] Multirun job 538 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 539/540
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [3] | Seed: 400
# Timeout: --time=0-04:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 539/540..."
sbatch --time=0-04:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 539 submitted successfully"
else
    echo "[ERROR] Multirun job 539 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 540/540
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [3] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 540/540..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 540 submitted successfully"
else
    echo "[ERROR] Multirun job 540 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

