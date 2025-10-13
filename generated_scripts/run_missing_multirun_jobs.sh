#!/bin/bash
# Generated multirun sbatch automation script
# Generated on: 2025-10-12 10:29:15
# Total missing multirun jobs: 135

set -e  # Exit on any error

echo "Starting multirun experiment automation..."
echo "Total multirun jobs to submit: 135"

# Multirun Job 1/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [2] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 1/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 1 submitted successfully"
else
    echo "[ERROR] Multirun job 1 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 2/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [3] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 2/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 2 submitted successfully"
else
    echo "[ERROR] Multirun job 2 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 3/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [8] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 3/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 3 submitted successfully"
else
    echo "[ERROR] Multirun job 3 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 4/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [9] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 4/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 4 submitted successfully"
else
    echo "[ERROR] Multirun job 4 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 5/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [9] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 5/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 5 submitted successfully"
else
    echo "[ERROR] Multirun job 5 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 6/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [6] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 6/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 6 submitted successfully"
else
    echo "[ERROR] Multirun job 6 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 7/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [7] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 7/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 7 submitted successfully"
else
    echo "[ERROR] Multirun job 7 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 8/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [4] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 8/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 8 submitted successfully"
else
    echo "[ERROR] Multirun job 8 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 9/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [1] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 9/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 9 submitted successfully"
else
    echo "[ERROR] Multirun job 9 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 10/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [5] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 10/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 10 submitted successfully"
else
    echo "[ERROR] Multirun job 10 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 11/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [2] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 11/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 11 submitted successfully"
else
    echo "[ERROR] Multirun job 11 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 12/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [3] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 12/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 12 submitted successfully"
else
    echo "[ERROR] Multirun job 12 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 13/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [1] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 13/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 13 submitted successfully"
else
    echo "[ERROR] Multirun job 13 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 14/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [5] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 14/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 14 submitted successfully"
else
    echo "[ERROR] Multirun job 14 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 15/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [2] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 15/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 15 submitted successfully"
else
    echo "[ERROR] Multirun job 15 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 16/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [7] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 16/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 16 submitted successfully"
else
    echo "[ERROR] Multirun job 16 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 17/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [1] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 17/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 17 submitted successfully"
else
    echo "[ERROR] Multirun job 17 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 18/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [1] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 18/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 18 submitted successfully"
else
    echo "[ERROR] Multirun job 18 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 19/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [8] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 19/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 19 submitted successfully"
else
    echo "[ERROR] Multirun job 19 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 20/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [5] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 20/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 20 submitted successfully"
else
    echo "[ERROR] Multirun job 20 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 21/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [2] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 21/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 21 submitted successfully"
else
    echo "[ERROR] Multirun job 21 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 22/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [6] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 22/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 22 submitted successfully"
else
    echo "[ERROR] Multirun job 22 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 23/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [9] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 23/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 23 submitted successfully"
else
    echo "[ERROR] Multirun job 23 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 24/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [8] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 24/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 24 submitted successfully"
else
    echo "[ERROR] Multirun job 24 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 25/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [6] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 25/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 25 submitted successfully"
else
    echo "[ERROR] Multirun job 25 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 26/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [3] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 26/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 26 submitted successfully"
else
    echo "[ERROR] Multirun job 26 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 27/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [5] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 27/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 27 submitted successfully"
else
    echo "[ERROR] Multirun job 27 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 28/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [1] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 28/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 28 submitted successfully"
else
    echo "[ERROR] Multirun job 28 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 29/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [5] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 29/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 29 submitted successfully"
else
    echo "[ERROR] Multirun job 29 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 30/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [4] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 30/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 30 submitted successfully"
else
    echo "[ERROR] Multirun job 30 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 31/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [1] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 31/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 31 submitted successfully"
else
    echo "[ERROR] Multirun job 31 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 32/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [8] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 32/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 32 submitted successfully"
else
    echo "[ERROR] Multirun job 32 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 33/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [8] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 33/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 33 submitted successfully"
else
    echo "[ERROR] Multirun job 33 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 34/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [6] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 34/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 34 submitted successfully"
else
    echo "[ERROR] Multirun job 34 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 35/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [1] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 35/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 35 submitted successfully"
else
    echo "[ERROR] Multirun job 35 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 36/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [8] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 36/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 36 submitted successfully"
else
    echo "[ERROR] Multirun job 36 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 37/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [6] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 37/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 37 submitted successfully"
else
    echo "[ERROR] Multirun job 37 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 38/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [4] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 38/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 38 submitted successfully"
else
    echo "[ERROR] Multirun job 38 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 39/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [7] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 39/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 39 submitted successfully"
else
    echo "[ERROR] Multirun job 39 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 40/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [2] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 40/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 40 submitted successfully"
else
    echo "[ERROR] Multirun job 40 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 41/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [3] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 41/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 41 submitted successfully"
else
    echo "[ERROR] Multirun job 41 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 42/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [8] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 42/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 42 submitted successfully"
else
    echo "[ERROR] Multirun job 42 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 43/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [7] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 43/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 43 submitted successfully"
else
    echo "[ERROR] Multirun job 43 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 44/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [9] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 44/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 44 submitted successfully"
else
    echo "[ERROR] Multirun job 44 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 45/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [2] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 45/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 45 submitted successfully"
else
    echo "[ERROR] Multirun job 45 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 46/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [9] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 46/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 46 submitted successfully"
else
    echo "[ERROR] Multirun job 46 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 47/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [3] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 47/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 47 submitted successfully"
else
    echo "[ERROR] Multirun job 47 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 48/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [6] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 48/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 48 submitted successfully"
else
    echo "[ERROR] Multirun job 48 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 49/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [7] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 49/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 49 submitted successfully"
else
    echo "[ERROR] Multirun job 49 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 50/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [4] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 50/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 50 submitted successfully"
else
    echo "[ERROR] Multirun job 50 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 51/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [7] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 51/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 51 submitted successfully"
else
    echo "[ERROR] Multirun job 51 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 52/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [9] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 52/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 52 submitted successfully"
else
    echo "[ERROR] Multirun job 52 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 53/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [6] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 53/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 53 submitted successfully"
else
    echo "[ERROR] Multirun job 53 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 54/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [3] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 54/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 54 submitted successfully"
else
    echo "[ERROR] Multirun job 54 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 55/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [7] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 55/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 55 submitted successfully"
else
    echo "[ERROR] Multirun job 55 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 56/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [4] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 56/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 56 submitted successfully"
else
    echo "[ERROR] Multirun job 56 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 57/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [3] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 57/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 57 submitted successfully"
else
    echo "[ERROR] Multirun job 57 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 58/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [5] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 58/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 58 submitted successfully"
else
    echo "[ERROR] Multirun job 58 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 59/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [2] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 59/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 59 submitted successfully"
else
    echo "[ERROR] Multirun job 59 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 60/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [7] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 60/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 60 submitted successfully"
else
    echo "[ERROR] Multirun job 60 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 61/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [1] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 61/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 61 submitted successfully"
else
    echo "[ERROR] Multirun job 61 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 62/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [1] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 62/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 62 submitted successfully"
else
    echo "[ERROR] Multirun job 62 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 63/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [5] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 63/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 63 submitted successfully"
else
    echo "[ERROR] Multirun job 63 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 64/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [2] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 64/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 64 submitted successfully"
else
    echo "[ERROR] Multirun job 64 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 65/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [2] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 65/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 65 submitted successfully"
else
    echo "[ERROR] Multirun job 65 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 66/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [9] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 66/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 66 submitted successfully"
else
    echo "[ERROR] Multirun job 66 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 67/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [6] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 67/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 67 submitted successfully"
else
    echo "[ERROR] Multirun job 67 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 68/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [3] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 68/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 68 submitted successfully"
else
    echo "[ERROR] Multirun job 68 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 69/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [2] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 69/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 69 submitted successfully"
else
    echo "[ERROR] Multirun job 69 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 70/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [5] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 70/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 70 submitted successfully"
else
    echo "[ERROR] Multirun job 70 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 71/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [5] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 71/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 71 submitted successfully"
else
    echo "[ERROR] Multirun job 71 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 72/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [4] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 72/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 72 submitted successfully"
else
    echo "[ERROR] Multirun job 72 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 73/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [1] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 73/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 73 submitted successfully"
else
    echo "[ERROR] Multirun job 73 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 74/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [8] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 74/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 74 submitted successfully"
else
    echo "[ERROR] Multirun job 74 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 75/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [5] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 75/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 75 submitted successfully"
else
    echo "[ERROR] Multirun job 75 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 76/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [2] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 76/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 76 submitted successfully"
else
    echo "[ERROR] Multirun job 76 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 77/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [6] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 77/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 77 submitted successfully"
else
    echo "[ERROR] Multirun job 77 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 78/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [1] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 78/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 78 submitted successfully"
else
    echo "[ERROR] Multirun job 78 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 79/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [5] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 79/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 79 submitted successfully"
else
    echo "[ERROR] Multirun job 79 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 80/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [3] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 80/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 80 submitted successfully"
else
    echo "[ERROR] Multirun job 80 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 81/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [8] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 81/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 81 submitted successfully"
else
    echo "[ERROR] Multirun job 81 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 82/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [6] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 82/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 82 submitted successfully"
else
    echo "[ERROR] Multirun job 82 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 83/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [4] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 83/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 83 submitted successfully"
else
    echo "[ERROR] Multirun job 83 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 84/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [1] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 84/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 84 submitted successfully"
else
    echo "[ERROR] Multirun job 84 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 85/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [5] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 85/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 85 submitted successfully"
else
    echo "[ERROR] Multirun job 85 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 86/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [4] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 86/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 86 submitted successfully"
else
    echo "[ERROR] Multirun job 86 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 87/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [8] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 87/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 87 submitted successfully"
else
    echo "[ERROR] Multirun job 87 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 88/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [4] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 88/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 88 submitted successfully"
else
    echo "[ERROR] Multirun job 88 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 89/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [8] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 89/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 89 submitted successfully"
else
    echo "[ERROR] Multirun job 89 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 90/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [9] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 90/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 90 submitted successfully"
else
    echo "[ERROR] Multirun job 90 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 91/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [2] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 91/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 91 submitted successfully"
else
    echo "[ERROR] Multirun job 91 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 92/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [3] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 92/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 92 submitted successfully"
else
    echo "[ERROR] Multirun job 92 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 93/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [8] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 93/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 93 submitted successfully"
else
    echo "[ERROR] Multirun job 93 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 94/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [7] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 94/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 94 submitted successfully"
else
    echo "[ERROR] Multirun job 94 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 95/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [2] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 95/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 95 submitted successfully"
else
    echo "[ERROR] Multirun job 95 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 96/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [9] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 96/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 96 submitted successfully"
else
    echo "[ERROR] Multirun job 96 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 97/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [3] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 97/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 97 submitted successfully"
else
    echo "[ERROR] Multirun job 97 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 98/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [6] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 98/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 98 submitted successfully"
else
    echo "[ERROR] Multirun job 98 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 99/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [7] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 99/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 99 submitted successfully"
else
    echo "[ERROR] Multirun job 99 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 100/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [4] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 100/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 100 submitted successfully"
else
    echo "[ERROR] Multirun job 100 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 101/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [9] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 101/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 101 submitted successfully"
else
    echo "[ERROR] Multirun job 101 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 102/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [6] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 102/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 102 submitted successfully"
else
    echo "[ERROR] Multirun job 102 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 103/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [3] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 103/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 103 submitted successfully"
else
    echo "[ERROR] Multirun job 103 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 104/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [7] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 104/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 104 submitted successfully"
else
    echo "[ERROR] Multirun job 104 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 105/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [4] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 105/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 105 submitted successfully"
else
    echo "[ERROR] Multirun job 105 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 106/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [7] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 106/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 106 submitted successfully"
else
    echo "[ERROR] Multirun job 106 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 107/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [1] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 107/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 107 submitted successfully"
else
    echo "[ERROR] Multirun job 107 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 108/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [5] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 108/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 108 submitted successfully"
else
    echo "[ERROR] Multirun job 108 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 109/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [2] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 109/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 109 submitted successfully"
else
    echo "[ERROR] Multirun job 109 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 110/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [3] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 110/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 110 submitted successfully"
else
    echo "[ERROR] Multirun job 110 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 111/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [9] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 111/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 111 submitted successfully"
else
    echo "[ERROR] Multirun job 111 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 112/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [7] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 112/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 112 submitted successfully"
else
    echo "[ERROR] Multirun job 112 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 113/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [7] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 113/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 113 submitted successfully"
else
    echo "[ERROR] Multirun job 113 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 114/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [9] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 114/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 114 submitted successfully"
else
    echo "[ERROR] Multirun job 114 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 115/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [4] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 115/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 115 submitted successfully"
else
    echo "[ERROR] Multirun job 115 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 116/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [1] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 116/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 116 submitted successfully"
else
    echo "[ERROR] Multirun job 116 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 117/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [8] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 117/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 117 submitted successfully"
else
    echo "[ERROR] Multirun job 117 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 118/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [5] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 118/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 118 submitted successfully"
else
    echo "[ERROR] Multirun job 118 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 119/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [2] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 119/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 119 submitted successfully"
else
    echo "[ERROR] Multirun job 119 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 120/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [6] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 120/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 120 submitted successfully"
else
    echo "[ERROR] Multirun job 120 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 121/135
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [9] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 121/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 121 submitted successfully"
else
    echo "[ERROR] Multirun job 121 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 122/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [3] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 122/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 122 submitted successfully"
else
    echo "[ERROR] Multirun job 122 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 123/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [6] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 123/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 123 submitted successfully"
else
    echo "[ERROR] Multirun job 123 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 124/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [3] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 124/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 124 submitted successfully"
else
    echo "[ERROR] Multirun job 124 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 125/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [1] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 125/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 125 submitted successfully"
else
    echo "[ERROR] Multirun job 125 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 126/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [5] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 126/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 126 submitted successfully"
else
    echo "[ERROR] Multirun job 126 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 127/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [4] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 127/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 127 submitted successfully"
else
    echo "[ERROR] Multirun job 127 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 128/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [8] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 128/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 128 submitted successfully"
else
    echo "[ERROR] Multirun job 128 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 129/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [4] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 129/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 129 submitted successfully"
else
    echo "[ERROR] Multirun job 129 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 130/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [6] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 130/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 130 submitted successfully"
else
    echo "[ERROR] Multirun job 130 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 131/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [9] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 131/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 131 submitted successfully"
else
    echo "[ERROR] Multirun job 131 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 132/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [4] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 132/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 132 submitted successfully"
else
    echo "[ERROR] Multirun job 132 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 133/135
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [9] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 133/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 133 submitted successfully"
else
    echo "[ERROR] Multirun job 133 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 134/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [8] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 134/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 134 submitted successfully"
else
    echo "[ERROR] Multirun job 134 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 135/135
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [7] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 135/135..."
sbatch --time=1-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 135 submitted successfully"
else
    echo "[ERROR] Multirun job 135 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

