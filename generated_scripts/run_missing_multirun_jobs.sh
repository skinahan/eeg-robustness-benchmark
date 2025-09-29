#!/bin/bash
# Generated multirun sbatch automation script
# Generated on: 2025-09-28 17:45:24
# Total missing multirun jobs: 108

set -e  # Exit on any error

echo "Starting multirun experiment automation..."
echo "Total multirun jobs to submit: 108"

# Multirun Job 1/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [6]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 1/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 1 submitted successfully"
else
    echo "[ERROR] Multirun job 1 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 2/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [7]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 2/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 2 submitted successfully"
else
    echo "[ERROR] Multirun job 2 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 3/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [4] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 3/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 3 submitted successfully"
else
    echo "[ERROR] Multirun job 3 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 4/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [2] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 4/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 4 submitted successfully"
else
    echo "[ERROR] Multirun job 4 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 5/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [1] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 5/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 5 submitted successfully"
else
    echo "[ERROR] Multirun job 5 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 6/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [3] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 6/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 6 submitted successfully"
else
    echo "[ERROR] Multirun job 6 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 7/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [5] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 7/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 7 submitted successfully"
else
    echo "[ERROR] Multirun job 7 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 8/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [7]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 8/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 8 submitted successfully"
else
    echo "[ERROR] Multirun job 8 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 9/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [8]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 9/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 9 submitted successfully"
else
    echo "[ERROR] Multirun job 9 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 10/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [6] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 10/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 10 submitted successfully"
else
    echo "[ERROR] Multirun job 10 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 11/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [1] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 11/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 11 submitted successfully"
else
    echo "[ERROR] Multirun job 11 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 12/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [2] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 12/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 12 submitted successfully"
else
    echo "[ERROR] Multirun job 12 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 13/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [6]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 13/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 13 submitted successfully"
else
    echo "[ERROR] Multirun job 13 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 14/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [3] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 14/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 14 submitted successfully"
else
    echo "[ERROR] Multirun job 14 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 15/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [7]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 15/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 15 submitted successfully"
else
    echo "[ERROR] Multirun job 15 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 16/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [6] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 16/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 16 submitted successfully"
else
    echo "[ERROR] Multirun job 16 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 17/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [4] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 17/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 17 submitted successfully"
else
    echo "[ERROR] Multirun job 17 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 18/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [5] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 18/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 18 submitted successfully"
else
    echo "[ERROR] Multirun job 18 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 19/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [6]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 19/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 19 submitted successfully"
else
    echo "[ERROR] Multirun job 19 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 20/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [8]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 20/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 20 submitted successfully"
else
    echo "[ERROR] Multirun job 20 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 21/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [8] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 21/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 21 submitted successfully"
else
    echo "[ERROR] Multirun job 21 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 22/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [9] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 22/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 22 submitted successfully"
else
    echo "[ERROR] Multirun job 22 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 23/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [8]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 23/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 23 submitted successfully"
else
    echo "[ERROR] Multirun job 23 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 24/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [3]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 24/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 24 submitted successfully"
else
    echo "[ERROR] Multirun job 24 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 25/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [9] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 25/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 25 submitted successfully"
else
    echo "[ERROR] Multirun job 25 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 26/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [1] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 26/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 26 submitted successfully"
else
    echo "[ERROR] Multirun job 26 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 27/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [2] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 27/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 27 submitted successfully"
else
    echo "[ERROR] Multirun job 27 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 28/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [1] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 28/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 28 submitted successfully"
else
    echo "[ERROR] Multirun job 28 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 29/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [6]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 29/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 29 submitted successfully"
else
    echo "[ERROR] Multirun job 29 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 30/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [7]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 30/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 30 submitted successfully"
else
    echo "[ERROR] Multirun job 30 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 31/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [3] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 31/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 31 submitted successfully"
else
    echo "[ERROR] Multirun job 31 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 32/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [6] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 32/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 32 submitted successfully"
else
    echo "[ERROR] Multirun job 32 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 33/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [8] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 33/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 33 submitted successfully"
else
    echo "[ERROR] Multirun job 33 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 34/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [9] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 34/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 34 submitted successfully"
else
    echo "[ERROR] Multirun job 34 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 35/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [7] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 35/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 35 submitted successfully"
else
    echo "[ERROR] Multirun job 35 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 36/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [6] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 36/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 36 submitted successfully"
else
    echo "[ERROR] Multirun job 36 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 37/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [1]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 37/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 37 submitted successfully"
else
    echo "[ERROR] Multirun job 37 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 38/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [3]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 38/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 38 submitted successfully"
else
    echo "[ERROR] Multirun job 38 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 39/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [5]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 39/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 39 submitted successfully"
else
    echo "[ERROR] Multirun job 39 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 40/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [9] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 40/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 40 submitted successfully"
else
    echo "[ERROR] Multirun job 40 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 41/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [6]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 41/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 41 submitted successfully"
else
    echo "[ERROR] Multirun job 41 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 42/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [5] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 42/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 42 submitted successfully"
else
    echo "[ERROR] Multirun job 42 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 43/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [8] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 43/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 43 submitted successfully"
else
    echo "[ERROR] Multirun job 43 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 44/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [9] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 44/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 44 submitted successfully"
else
    echo "[ERROR] Multirun job 44 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 45/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [3]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 45/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 45 submitted successfully"
else
    echo "[ERROR] Multirun job 45 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 46/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [1] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 46/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 46 submitted successfully"
else
    echo "[ERROR] Multirun job 46 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 47/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [4]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 47/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 47 submitted successfully"
else
    echo "[ERROR] Multirun job 47 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 48/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [8] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 48/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 48 submitted successfully"
else
    echo "[ERROR] Multirun job 48 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 49/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [9] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 49/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 49 submitted successfully"
else
    echo "[ERROR] Multirun job 49 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 50/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [1]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 50/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 50 submitted successfully"
else
    echo "[ERROR] Multirun job 50 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 51/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [5] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 51/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 51 submitted successfully"
else
    echo "[ERROR] Multirun job 51 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 52/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [6] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 52/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 52 submitted successfully"
else
    echo "[ERROR] Multirun job 52 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 53/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [7] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 53/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 53 submitted successfully"
else
    echo "[ERROR] Multirun job 53 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 54/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [1]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 54/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 54 submitted successfully"
else
    echo "[ERROR] Multirun job 54 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 55/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [4]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 55/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 55 submitted successfully"
else
    echo "[ERROR] Multirun job 55 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 56/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [2]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 56/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 56 submitted successfully"
else
    echo "[ERROR] Multirun job 56 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 57/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [5]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 57/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 57 submitted successfully"
else
    echo "[ERROR] Multirun job 57 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 58/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [8] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 58/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 58 submitted successfully"
else
    echo "[ERROR] Multirun job 58 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 59/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [4] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 59/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 59 submitted successfully"
else
    echo "[ERROR] Multirun job 59 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 60/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [5] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 60/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 60 submitted successfully"
else
    echo "[ERROR] Multirun job 60 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 61/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [2]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 61/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 61 submitted successfully"
else
    echo "[ERROR] Multirun job 61 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 62/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [6] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 62/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 62 submitted successfully"
else
    echo "[ERROR] Multirun job 62 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 63/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [4]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 63/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 63 submitted successfully"
else
    echo "[ERROR] Multirun job 63 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 64/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [7] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 64/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 64 submitted successfully"
else
    echo "[ERROR] Multirun job 64 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 65/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [8] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 65/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 65 submitted successfully"
else
    echo "[ERROR] Multirun job 65 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 66/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [9]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 66/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 66 submitted successfully"
else
    echo "[ERROR] Multirun job 66 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 67/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [1]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 67/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 67 submitted successfully"
else
    echo "[ERROR] Multirun job 67 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 68/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [4] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 68/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 68 submitted successfully"
else
    echo "[ERROR] Multirun job 68 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 69/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [2] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 69/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 69 submitted successfully"
else
    echo "[ERROR] Multirun job 69 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 70/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [4]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 70/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 70 submitted successfully"
else
    echo "[ERROR] Multirun job 70 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 71/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [5]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 71/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 71 submitted successfully"
else
    echo "[ERROR] Multirun job 71 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 72/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [3]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 72/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 72 submitted successfully"
else
    echo "[ERROR] Multirun job 72 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 73/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [2]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 73/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 73 submitted successfully"
else
    echo "[ERROR] Multirun job 73 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 74/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [9]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 74/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 74 submitted successfully"
else
    echo "[ERROR] Multirun job 74 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 75/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [2] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 75/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 75 submitted successfully"
else
    echo "[ERROR] Multirun job 75 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 76/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [7] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 76/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 76 submitted successfully"
else
    echo "[ERROR] Multirun job 76 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 77/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [4]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 77/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 77 submitted successfully"
else
    echo "[ERROR] Multirun job 77 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 78/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [2]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 78/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 78 submitted successfully"
else
    echo "[ERROR] Multirun job 78 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 79/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [4] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 79/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 79 submitted successfully"
else
    echo "[ERROR] Multirun job 79 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 80/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [3]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 80/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 80 submitted successfully"
else
    echo "[ERROR] Multirun job 80 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 81/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [4] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 81/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 81 submitted successfully"
else
    echo "[ERROR] Multirun job 81 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 82/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [5]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 82/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 82 submitted successfully"
else
    echo "[ERROR] Multirun job 82 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 83/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [9]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 83/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 83 submitted successfully"
else
    echo "[ERROR] Multirun job 83 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 84/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [1]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 84/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 84 submitted successfully"
else
    echo "[ERROR] Multirun job 84 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 85/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [7] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 85/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 85 submitted successfully"
else
    echo "[ERROR] Multirun job 85 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 86/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [2]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 86/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 86 submitted successfully"
else
    echo "[ERROR] Multirun job 86 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 87/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [9]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 87/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 87 submitted successfully"
else
    echo "[ERROR] Multirun job 87 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 88/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [2] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 88/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 88 submitted successfully"
else
    echo "[ERROR] Multirun job 88 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 89/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [5]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 89/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 89 submitted successfully"
else
    echo "[ERROR] Multirun job 89 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 90/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [4]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 90/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 90 submitted successfully"
else
    echo "[ERROR] Multirun job 90 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 91/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [3] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 91/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 91 submitted successfully"
else
    echo "[ERROR] Multirun job 91 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 92/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [8]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 92/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 92 submitted successfully"
else
    echo "[ERROR] Multirun job 92 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 93/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [3]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 93/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 93 submitted successfully"
else
    echo "[ERROR] Multirun job 93 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 94/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [9]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 94/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 94 submitted successfully"
else
    echo "[ERROR] Multirun job 94 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 95/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [7]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 95/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 95 submitted successfully"
else
    echo "[ERROR] Multirun job 95 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 96/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [1]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 96/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 96 submitted successfully"
else
    echo "[ERROR] Multirun job 96 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 97/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [7] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 97/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 97 submitted successfully"
else
    echo "[ERROR] Multirun job 97 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 98/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [2]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 98/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 98 submitted successfully"
else
    echo "[ERROR] Multirun job 98 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 99/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [3] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 99/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 99 submitted successfully"
else
    echo "[ERROR] Multirun job 99 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 100/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [1] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 100/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 100 submitted successfully"
else
    echo "[ERROR] Multirun job 100 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 101/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [5]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 101/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 101 submitted successfully"
else
    echo "[ERROR] Multirun job 101 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 102/108
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [7]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 102/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 102 submitted successfully"
else
    echo "[ERROR] Multirun job 102 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 103/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [6]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 103/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 103 submitted successfully"
else
    echo "[ERROR] Multirun job 103 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 104/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [9]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 104/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 104 submitted successfully"
else
    echo "[ERROR] Multirun job 104 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 105/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [5] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 105/108..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 105 submitted successfully"
else
    echo "[ERROR] Multirun job 105 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 106/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [3] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 106/108..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 106 submitted successfully"
else
    echo "[ERROR] Multirun job 106 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 107/108
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [8]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 107/108..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 107 submitted successfully"
else
    echo "[ERROR] Multirun job 107 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 108/108
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [8]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 108/108..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 108 submitted successfully"
else
    echo "[ERROR] Multirun job 108 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

