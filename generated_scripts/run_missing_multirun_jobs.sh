#!/bin/bash
# Generated multirun sbatch automation script
# Generated on: 2025-09-29 11:23:32
# Total missing multirun jobs: 114

set -e  # Exit on any error

echo "Starting multirun experiment automation..."
echo "Total multirun jobs to submit: 114"

# Multirun Job 1/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [3]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 1/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 1 submitted successfully"
else
    echo "[ERROR] Multirun job 1 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 2/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [6]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 2/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 2 submitted successfully"
else
    echo "[ERROR] Multirun job 2 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 3/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [2] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 3/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 3 submitted successfully"
else
    echo "[ERROR] Multirun job 3 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 4/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 4/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 4 submitted successfully"
else
    echo "[ERROR] Multirun job 4 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 5/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [9]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 5/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 5 submitted successfully"
else
    echo "[ERROR] Multirun job 5 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 6/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [1]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 6/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 6 submitted successfully"
else
    echo "[ERROR] Multirun job 6 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 7/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [3]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 7/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 7 submitted successfully"
else
    echo "[ERROR] Multirun job 7 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 8/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [8]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 8/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 8 submitted successfully"
else
    echo "[ERROR] Multirun job 8 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 9/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [4] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 9/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 9 submitted successfully"
else
    echo "[ERROR] Multirun job 9 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 10/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [7] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 10/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 10 submitted successfully"
else
    echo "[ERROR] Multirun job 10 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 11/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [7] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 11/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 11 submitted successfully"
else
    echo "[ERROR] Multirun job 11 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 12/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [2]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 12/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 12 submitted successfully"
else
    echo "[ERROR] Multirun job 12 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 13/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [4] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 13/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 13 submitted successfully"
else
    echo "[ERROR] Multirun job 13 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 14/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [5] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 14/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 14 submitted successfully"
else
    echo "[ERROR] Multirun job 14 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 15/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [8] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 15/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 15 submitted successfully"
else
    echo "[ERROR] Multirun job 15 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 16/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [9] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 16/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 16 submitted successfully"
else
    echo "[ERROR] Multirun job 16 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 17/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 17/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 17 submitted successfully"
else
    echo "[ERROR] Multirun job 17 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 18/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [4]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 18/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 18 submitted successfully"
else
    echo "[ERROR] Multirun job 18 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 19/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [6]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 19/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 19 submitted successfully"
else
    echo "[ERROR] Multirun job 19 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 20/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [3] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 20/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 20 submitted successfully"
else
    echo "[ERROR] Multirun job 20 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 21/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [1]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 21/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 21 submitted successfully"
else
    echo "[ERROR] Multirun job 21 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 22/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [7]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 22/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 22 submitted successfully"
else
    echo "[ERROR] Multirun job 22 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 23/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [3] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 23/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 23 submitted successfully"
else
    echo "[ERROR] Multirun job 23 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 24/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [1]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 24/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 24 submitted successfully"
else
    echo "[ERROR] Multirun job 24 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 25/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [7]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 25/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 25 submitted successfully"
else
    echo "[ERROR] Multirun job 25 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 26/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [4] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 26/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 26 submitted successfully"
else
    echo "[ERROR] Multirun job 26 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 27/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [6]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 27/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 27 submitted successfully"
else
    echo "[ERROR] Multirun job 27 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 28/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [5] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 28/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 28 submitted successfully"
else
    echo "[ERROR] Multirun job 28 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 29/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [5]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 29/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 29 submitted successfully"
else
    echo "[ERROR] Multirun job 29 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 30/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [9] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 30/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 30 submitted successfully"
else
    echo "[ERROR] Multirun job 30 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 31/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 31/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 31 submitted successfully"
else
    echo "[ERROR] Multirun job 31 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 32/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [9]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 32/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 32 submitted successfully"
else
    echo "[ERROR] Multirun job 32 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 33/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [6] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 33/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 33 submitted successfully"
else
    echo "[ERROR] Multirun job 33 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 34/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [2]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 34/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 34 submitted successfully"
else
    echo "[ERROR] Multirun job 34 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 35/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [1] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 35/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 35 submitted successfully"
else
    echo "[ERROR] Multirun job 35 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 36/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [4] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 36/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 36 submitted successfully"
else
    echo "[ERROR] Multirun job 36 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 37/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [3] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 37/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 37 submitted successfully"
else
    echo "[ERROR] Multirun job 37 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 38/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [3]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 38/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 38 submitted successfully"
else
    echo "[ERROR] Multirun job 38 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 39/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [7] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 39/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 39 submitted successfully"
else
    echo "[ERROR] Multirun job 39 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 40/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [9]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 40/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 40 submitted successfully"
else
    echo "[ERROR] Multirun job 40 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 41/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [5] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 41/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 41 submitted successfully"
else
    echo "[ERROR] Multirun job 41 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 42/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [8]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 42/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 42 submitted successfully"
else
    echo "[ERROR] Multirun job 42 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 43/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [7] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 43/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 43 submitted successfully"
else
    echo "[ERROR] Multirun job 43 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 44/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [2]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 44/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 44 submitted successfully"
else
    echo "[ERROR] Multirun job 44 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 45/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 45/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 45 submitted successfully"
else
    echo "[ERROR] Multirun job 45 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 46/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [5]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 46/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 46 submitted successfully"
else
    echo "[ERROR] Multirun job 46 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 47/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [9] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 47/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 47 submitted successfully"
else
    echo "[ERROR] Multirun job 47 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 48/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [2] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 48/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 48 submitted successfully"
else
    echo "[ERROR] Multirun job 48 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 49/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [7]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 49/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 49 submitted successfully"
else
    echo "[ERROR] Multirun job 49 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 50/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [9] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 50/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 50 submitted successfully"
else
    echo "[ERROR] Multirun job 50 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 51/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [4]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 51/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 51 submitted successfully"
else
    echo "[ERROR] Multirun job 51 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 52/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [6] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 52/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 52 submitted successfully"
else
    echo "[ERROR] Multirun job 52 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 53/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [3] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 53/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 53 submitted successfully"
else
    echo "[ERROR] Multirun job 53 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 54/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [5]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 54/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 54 submitted successfully"
else
    echo "[ERROR] Multirun job 54 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 55/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [1] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 55/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 55 submitted successfully"
else
    echo "[ERROR] Multirun job 55 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 56/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [1]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 56/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 56 submitted successfully"
else
    echo "[ERROR] Multirun job 56 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 57/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [3]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 57/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 57 submitted successfully"
else
    echo "[ERROR] Multirun job 57 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 58/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [4]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 58/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 58 submitted successfully"
else
    echo "[ERROR] Multirun job 58 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 59/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [9]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 59/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 59 submitted successfully"
else
    echo "[ERROR] Multirun job 59 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 60/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [8] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 60/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 60 submitted successfully"
else
    echo "[ERROR] Multirun job 60 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 61/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [5] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 61/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 61 submitted successfully"
else
    echo "[ERROR] Multirun job 61 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 62/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [8] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 62/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 62 submitted successfully"
else
    echo "[ERROR] Multirun job 62 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 63/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [3]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 63/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 63 submitted successfully"
else
    echo "[ERROR] Multirun job 63 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 64/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [2] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 64/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 64 submitted successfully"
else
    echo "[ERROR] Multirun job 64 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 65/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [2]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 65/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 65 submitted successfully"
else
    echo "[ERROR] Multirun job 65 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 66/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [9] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 66/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 66 submitted successfully"
else
    echo "[ERROR] Multirun job 66 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 67/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 67/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 67 submitted successfully"
else
    echo "[ERROR] Multirun job 67 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 68/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [8]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 68/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 68 submitted successfully"
else
    echo "[ERROR] Multirun job 68 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 69/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [6] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 69/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 69 submitted successfully"
else
    echo "[ERROR] Multirun job 69 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 70/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [3] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 70/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 70 submitted successfully"
else
    echo "[ERROR] Multirun job 70 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 71/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [3] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 71/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 71 submitted successfully"
else
    echo "[ERROR] Multirun job 71 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 72/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [6] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 72/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 72 submitted successfully"
else
    echo "[ERROR] Multirun job 72 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 73/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [5]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 73/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 73 submitted successfully"
else
    echo "[ERROR] Multirun job 73 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 74/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [1] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 74/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 74 submitted successfully"
else
    echo "[ERROR] Multirun job 74 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 75/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [4]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 75/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 75 submitted successfully"
else
    echo "[ERROR] Multirun job 75 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 76/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [1] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 76/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 76 submitted successfully"
else
    echo "[ERROR] Multirun job 76 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 77/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [6]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 77/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 77 submitted successfully"
else
    echo "[ERROR] Multirun job 77 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 78/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [5] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 78/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 78 submitted successfully"
else
    echo "[ERROR] Multirun job 78 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 79/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [9] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 79/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 79 submitted successfully"
else
    echo "[ERROR] Multirun job 79 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 80/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [8] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 80/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 80 submitted successfully"
else
    echo "[ERROR] Multirun job 80 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 81/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [8]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 81/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 81 submitted successfully"
else
    echo "[ERROR] Multirun job 81 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 82/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [1]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 82/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 82 submitted successfully"
else
    echo "[ERROR] Multirun job 82 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 83/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [3]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 83/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 83 submitted successfully"
else
    echo "[ERROR] Multirun job 83 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 84/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 84/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 84 submitted successfully"
else
    echo "[ERROR] Multirun job 84 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 85/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [2] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 85/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 85 submitted successfully"
else
    echo "[ERROR] Multirun job 85 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 86/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [2] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 86/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 86 submitted successfully"
else
    echo "[ERROR] Multirun job 86 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 87/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [7]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 87/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 87 submitted successfully"
else
    echo "[ERROR] Multirun job 87 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 88/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [4] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 88/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 88 submitted successfully"
else
    echo "[ERROR] Multirun job 88 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 89/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [6] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 89/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 89 submitted successfully"
else
    echo "[ERROR] Multirun job 89 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 90/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [9]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 90/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 90 submitted successfully"
else
    echo "[ERROR] Multirun job 90 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 91/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [8] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 91/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 91 submitted successfully"
else
    echo "[ERROR] Multirun job 91 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 92/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [1] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 92/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 92 submitted successfully"
else
    echo "[ERROR] Multirun job 92 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 93/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [7] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 93/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 93 submitted successfully"
else
    echo "[ERROR] Multirun job 93 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 94/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [1]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 94/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 94 submitted successfully"
else
    echo "[ERROR] Multirun job 94 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 95/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [7]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 95/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 95 submitted successfully"
else
    echo "[ERROR] Multirun job 95 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 96/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [2]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 96/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 96 submitted successfully"
else
    echo "[ERROR] Multirun job 96 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 97/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [6]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 97/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 97 submitted successfully"
else
    echo "[ERROR] Multirun job 97 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 98/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [5] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 98/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 98 submitted successfully"
else
    echo "[ERROR] Multirun job 98 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 99/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [8]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 99/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 99 submitted successfully"
else
    echo "[ERROR] Multirun job 99 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 100/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [9]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 100/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 100 submitted successfully"
else
    echo "[ERROR] Multirun job 100 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 101/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [4]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 101/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 101 submitted successfully"
else
    echo "[ERROR] Multirun job 101 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 102/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [6] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 102/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 102 submitted successfully"
else
    echo "[ERROR] Multirun job 102 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 103/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: CrossSession | Subjects: [2] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 103/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 103 submitted successfully"
else
    echo "[ERROR] Multirun job 103 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 104/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [5]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 104/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 104 submitted successfully"
else
    echo "[ERROR] Multirun job 104 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 105/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [2]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 105/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 105 submitted successfully"
else
    echo "[ERROR] Multirun job 105 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 106/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [1] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 106/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 106 submitted successfully"
else
    echo "[ERROR] Multirun job 106 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 107/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: CrossSession | Subjects: [8]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 107/114..."
sbatch --time=0-06:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 107 submitted successfully"
else
    echo "[ERROR] Multirun job 107 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 108/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [4] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 108/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 108 submitted successfully"
else
    echo "[ERROR] Multirun job 108 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 109/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [4]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 109/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 109 submitted successfully"
else
    echo "[ERROR] Multirun job 109 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 110/114
# Dataset: BNCI2014_001 | Model: reegnet | Eval: WithinSession | Subjects: [7] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 110/114..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 110 submitted successfully"
else
    echo "[ERROR] Multirun job 110 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 111/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: WithinSession | Subjects: [5]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 111/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 111 submitted successfully"
else
    echo "[ERROR] Multirun job 111 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 112/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [7]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 112/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 112 submitted successfully"
else
    echo "[ERROR] Multirun job 112 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 113/114
# Dataset: BNCI2014_001 | Model: eegnet | Eval: CrossSession | Subjects: [8] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 113/114..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 113 submitted successfully"
else
    echo "[ERROR] Multirun job 113 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 114/114
# Dataset: BNCI2014_001 | Model: cnn_ncp | Eval: WithinSession | Subjects: [6]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 114/114..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 114 submitted successfully"
else
    echo "[ERROR] Multirun job 114 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

