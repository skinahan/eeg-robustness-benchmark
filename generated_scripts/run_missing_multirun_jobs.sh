#!/bin/bash
# Generated multirun sbatch automation script
# Generated on: 2025-09-17 21:20:14
# Total missing multirun jobs: 36

set -e  # Exit on any error

echo "Starting multirun experiment automation..."
echo "Total multirun jobs to submit: 36"

# Multirun Job 1/36
# Dataset: BNCI2014_001 | Eval: WithinSession | Subjects: [3]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 1/36..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 1 submitted successfully"
else
    echo "[ERROR] Multirun job 1 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 2/36
# Dataset: BNCI2014_001 | Eval: WithinSession | Subjects: [5]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 2/36..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 2 submitted successfully"
else
    echo "[ERROR] Multirun job 2 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 3/36
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [4] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 3/36..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 3 submitted successfully"
else
    echo "[ERROR] Multirun job 3 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 4/36
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [5]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 4/36..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 4 submitted successfully"
else
    echo "[ERROR] Multirun job 4 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 5/36
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [2] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 5/36..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 5 submitted successfully"
else
    echo "[ERROR] Multirun job 5 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 6/36
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [7]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 6/36..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 6 submitted successfully"
else
    echo "[ERROR] Multirun job 6 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 7/36
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [9]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 7/36..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 7 submitted successfully"
else
    echo "[ERROR] Multirun job 7 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 8/36
# Dataset: BNCI2014_001 | Eval: WithinSession | Subjects: [2] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 8/36..."
sbatch --time=14-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 8 submitted successfully"
else
    echo "[ERROR] Multirun job 8 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 9/36
# Dataset: BNCI2014_001 | Eval: WithinSession | Subjects: [7]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 9/36..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 9 submitted successfully"
else
    echo "[ERROR] Multirun job 9 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 10/36
# Dataset: BNCI2014_001 | Eval: WithinSession | Subjects: [4] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 10/36..."
sbatch --time=14-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 10 submitted successfully"
else
    echo "[ERROR] Multirun job 10 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 11/36
# Dataset: BNCI2014_001 | Eval: WithinSession | Subjects: [9]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 11/36..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 11 submitted successfully"
else
    echo "[ERROR] Multirun job 11 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 12/36
# Dataset: BNCI2014_001 | Eval: WithinSession | Subjects: [6] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 12/36..."
sbatch --time=14-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 12 submitted successfully"
else
    echo "[ERROR] Multirun job 12 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 13/36
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [8] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 13/36..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 13 submitted successfully"
else
    echo "[ERROR] Multirun job 13 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 14/36
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [6] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 14/36..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 14 submitted successfully"
else
    echo "[ERROR] Multirun job 14 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 15/36
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [2]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 15/36..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 15 submitted successfully"
else
    echo "[ERROR] Multirun job 15 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 16/36
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [4]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 16/36..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 16 submitted successfully"
else
    echo "[ERROR] Multirun job 16 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 17/36
# Dataset: BNCI2014_001 | Eval: WithinSession | Subjects: [8] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 17/36..."
sbatch --time=14-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 17 submitted successfully"
else
    echo "[ERROR] Multirun job 17 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 18/36
# Dataset: BNCI2014_001 | Eval: WithinSession | Subjects: [2]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 18/36..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 WithinSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 18 submitted successfully"
else
    echo "[ERROR] Multirun job 18 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 19/36
# Dataset: BNCI2014_001 | Eval: WithinSession | Subjects: [1] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 19/36..."
sbatch --time=14-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 19 submitted successfully"
else
    echo "[ERROR] Multirun job 19 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 20/36
# Dataset: BNCI2014_001 | Eval: WithinSession | Subjects: [4]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 20/36..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 WithinSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 20 submitted successfully"
else
    echo "[ERROR] Multirun job 20 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 21/36
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [3] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 21/36..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 21 submitted successfully"
else
    echo "[ERROR] Multirun job 21 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 22/36
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [1] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 22/36..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 22 submitted successfully"
else
    echo "[ERROR] Multirun job 22 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 23/36
# Dataset: BNCI2014_001 | Eval: WithinSession | Subjects: [6]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 23/36..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 WithinSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 23 submitted successfully"
else
    echo "[ERROR] Multirun job 23 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 24/36
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [6]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 24/36..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 24 submitted successfully"
else
    echo "[ERROR] Multirun job 24 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 25/36
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [8]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 25/36..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 25 submitted successfully"
else
    echo "[ERROR] Multirun job 25 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 26/36
# Dataset: BNCI2014_001 | Eval: WithinSession | Subjects: [3] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 26/36..."
sbatch --time=14-00:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 WithinSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 26 submitted successfully"
else
    echo "[ERROR] Multirun job 26 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 27/36
# Dataset: BNCI2014_001 | Eval: WithinSession | Subjects: [5] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 27/36..."
sbatch --time=14-00:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 WithinSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 27 submitted successfully"
else
    echo "[ERROR] Multirun job 27 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 28/36
# Dataset: BNCI2014_001 | Eval: WithinSession | Subjects: [8]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 28/36..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 WithinSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 28 submitted successfully"
else
    echo "[ERROR] Multirun job 28 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 29/36
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [5] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 29/36..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 29 submitted successfully"
else
    echo "[ERROR] Multirun job 29 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 30/36
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [9] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 30/36..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 30 submitted successfully"
else
    echo "[ERROR] Multirun job 30 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 31/36
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [7] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 31/36..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 31 submitted successfully"
else
    echo "[ERROR] Multirun job 31 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 32/36
# Dataset: BNCI2014_001 | Eval: WithinSession | Subjects: [1]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 32/36..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 WithinSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 32 submitted successfully"
else
    echo "[ERROR] Multirun job 32 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 33/36
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [1]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 33/36..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 33 submitted successfully"
else
    echo "[ERROR] Multirun job 33 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 34/36
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [3]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 34/36..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 34 submitted successfully"
else
    echo "[ERROR] Multirun job 34 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 35/36
# Dataset: BNCI2014_001 | Eval: WithinSession | Subjects: [7] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 35/36..."
sbatch --time=14-00:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 WithinSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 35 submitted successfully"
else
    echo "[ERROR] Multirun job 35 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 36/36
# Dataset: BNCI2014_001 | Eval: WithinSession | Subjects: [9] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 36/36..."
sbatch --time=14-00:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 WithinSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 36 submitted successfully"
else
    echo "[ERROR] Multirun job 36 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

