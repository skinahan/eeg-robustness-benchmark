#!/bin/bash
# Generated multirun sbatch automation script
# Generated on: 2025-09-17 14:03:05
# Total missing multirun jobs: 20

set -e  # Exit on any error

echo "Starting multirun experiment automation..."
echo "Total multirun jobs to submit: 20"

# Multirun Job 1/20
# Dataset: BNCI2014_001 | Eval: WithinSession | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 1/20..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 WithinSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 1 submitted successfully"
else
    echo "[ERROR] Multirun job 1 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 2/20
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [1] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 2/20..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 2 submitted successfully"
else
    echo "[ERROR] Multirun job 2 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 3/20
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [4]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 3/20..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 3 submitted successfully"
else
    echo "[ERROR] Multirun job 3 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 4/20
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [5] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 4/20..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 4 submitted successfully"
else
    echo "[ERROR] Multirun job 4 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 5/20
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [6]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 5/20..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 5 submitted successfully"
else
    echo "[ERROR] Multirun job 5 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 6/20
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [3] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 6/20..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 6 submitted successfully"
else
    echo "[ERROR] Multirun job 6 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 7/20
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [7] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 7/20..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 7 submitted successfully"
else
    echo "[ERROR] Multirun job 7 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 8/20
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [8]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 8/20..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 8 submitted successfully"
else
    echo "[ERROR] Multirun job 8 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 9/20
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [9] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 9/20..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 9 submitted successfully"
else
    echo "[ERROR] Multirun job 9 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 10/20
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [1]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 10/20..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 1 BNCI2014_001 CrossSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 10 submitted successfully"
else
    echo "[ERROR] Multirun job 10 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 11/20
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [5]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 11/20..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 5 BNCI2014_001 CrossSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 11 submitted successfully"
else
    echo "[ERROR] Multirun job 11 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 12/20
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [2] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 12/20..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 12 submitted successfully"
else
    echo "[ERROR] Multirun job 12 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 13/20
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [3]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 13/20..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 3 BNCI2014_001 CrossSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 13 submitted successfully"
else
    echo "[ERROR] Multirun job 13 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 14/20
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [7]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 14/20..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 7 BNCI2014_001 CrossSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 14 submitted successfully"
else
    echo "[ERROR] Multirun job 14 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 15/20
# Dataset: BNCI2014_001 | Eval: WithinSession | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 15/20..."
sbatch --time=14-00:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 WithinSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 15 submitted successfully"
else
    echo "[ERROR] Multirun job 15 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 16/20
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [9]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 16/20..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 9 BNCI2014_001 CrossSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 16 submitted successfully"
else
    echo "[ERROR] Multirun job 16 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 17/20
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [6] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 17/20..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 6 BNCI2014_001 CrossSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 17 submitted successfully"
else
    echo "[ERROR] Multirun job 17 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 18/20
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [4] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 18/20..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 4 BNCI2014_001 CrossSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 18 submitted successfully"
else
    echo "[ERROR] Multirun job 18 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 19/20
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [2]
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 19/20..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 2 BNCI2014_001 CrossSession false
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 19 submitted successfully"
else
    echo "[ERROR] Multirun job 19 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 20/20
# Dataset: BNCI2014_001 | Eval: CrossSession | Subjects: [8] | TUNED
# This multirun will generate test_perturb results for models: eegnet, reegnet, cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 20/20..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 8 BNCI2014_001 CrossSession true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 20 submitted successfully"
else
    echo "[ERROR] Multirun job 20 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

