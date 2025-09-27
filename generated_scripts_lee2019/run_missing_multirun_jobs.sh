#!/bin/bash
# Generated multirun sbatch automation script
# Generated on: 2025-09-26 17:15:57
# Total missing multirun jobs: 324

set -e  # Exit on any error

echo "Starting multirun experiment automation..."
echo "Total multirun jobs to submit: 324"

# Multirun Job 1/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [13] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 1/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 1 submitted successfully"
else
    echo "[ERROR] Multirun job 1 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 2/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [18] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 2/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 2 submitted successfully"
else
    echo "[ERROR] Multirun job 2 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 3/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [23]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 3/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 3 submitted successfully"
else
    echo "[ERROR] Multirun job 3 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 4/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [13]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 4/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 4 submitted successfully"
else
    echo "[ERROR] Multirun job 4 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 5/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [14]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 5/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 5 submitted successfully"
else
    echo "[ERROR] Multirun job 5 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 6/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [19] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 6/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 6 submitted successfully"
else
    echo "[ERROR] Multirun job 6 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 7/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [26] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 7/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 7 submitted successfully"
else
    echo "[ERROR] Multirun job 7 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 8/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [11] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 8/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 8 submitted successfully"
else
    echo "[ERROR] Multirun job 8 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 9/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [21]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 9/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 9 submitted successfully"
else
    echo "[ERROR] Multirun job 9 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 10/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [1]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 10/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 10 submitted successfully"
else
    echo "[ERROR] Multirun job 10 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 11/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [24] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 11/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 11 submitted successfully"
else
    echo "[ERROR] Multirun job 11 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 12/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [27] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 12/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 12 submitted successfully"
else
    echo "[ERROR] Multirun job 12 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 13/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [1] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 13/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 13 submitted successfully"
else
    echo "[ERROR] Multirun job 13 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 14/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [16] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 14/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 14 submitted successfully"
else
    echo "[ERROR] Multirun job 14 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 15/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [22]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 15/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 15 submitted successfully"
else
    echo "[ERROR] Multirun job 15 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 16/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [27] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 16/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 16 submitted successfully"
else
    echo "[ERROR] Multirun job 16 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 17/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [26] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 17/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 17 submitted successfully"
else
    echo "[ERROR] Multirun job 17 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 18/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [21]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 18/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 18 submitted successfully"
else
    echo "[ERROR] Multirun job 18 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 19/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [14] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 19/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 19 submitted successfully"
else
    echo "[ERROR] Multirun job 19 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 20/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [10] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 20/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 20 submitted successfully"
else
    echo "[ERROR] Multirun job 20 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 21/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [24]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 21/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 21 submitted successfully"
else
    echo "[ERROR] Multirun job 21 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 22/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [5]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 22/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 22 submitted successfully"
else
    echo "[ERROR] Multirun job 22 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 23/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [8]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 23/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 23 submitted successfully"
else
    echo "[ERROR] Multirun job 23 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 24/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [12] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 24/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 24 submitted successfully"
else
    echo "[ERROR] Multirun job 24 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 25/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [7]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 25/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 25 submitted successfully"
else
    echo "[ERROR] Multirun job 25 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 26/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [10] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 26/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 26 submitted successfully"
else
    echo "[ERROR] Multirun job 26 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 27/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [25] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 27/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 27 submitted successfully"
else
    echo "[ERROR] Multirun job 27 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 28/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [9] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 28/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 28 submitted successfully"
else
    echo "[ERROR] Multirun job 28 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 29/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [20]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 29/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 29 submitted successfully"
else
    echo "[ERROR] Multirun job 29 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 30/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [15] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 30/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 30 submitted successfully"
else
    echo "[ERROR] Multirun job 30 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 31/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [21] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 31/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 31 submitted successfully"
else
    echo "[ERROR] Multirun job 31 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 32/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [16]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 32/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 32 submitted successfully"
else
    echo "[ERROR] Multirun job 32 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 33/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [23] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 33/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 33 submitted successfully"
else
    echo "[ERROR] Multirun job 33 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 34/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [8] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 34/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 34 submitted successfully"
else
    echo "[ERROR] Multirun job 34 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 35/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [11] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 35/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 35 submitted successfully"
else
    echo "[ERROR] Multirun job 35 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 36/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [19]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 36/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 36 submitted successfully"
else
    echo "[ERROR] Multirun job 36 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 37/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [22]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 37/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 37 submitted successfully"
else
    echo "[ERROR] Multirun job 37 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 38/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [3]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 38/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 38 submitted successfully"
else
    echo "[ERROR] Multirun job 38 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 39/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [18]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 39/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 39 submitted successfully"
else
    echo "[ERROR] Multirun job 39 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 40/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [6]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 40/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 40 submitted successfully"
else
    echo "[ERROR] Multirun job 40 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 41/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [4] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 41/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 41 submitted successfully"
else
    echo "[ERROR] Multirun job 41 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 42/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [9] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 42/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 42 submitted successfully"
else
    echo "[ERROR] Multirun job 42 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 43/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [4]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 43/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 43 submitted successfully"
else
    echo "[ERROR] Multirun job 43 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 44/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [6] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 44/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 44 submitted successfully"
else
    echo "[ERROR] Multirun job 44 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 45/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [5]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 45/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 45 submitted successfully"
else
    echo "[ERROR] Multirun job 45 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 46/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [1]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 46/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 46 submitted successfully"
else
    echo "[ERROR] Multirun job 46 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 47/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [4]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 47/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 47 submitted successfully"
else
    echo "[ERROR] Multirun job 47 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 48/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [15] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 48/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 48 submitted successfully"
else
    echo "[ERROR] Multirun job 48 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 49/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [20] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 49/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 49 submitted successfully"
else
    echo "[ERROR] Multirun job 49 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 50/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [25]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 50/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 50 submitted successfully"
else
    echo "[ERROR] Multirun job 50 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 51/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [17] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 51/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 51 submitted successfully"
else
    echo "[ERROR] Multirun job 51 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 52/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [12]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 52/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 52 submitted successfully"
else
    echo "[ERROR] Multirun job 52 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 53/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [18] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 53/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 53 submitted successfully"
else
    echo "[ERROR] Multirun job 53 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 54/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [3]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 54/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 54 submitted successfully"
else
    echo "[ERROR] Multirun job 54 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 55/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [2] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 55/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 55 submitted successfully"
else
    echo "[ERROR] Multirun job 55 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 56/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [12]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 56/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 56 submitted successfully"
else
    echo "[ERROR] Multirun job 56 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 57/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [8]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 57/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 57 submitted successfully"
else
    echo "[ERROR] Multirun job 57 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 58/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [7] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 58/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 58 submitted successfully"
else
    echo "[ERROR] Multirun job 58 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 59/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [2]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 59/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 59 submitted successfully"
else
    echo "[ERROR] Multirun job 59 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 60/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [23]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 60/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 60 submitted successfully"
else
    echo "[ERROR] Multirun job 60 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 61/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [1] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 61/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 61 submitted successfully"
else
    echo "[ERROR] Multirun job 61 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 62/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [26]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 62/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 62 submitted successfully"
else
    echo "[ERROR] Multirun job 62 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 63/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [11]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 63/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 63 submitted successfully"
else
    echo "[ERROR] Multirun job 63 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 64/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [14]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 64/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 64 submitted successfully"
else
    echo "[ERROR] Multirun job 64 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 65/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [19]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 65/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 65 submitted successfully"
else
    echo "[ERROR] Multirun job 65 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 66/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [27] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 66/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 66 submitted successfully"
else
    echo "[ERROR] Multirun job 66 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 67/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [26] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 67/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 67 submitted successfully"
else
    echo "[ERROR] Multirun job 67 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 68/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [11] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 68/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 68 submitted successfully"
else
    echo "[ERROR] Multirun job 68 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 69/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [14] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 69/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 69 submitted successfully"
else
    echo "[ERROR] Multirun job 69 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 70/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [22]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 70/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 70 submitted successfully"
else
    echo "[ERROR] Multirun job 70 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 71/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [21]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 71/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 71 submitted successfully"
else
    echo "[ERROR] Multirun job 71 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 72/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [9]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 72/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 72 submitted successfully"
else
    echo "[ERROR] Multirun job 72 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 73/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [13]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 73/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 73 submitted successfully"
else
    echo "[ERROR] Multirun job 73 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 74/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [7] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 74/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 74 submitted successfully"
else
    echo "[ERROR] Multirun job 74 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 75/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [10] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 75/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 75 submitted successfully"
else
    echo "[ERROR] Multirun job 75 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 76/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [13] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 76/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 76 submitted successfully"
else
    echo "[ERROR] Multirun job 76 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 77/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [24]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 77/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 77 submitted successfully"
else
    echo "[ERROR] Multirun job 77 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 78/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [5]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 78/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 78 submitted successfully"
else
    echo "[ERROR] Multirun job 78 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 79/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [8]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 79/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 79 submitted successfully"
else
    echo "[ERROR] Multirun job 79 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 80/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [23] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 80/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 80 submitted successfully"
else
    echo "[ERROR] Multirun job 80 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 81/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [24] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 81/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 81 submitted successfully"
else
    echo "[ERROR] Multirun job 81 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 82/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [21] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 82/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 82 submitted successfully"
else
    echo "[ERROR] Multirun job 82 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 83/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [26]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 83/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 83 submitted successfully"
else
    echo "[ERROR] Multirun job 83 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 84/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [16]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 84/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 84 submitted successfully"
else
    echo "[ERROR] Multirun job 84 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 85/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [22] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 85/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 85 submitted successfully"
else
    echo "[ERROR] Multirun job 85 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 86/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [6] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 86/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 86 submitted successfully"
else
    echo "[ERROR] Multirun job 86 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 87/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [20] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 87/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 87 submitted successfully"
else
    echo "[ERROR] Multirun job 87 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 88/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [4] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 88/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 88 submitted successfully"
else
    echo "[ERROR] Multirun job 88 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 89/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [19]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 89/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 89 submitted successfully"
else
    echo "[ERROR] Multirun job 89 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 90/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [22]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 90/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 90 submitted successfully"
else
    echo "[ERROR] Multirun job 90 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 91/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [14]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 91/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 91 submitted successfully"
else
    echo "[ERROR] Multirun job 91 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 92/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [5] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 92/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 92 submitted successfully"
else
    echo "[ERROR] Multirun job 92 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 93/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [9] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 93/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 93 submitted successfully"
else
    echo "[ERROR] Multirun job 93 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 94/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [15]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 94/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 94 submitted successfully"
else
    echo "[ERROR] Multirun job 94 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 95/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [18]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 95/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 95 submitted successfully"
else
    echo "[ERROR] Multirun job 95 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 96/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [4]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 96/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 96 submitted successfully"
else
    echo "[ERROR] Multirun job 96 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 97/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [15] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 97/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 97 submitted successfully"
else
    echo "[ERROR] Multirun job 97 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 98/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [25]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 98/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 98 submitted successfully"
else
    echo "[ERROR] Multirun job 98 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 99/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [3] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 99/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 99 submitted successfully"
else
    echo "[ERROR] Multirun job 99 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 100/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [2] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 100/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 100 submitted successfully"
else
    echo "[ERROR] Multirun job 100 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 101/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [5]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 101/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 101 submitted successfully"
else
    echo "[ERROR] Multirun job 101 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 102/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [20] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 102/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 102 submitted successfully"
else
    echo "[ERROR] Multirun job 102 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 103/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [3] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 103/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 103 submitted successfully"
else
    echo "[ERROR] Multirun job 103 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 104/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [17]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 104/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 104 submitted successfully"
else
    echo "[ERROR] Multirun job 104 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 105/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [1]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 105/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 105 submitted successfully"
else
    echo "[ERROR] Multirun job 105 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 106/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [16] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 106/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 106 submitted successfully"
else
    echo "[ERROR] Multirun job 106 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 107/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [24]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 107/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 107 submitted successfully"
else
    echo "[ERROR] Multirun job 107 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 108/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [8]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 108/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 108 submitted successfully"
else
    echo "[ERROR] Multirun job 108 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 109/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [18] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 109/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 109 submitted successfully"
else
    echo "[ERROR] Multirun job 109 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 110/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [14] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 110/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 110 submitted successfully"
else
    echo "[ERROR] Multirun job 110 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 111/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [12] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 111/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 111 submitted successfully"
else
    echo "[ERROR] Multirun job 111 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 112/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [1] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 112/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 112 submitted successfully"
else
    echo "[ERROR] Multirun job 112 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 113/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [19]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 113/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 113 submitted successfully"
else
    echo "[ERROR] Multirun job 113 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 114/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [11]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 114/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 114 submitted successfully"
else
    echo "[ERROR] Multirun job 114 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 115/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [14]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 115/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 115 submitted successfully"
else
    echo "[ERROR] Multirun job 115 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 116/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [2]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 116/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 116 submitted successfully"
else
    echo "[ERROR] Multirun job 116 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 117/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [25] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 117/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 117 submitted successfully"
else
    echo "[ERROR] Multirun job 117 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 118/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [20]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 118/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 118 submitted successfully"
else
    echo "[ERROR] Multirun job 118 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 119/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [23]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 119/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 119 submitted successfully"
else
    echo "[ERROR] Multirun job 119 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 120/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [7]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 120/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 120 submitted successfully"
else
    echo "[ERROR] Multirun job 120 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 121/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [27] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 121/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 121 submitted successfully"
else
    echo "[ERROR] Multirun job 121 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 122/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [7] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 122/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 122 submitted successfully"
else
    echo "[ERROR] Multirun job 122 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 123/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [22]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 123/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 123 submitted successfully"
else
    echo "[ERROR] Multirun job 123 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 124/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [10]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 124/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 124 submitted successfully"
else
    echo "[ERROR] Multirun job 124 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 125/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [13]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 125/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 125 submitted successfully"
else
    echo "[ERROR] Multirun job 125 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 126/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [23] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 126/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 126 submitted successfully"
else
    echo "[ERROR] Multirun job 126 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 127/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [8] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 127/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 127 submitted successfully"
else
    echo "[ERROR] Multirun job 127 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 128/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [13] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 128/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 128 submitted successfully"
else
    echo "[ERROR] Multirun job 128 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 129/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [15]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 129/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 129 submitted successfully"
else
    echo "[ERROR] Multirun job 129 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 130/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [18]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 130/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 130 submitted successfully"
else
    echo "[ERROR] Multirun job 130 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 131/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [6]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 131/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 131 submitted successfully"
else
    echo "[ERROR] Multirun job 131 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 132/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [9]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 132/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 132 submitted successfully"
else
    echo "[ERROR] Multirun job 132 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 133/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [10] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 133/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 133 submitted successfully"
else
    echo "[ERROR] Multirun job 133 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 134/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [21] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 134/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 134 submitted successfully"
else
    echo "[ERROR] Multirun job 134 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 135/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [24]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 135/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 135 submitted successfully"
else
    echo "[ERROR] Multirun job 135 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 136/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [5]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 136/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 136 submitted successfully"
else
    echo "[ERROR] Multirun job 136 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 137/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [16]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 137/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 137 submitted successfully"
else
    echo "[ERROR] Multirun job 137 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 138/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [22] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 138/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 138 submitted successfully"
else
    echo "[ERROR] Multirun job 138 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 139/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [6] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 139/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 139 submitted successfully"
else
    echo "[ERROR] Multirun job 139 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 140/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [17]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 140/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 140 submitted successfully"
else
    echo "[ERROR] Multirun job 140 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 141/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [1]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 141/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 141 submitted successfully"
else
    echo "[ERROR] Multirun job 141 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 142/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [4]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 142/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 142 submitted successfully"
else
    echo "[ERROR] Multirun job 142 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 143/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [26]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 143/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 143 submitted successfully"
else
    echo "[ERROR] Multirun job 143 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 144/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [4] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 144/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 144 submitted successfully"
else
    echo "[ERROR] Multirun job 144 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 145/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [22] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 145/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 145 submitted successfully"
else
    echo "[ERROR] Multirun job 145 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 146/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [21] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 146/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 146 submitted successfully"
else
    echo "[ERROR] Multirun job 146 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 147/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [16]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 147/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 147 submitted successfully"
else
    echo "[ERROR] Multirun job 147 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 148/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [17] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 148/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 148 submitted successfully"
else
    echo "[ERROR] Multirun job 148 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 149/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [12]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 149/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 149 submitted successfully"
else
    echo "[ERROR] Multirun job 149 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 150/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [5] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 150/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 150 submitted successfully"
else
    echo "[ERROR] Multirun job 150 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 151/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [3]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 151/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 151 submitted successfully"
else
    echo "[ERROR] Multirun job 151 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 152/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [25]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 152/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 152 submitted successfully"
else
    echo "[ERROR] Multirun job 152 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 153/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [12]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 153/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 153 submitted successfully"
else
    echo "[ERROR] Multirun job 153 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 154/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [5] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 154/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 154 submitted successfully"
else
    echo "[ERROR] Multirun job 154 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 155/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [20] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 155/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 155 submitted successfully"
else
    echo "[ERROR] Multirun job 155 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 156/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [4] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 156/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 156 submitted successfully"
else
    echo "[ERROR] Multirun job 156 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 157/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [15]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 157/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 157 submitted successfully"
else
    echo "[ERROR] Multirun job 157 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 158/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [14]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 158/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 158 submitted successfully"
else
    echo "[ERROR] Multirun job 158 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 159/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [13] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 159/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 159 submitted successfully"
else
    echo "[ERROR] Multirun job 159 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 160/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [16] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 160/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 160 submitted successfully"
else
    echo "[ERROR] Multirun job 160 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 161/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [11]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 161/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 161 submitted successfully"
else
    echo "[ERROR] Multirun job 161 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 162/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [19] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 162/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 162 submitted successfully"
else
    echo "[ERROR] Multirun job 162 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 163/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [3] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 163/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 163 submitted successfully"
else
    echo "[ERROR] Multirun job 163 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 164/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [25]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 164/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 164 submitted successfully"
else
    echo "[ERROR] Multirun job 164 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 165/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [3] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 165/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 165 submitted successfully"
else
    echo "[ERROR] Multirun job 165 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 166/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [2] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 166/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 166 submitted successfully"
else
    echo "[ERROR] Multirun job 166 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 167/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [18] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 167/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 167 submitted successfully"
else
    echo "[ERROR] Multirun job 167 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 168/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [27]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 168/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 168 submitted successfully"
else
    echo "[ERROR] Multirun job 168 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 169/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [17]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 169/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 169 submitted successfully"
else
    echo "[ERROR] Multirun job 169 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 170/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [13]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 170/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 170 submitted successfully"
else
    echo "[ERROR] Multirun job 170 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 171/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [24] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 171/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 171 submitted successfully"
else
    echo "[ERROR] Multirun job 171 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 172/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [26] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 172/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 172 submitted successfully"
else
    echo "[ERROR] Multirun job 172 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 173/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [11] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 173/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 173 submitted successfully"
else
    echo "[ERROR] Multirun job 173 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 174/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [21]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 174/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 174 submitted successfully"
else
    echo "[ERROR] Multirun job 174 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 175/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [24]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 175/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 175 submitted successfully"
else
    echo "[ERROR] Multirun job 175 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 176/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [8]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 176/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 176 submitted successfully"
else
    echo "[ERROR] Multirun job 176 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 177/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [1] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 177/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 177 submitted successfully"
else
    echo "[ERROR] Multirun job 177 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 178/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [10]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 178/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 178 submitted successfully"
else
    echo "[ERROR] Multirun job 178 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 179/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [14] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 179/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 179 submitted successfully"
else
    echo "[ERROR] Multirun job 179 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 180/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [25] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 180/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 180 submitted successfully"
else
    echo "[ERROR] Multirun job 180 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 181/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [20]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 181/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 181 submitted successfully"
else
    echo "[ERROR] Multirun job 181 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 182/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [27] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 182/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 182 submitted successfully"
else
    echo "[ERROR] Multirun job 182 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 183/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [12] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 183/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 183 submitted successfully"
else
    echo "[ERROR] Multirun job 183 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 184/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [12] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 184/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 184 submitted successfully"
else
    echo "[ERROR] Multirun job 184 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 185/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [7]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 185/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 185 submitted successfully"
else
    echo "[ERROR] Multirun job 185 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 186/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [7]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 186/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 186 submitted successfully"
else
    echo "[ERROR] Multirun job 186 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 187/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [8] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 187/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 187 submitted successfully"
else
    echo "[ERROR] Multirun job 187 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 188/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [10] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 188/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 188 submitted successfully"
else
    echo "[ERROR] Multirun job 188 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 189/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [23]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 189/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 189 submitted successfully"
else
    echo "[ERROR] Multirun job 189 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 190/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [23] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 190/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 190 submitted successfully"
else
    echo "[ERROR] Multirun job 190 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 191/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [7] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 191/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 191 submitted successfully"
else
    echo "[ERROR] Multirun job 191 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 192/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [18]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 192/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 192 submitted successfully"
else
    echo "[ERROR] Multirun job 192 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 193/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [11] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 193/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 193 submitted successfully"
else
    echo "[ERROR] Multirun job 193 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 194/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [6]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 194/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 194 submitted successfully"
else
    echo "[ERROR] Multirun job 194 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 195/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [9]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 195/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 195 submitted successfully"
else
    echo "[ERROR] Multirun job 195 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 196/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [19] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 196/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 196 submitted successfully"
else
    echo "[ERROR] Multirun job 196 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 197/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [19]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 197/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 197 submitted successfully"
else
    echo "[ERROR] Multirun job 197 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 198/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [3]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 198/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 198 submitted successfully"
else
    echo "[ERROR] Multirun job 198 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 199/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [6]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 199/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 199 submitted successfully"
else
    echo "[ERROR] Multirun job 199 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 200/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [17] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 200/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 200 submitted successfully"
else
    echo "[ERROR] Multirun job 200 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 201/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [22] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 201/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 201 submitted successfully"
else
    echo "[ERROR] Multirun job 201 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 202/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [6] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 202/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 202 submitted successfully"
else
    echo "[ERROR] Multirun job 202 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 203/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [21] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 203/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 203 submitted successfully"
else
    echo "[ERROR] Multirun job 203 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 204/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [9] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 204/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 204 submitted successfully"
else
    echo "[ERROR] Multirun job 204 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 205/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [27]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 205/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 205 submitted successfully"
else
    echo "[ERROR] Multirun job 205 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 206/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [17]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 206/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 206 submitted successfully"
else
    echo "[ERROR] Multirun job 206 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 207/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [1]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 207/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 207 submitted successfully"
else
    echo "[ERROR] Multirun job 207 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 208/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [16]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 208/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 208 submitted successfully"
else
    echo "[ERROR] Multirun job 208 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 209/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [4]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 209/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 209 submitted successfully"
else
    echo "[ERROR] Multirun job 209 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 210/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [26]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 210/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 210 submitted successfully"
else
    echo "[ERROR] Multirun job 210 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 211/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [5] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 211/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 211 submitted successfully"
else
    echo "[ERROR] Multirun job 211 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 212/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [4] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 212/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 212 submitted successfully"
else
    echo "[ERROR] Multirun job 212 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 213/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [3]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 213/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 213 submitted successfully"
else
    echo "[ERROR] Multirun job 213 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 214/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [12]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 214/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 214 submitted successfully"
else
    echo "[ERROR] Multirun job 214 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 215/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [18] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 215/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 215 submitted successfully"
else
    echo "[ERROR] Multirun job 215 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 216/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [25]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 216/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 216 submitted successfully"
else
    echo "[ERROR] Multirun job 216 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 217/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [2] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 217/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 217 submitted successfully"
else
    echo "[ERROR] Multirun job 217 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 218/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [26] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 218/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 218 submitted successfully"
else
    echo "[ERROR] Multirun job 218 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 219/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [16] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 219/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 219 submitted successfully"
else
    echo "[ERROR] Multirun job 219 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 220/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [21]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 220/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 220 submitted successfully"
else
    echo "[ERROR] Multirun job 220 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 221/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [11]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 221/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 221 submitted successfully"
else
    echo "[ERROR] Multirun job 221 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 222/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [26]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 222/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 222 submitted successfully"
else
    echo "[ERROR] Multirun job 222 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 223/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [13] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 223/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 223 submitted successfully"
else
    echo "[ERROR] Multirun job 223 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 224/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [1] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 224/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 224 submitted successfully"
else
    echo "[ERROR] Multirun job 224 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 225/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [27]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 225/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 225 submitted successfully"
else
    echo "[ERROR] Multirun job 225 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 226/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [23]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 226/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 226 submitted successfully"
else
    echo "[ERROR] Multirun job 226 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 227/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [11]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 227/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 227 submitted successfully"
else
    echo "[ERROR] Multirun job 227 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 228/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [2]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 228/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 228 submitted successfully"
else
    echo "[ERROR] Multirun job 228 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 229/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [19] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 229/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 229 submitted successfully"
else
    echo "[ERROR] Multirun job 229 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 230/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [3] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 230/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 230 submitted successfully"
else
    echo "[ERROR] Multirun job 230 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 231/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [9] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 231/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 231 submitted successfully"
else
    echo "[ERROR] Multirun job 231 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 232/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [14] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 232/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 232 submitted successfully"
else
    echo "[ERROR] Multirun job 232 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 233/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [17]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 233/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 233 submitted successfully"
else
    echo "[ERROR] Multirun job 233 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 234/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [9]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 234/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 234 submitted successfully"
else
    echo "[ERROR] Multirun job 234 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 235/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [24] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 235/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 235 submitted successfully"
else
    echo "[ERROR] Multirun job 235 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 236/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [15] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 236/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 236 submitted successfully"
else
    echo "[ERROR] Multirun job 236 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 237/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [10]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 237/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 237 submitted successfully"
else
    echo "[ERROR] Multirun job 237 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 238/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [10]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 238/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 238 submitted successfully"
else
    echo "[ERROR] Multirun job 238 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 239/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [13]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 239/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 239 submitted successfully"
else
    echo "[ERROR] Multirun job 239 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 240/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [25] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 240/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 240 submitted successfully"
else
    echo "[ERROR] Multirun job 240 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 241/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [15]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 241/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 241 submitted successfully"
else
    echo "[ERROR] Multirun job 241 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 242/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [20]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 242/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 242 submitted successfully"
else
    echo "[ERROR] Multirun job 242 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 243/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [23] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 243/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 243 submitted successfully"
else
    echo "[ERROR] Multirun job 243 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 244/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [8]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 244/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 244 submitted successfully"
else
    echo "[ERROR] Multirun job 244 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 245/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [24] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 245/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 245 submitted successfully"
else
    echo "[ERROR] Multirun job 245 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 246/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [8] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 246/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 246 submitted successfully"
else
    echo "[ERROR] Multirun job 246 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 247/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [7]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 247/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 247 submitted successfully"
else
    echo "[ERROR] Multirun job 247 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 248/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [6] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 248/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 248 submitted successfully"
else
    echo "[ERROR] Multirun job 248 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 249/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [19] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 249/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 249 submitted successfully"
else
    echo "[ERROR] Multirun job 249 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 250/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [11] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 250/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 250 submitted successfully"
else
    echo "[ERROR] Multirun job 250 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 251/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [6]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 251/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 251 submitted successfully"
else
    echo "[ERROR] Multirun job 251 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 252/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [9]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 252/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 252 submitted successfully"
else
    echo "[ERROR] Multirun job 252 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 253/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [17] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 253/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 253 submitted successfully"
else
    echo "[ERROR] Multirun job 253 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 254/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [20] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 254/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 254 submitted successfully"
else
    echo "[ERROR] Multirun job 254 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 255/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [27]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 255/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 255 submitted successfully"
else
    echo "[ERROR] Multirun job 255 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 256/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [14]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 256/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 256 submitted successfully"
else
    echo "[ERROR] Multirun job 256 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 257/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [7] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 257/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 257 submitted successfully"
else
    echo "[ERROR] Multirun job 257 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 258/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [18]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 258/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 258 submitted successfully"
else
    echo "[ERROR] Multirun job 258 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 259/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [2]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 259/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 259 submitted successfully"
else
    echo "[ERROR] Multirun job 259 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 260/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [22] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 260/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 260 submitted successfully"
else
    echo "[ERROR] Multirun job 260 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 261/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [17]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 261/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 261 submitted successfully"
else
    echo "[ERROR] Multirun job 261 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 262/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [15] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 262/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 262 submitted successfully"
else
    echo "[ERROR] Multirun job 262 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 263/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [18] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 263/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 263 submitted successfully"
else
    echo "[ERROR] Multirun job 263 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 264/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [3] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 264/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 264 submitted successfully"
else
    echo "[ERROR] Multirun job 264 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 265/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [2] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 265/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 265 submitted successfully"
else
    echo "[ERROR] Multirun job 265 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 266/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [13]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 266/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 266 submitted successfully"
else
    echo "[ERROR] Multirun job 266 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 267/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [1]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 267/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 267 submitted successfully"
else
    echo "[ERROR] Multirun job 267 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 268/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [5] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 268/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 268 submitted successfully"
else
    echo "[ERROR] Multirun job 268 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 269/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [16] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 269/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 269 submitted successfully"
else
    echo "[ERROR] Multirun job 269 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 270/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [24]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 270/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 270 submitted successfully"
else
    echo "[ERROR] Multirun job 270 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 271/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [1] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 271/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 271 submitted successfully"
else
    echo "[ERROR] Multirun job 271 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 272/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [26] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 272/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 272 submitted successfully"
else
    echo "[ERROR] Multirun job 272 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 273/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [21]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 273/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 273 submitted successfully"
else
    echo "[ERROR] Multirun job 273 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 274/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [27] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 274/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 274 submitted successfully"
else
    echo "[ERROR] Multirun job 274 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 275/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [16] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 275/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 275 submitted successfully"
else
    echo "[ERROR] Multirun job 275 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 276/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [26]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 276/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 276 submitted successfully"
else
    echo "[ERROR] Multirun job 276 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 277/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [11]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 277/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 277 submitted successfully"
else
    echo "[ERROR] Multirun job 277 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 278/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [12] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 278/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 278 submitted successfully"
else
    echo "[ERROR] Multirun job 278 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 279/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [2]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 279/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 279 submitted successfully"
else
    echo "[ERROR] Multirun job 279 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 280/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [25] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 280/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 280 submitted successfully"
else
    echo "[ERROR] Multirun job 280 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 281/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [9] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 281/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 281 submitted successfully"
else
    echo "[ERROR] Multirun job 281 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 282/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [12] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 282/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 282 submitted successfully"
else
    echo "[ERROR] Multirun job 282 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 283/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [20]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 283/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 283 submitted successfully"
else
    echo "[ERROR] Multirun job 283 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 284/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [27]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 284/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 284 submitted successfully"
else
    echo "[ERROR] Multirun job 284 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 285/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [19]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 285/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 285 submitted successfully"
else
    echo "[ERROR] Multirun job 285 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 286/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [7]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 286/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 286 submitted successfully"
else
    echo "[ERROR] Multirun job 286 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 287/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [10] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 287/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 287 submitted successfully"
else
    echo "[ERROR] Multirun job 287 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 288/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [15] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 288/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 288 submitted successfully"
else
    echo "[ERROR] Multirun job 288 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 289/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [14] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 289/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 289 submitted successfully"
else
    echo "[ERROR] Multirun job 289 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 290/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [10]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 290/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 290 submitted successfully"
else
    echo "[ERROR] Multirun job 290 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 291/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [23]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 291/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 291 submitted successfully"
else
    echo "[ERROR] Multirun job 291 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 292/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [9]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 292/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 292 submitted successfully"
else
    echo "[ERROR] Multirun job 292 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 293/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [15]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 293/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 293 submitted successfully"
else
    echo "[ERROR] Multirun job 293 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 294/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [8] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 294/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP CrossSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 294 submitted successfully"
else
    echo "[ERROR] Multirun job 294 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 295/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [10]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 295/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 295 submitted successfully"
else
    echo "[ERROR] Multirun job 295 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 296/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [25] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 296/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 296 submitted successfully"
else
    echo "[ERROR] Multirun job 296 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 297/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [20]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 297/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 297 submitted successfully"
else
    echo "[ERROR] Multirun job 297 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 298/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [13] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 298/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 298 submitted successfully"
else
    echo "[ERROR] Multirun job 298 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 299/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [22]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 299/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 299 submitted successfully"
else
    echo "[ERROR] Multirun job 299 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 300/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [3]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 300/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 300 submitted successfully"
else
    echo "[ERROR] Multirun job 300 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 301/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [6]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 301/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 301 submitted successfully"
else
    echo "[ERROR] Multirun job 301 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 302/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [21] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 302/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 302 submitted successfully"
else
    echo "[ERROR] Multirun job 302 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 303/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [16]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 303/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP WithinSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 303 submitted successfully"
else
    echo "[ERROR] Multirun job 303 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 304/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [23] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 304/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 304 submitted successfully"
else
    echo "[ERROR] Multirun job 304 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 305/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSession | Subjects: [4]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 305/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP CrossSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 305 submitted successfully"
else
    echo "[ERROR] Multirun job 305 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 306/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [24] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 306/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 306 submitted successfully"
else
    echo "[ERROR] Multirun job 306 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 307/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [8] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 307/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP WithinSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 307 submitted successfully"
else
    echo "[ERROR] Multirun job 307 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 308/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [5]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 308/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 308 submitted successfully"
else
    echo "[ERROR] Multirun job 308 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 309/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [17] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 309/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 309 submitted successfully"
else
    echo "[ERROR] Multirun job 309 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 310/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [20] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 310/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 310 submitted successfully"
else
    echo "[ERROR] Multirun job 310 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 311/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [4] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 311/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP WithinSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 311 submitted successfully"
else
    echo "[ERROR] Multirun job 311 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 312/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [12]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 312/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 312 submitted successfully"
else
    echo "[ERROR] Multirun job 312 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 313/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [22] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 313/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 313 submitted successfully"
else
    echo "[ERROR] Multirun job 313 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 314/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [6] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 314/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 314 submitted successfully"
else
    echo "[ERROR] Multirun job 314 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 315/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [19] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 315/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 315 submitted successfully"
else
    echo "[ERROR] Multirun job 315 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 316/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [25]
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 316/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP CrossSession false reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 316 submitted successfully"
else
    echo "[ERROR] Multirun job 316 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 317/324
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSession | Subjects: [2] | TUNED
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 317/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP CrossSession true reegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 317 submitted successfully"
else
    echo "[ERROR] Multirun job 317 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 318/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [7] | TUNED
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 318/324..."
sbatch --time=7-00:00:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP WithinSession true cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 318 submitted successfully"
else
    echo "[ERROR] Multirun job 318 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 319/324
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [2]
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 319/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP WithinSession false cnn_ncp
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 319 submitted successfully"
else
    echo "[ERROR] Multirun job 319 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 320/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [17] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 320/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 320 submitted successfully"
else
    echo "[ERROR] Multirun job 320 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 321/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [27]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 321/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 321 submitted successfully"
else
    echo "[ERROR] Multirun job 321 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 322/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [5] | TUNED
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 322/324..."
sbatch --time=2-12:00:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP CrossSession true eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 322 submitted successfully"
else
    echo "[ERROR] Multirun job 322 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 323/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [15]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 323/324..."
sbatch --time=1-00:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP WithinSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 323 submitted successfully"
else
    echo "[ERROR] Multirun job 323 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 324/324
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSession | Subjects: [18]
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seeds: 100, 200, 300, 400, 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 324/324..."
sbatch --time=0-04:30:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP CrossSession false eegnet
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 324 submitted successfully"
else
    echo "[ERROR] Multirun job 324 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

