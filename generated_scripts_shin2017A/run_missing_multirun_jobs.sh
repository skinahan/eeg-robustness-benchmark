#!/bin/bash
# Generated multirun sbatch automation script
# Generated on: 2026-03-29 14:16:42
# Total missing multirun jobs: 725

set -e  # Exit on any error

echo "Starting multirun experiment automation..."
echo "Total multirun jobs to submit: 725"

# Multirun Job 1/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [1] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 1/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 1 submitted successfully"
else
    echo "[ERROR] Multirun job 1 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 2/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [2] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 2/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 2 submitted successfully"
else
    echo "[ERROR] Multirun job 2 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 3/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [3] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 3/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 3 submitted successfully"
else
    echo "[ERROR] Multirun job 3 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 4/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [4] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 4/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 4 submitted successfully"
else
    echo "[ERROR] Multirun job 4 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 5/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [5] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 5/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 5 submitted successfully"
else
    echo "[ERROR] Multirun job 5 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 6/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [6] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 6/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 6 submitted successfully"
else
    echo "[ERROR] Multirun job 6 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 7/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [7] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 7/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 7 submitted successfully"
else
    echo "[ERROR] Multirun job 7 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 8/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [8] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 8/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 8 submitted successfully"
else
    echo "[ERROR] Multirun job 8 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 9/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [9] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 9/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 9 submitted successfully"
else
    echo "[ERROR] Multirun job 9 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 10/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [10] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 10/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 10 submitted successfully"
else
    echo "[ERROR] Multirun job 10 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 11/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [11] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 11/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 11 submitted successfully"
else
    echo "[ERROR] Multirun job 11 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 12/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [12] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 12/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 12 submitted successfully"
else
    echo "[ERROR] Multirun job 12 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 13/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [13] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 13/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 13 submitted successfully"
else
    echo "[ERROR] Multirun job 13 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 14/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [14] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 14/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 14 submitted successfully"
else
    echo "[ERROR] Multirun job 14 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 15/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [15] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 15/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 15 submitted successfully"
else
    echo "[ERROR] Multirun job 15 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 16/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [16] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 16/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 16 submitted successfully"
else
    echo "[ERROR] Multirun job 16 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 17/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [17] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 17/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 17 submitted successfully"
else
    echo "[ERROR] Multirun job 17 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 18/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [18] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 18/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 18 submitted successfully"
else
    echo "[ERROR] Multirun job 18 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 19/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [19] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 19/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 19 submitted successfully"
else
    echo "[ERROR] Multirun job 19 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 20/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [20] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 20/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 20 submitted successfully"
else
    echo "[ERROR] Multirun job 20 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 21/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [21] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 21/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 21 submitted successfully"
else
    echo "[ERROR] Multirun job 21 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 22/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [22] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 22/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 22 submitted successfully"
else
    echo "[ERROR] Multirun job 22 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 23/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [23] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 23/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 23 submitted successfully"
else
    echo "[ERROR] Multirun job 23 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 24/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [24] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 24/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 24 submitted successfully"
else
    echo "[ERROR] Multirun job 24 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 25/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [25] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 25/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 25 submitted successfully"
else
    echo "[ERROR] Multirun job 25 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 26/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [26] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 26/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 26 submitted successfully"
else
    echo "[ERROR] Multirun job 26 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 27/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [27] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 27/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 27 submitted successfully"
else
    echo "[ERROR] Multirun job 27 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 28/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [28] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 28/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 28 submitted successfully"
else
    echo "[ERROR] Multirun job 28 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 29/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [29] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 29/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false eegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 29 submitted successfully"
else
    echo "[ERROR] Multirun job 29 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 30/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [1] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 30/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 30 submitted successfully"
else
    echo "[ERROR] Multirun job 30 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 31/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [2] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 31/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 31 submitted successfully"
else
    echo "[ERROR] Multirun job 31 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 32/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [3] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 32/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 32 submitted successfully"
else
    echo "[ERROR] Multirun job 32 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 33/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [4] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 33/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 33 submitted successfully"
else
    echo "[ERROR] Multirun job 33 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 34/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [5] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 34/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 34 submitted successfully"
else
    echo "[ERROR] Multirun job 34 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 35/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [6] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 35/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 35 submitted successfully"
else
    echo "[ERROR] Multirun job 35 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 36/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [7] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 36/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 36 submitted successfully"
else
    echo "[ERROR] Multirun job 36 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 37/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [8] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 37/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 37 submitted successfully"
else
    echo "[ERROR] Multirun job 37 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 38/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [9] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 38/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 38 submitted successfully"
else
    echo "[ERROR] Multirun job 38 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 39/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [10] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 39/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 39 submitted successfully"
else
    echo "[ERROR] Multirun job 39 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 40/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [11] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 40/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 40 submitted successfully"
else
    echo "[ERROR] Multirun job 40 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 41/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [12] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 41/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 41 submitted successfully"
else
    echo "[ERROR] Multirun job 41 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 42/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [13] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 42/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 42 submitted successfully"
else
    echo "[ERROR] Multirun job 42 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 43/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [14] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 43/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 43 submitted successfully"
else
    echo "[ERROR] Multirun job 43 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 44/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [15] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 44/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 44 submitted successfully"
else
    echo "[ERROR] Multirun job 44 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 45/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [16] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 45/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 45 submitted successfully"
else
    echo "[ERROR] Multirun job 45 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 46/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [17] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 46/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 46 submitted successfully"
else
    echo "[ERROR] Multirun job 46 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 47/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [18] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 47/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 47 submitted successfully"
else
    echo "[ERROR] Multirun job 47 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 48/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [19] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 48/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 48 submitted successfully"
else
    echo "[ERROR] Multirun job 48 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 49/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [20] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 49/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 49 submitted successfully"
else
    echo "[ERROR] Multirun job 49 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 50/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [21] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 50/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 50 submitted successfully"
else
    echo "[ERROR] Multirun job 50 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 51/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [22] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 51/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 51 submitted successfully"
else
    echo "[ERROR] Multirun job 51 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 52/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [23] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 52/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 52 submitted successfully"
else
    echo "[ERROR] Multirun job 52 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 53/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [24] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 53/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 53 submitted successfully"
else
    echo "[ERROR] Multirun job 53 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 54/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [25] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 54/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 54 submitted successfully"
else
    echo "[ERROR] Multirun job 54 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 55/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [26] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 55/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 55 submitted successfully"
else
    echo "[ERROR] Multirun job 55 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 56/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [27] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 56/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 56 submitted successfully"
else
    echo "[ERROR] Multirun job 56 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 57/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [28] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 57/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 57 submitted successfully"
else
    echo "[ERROR] Multirun job 57 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 58/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [29] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 58/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false eegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 58 submitted successfully"
else
    echo "[ERROR] Multirun job 58 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 59/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [1] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 59/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 59 submitted successfully"
else
    echo "[ERROR] Multirun job 59 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 60/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [2] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 60/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 60 submitted successfully"
else
    echo "[ERROR] Multirun job 60 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 61/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [3] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 61/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 61 submitted successfully"
else
    echo "[ERROR] Multirun job 61 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 62/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [4] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 62/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 62 submitted successfully"
else
    echo "[ERROR] Multirun job 62 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 63/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [5] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 63/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 63 submitted successfully"
else
    echo "[ERROR] Multirun job 63 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 64/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [6] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 64/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 64 submitted successfully"
else
    echo "[ERROR] Multirun job 64 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 65/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [7] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 65/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 65 submitted successfully"
else
    echo "[ERROR] Multirun job 65 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 66/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [8] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 66/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 66 submitted successfully"
else
    echo "[ERROR] Multirun job 66 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 67/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [9] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 67/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 67 submitted successfully"
else
    echo "[ERROR] Multirun job 67 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 68/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [10] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 68/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 68 submitted successfully"
else
    echo "[ERROR] Multirun job 68 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 69/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [11] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 69/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 69 submitted successfully"
else
    echo "[ERROR] Multirun job 69 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 70/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [12] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 70/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 70 submitted successfully"
else
    echo "[ERROR] Multirun job 70 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 71/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [13] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 71/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 71 submitted successfully"
else
    echo "[ERROR] Multirun job 71 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 72/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [14] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 72/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 72 submitted successfully"
else
    echo "[ERROR] Multirun job 72 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 73/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [15] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 73/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 73 submitted successfully"
else
    echo "[ERROR] Multirun job 73 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 74/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [16] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 74/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 74 submitted successfully"
else
    echo "[ERROR] Multirun job 74 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 75/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [17] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 75/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 75 submitted successfully"
else
    echo "[ERROR] Multirun job 75 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 76/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [18] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 76/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 76 submitted successfully"
else
    echo "[ERROR] Multirun job 76 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 77/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [19] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 77/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 77 submitted successfully"
else
    echo "[ERROR] Multirun job 77 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 78/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [20] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 78/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 78 submitted successfully"
else
    echo "[ERROR] Multirun job 78 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 79/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [21] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 79/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 79 submitted successfully"
else
    echo "[ERROR] Multirun job 79 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 80/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [22] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 80/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 80 submitted successfully"
else
    echo "[ERROR] Multirun job 80 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 81/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [23] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 81/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 81 submitted successfully"
else
    echo "[ERROR] Multirun job 81 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 82/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [24] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 82/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 82 submitted successfully"
else
    echo "[ERROR] Multirun job 82 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 83/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [25] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 83/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 83 submitted successfully"
else
    echo "[ERROR] Multirun job 83 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 84/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [26] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 84/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 84 submitted successfully"
else
    echo "[ERROR] Multirun job 84 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 85/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [27] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 85/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 85 submitted successfully"
else
    echo "[ERROR] Multirun job 85 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 86/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [28] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 86/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 86 submitted successfully"
else
    echo "[ERROR] Multirun job 86 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 87/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [29] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 87/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false eegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 87 submitted successfully"
else
    echo "[ERROR] Multirun job 87 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 88/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [1] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 88/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 88 submitted successfully"
else
    echo "[ERROR] Multirun job 88 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 89/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [2] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 89/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 89 submitted successfully"
else
    echo "[ERROR] Multirun job 89 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 90/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [3] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 90/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 90 submitted successfully"
else
    echo "[ERROR] Multirun job 90 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 91/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [4] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 91/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 91 submitted successfully"
else
    echo "[ERROR] Multirun job 91 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 92/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [5] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 92/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 92 submitted successfully"
else
    echo "[ERROR] Multirun job 92 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 93/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [6] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 93/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 93 submitted successfully"
else
    echo "[ERROR] Multirun job 93 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 94/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [7] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 94/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 94 submitted successfully"
else
    echo "[ERROR] Multirun job 94 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 95/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [8] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 95/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 95 submitted successfully"
else
    echo "[ERROR] Multirun job 95 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 96/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [9] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 96/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 96 submitted successfully"
else
    echo "[ERROR] Multirun job 96 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 97/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [10] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 97/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 97 submitted successfully"
else
    echo "[ERROR] Multirun job 97 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 98/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [11] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 98/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 98 submitted successfully"
else
    echo "[ERROR] Multirun job 98 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 99/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [12] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 99/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 99 submitted successfully"
else
    echo "[ERROR] Multirun job 99 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 100/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [13] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 100/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 100 submitted successfully"
else
    echo "[ERROR] Multirun job 100 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 101/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [14] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 101/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 101 submitted successfully"
else
    echo "[ERROR] Multirun job 101 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 102/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [15] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 102/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 102 submitted successfully"
else
    echo "[ERROR] Multirun job 102 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 103/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [16] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 103/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 103 submitted successfully"
else
    echo "[ERROR] Multirun job 103 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 104/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [17] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 104/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 104 submitted successfully"
else
    echo "[ERROR] Multirun job 104 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 105/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [18] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 105/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 105 submitted successfully"
else
    echo "[ERROR] Multirun job 105 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 106/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [19] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 106/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 106 submitted successfully"
else
    echo "[ERROR] Multirun job 106 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 107/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [20] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 107/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 107 submitted successfully"
else
    echo "[ERROR] Multirun job 107 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 108/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [21] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 108/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 108 submitted successfully"
else
    echo "[ERROR] Multirun job 108 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 109/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [22] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 109/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 109 submitted successfully"
else
    echo "[ERROR] Multirun job 109 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 110/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [23] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 110/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 110 submitted successfully"
else
    echo "[ERROR] Multirun job 110 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 111/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [24] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 111/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 111 submitted successfully"
else
    echo "[ERROR] Multirun job 111 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 112/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [25] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 112/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 112 submitted successfully"
else
    echo "[ERROR] Multirun job 112 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 113/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [26] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 113/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 113 submitted successfully"
else
    echo "[ERROR] Multirun job 113 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 114/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [27] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 114/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 114 submitted successfully"
else
    echo "[ERROR] Multirun job 114 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 115/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [28] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 115/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 115 submitted successfully"
else
    echo "[ERROR] Multirun job 115 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 116/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [29] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 116/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false eegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 116 submitted successfully"
else
    echo "[ERROR] Multirun job 116 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 117/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [1] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 117/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 117 submitted successfully"
else
    echo "[ERROR] Multirun job 117 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 118/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [2] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 118/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 118 submitted successfully"
else
    echo "[ERROR] Multirun job 118 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 119/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [3] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 119/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 119 submitted successfully"
else
    echo "[ERROR] Multirun job 119 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 120/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [4] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 120/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 120 submitted successfully"
else
    echo "[ERROR] Multirun job 120 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 121/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [5] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 121/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 121 submitted successfully"
else
    echo "[ERROR] Multirun job 121 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 122/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [6] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 122/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 122 submitted successfully"
else
    echo "[ERROR] Multirun job 122 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 123/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [7] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 123/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 123 submitted successfully"
else
    echo "[ERROR] Multirun job 123 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 124/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [8] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 124/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 124 submitted successfully"
else
    echo "[ERROR] Multirun job 124 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 125/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [9] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 125/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 125 submitted successfully"
else
    echo "[ERROR] Multirun job 125 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 126/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [10] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 126/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 126 submitted successfully"
else
    echo "[ERROR] Multirun job 126 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 127/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [11] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 127/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 127 submitted successfully"
else
    echo "[ERROR] Multirun job 127 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 128/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [12] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 128/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 128 submitted successfully"
else
    echo "[ERROR] Multirun job 128 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 129/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [13] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 129/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 129 submitted successfully"
else
    echo "[ERROR] Multirun job 129 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 130/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [14] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 130/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 130 submitted successfully"
else
    echo "[ERROR] Multirun job 130 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 131/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [15] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 131/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 131 submitted successfully"
else
    echo "[ERROR] Multirun job 131 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 132/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [16] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 132/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 132 submitted successfully"
else
    echo "[ERROR] Multirun job 132 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 133/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [17] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 133/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 133 submitted successfully"
else
    echo "[ERROR] Multirun job 133 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 134/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [18] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 134/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 134 submitted successfully"
else
    echo "[ERROR] Multirun job 134 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 135/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [19] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 135/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 135 submitted successfully"
else
    echo "[ERROR] Multirun job 135 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 136/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [20] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 136/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 136 submitted successfully"
else
    echo "[ERROR] Multirun job 136 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 137/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [21] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 137/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 137 submitted successfully"
else
    echo "[ERROR] Multirun job 137 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 138/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [22] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 138/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 138 submitted successfully"
else
    echo "[ERROR] Multirun job 138 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 139/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [23] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 139/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 139 submitted successfully"
else
    echo "[ERROR] Multirun job 139 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 140/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [24] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 140/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 140 submitted successfully"
else
    echo "[ERROR] Multirun job 140 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 141/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [25] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 141/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 141 submitted successfully"
else
    echo "[ERROR] Multirun job 141 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 142/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [26] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 142/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 142 submitted successfully"
else
    echo "[ERROR] Multirun job 142 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 143/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [27] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 143/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 143 submitted successfully"
else
    echo "[ERROR] Multirun job 143 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 144/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [28] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 144/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 144 submitted successfully"
else
    echo "[ERROR] Multirun job 144 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 145/725
# Dataset: Shin2017A | Model: eegnet | Eval: WithinSession | Subjects: [29] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 145/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false eegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 145 submitted successfully"
else
    echo "[ERROR] Multirun job 145 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 146/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [1] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 146/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 146 submitted successfully"
else
    echo "[ERROR] Multirun job 146 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 147/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [2] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 147/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 147 submitted successfully"
else
    echo "[ERROR] Multirun job 147 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 148/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [3] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 148/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 148 submitted successfully"
else
    echo "[ERROR] Multirun job 148 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 149/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [4] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 149/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 149 submitted successfully"
else
    echo "[ERROR] Multirun job 149 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 150/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [5] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 150/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 150 submitted successfully"
else
    echo "[ERROR] Multirun job 150 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 151/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [6] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 151/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 151 submitted successfully"
else
    echo "[ERROR] Multirun job 151 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 152/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [7] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 152/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 152 submitted successfully"
else
    echo "[ERROR] Multirun job 152 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 153/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [8] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 153/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 153 submitted successfully"
else
    echo "[ERROR] Multirun job 153 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 154/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [9] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 154/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 154 submitted successfully"
else
    echo "[ERROR] Multirun job 154 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 155/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [10] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 155/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 155 submitted successfully"
else
    echo "[ERROR] Multirun job 155 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 156/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [11] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 156/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 156 submitted successfully"
else
    echo "[ERROR] Multirun job 156 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 157/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [12] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 157/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 157 submitted successfully"
else
    echo "[ERROR] Multirun job 157 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 158/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [13] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 158/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 158 submitted successfully"
else
    echo "[ERROR] Multirun job 158 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 159/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [14] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 159/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 159 submitted successfully"
else
    echo "[ERROR] Multirun job 159 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 160/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [15] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 160/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 160 submitted successfully"
else
    echo "[ERROR] Multirun job 160 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 161/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [16] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 161/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 161 submitted successfully"
else
    echo "[ERROR] Multirun job 161 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 162/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [17] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 162/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 162 submitted successfully"
else
    echo "[ERROR] Multirun job 162 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 163/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [18] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 163/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 163 submitted successfully"
else
    echo "[ERROR] Multirun job 163 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 164/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [19] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 164/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 164 submitted successfully"
else
    echo "[ERROR] Multirun job 164 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 165/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [20] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 165/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 165 submitted successfully"
else
    echo "[ERROR] Multirun job 165 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 166/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [21] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 166/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 166 submitted successfully"
else
    echo "[ERROR] Multirun job 166 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 167/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [22] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 167/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 167 submitted successfully"
else
    echo "[ERROR] Multirun job 167 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 168/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [23] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 168/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 168 submitted successfully"
else
    echo "[ERROR] Multirun job 168 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 169/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [24] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 169/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 169 submitted successfully"
else
    echo "[ERROR] Multirun job 169 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 170/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [25] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 170/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 170 submitted successfully"
else
    echo "[ERROR] Multirun job 170 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 171/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [26] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 171/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 171 submitted successfully"
else
    echo "[ERROR] Multirun job 171 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 172/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [27] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 172/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 172 submitted successfully"
else
    echo "[ERROR] Multirun job 172 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 173/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [28] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 173/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 173 submitted successfully"
else
    echo "[ERROR] Multirun job 173 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 174/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [29] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 174/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false ctnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 174 submitted successfully"
else
    echo "[ERROR] Multirun job 174 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 175/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [1] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 175/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 175 submitted successfully"
else
    echo "[ERROR] Multirun job 175 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 176/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [2] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 176/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 176 submitted successfully"
else
    echo "[ERROR] Multirun job 176 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 177/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [3] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 177/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 177 submitted successfully"
else
    echo "[ERROR] Multirun job 177 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 178/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [4] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 178/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 178 submitted successfully"
else
    echo "[ERROR] Multirun job 178 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 179/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [5] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 179/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 179 submitted successfully"
else
    echo "[ERROR] Multirun job 179 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 180/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [6] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 180/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 180 submitted successfully"
else
    echo "[ERROR] Multirun job 180 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 181/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [7] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 181/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 181 submitted successfully"
else
    echo "[ERROR] Multirun job 181 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 182/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [8] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 182/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 182 submitted successfully"
else
    echo "[ERROR] Multirun job 182 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 183/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [9] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 183/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 183 submitted successfully"
else
    echo "[ERROR] Multirun job 183 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 184/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [10] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 184/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 184 submitted successfully"
else
    echo "[ERROR] Multirun job 184 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 185/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [11] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 185/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 185 submitted successfully"
else
    echo "[ERROR] Multirun job 185 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 186/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [12] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 186/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 186 submitted successfully"
else
    echo "[ERROR] Multirun job 186 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 187/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [13] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 187/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 187 submitted successfully"
else
    echo "[ERROR] Multirun job 187 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 188/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [14] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 188/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 188 submitted successfully"
else
    echo "[ERROR] Multirun job 188 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 189/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [15] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 189/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 189 submitted successfully"
else
    echo "[ERROR] Multirun job 189 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 190/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [16] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 190/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 190 submitted successfully"
else
    echo "[ERROR] Multirun job 190 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 191/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [17] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 191/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 191 submitted successfully"
else
    echo "[ERROR] Multirun job 191 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 192/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [18] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 192/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 192 submitted successfully"
else
    echo "[ERROR] Multirun job 192 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 193/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [19] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 193/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 193 submitted successfully"
else
    echo "[ERROR] Multirun job 193 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 194/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [20] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 194/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 194 submitted successfully"
else
    echo "[ERROR] Multirun job 194 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 195/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [21] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 195/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 195 submitted successfully"
else
    echo "[ERROR] Multirun job 195 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 196/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [22] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 196/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 196 submitted successfully"
else
    echo "[ERROR] Multirun job 196 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 197/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [23] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 197/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 197 submitted successfully"
else
    echo "[ERROR] Multirun job 197 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 198/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [24] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 198/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 198 submitted successfully"
else
    echo "[ERROR] Multirun job 198 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 199/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [25] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 199/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 199 submitted successfully"
else
    echo "[ERROR] Multirun job 199 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 200/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [26] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 200/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 200 submitted successfully"
else
    echo "[ERROR] Multirun job 200 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 201/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [27] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 201/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 201 submitted successfully"
else
    echo "[ERROR] Multirun job 201 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 202/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [28] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 202/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 202 submitted successfully"
else
    echo "[ERROR] Multirun job 202 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 203/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [29] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 203/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false ctnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 203 submitted successfully"
else
    echo "[ERROR] Multirun job 203 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 204/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [1] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 204/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 204 submitted successfully"
else
    echo "[ERROR] Multirun job 204 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 205/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [2] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 205/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 205 submitted successfully"
else
    echo "[ERROR] Multirun job 205 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 206/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [3] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 206/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 206 submitted successfully"
else
    echo "[ERROR] Multirun job 206 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 207/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [4] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 207/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 207 submitted successfully"
else
    echo "[ERROR] Multirun job 207 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 208/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [5] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 208/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 208 submitted successfully"
else
    echo "[ERROR] Multirun job 208 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 209/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [6] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 209/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 209 submitted successfully"
else
    echo "[ERROR] Multirun job 209 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 210/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [7] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 210/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 210 submitted successfully"
else
    echo "[ERROR] Multirun job 210 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 211/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [8] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 211/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 211 submitted successfully"
else
    echo "[ERROR] Multirun job 211 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 212/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [9] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 212/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 212 submitted successfully"
else
    echo "[ERROR] Multirun job 212 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 213/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [10] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 213/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 213 submitted successfully"
else
    echo "[ERROR] Multirun job 213 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 214/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [11] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 214/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 214 submitted successfully"
else
    echo "[ERROR] Multirun job 214 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 215/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [12] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 215/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 215 submitted successfully"
else
    echo "[ERROR] Multirun job 215 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 216/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [13] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 216/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 216 submitted successfully"
else
    echo "[ERROR] Multirun job 216 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 217/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [14] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 217/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 217 submitted successfully"
else
    echo "[ERROR] Multirun job 217 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 218/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [15] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 218/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 218 submitted successfully"
else
    echo "[ERROR] Multirun job 218 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 219/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [16] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 219/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 219 submitted successfully"
else
    echo "[ERROR] Multirun job 219 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 220/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [17] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 220/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 220 submitted successfully"
else
    echo "[ERROR] Multirun job 220 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 221/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [18] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 221/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 221 submitted successfully"
else
    echo "[ERROR] Multirun job 221 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 222/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [19] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 222/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 222 submitted successfully"
else
    echo "[ERROR] Multirun job 222 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 223/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [20] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 223/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 223 submitted successfully"
else
    echo "[ERROR] Multirun job 223 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 224/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [21] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 224/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 224 submitted successfully"
else
    echo "[ERROR] Multirun job 224 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 225/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [22] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 225/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 225 submitted successfully"
else
    echo "[ERROR] Multirun job 225 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 226/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [23] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 226/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 226 submitted successfully"
else
    echo "[ERROR] Multirun job 226 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 227/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [24] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 227/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 227 submitted successfully"
else
    echo "[ERROR] Multirun job 227 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 228/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [25] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 228/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 228 submitted successfully"
else
    echo "[ERROR] Multirun job 228 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 229/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [26] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 229/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 229 submitted successfully"
else
    echo "[ERROR] Multirun job 229 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 230/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [27] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 230/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 230 submitted successfully"
else
    echo "[ERROR] Multirun job 230 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 231/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [28] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 231/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 231 submitted successfully"
else
    echo "[ERROR] Multirun job 231 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 232/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [29] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 232/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false ctnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 232 submitted successfully"
else
    echo "[ERROR] Multirun job 232 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 233/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [1] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 233/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 233 submitted successfully"
else
    echo "[ERROR] Multirun job 233 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 234/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [2] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 234/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 234 submitted successfully"
else
    echo "[ERROR] Multirun job 234 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 235/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [3] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 235/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 235 submitted successfully"
else
    echo "[ERROR] Multirun job 235 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 236/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [4] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 236/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 236 submitted successfully"
else
    echo "[ERROR] Multirun job 236 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 237/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [5] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 237/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 237 submitted successfully"
else
    echo "[ERROR] Multirun job 237 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 238/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [6] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 238/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 238 submitted successfully"
else
    echo "[ERROR] Multirun job 238 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 239/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [7] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 239/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 239 submitted successfully"
else
    echo "[ERROR] Multirun job 239 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 240/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [8] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 240/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 240 submitted successfully"
else
    echo "[ERROR] Multirun job 240 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 241/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [9] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 241/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 241 submitted successfully"
else
    echo "[ERROR] Multirun job 241 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 242/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [10] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 242/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 242 submitted successfully"
else
    echo "[ERROR] Multirun job 242 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 243/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [11] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 243/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 243 submitted successfully"
else
    echo "[ERROR] Multirun job 243 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 244/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [12] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 244/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 244 submitted successfully"
else
    echo "[ERROR] Multirun job 244 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 245/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [13] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 245/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 245 submitted successfully"
else
    echo "[ERROR] Multirun job 245 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 246/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [14] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 246/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 246 submitted successfully"
else
    echo "[ERROR] Multirun job 246 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 247/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [15] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 247/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 247 submitted successfully"
else
    echo "[ERROR] Multirun job 247 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 248/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [16] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 248/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 248 submitted successfully"
else
    echo "[ERROR] Multirun job 248 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 249/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [17] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 249/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 249 submitted successfully"
else
    echo "[ERROR] Multirun job 249 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 250/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [18] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 250/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 250 submitted successfully"
else
    echo "[ERROR] Multirun job 250 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 251/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [19] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 251/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 251 submitted successfully"
else
    echo "[ERROR] Multirun job 251 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 252/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [20] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 252/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 252 submitted successfully"
else
    echo "[ERROR] Multirun job 252 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 253/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [21] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 253/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 253 submitted successfully"
else
    echo "[ERROR] Multirun job 253 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 254/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [22] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 254/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 254 submitted successfully"
else
    echo "[ERROR] Multirun job 254 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 255/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [23] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 255/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 255 submitted successfully"
else
    echo "[ERROR] Multirun job 255 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 256/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [24] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 256/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 256 submitted successfully"
else
    echo "[ERROR] Multirun job 256 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 257/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [25] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 257/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 257 submitted successfully"
else
    echo "[ERROR] Multirun job 257 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 258/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [26] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 258/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 258 submitted successfully"
else
    echo "[ERROR] Multirun job 258 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 259/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [27] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 259/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 259 submitted successfully"
else
    echo "[ERROR] Multirun job 259 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 260/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [28] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 260/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 260 submitted successfully"
else
    echo "[ERROR] Multirun job 260 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 261/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [29] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 261/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false ctnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 261 submitted successfully"
else
    echo "[ERROR] Multirun job 261 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 262/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [1] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 262/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 262 submitted successfully"
else
    echo "[ERROR] Multirun job 262 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 263/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [2] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 263/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 263 submitted successfully"
else
    echo "[ERROR] Multirun job 263 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 264/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [3] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 264/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 264 submitted successfully"
else
    echo "[ERROR] Multirun job 264 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 265/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [4] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 265/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 265 submitted successfully"
else
    echo "[ERROR] Multirun job 265 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 266/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [5] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 266/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 266 submitted successfully"
else
    echo "[ERROR] Multirun job 266 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 267/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [6] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 267/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 267 submitted successfully"
else
    echo "[ERROR] Multirun job 267 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 268/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [7] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 268/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 268 submitted successfully"
else
    echo "[ERROR] Multirun job 268 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 269/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [8] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 269/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 269 submitted successfully"
else
    echo "[ERROR] Multirun job 269 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 270/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [9] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 270/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 270 submitted successfully"
else
    echo "[ERROR] Multirun job 270 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 271/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [10] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 271/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 271 submitted successfully"
else
    echo "[ERROR] Multirun job 271 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 272/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [11] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 272/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 272 submitted successfully"
else
    echo "[ERROR] Multirun job 272 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 273/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [12] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 273/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 273 submitted successfully"
else
    echo "[ERROR] Multirun job 273 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 274/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [13] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 274/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 274 submitted successfully"
else
    echo "[ERROR] Multirun job 274 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 275/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [14] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 275/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 275 submitted successfully"
else
    echo "[ERROR] Multirun job 275 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 276/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [15] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 276/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 276 submitted successfully"
else
    echo "[ERROR] Multirun job 276 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 277/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [16] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 277/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 277 submitted successfully"
else
    echo "[ERROR] Multirun job 277 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 278/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [17] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 278/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 278 submitted successfully"
else
    echo "[ERROR] Multirun job 278 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 279/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [18] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 279/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 279 submitted successfully"
else
    echo "[ERROR] Multirun job 279 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 280/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [19] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 280/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 280 submitted successfully"
else
    echo "[ERROR] Multirun job 280 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 281/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [20] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 281/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 281 submitted successfully"
else
    echo "[ERROR] Multirun job 281 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 282/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [21] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 282/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 282 submitted successfully"
else
    echo "[ERROR] Multirun job 282 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 283/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [22] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 283/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 283 submitted successfully"
else
    echo "[ERROR] Multirun job 283 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 284/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [23] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 284/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 284 submitted successfully"
else
    echo "[ERROR] Multirun job 284 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 285/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [24] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 285/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 285 submitted successfully"
else
    echo "[ERROR] Multirun job 285 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 286/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [25] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 286/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 286 submitted successfully"
else
    echo "[ERROR] Multirun job 286 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 287/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [26] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 287/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 287 submitted successfully"
else
    echo "[ERROR] Multirun job 287 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 288/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [27] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 288/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 288 submitted successfully"
else
    echo "[ERROR] Multirun job 288 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 289/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [28] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 289/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 289 submitted successfully"
else
    echo "[ERROR] Multirun job 289 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 290/725
# Dataset: Shin2017A | Model: ctnet | Eval: WithinSession | Subjects: [29] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: ctnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 290/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false ctnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 290 submitted successfully"
else
    echo "[ERROR] Multirun job 290 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 291/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [1] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 291/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 291 submitted successfully"
else
    echo "[ERROR] Multirun job 291 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 292/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [2] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 292/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 292 submitted successfully"
else
    echo "[ERROR] Multirun job 292 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 293/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [3] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 293/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 293 submitted successfully"
else
    echo "[ERROR] Multirun job 293 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 294/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [4] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 294/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 294 submitted successfully"
else
    echo "[ERROR] Multirun job 294 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 295/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [5] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 295/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 295 submitted successfully"
else
    echo "[ERROR] Multirun job 295 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 296/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [6] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 296/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 296 submitted successfully"
else
    echo "[ERROR] Multirun job 296 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 297/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [7] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 297/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 297 submitted successfully"
else
    echo "[ERROR] Multirun job 297 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 298/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [8] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 298/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 298 submitted successfully"
else
    echo "[ERROR] Multirun job 298 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 299/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [9] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 299/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 299 submitted successfully"
else
    echo "[ERROR] Multirun job 299 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 300/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [10] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 300/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 300 submitted successfully"
else
    echo "[ERROR] Multirun job 300 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 301/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [11] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 301/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 301 submitted successfully"
else
    echo "[ERROR] Multirun job 301 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 302/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [12] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 302/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 302 submitted successfully"
else
    echo "[ERROR] Multirun job 302 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 303/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [13] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 303/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 303 submitted successfully"
else
    echo "[ERROR] Multirun job 303 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 304/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [14] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 304/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 304 submitted successfully"
else
    echo "[ERROR] Multirun job 304 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 305/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [15] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 305/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 305 submitted successfully"
else
    echo "[ERROR] Multirun job 305 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 306/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [16] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 306/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 306 submitted successfully"
else
    echo "[ERROR] Multirun job 306 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 307/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [17] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 307/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 307 submitted successfully"
else
    echo "[ERROR] Multirun job 307 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 308/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [18] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 308/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 308 submitted successfully"
else
    echo "[ERROR] Multirun job 308 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 309/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [19] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 309/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 309 submitted successfully"
else
    echo "[ERROR] Multirun job 309 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 310/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [20] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 310/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 310 submitted successfully"
else
    echo "[ERROR] Multirun job 310 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 311/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [21] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 311/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 311 submitted successfully"
else
    echo "[ERROR] Multirun job 311 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 312/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [22] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 312/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 312 submitted successfully"
else
    echo "[ERROR] Multirun job 312 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 313/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [23] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 313/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 313 submitted successfully"
else
    echo "[ERROR] Multirun job 313 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 314/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [24] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 314/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 314 submitted successfully"
else
    echo "[ERROR] Multirun job 314 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 315/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [25] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 315/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 315 submitted successfully"
else
    echo "[ERROR] Multirun job 315 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 316/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [26] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 316/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 316 submitted successfully"
else
    echo "[ERROR] Multirun job 316 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 317/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [27] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 317/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 317 submitted successfully"
else
    echo "[ERROR] Multirun job 317 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 318/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [28] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 318/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 318 submitted successfully"
else
    echo "[ERROR] Multirun job 318 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 319/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [29] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 319/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false reegnet 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 319 submitted successfully"
else
    echo "[ERROR] Multirun job 319 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 320/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [1] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 320/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 320 submitted successfully"
else
    echo "[ERROR] Multirun job 320 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 321/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [2] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 321/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 321 submitted successfully"
else
    echo "[ERROR] Multirun job 321 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 322/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [3] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 322/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 322 submitted successfully"
else
    echo "[ERROR] Multirun job 322 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 323/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [4] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 323/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 323 submitted successfully"
else
    echo "[ERROR] Multirun job 323 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 324/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [5] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 324/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 324 submitted successfully"
else
    echo "[ERROR] Multirun job 324 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 325/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [6] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 325/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 325 submitted successfully"
else
    echo "[ERROR] Multirun job 325 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 326/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [7] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 326/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 326 submitted successfully"
else
    echo "[ERROR] Multirun job 326 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 327/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [8] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 327/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 327 submitted successfully"
else
    echo "[ERROR] Multirun job 327 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 328/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [9] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 328/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 328 submitted successfully"
else
    echo "[ERROR] Multirun job 328 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 329/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [10] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 329/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 329 submitted successfully"
else
    echo "[ERROR] Multirun job 329 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 330/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [11] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 330/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 330 submitted successfully"
else
    echo "[ERROR] Multirun job 330 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 331/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [12] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 331/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 331 submitted successfully"
else
    echo "[ERROR] Multirun job 331 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 332/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [13] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 332/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 332 submitted successfully"
else
    echo "[ERROR] Multirun job 332 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 333/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [14] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 333/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 333 submitted successfully"
else
    echo "[ERROR] Multirun job 333 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 334/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [15] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 334/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 334 submitted successfully"
else
    echo "[ERROR] Multirun job 334 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 335/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [16] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 335/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 335 submitted successfully"
else
    echo "[ERROR] Multirun job 335 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 336/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [17] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 336/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 336 submitted successfully"
else
    echo "[ERROR] Multirun job 336 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 337/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [18] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 337/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 337 submitted successfully"
else
    echo "[ERROR] Multirun job 337 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 338/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [19] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 338/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 338 submitted successfully"
else
    echo "[ERROR] Multirun job 338 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 339/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [20] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 339/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 339 submitted successfully"
else
    echo "[ERROR] Multirun job 339 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 340/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [21] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 340/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 340 submitted successfully"
else
    echo "[ERROR] Multirun job 340 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 341/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [22] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 341/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 341 submitted successfully"
else
    echo "[ERROR] Multirun job 341 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 342/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [23] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 342/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 342 submitted successfully"
else
    echo "[ERROR] Multirun job 342 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 343/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [24] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 343/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 343 submitted successfully"
else
    echo "[ERROR] Multirun job 343 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 344/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [25] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 344/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 344 submitted successfully"
else
    echo "[ERROR] Multirun job 344 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 345/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [26] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 345/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 345 submitted successfully"
else
    echo "[ERROR] Multirun job 345 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 346/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [27] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 346/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 346 submitted successfully"
else
    echo "[ERROR] Multirun job 346 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 347/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [28] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 347/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 347 submitted successfully"
else
    echo "[ERROR] Multirun job 347 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 348/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [29] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 348/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false reegnet 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 348 submitted successfully"
else
    echo "[ERROR] Multirun job 348 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 349/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [1] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 349/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 349 submitted successfully"
else
    echo "[ERROR] Multirun job 349 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 350/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [2] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 350/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 350 submitted successfully"
else
    echo "[ERROR] Multirun job 350 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 351/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [3] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 351/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 351 submitted successfully"
else
    echo "[ERROR] Multirun job 351 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 352/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [4] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 352/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 352 submitted successfully"
else
    echo "[ERROR] Multirun job 352 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 353/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [5] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 353/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 353 submitted successfully"
else
    echo "[ERROR] Multirun job 353 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 354/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [6] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 354/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 354 submitted successfully"
else
    echo "[ERROR] Multirun job 354 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 355/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [7] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 355/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 355 submitted successfully"
else
    echo "[ERROR] Multirun job 355 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 356/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [8] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 356/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 356 submitted successfully"
else
    echo "[ERROR] Multirun job 356 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 357/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [9] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 357/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 357 submitted successfully"
else
    echo "[ERROR] Multirun job 357 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 358/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [10] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 358/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 358 submitted successfully"
else
    echo "[ERROR] Multirun job 358 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 359/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [11] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 359/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 359 submitted successfully"
else
    echo "[ERROR] Multirun job 359 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 360/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [12] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 360/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 360 submitted successfully"
else
    echo "[ERROR] Multirun job 360 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 361/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [13] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 361/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 361 submitted successfully"
else
    echo "[ERROR] Multirun job 361 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 362/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [14] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 362/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 362 submitted successfully"
else
    echo "[ERROR] Multirun job 362 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 363/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [15] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 363/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 363 submitted successfully"
else
    echo "[ERROR] Multirun job 363 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 364/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [16] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 364/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 364 submitted successfully"
else
    echo "[ERROR] Multirun job 364 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 365/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [17] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 365/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 365 submitted successfully"
else
    echo "[ERROR] Multirun job 365 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 366/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [18] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 366/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 366 submitted successfully"
else
    echo "[ERROR] Multirun job 366 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 367/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [19] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 367/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 367 submitted successfully"
else
    echo "[ERROR] Multirun job 367 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 368/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [20] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 368/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 368 submitted successfully"
else
    echo "[ERROR] Multirun job 368 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 369/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [21] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 369/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 369 submitted successfully"
else
    echo "[ERROR] Multirun job 369 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 370/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [22] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 370/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 370 submitted successfully"
else
    echo "[ERROR] Multirun job 370 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 371/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [23] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 371/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 371 submitted successfully"
else
    echo "[ERROR] Multirun job 371 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 372/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [24] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 372/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 372 submitted successfully"
else
    echo "[ERROR] Multirun job 372 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 373/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [25] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 373/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 373 submitted successfully"
else
    echo "[ERROR] Multirun job 373 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 374/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [26] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 374/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 374 submitted successfully"
else
    echo "[ERROR] Multirun job 374 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 375/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [27] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 375/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 375 submitted successfully"
else
    echo "[ERROR] Multirun job 375 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 376/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [28] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 376/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 376 submitted successfully"
else
    echo "[ERROR] Multirun job 376 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 377/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [29] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 377/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false reegnet 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 377 submitted successfully"
else
    echo "[ERROR] Multirun job 377 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 378/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [1] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 378/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 378 submitted successfully"
else
    echo "[ERROR] Multirun job 378 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 379/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [2] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 379/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 379 submitted successfully"
else
    echo "[ERROR] Multirun job 379 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 380/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [3] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 380/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 380 submitted successfully"
else
    echo "[ERROR] Multirun job 380 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 381/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [4] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 381/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 381 submitted successfully"
else
    echo "[ERROR] Multirun job 381 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 382/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [5] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 382/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 382 submitted successfully"
else
    echo "[ERROR] Multirun job 382 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 383/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [6] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 383/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 383 submitted successfully"
else
    echo "[ERROR] Multirun job 383 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 384/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [7] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 384/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 384 submitted successfully"
else
    echo "[ERROR] Multirun job 384 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 385/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [8] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 385/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 385 submitted successfully"
else
    echo "[ERROR] Multirun job 385 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 386/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [9] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 386/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 386 submitted successfully"
else
    echo "[ERROR] Multirun job 386 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 387/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [10] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 387/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 387 submitted successfully"
else
    echo "[ERROR] Multirun job 387 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 388/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [11] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 388/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 388 submitted successfully"
else
    echo "[ERROR] Multirun job 388 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 389/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [12] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 389/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 389 submitted successfully"
else
    echo "[ERROR] Multirun job 389 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 390/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [13] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 390/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 390 submitted successfully"
else
    echo "[ERROR] Multirun job 390 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 391/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [14] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 391/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 391 submitted successfully"
else
    echo "[ERROR] Multirun job 391 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 392/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [15] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 392/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 392 submitted successfully"
else
    echo "[ERROR] Multirun job 392 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 393/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [16] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 393/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 393 submitted successfully"
else
    echo "[ERROR] Multirun job 393 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 394/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [17] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 394/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 394 submitted successfully"
else
    echo "[ERROR] Multirun job 394 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 395/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [18] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 395/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 395 submitted successfully"
else
    echo "[ERROR] Multirun job 395 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 396/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [19] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 396/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 396 submitted successfully"
else
    echo "[ERROR] Multirun job 396 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 397/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [20] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 397/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 397 submitted successfully"
else
    echo "[ERROR] Multirun job 397 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 398/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [21] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 398/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 398 submitted successfully"
else
    echo "[ERROR] Multirun job 398 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 399/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [22] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 399/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 399 submitted successfully"
else
    echo "[ERROR] Multirun job 399 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 400/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [23] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 400/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 400 submitted successfully"
else
    echo "[ERROR] Multirun job 400 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 401/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [24] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 401/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 401 submitted successfully"
else
    echo "[ERROR] Multirun job 401 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 402/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [25] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 402/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 402 submitted successfully"
else
    echo "[ERROR] Multirun job 402 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 403/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [26] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 403/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 403 submitted successfully"
else
    echo "[ERROR] Multirun job 403 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 404/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [27] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 404/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 404 submitted successfully"
else
    echo "[ERROR] Multirun job 404 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 405/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [28] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 405/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 405 submitted successfully"
else
    echo "[ERROR] Multirun job 405 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 406/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [29] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 406/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false reegnet 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 406 submitted successfully"
else
    echo "[ERROR] Multirun job 406 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 407/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [1] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 407/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 407 submitted successfully"
else
    echo "[ERROR] Multirun job 407 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 408/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [2] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 408/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 408 submitted successfully"
else
    echo "[ERROR] Multirun job 408 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 409/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [3] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 409/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 409 submitted successfully"
else
    echo "[ERROR] Multirun job 409 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 410/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [4] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 410/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 410 submitted successfully"
else
    echo "[ERROR] Multirun job 410 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 411/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [5] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 411/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 411 submitted successfully"
else
    echo "[ERROR] Multirun job 411 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 412/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [6] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 412/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 412 submitted successfully"
else
    echo "[ERROR] Multirun job 412 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 413/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [7] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 413/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 413 submitted successfully"
else
    echo "[ERROR] Multirun job 413 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 414/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [8] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 414/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 414 submitted successfully"
else
    echo "[ERROR] Multirun job 414 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 415/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [9] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 415/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 415 submitted successfully"
else
    echo "[ERROR] Multirun job 415 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 416/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [10] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 416/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 416 submitted successfully"
else
    echo "[ERROR] Multirun job 416 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 417/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [11] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 417/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 417 submitted successfully"
else
    echo "[ERROR] Multirun job 417 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 418/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [12] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 418/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 418 submitted successfully"
else
    echo "[ERROR] Multirun job 418 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 419/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [13] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 419/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 419 submitted successfully"
else
    echo "[ERROR] Multirun job 419 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 420/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [14] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 420/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 420 submitted successfully"
else
    echo "[ERROR] Multirun job 420 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 421/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [15] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 421/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 421 submitted successfully"
else
    echo "[ERROR] Multirun job 421 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 422/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [16] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 422/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 422 submitted successfully"
else
    echo "[ERROR] Multirun job 422 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 423/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [17] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 423/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 423 submitted successfully"
else
    echo "[ERROR] Multirun job 423 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 424/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [18] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 424/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 424 submitted successfully"
else
    echo "[ERROR] Multirun job 424 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 425/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [19] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 425/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 425 submitted successfully"
else
    echo "[ERROR] Multirun job 425 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 426/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [20] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 426/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 426 submitted successfully"
else
    echo "[ERROR] Multirun job 426 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 427/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [21] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 427/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 427 submitted successfully"
else
    echo "[ERROR] Multirun job 427 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 428/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [22] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 428/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 428 submitted successfully"
else
    echo "[ERROR] Multirun job 428 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 429/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [23] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 429/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 429 submitted successfully"
else
    echo "[ERROR] Multirun job 429 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 430/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [24] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 430/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 430 submitted successfully"
else
    echo "[ERROR] Multirun job 430 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 431/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [25] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 431/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 431 submitted successfully"
else
    echo "[ERROR] Multirun job 431 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 432/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [26] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 432/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 432 submitted successfully"
else
    echo "[ERROR] Multirun job 432 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 433/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [27] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 433/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 433 submitted successfully"
else
    echo "[ERROR] Multirun job 433 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 434/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [28] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 434/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 434 submitted successfully"
else
    echo "[ERROR] Multirun job 434 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 435/725
# Dataset: Shin2017A | Model: reegnet | Eval: WithinSession | Subjects: [29] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 435/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false reegnet 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 435 submitted successfully"
else
    echo "[ERROR] Multirun job 435 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 436/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [1] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 436/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 436 submitted successfully"
else
    echo "[ERROR] Multirun job 436 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 437/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [2] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 437/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 437 submitted successfully"
else
    echo "[ERROR] Multirun job 437 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 438/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [3] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 438/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 438 submitted successfully"
else
    echo "[ERROR] Multirun job 438 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 439/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [4] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 439/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 439 submitted successfully"
else
    echo "[ERROR] Multirun job 439 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 440/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [5] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 440/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 440 submitted successfully"
else
    echo "[ERROR] Multirun job 440 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 441/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [6] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 441/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 441 submitted successfully"
else
    echo "[ERROR] Multirun job 441 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 442/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [7] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 442/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 442 submitted successfully"
else
    echo "[ERROR] Multirun job 442 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 443/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [8] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 443/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 443 submitted successfully"
else
    echo "[ERROR] Multirun job 443 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 444/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [9] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 444/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 444 submitted successfully"
else
    echo "[ERROR] Multirun job 444 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 445/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [10] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 445/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 445 submitted successfully"
else
    echo "[ERROR] Multirun job 445 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 446/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [11] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 446/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 446 submitted successfully"
else
    echo "[ERROR] Multirun job 446 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 447/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [12] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 447/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 447 submitted successfully"
else
    echo "[ERROR] Multirun job 447 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 448/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [13] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 448/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 448 submitted successfully"
else
    echo "[ERROR] Multirun job 448 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 449/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [14] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 449/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 449 submitted successfully"
else
    echo "[ERROR] Multirun job 449 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 450/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [15] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 450/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 450 submitted successfully"
else
    echo "[ERROR] Multirun job 450 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 451/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [16] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 451/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 451 submitted successfully"
else
    echo "[ERROR] Multirun job 451 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 452/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [17] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 452/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 452 submitted successfully"
else
    echo "[ERROR] Multirun job 452 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 453/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [18] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 453/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 453 submitted successfully"
else
    echo "[ERROR] Multirun job 453 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 454/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [19] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 454/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 454 submitted successfully"
else
    echo "[ERROR] Multirun job 454 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 455/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [20] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 455/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 455 submitted successfully"
else
    echo "[ERROR] Multirun job 455 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 456/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [21] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 456/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 456 submitted successfully"
else
    echo "[ERROR] Multirun job 456 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 457/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [22] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 457/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 457 submitted successfully"
else
    echo "[ERROR] Multirun job 457 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 458/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [23] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 458/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 458 submitted successfully"
else
    echo "[ERROR] Multirun job 458 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 459/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [24] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 459/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 459 submitted successfully"
else
    echo "[ERROR] Multirun job 459 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 460/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [25] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 460/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 460 submitted successfully"
else
    echo "[ERROR] Multirun job 460 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 461/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [26] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 461/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 461 submitted successfully"
else
    echo "[ERROR] Multirun job 461 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 462/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [27] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 462/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 462 submitted successfully"
else
    echo "[ERROR] Multirun job 462 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 463/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [28] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 463/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 463 submitted successfully"
else
    echo "[ERROR] Multirun job 463 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 464/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [29] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 464/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false cnn_ncp 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 464 submitted successfully"
else
    echo "[ERROR] Multirun job 464 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 465/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [1] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 465/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 465 submitted successfully"
else
    echo "[ERROR] Multirun job 465 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 466/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [2] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 466/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 466 submitted successfully"
else
    echo "[ERROR] Multirun job 466 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 467/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [3] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 467/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 467 submitted successfully"
else
    echo "[ERROR] Multirun job 467 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 468/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [4] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 468/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 468 submitted successfully"
else
    echo "[ERROR] Multirun job 468 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 469/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [5] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 469/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 469 submitted successfully"
else
    echo "[ERROR] Multirun job 469 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 470/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [6] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 470/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 470 submitted successfully"
else
    echo "[ERROR] Multirun job 470 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 471/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [7] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 471/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 471 submitted successfully"
else
    echo "[ERROR] Multirun job 471 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 472/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [8] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 472/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 472 submitted successfully"
else
    echo "[ERROR] Multirun job 472 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 473/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [9] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 473/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 473 submitted successfully"
else
    echo "[ERROR] Multirun job 473 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 474/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [10] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 474/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 474 submitted successfully"
else
    echo "[ERROR] Multirun job 474 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 475/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [11] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 475/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 475 submitted successfully"
else
    echo "[ERROR] Multirun job 475 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 476/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [12] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 476/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 476 submitted successfully"
else
    echo "[ERROR] Multirun job 476 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 477/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [13] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 477/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 477 submitted successfully"
else
    echo "[ERROR] Multirun job 477 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 478/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [14] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 478/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 478 submitted successfully"
else
    echo "[ERROR] Multirun job 478 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 479/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [15] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 479/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 479 submitted successfully"
else
    echo "[ERROR] Multirun job 479 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 480/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [16] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 480/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 480 submitted successfully"
else
    echo "[ERROR] Multirun job 480 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 481/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [17] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 481/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 481 submitted successfully"
else
    echo "[ERROR] Multirun job 481 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 482/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [18] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 482/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 482 submitted successfully"
else
    echo "[ERROR] Multirun job 482 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 483/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [19] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 483/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 483 submitted successfully"
else
    echo "[ERROR] Multirun job 483 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 484/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [20] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 484/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 484 submitted successfully"
else
    echo "[ERROR] Multirun job 484 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 485/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [21] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 485/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 485 submitted successfully"
else
    echo "[ERROR] Multirun job 485 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 486/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [22] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 486/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 486 submitted successfully"
else
    echo "[ERROR] Multirun job 486 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 487/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [23] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 487/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 487 submitted successfully"
else
    echo "[ERROR] Multirun job 487 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 488/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [24] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 488/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 488 submitted successfully"
else
    echo "[ERROR] Multirun job 488 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 489/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [25] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 489/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 489 submitted successfully"
else
    echo "[ERROR] Multirun job 489 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 490/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [26] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 490/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 490 submitted successfully"
else
    echo "[ERROR] Multirun job 490 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 491/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [27] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 491/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 491 submitted successfully"
else
    echo "[ERROR] Multirun job 491 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 492/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [28] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 492/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 492 submitted successfully"
else
    echo "[ERROR] Multirun job 492 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 493/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [29] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 493/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false cnn_ncp 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 493 submitted successfully"
else
    echo "[ERROR] Multirun job 493 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 494/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [1] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 494/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 494 submitted successfully"
else
    echo "[ERROR] Multirun job 494 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 495/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [2] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 495/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 495 submitted successfully"
else
    echo "[ERROR] Multirun job 495 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 496/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [3] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 496/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 496 submitted successfully"
else
    echo "[ERROR] Multirun job 496 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 497/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [4] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 497/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 497 submitted successfully"
else
    echo "[ERROR] Multirun job 497 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 498/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [5] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 498/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 498 submitted successfully"
else
    echo "[ERROR] Multirun job 498 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 499/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [6] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 499/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 499 submitted successfully"
else
    echo "[ERROR] Multirun job 499 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 500/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [7] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 500/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 500 submitted successfully"
else
    echo "[ERROR] Multirun job 500 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 501/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [8] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 501/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 501 submitted successfully"
else
    echo "[ERROR] Multirun job 501 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 502/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [9] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 502/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 502 submitted successfully"
else
    echo "[ERROR] Multirun job 502 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 503/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [10] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 503/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 503 submitted successfully"
else
    echo "[ERROR] Multirun job 503 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 504/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [11] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 504/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 504 submitted successfully"
else
    echo "[ERROR] Multirun job 504 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 505/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [12] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 505/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 505 submitted successfully"
else
    echo "[ERROR] Multirun job 505 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 506/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [13] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 506/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 506 submitted successfully"
else
    echo "[ERROR] Multirun job 506 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 507/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [14] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 507/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 507 submitted successfully"
else
    echo "[ERROR] Multirun job 507 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 508/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [15] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 508/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 508 submitted successfully"
else
    echo "[ERROR] Multirun job 508 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 509/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [16] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 509/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 509 submitted successfully"
else
    echo "[ERROR] Multirun job 509 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 510/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [17] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 510/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 510 submitted successfully"
else
    echo "[ERROR] Multirun job 510 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 511/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [18] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 511/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 511 submitted successfully"
else
    echo "[ERROR] Multirun job 511 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 512/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [19] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 512/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 512 submitted successfully"
else
    echo "[ERROR] Multirun job 512 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 513/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [20] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 513/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 513 submitted successfully"
else
    echo "[ERROR] Multirun job 513 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 514/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [21] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 514/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 514 submitted successfully"
else
    echo "[ERROR] Multirun job 514 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 515/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [22] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 515/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 515 submitted successfully"
else
    echo "[ERROR] Multirun job 515 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 516/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [23] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 516/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 516 submitted successfully"
else
    echo "[ERROR] Multirun job 516 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 517/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [24] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 517/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 517 submitted successfully"
else
    echo "[ERROR] Multirun job 517 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 518/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [25] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 518/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 518 submitted successfully"
else
    echo "[ERROR] Multirun job 518 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 519/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [26] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 519/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 519 submitted successfully"
else
    echo "[ERROR] Multirun job 519 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 520/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [27] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 520/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 520 submitted successfully"
else
    echo "[ERROR] Multirun job 520 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 521/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [28] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 521/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 521 submitted successfully"
else
    echo "[ERROR] Multirun job 521 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 522/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [29] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 522/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false cnn_ncp 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 522 submitted successfully"
else
    echo "[ERROR] Multirun job 522 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 523/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [1] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 523/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 523 submitted successfully"
else
    echo "[ERROR] Multirun job 523 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 524/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [2] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 524/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 524 submitted successfully"
else
    echo "[ERROR] Multirun job 524 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 525/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [3] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 525/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 525 submitted successfully"
else
    echo "[ERROR] Multirun job 525 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 526/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [4] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 526/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 526 submitted successfully"
else
    echo "[ERROR] Multirun job 526 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 527/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [5] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 527/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 527 submitted successfully"
else
    echo "[ERROR] Multirun job 527 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 528/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [6] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 528/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 528 submitted successfully"
else
    echo "[ERROR] Multirun job 528 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 529/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [7] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 529/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 529 submitted successfully"
else
    echo "[ERROR] Multirun job 529 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 530/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [8] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 530/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 530 submitted successfully"
else
    echo "[ERROR] Multirun job 530 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 531/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [9] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 531/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 531 submitted successfully"
else
    echo "[ERROR] Multirun job 531 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 532/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [10] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 532/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 532 submitted successfully"
else
    echo "[ERROR] Multirun job 532 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 533/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [11] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 533/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 533 submitted successfully"
else
    echo "[ERROR] Multirun job 533 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 534/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [12] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 534/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 534 submitted successfully"
else
    echo "[ERROR] Multirun job 534 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 535/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [13] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 535/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 535 submitted successfully"
else
    echo "[ERROR] Multirun job 535 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 536/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [14] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 536/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 536 submitted successfully"
else
    echo "[ERROR] Multirun job 536 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 537/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [15] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 537/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 537 submitted successfully"
else
    echo "[ERROR] Multirun job 537 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 538/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [16] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 538/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 538 submitted successfully"
else
    echo "[ERROR] Multirun job 538 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 539/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [17] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 539/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 539 submitted successfully"
else
    echo "[ERROR] Multirun job 539 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 540/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [18] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 540/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 540 submitted successfully"
else
    echo "[ERROR] Multirun job 540 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 541/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [19] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 541/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 541 submitted successfully"
else
    echo "[ERROR] Multirun job 541 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 542/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [20] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 542/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 542 submitted successfully"
else
    echo "[ERROR] Multirun job 542 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 543/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [21] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 543/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 543 submitted successfully"
else
    echo "[ERROR] Multirun job 543 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 544/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [22] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 544/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 544 submitted successfully"
else
    echo "[ERROR] Multirun job 544 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 545/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [23] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 545/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 545 submitted successfully"
else
    echo "[ERROR] Multirun job 545 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 546/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [24] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 546/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 546 submitted successfully"
else
    echo "[ERROR] Multirun job 546 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 547/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [25] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 547/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 547 submitted successfully"
else
    echo "[ERROR] Multirun job 547 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 548/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [26] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 548/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 548 submitted successfully"
else
    echo "[ERROR] Multirun job 548 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 549/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [27] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 549/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 549 submitted successfully"
else
    echo "[ERROR] Multirun job 549 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 550/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [28] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 550/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 550 submitted successfully"
else
    echo "[ERROR] Multirun job 550 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 551/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [29] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 551/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false cnn_ncp 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 551 submitted successfully"
else
    echo "[ERROR] Multirun job 551 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 552/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [1] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 552/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 552 submitted successfully"
else
    echo "[ERROR] Multirun job 552 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 553/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [2] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 553/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 553 submitted successfully"
else
    echo "[ERROR] Multirun job 553 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 554/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [3] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 554/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 554 submitted successfully"
else
    echo "[ERROR] Multirun job 554 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 555/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [4] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 555/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 555 submitted successfully"
else
    echo "[ERROR] Multirun job 555 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 556/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [5] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 556/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 556 submitted successfully"
else
    echo "[ERROR] Multirun job 556 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 557/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [6] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 557/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 557 submitted successfully"
else
    echo "[ERROR] Multirun job 557 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 558/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [7] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 558/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 558 submitted successfully"
else
    echo "[ERROR] Multirun job 558 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 559/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [8] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 559/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 559 submitted successfully"
else
    echo "[ERROR] Multirun job 559 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 560/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [9] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 560/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 560 submitted successfully"
else
    echo "[ERROR] Multirun job 560 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 561/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [10] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 561/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 561 submitted successfully"
else
    echo "[ERROR] Multirun job 561 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 562/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [11] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 562/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 562 submitted successfully"
else
    echo "[ERROR] Multirun job 562 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 563/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [12] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 563/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 563 submitted successfully"
else
    echo "[ERROR] Multirun job 563 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 564/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [13] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 564/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 564 submitted successfully"
else
    echo "[ERROR] Multirun job 564 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 565/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [14] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 565/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 565 submitted successfully"
else
    echo "[ERROR] Multirun job 565 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 566/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [15] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 566/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 566 submitted successfully"
else
    echo "[ERROR] Multirun job 566 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 567/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [16] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 567/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 567 submitted successfully"
else
    echo "[ERROR] Multirun job 567 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 568/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [17] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 568/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 568 submitted successfully"
else
    echo "[ERROR] Multirun job 568 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 569/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [18] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 569/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 569 submitted successfully"
else
    echo "[ERROR] Multirun job 569 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 570/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [19] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 570/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 570 submitted successfully"
else
    echo "[ERROR] Multirun job 570 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 571/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [20] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 571/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 571 submitted successfully"
else
    echo "[ERROR] Multirun job 571 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 572/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [21] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 572/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 572 submitted successfully"
else
    echo "[ERROR] Multirun job 572 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 573/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [22] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 573/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 573 submitted successfully"
else
    echo "[ERROR] Multirun job 573 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 574/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [23] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 574/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 574 submitted successfully"
else
    echo "[ERROR] Multirun job 574 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 575/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [24] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 575/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 575 submitted successfully"
else
    echo "[ERROR] Multirun job 575 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 576/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [25] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 576/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 576 submitted successfully"
else
    echo "[ERROR] Multirun job 576 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 577/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [26] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 577/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 577 submitted successfully"
else
    echo "[ERROR] Multirun job 577 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 578/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [27] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 578/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 578 submitted successfully"
else
    echo "[ERROR] Multirun job 578 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 579/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [28] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 579/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 579 submitted successfully"
else
    echo "[ERROR] Multirun job 579 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 580/725
# Dataset: Shin2017A | Model: cnn_ncp | Eval: WithinSession | Subjects: [29] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 580/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false cnn_ncp 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 580 submitted successfully"
else
    echo "[ERROR] Multirun job 580 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 581/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [1] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 581/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 581 submitted successfully"
else
    echo "[ERROR] Multirun job 581 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 582/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [2] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 582/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 582 submitted successfully"
else
    echo "[ERROR] Multirun job 582 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 583/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [3] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 583/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 583 submitted successfully"
else
    echo "[ERROR] Multirun job 583 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 584/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [4] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 584/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 584 submitted successfully"
else
    echo "[ERROR] Multirun job 584 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 585/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [5] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 585/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 585 submitted successfully"
else
    echo "[ERROR] Multirun job 585 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 586/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [6] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 586/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 586 submitted successfully"
else
    echo "[ERROR] Multirun job 586 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 587/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [7] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 587/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 587 submitted successfully"
else
    echo "[ERROR] Multirun job 587 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 588/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [8] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 588/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 588 submitted successfully"
else
    echo "[ERROR] Multirun job 588 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 589/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [9] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 589/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 589 submitted successfully"
else
    echo "[ERROR] Multirun job 589 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 590/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [10] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 590/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 590 submitted successfully"
else
    echo "[ERROR] Multirun job 590 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 591/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [11] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 591/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 591 submitted successfully"
else
    echo "[ERROR] Multirun job 591 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 592/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [12] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 592/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 592 submitted successfully"
else
    echo "[ERROR] Multirun job 592 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 593/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [13] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 593/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 593 submitted successfully"
else
    echo "[ERROR] Multirun job 593 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 594/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [14] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 594/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 594 submitted successfully"
else
    echo "[ERROR] Multirun job 594 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 595/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [15] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 595/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 595 submitted successfully"
else
    echo "[ERROR] Multirun job 595 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 596/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [16] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 596/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 596 submitted successfully"
else
    echo "[ERROR] Multirun job 596 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 597/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [17] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 597/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 597 submitted successfully"
else
    echo "[ERROR] Multirun job 597 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 598/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [18] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 598/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 598 submitted successfully"
else
    echo "[ERROR] Multirun job 598 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 599/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [19] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 599/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 599 submitted successfully"
else
    echo "[ERROR] Multirun job 599 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 600/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [20] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 600/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 600 submitted successfully"
else
    echo "[ERROR] Multirun job 600 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 601/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [21] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 601/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 601 submitted successfully"
else
    echo "[ERROR] Multirun job 601 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 602/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [22] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 602/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 602 submitted successfully"
else
    echo "[ERROR] Multirun job 602 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 603/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [23] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 603/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 603 submitted successfully"
else
    echo "[ERROR] Multirun job 603 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 604/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [24] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 604/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 604 submitted successfully"
else
    echo "[ERROR] Multirun job 604 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 605/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [25] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 605/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 605 submitted successfully"
else
    echo "[ERROR] Multirun job 605 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 606/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [26] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 606/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 606 submitted successfully"
else
    echo "[ERROR] Multirun job 606 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 607/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [27] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 607/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 607 submitted successfully"
else
    echo "[ERROR] Multirun job 607 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 608/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [28] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 608/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 608 submitted successfully"
else
    echo "[ERROR] Multirun job 608 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 609/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [29] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 609/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false branched_wiredcfc_arch4 100 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 609 submitted successfully"
else
    echo "[ERROR] Multirun job 609 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 610/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [1] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 610/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 610 submitted successfully"
else
    echo "[ERROR] Multirun job 610 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 611/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [2] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 611/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 611 submitted successfully"
else
    echo "[ERROR] Multirun job 611 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 612/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [3] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 612/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 612 submitted successfully"
else
    echo "[ERROR] Multirun job 612 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 613/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [4] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 613/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 613 submitted successfully"
else
    echo "[ERROR] Multirun job 613 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 614/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [5] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 614/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 614 submitted successfully"
else
    echo "[ERROR] Multirun job 614 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 615/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [6] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 615/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 615 submitted successfully"
else
    echo "[ERROR] Multirun job 615 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 616/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [7] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 616/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 616 submitted successfully"
else
    echo "[ERROR] Multirun job 616 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 617/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [8] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 617/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 617 submitted successfully"
else
    echo "[ERROR] Multirun job 617 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 618/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [9] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 618/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 618 submitted successfully"
else
    echo "[ERROR] Multirun job 618 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 619/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [10] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 619/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 619 submitted successfully"
else
    echo "[ERROR] Multirun job 619 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 620/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [11] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 620/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 620 submitted successfully"
else
    echo "[ERROR] Multirun job 620 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 621/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [12] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 621/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 621 submitted successfully"
else
    echo "[ERROR] Multirun job 621 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 622/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [13] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 622/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 622 submitted successfully"
else
    echo "[ERROR] Multirun job 622 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 623/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [14] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 623/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 623 submitted successfully"
else
    echo "[ERROR] Multirun job 623 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 624/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [15] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 624/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 624 submitted successfully"
else
    echo "[ERROR] Multirun job 624 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 625/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [16] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 625/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 625 submitted successfully"
else
    echo "[ERROR] Multirun job 625 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 626/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [17] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 626/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 626 submitted successfully"
else
    echo "[ERROR] Multirun job 626 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 627/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [18] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 627/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 627 submitted successfully"
else
    echo "[ERROR] Multirun job 627 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 628/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [19] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 628/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 628 submitted successfully"
else
    echo "[ERROR] Multirun job 628 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 629/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [20] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 629/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 629 submitted successfully"
else
    echo "[ERROR] Multirun job 629 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 630/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [21] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 630/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 630 submitted successfully"
else
    echo "[ERROR] Multirun job 630 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 631/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [22] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 631/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 631 submitted successfully"
else
    echo "[ERROR] Multirun job 631 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 632/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [23] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 632/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 632 submitted successfully"
else
    echo "[ERROR] Multirun job 632 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 633/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [24] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 633/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 633 submitted successfully"
else
    echo "[ERROR] Multirun job 633 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 634/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [25] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 634/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 634 submitted successfully"
else
    echo "[ERROR] Multirun job 634 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 635/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [26] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 635/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 635 submitted successfully"
else
    echo "[ERROR] Multirun job 635 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 636/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [27] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 636/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 636 submitted successfully"
else
    echo "[ERROR] Multirun job 636 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 637/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [28] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 637/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 637 submitted successfully"
else
    echo "[ERROR] Multirun job 637 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 638/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [29] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 638/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false branched_wiredcfc_arch4 200 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 638 submitted successfully"
else
    echo "[ERROR] Multirun job 638 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 639/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [1] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 639/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 639 submitted successfully"
else
    echo "[ERROR] Multirun job 639 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 640/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [2] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 640/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 640 submitted successfully"
else
    echo "[ERROR] Multirun job 640 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 641/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [3] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 641/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 641 submitted successfully"
else
    echo "[ERROR] Multirun job 641 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 642/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [4] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 642/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 642 submitted successfully"
else
    echo "[ERROR] Multirun job 642 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 643/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [5] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 643/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 643 submitted successfully"
else
    echo "[ERROR] Multirun job 643 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 644/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [6] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 644/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 644 submitted successfully"
else
    echo "[ERROR] Multirun job 644 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 645/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [7] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 645/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 645 submitted successfully"
else
    echo "[ERROR] Multirun job 645 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 646/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [8] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 646/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 646 submitted successfully"
else
    echo "[ERROR] Multirun job 646 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 647/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [9] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 647/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 647 submitted successfully"
else
    echo "[ERROR] Multirun job 647 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 648/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [10] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 648/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 648 submitted successfully"
else
    echo "[ERROR] Multirun job 648 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 649/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [11] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 649/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 649 submitted successfully"
else
    echo "[ERROR] Multirun job 649 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 650/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [12] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 650/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 650 submitted successfully"
else
    echo "[ERROR] Multirun job 650 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 651/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [13] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 651/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 651 submitted successfully"
else
    echo "[ERROR] Multirun job 651 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 652/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [14] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 652/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 652 submitted successfully"
else
    echo "[ERROR] Multirun job 652 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 653/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [15] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 653/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 653 submitted successfully"
else
    echo "[ERROR] Multirun job 653 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 654/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [16] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 654/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 654 submitted successfully"
else
    echo "[ERROR] Multirun job 654 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 655/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [17] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 655/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 655 submitted successfully"
else
    echo "[ERROR] Multirun job 655 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 656/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [18] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 656/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 656 submitted successfully"
else
    echo "[ERROR] Multirun job 656 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 657/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [19] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 657/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 657 submitted successfully"
else
    echo "[ERROR] Multirun job 657 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 658/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [20] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 658/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 658 submitted successfully"
else
    echo "[ERROR] Multirun job 658 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 659/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [21] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 659/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 659 submitted successfully"
else
    echo "[ERROR] Multirun job 659 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 660/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [22] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 660/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 660 submitted successfully"
else
    echo "[ERROR] Multirun job 660 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 661/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [23] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 661/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 661 submitted successfully"
else
    echo "[ERROR] Multirun job 661 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 662/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [24] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 662/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 662 submitted successfully"
else
    echo "[ERROR] Multirun job 662 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 663/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [25] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 663/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 663 submitted successfully"
else
    echo "[ERROR] Multirun job 663 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 664/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [26] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 664/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 664 submitted successfully"
else
    echo "[ERROR] Multirun job 664 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 665/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [27] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 665/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 665 submitted successfully"
else
    echo "[ERROR] Multirun job 665 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 666/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [28] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 666/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 666 submitted successfully"
else
    echo "[ERROR] Multirun job 666 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 667/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [29] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 667/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false branched_wiredcfc_arch4 300 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 667 submitted successfully"
else
    echo "[ERROR] Multirun job 667 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 668/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [1] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 668/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 668 submitted successfully"
else
    echo "[ERROR] Multirun job 668 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 669/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [2] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 669/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 669 submitted successfully"
else
    echo "[ERROR] Multirun job 669 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 670/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [3] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 670/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 670 submitted successfully"
else
    echo "[ERROR] Multirun job 670 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 671/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [4] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 671/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 671 submitted successfully"
else
    echo "[ERROR] Multirun job 671 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 672/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [5] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 672/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 672 submitted successfully"
else
    echo "[ERROR] Multirun job 672 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 673/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [6] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 673/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 673 submitted successfully"
else
    echo "[ERROR] Multirun job 673 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 674/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [7] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 674/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 674 submitted successfully"
else
    echo "[ERROR] Multirun job 674 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 675/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [8] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 675/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 675 submitted successfully"
else
    echo "[ERROR] Multirun job 675 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 676/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [9] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 676/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 676 submitted successfully"
else
    echo "[ERROR] Multirun job 676 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 677/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [10] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 677/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 677 submitted successfully"
else
    echo "[ERROR] Multirun job 677 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 678/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [11] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 678/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 678 submitted successfully"
else
    echo "[ERROR] Multirun job 678 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 679/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [12] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 679/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 679 submitted successfully"
else
    echo "[ERROR] Multirun job 679 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 680/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [13] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 680/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 680 submitted successfully"
else
    echo "[ERROR] Multirun job 680 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 681/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [14] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 681/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 681 submitted successfully"
else
    echo "[ERROR] Multirun job 681 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 682/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [15] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 682/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 682 submitted successfully"
else
    echo "[ERROR] Multirun job 682 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 683/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [16] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 683/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 683 submitted successfully"
else
    echo "[ERROR] Multirun job 683 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 684/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [17] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 684/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 684 submitted successfully"
else
    echo "[ERROR] Multirun job 684 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 685/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [18] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 685/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 685 submitted successfully"
else
    echo "[ERROR] Multirun job 685 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 686/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [19] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 686/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 686 submitted successfully"
else
    echo "[ERROR] Multirun job 686 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 687/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [20] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 687/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 687 submitted successfully"
else
    echo "[ERROR] Multirun job 687 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 688/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [21] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 688/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 688 submitted successfully"
else
    echo "[ERROR] Multirun job 688 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 689/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [22] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 689/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 689 submitted successfully"
else
    echo "[ERROR] Multirun job 689 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 690/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [23] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 690/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 690 submitted successfully"
else
    echo "[ERROR] Multirun job 690 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 691/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [24] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 691/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 691 submitted successfully"
else
    echo "[ERROR] Multirun job 691 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 692/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [25] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 692/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 692 submitted successfully"
else
    echo "[ERROR] Multirun job 692 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 693/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [26] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 693/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 693 submitted successfully"
else
    echo "[ERROR] Multirun job 693 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 694/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [27] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 694/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 694 submitted successfully"
else
    echo "[ERROR] Multirun job 694 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 695/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [28] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 695/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 695 submitted successfully"
else
    echo "[ERROR] Multirun job 695 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 696/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [29] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 696/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false branched_wiredcfc_arch4 400 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 696 submitted successfully"
else
    echo "[ERROR] Multirun job 696 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 697/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [1] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 697/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 697 submitted successfully"
else
    echo "[ERROR] Multirun job 697 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 698/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [2] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 698/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 698 submitted successfully"
else
    echo "[ERROR] Multirun job 698 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 699/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [3] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 699/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 699 submitted successfully"
else
    echo "[ERROR] Multirun job 699 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 700/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [4] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 700/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 700 submitted successfully"
else
    echo "[ERROR] Multirun job 700 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 701/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [5] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 701/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 701 submitted successfully"
else
    echo "[ERROR] Multirun job 701 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 702/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [6] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 702/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 702 submitted successfully"
else
    echo "[ERROR] Multirun job 702 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 703/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [7] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 703/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 703 submitted successfully"
else
    echo "[ERROR] Multirun job 703 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 704/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [8] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 704/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 704 submitted successfully"
else
    echo "[ERROR] Multirun job 704 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 705/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [9] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 705/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 705 submitted successfully"
else
    echo "[ERROR] Multirun job 705 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 706/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [10] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 706/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 706 submitted successfully"
else
    echo "[ERROR] Multirun job 706 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 707/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [11] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 707/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 707 submitted successfully"
else
    echo "[ERROR] Multirun job 707 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 708/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [12] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 708/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 708 submitted successfully"
else
    echo "[ERROR] Multirun job 708 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 709/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [13] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 709/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 709 submitted successfully"
else
    echo "[ERROR] Multirun job 709 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 710/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [14] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 710/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 710 submitted successfully"
else
    echo "[ERROR] Multirun job 710 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 711/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [15] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 711/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 711 submitted successfully"
else
    echo "[ERROR] Multirun job 711 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 712/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [16] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 712/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 712 submitted successfully"
else
    echo "[ERROR] Multirun job 712 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 713/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [17] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 713/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 713 submitted successfully"
else
    echo "[ERROR] Multirun job 713 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 714/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [18] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 714/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 714 submitted successfully"
else
    echo "[ERROR] Multirun job 714 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 715/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [19] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 715/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 715 submitted successfully"
else
    echo "[ERROR] Multirun job 715 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 716/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [20] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 716/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 716 submitted successfully"
else
    echo "[ERROR] Multirun job 716 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 717/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [21] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 717/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 717 submitted successfully"
else
    echo "[ERROR] Multirun job 717 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 718/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [22] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 718/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 718 submitted successfully"
else
    echo "[ERROR] Multirun job 718 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 719/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [23] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 719/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 719 submitted successfully"
else
    echo "[ERROR] Multirun job 719 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 720/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [24] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 720/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 720 submitted successfully"
else
    echo "[ERROR] Multirun job 720 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 721/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [25] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 721/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 721 submitted successfully"
else
    echo "[ERROR] Multirun job 721 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 722/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [26] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 722/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 722 submitted successfully"
else
    echo "[ERROR] Multirun job 722 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 723/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [27] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 723/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 723 submitted successfully"
else
    echo "[ERROR] Multirun job 723 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 724/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [28] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 724/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 724 submitted successfully"
else
    echo "[ERROR] Multirun job 724 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 725/725
# Dataset: Shin2017A | Model: branched_wiredcfc_arch4 | Eval: WithinSession | Subjects: [29] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 725/725..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Shin2017A WithinSession false branched_wiredcfc_arch4 500 true
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 725 submitted successfully"
else
    echo "[ERROR] Multirun job 725 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

