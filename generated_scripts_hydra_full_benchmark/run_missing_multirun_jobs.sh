#!/bin/bash
# Generated multirun sbatch automation script
# Generated on: 2025-12-30 19:39:29
# Total missing multirun jobs: 1000

set -e  # Exit on any error

echo "Starting multirun experiment automation..."
echo "Total multirun jobs to submit: 1000"

# Multirun Job 1/1000
# Dataset: BNCI2014_001 | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | Seed: 100
# Timeout: --time=1-08:00:00 --mem=16G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 1/1000..."
sbatch --time=1-08:00:00 --mem=16G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 1 submitted successfully"
else
    echo "[ERROR] Multirun job 1 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 2/1000
# Dataset: BNCI2014_001 | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | Seed: 100 | TUNED
# Timeout: --time=1-08:00:00 --mem=16G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 2/1000..."
sbatch --time=1-08:00:00 --mem=16G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 2 submitted successfully"
else
    echo "[ERROR] Multirun job 2 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 3/1000
# Dataset: BNCI2014_001 | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | Seed: 200
# Timeout: --time=1-08:00:00 --mem=16G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 3/1000..."
sbatch --time=1-08:00:00 --mem=16G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 3 submitted successfully"
else
    echo "[ERROR] Multirun job 3 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 4/1000
# Dataset: BNCI2014_001 | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | Seed: 200 | TUNED
# Timeout: --time=1-08:00:00 --mem=16G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 4/1000..."
sbatch --time=1-08:00:00 --mem=16G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 4 submitted successfully"
else
    echo "[ERROR] Multirun job 4 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 5/1000
# Dataset: BNCI2014_001 | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | Seed: 300
# Timeout: --time=1-08:00:00 --mem=16G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 5/1000..."
sbatch --time=1-08:00:00 --mem=16G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 5 submitted successfully"
else
    echo "[ERROR] Multirun job 5 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 6/1000
# Dataset: BNCI2014_001 | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | Seed: 300 | TUNED
# Timeout: --time=1-08:00:00 --mem=16G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 6/1000..."
sbatch --time=1-08:00:00 --mem=16G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 6 submitted successfully"
else
    echo "[ERROR] Multirun job 6 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 7/1000
# Dataset: BNCI2014_001 | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | Seed: 400
# Timeout: --time=1-08:00:00 --mem=16G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 7/1000..."
sbatch --time=1-08:00:00 --mem=16G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 7 submitted successfully"
else
    echo "[ERROR] Multirun job 7 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 8/1000
# Dataset: BNCI2014_001 | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | Seed: 400 | TUNED
# Timeout: --time=1-08:00:00 --mem=16G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 8/1000..."
sbatch --time=1-08:00:00 --mem=16G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 8 submitted successfully"
else
    echo "[ERROR] Multirun job 8 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 9/1000
# Dataset: BNCI2014_001 | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | Seed: 500
# Timeout: --time=1-08:00:00 --mem=16G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 9/1000..."
sbatch --time=1-08:00:00 --mem=16G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 9 submitted successfully"
else
    echo "[ERROR] Multirun job 9 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 10/1000
# Dataset: BNCI2014_001 | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | Seed: 500 | TUNED
# Timeout: --time=1-08:00:00 --mem=16G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 10/1000..."
sbatch --time=1-08:00:00 --mem=16G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 10 submitted successfully"
else
    echo "[ERROR] Multirun job 10 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 11/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [1] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 11/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 11 submitted successfully"
else
    echo "[ERROR] Multirun job 11 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 12/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [2] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 12/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 12 submitted successfully"
else
    echo "[ERROR] Multirun job 12 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 13/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [3] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 13/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 13 submitted successfully"
else
    echo "[ERROR] Multirun job 13 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 14/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [4] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 14/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 14 submitted successfully"
else
    echo "[ERROR] Multirun job 14 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 15/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [5] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 15/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 15 submitted successfully"
else
    echo "[ERROR] Multirun job 15 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 16/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [6] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 16/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 16 submitted successfully"
else
    echo "[ERROR] Multirun job 16 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 17/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [7] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 17/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 17 submitted successfully"
else
    echo "[ERROR] Multirun job 17 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 18/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [8] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 18/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 18 submitted successfully"
else
    echo "[ERROR] Multirun job 18 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 19/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [9] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 19/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 19 submitted successfully"
else
    echo "[ERROR] Multirun job 19 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 20/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [10] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 20/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 20 submitted successfully"
else
    echo "[ERROR] Multirun job 20 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 21/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [11] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 21/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 21 submitted successfully"
else
    echo "[ERROR] Multirun job 21 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 22/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [12] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 22/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 22 submitted successfully"
else
    echo "[ERROR] Multirun job 22 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 23/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [13] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 23/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 23 submitted successfully"
else
    echo "[ERROR] Multirun job 23 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 24/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [14] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 24/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 24 submitted successfully"
else
    echo "[ERROR] Multirun job 24 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 25/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [15] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 25/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 25 submitted successfully"
else
    echo "[ERROR] Multirun job 25 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 26/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [16] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 26/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 26 submitted successfully"
else
    echo "[ERROR] Multirun job 26 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 27/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [17] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 27/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 27 submitted successfully"
else
    echo "[ERROR] Multirun job 27 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 28/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [18] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 28/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 28 submitted successfully"
else
    echo "[ERROR] Multirun job 28 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 29/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [19] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 29/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 29 submitted successfully"
else
    echo "[ERROR] Multirun job 29 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 30/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [20] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 30/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 30 submitted successfully"
else
    echo "[ERROR] Multirun job 30 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 31/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [21] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 31/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 31 submitted successfully"
else
    echo "[ERROR] Multirun job 31 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 32/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [22] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 32/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 32 submitted successfully"
else
    echo "[ERROR] Multirun job 32 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 33/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [23] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 33/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 33 submitted successfully"
else
    echo "[ERROR] Multirun job 33 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 34/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [24] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 34/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 34 submitted successfully"
else
    echo "[ERROR] Multirun job 34 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 35/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [25] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 35/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 35 submitted successfully"
else
    echo "[ERROR] Multirun job 35 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 36/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [26] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 36/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 36 submitted successfully"
else
    echo "[ERROR] Multirun job 36 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 37/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [27] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 37/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 37 submitted successfully"
else
    echo "[ERROR] Multirun job 37 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 38/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [28] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 38/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 38 submitted successfully"
else
    echo "[ERROR] Multirun job 38 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 39/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [29] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 39/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 39 submitted successfully"
else
    echo "[ERROR] Multirun job 39 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 40/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [30] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 40/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 30 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 40 submitted successfully"
else
    echo "[ERROR] Multirun job 40 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 41/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [31] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 41/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 31 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 41 submitted successfully"
else
    echo "[ERROR] Multirun job 41 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 42/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [32] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 42/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 32 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 42 submitted successfully"
else
    echo "[ERROR] Multirun job 42 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 43/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [33] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 43/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 33 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 43 submitted successfully"
else
    echo "[ERROR] Multirun job 43 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 44/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [34] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 44/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 34 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 44 submitted successfully"
else
    echo "[ERROR] Multirun job 44 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 45/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [35] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 45/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 35 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 45 submitted successfully"
else
    echo "[ERROR] Multirun job 45 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 46/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [36] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 46/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 36 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 46 submitted successfully"
else
    echo "[ERROR] Multirun job 46 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 47/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [37] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 47/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 37 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 47 submitted successfully"
else
    echo "[ERROR] Multirun job 47 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 48/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [38] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 48/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 38 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 48 submitted successfully"
else
    echo "[ERROR] Multirun job 48 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 49/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [39] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 49/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 49 submitted successfully"
else
    echo "[ERROR] Multirun job 49 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 50/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [40] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 50/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 40 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 50 submitted successfully"
else
    echo "[ERROR] Multirun job 50 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 51/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [41] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 51/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 51 submitted successfully"
else
    echo "[ERROR] Multirun job 51 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 52/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [42] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 52/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 42 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 52 submitted successfully"
else
    echo "[ERROR] Multirun job 52 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 53/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [43] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 53/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 43 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 53 submitted successfully"
else
    echo "[ERROR] Multirun job 53 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 54/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [44] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 54/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 44 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 54 submitted successfully"
else
    echo "[ERROR] Multirun job 54 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 55/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [45] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 55/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 45 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 55 submitted successfully"
else
    echo "[ERROR] Multirun job 55 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 56/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [46] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 56/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 46 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 56 submitted successfully"
else
    echo "[ERROR] Multirun job 56 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 57/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [47] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 57/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 47 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 57 submitted successfully"
else
    echo "[ERROR] Multirun job 57 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 58/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [48] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 58/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 48 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 58 submitted successfully"
else
    echo "[ERROR] Multirun job 58 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 59/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [49] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 59/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 49 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 59 submitted successfully"
else
    echo "[ERROR] Multirun job 59 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 60/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [50] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 60/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 50 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 60 submitted successfully"
else
    echo "[ERROR] Multirun job 60 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 61/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [51] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 61/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 51 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 61 submitted successfully"
else
    echo "[ERROR] Multirun job 61 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 62/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [52] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 62/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 52 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 62 submitted successfully"
else
    echo "[ERROR] Multirun job 62 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 63/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [53] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 63/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 53 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 63 submitted successfully"
else
    echo "[ERROR] Multirun job 63 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 64/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [54] | Seed: 100
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 64/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 54 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 64 submitted successfully"
else
    echo "[ERROR] Multirun job 64 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 65/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [1] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 65/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 65 submitted successfully"
else
    echo "[ERROR] Multirun job 65 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 66/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [2] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 66/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 66 submitted successfully"
else
    echo "[ERROR] Multirun job 66 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 67/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [3] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 67/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 67 submitted successfully"
else
    echo "[ERROR] Multirun job 67 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 68/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [4] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 68/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 68 submitted successfully"
else
    echo "[ERROR] Multirun job 68 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 69/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [5] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 69/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 69 submitted successfully"
else
    echo "[ERROR] Multirun job 69 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 70/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [6] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 70/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 70 submitted successfully"
else
    echo "[ERROR] Multirun job 70 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 71/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [7] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 71/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 71 submitted successfully"
else
    echo "[ERROR] Multirun job 71 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 72/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [8] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 72/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 72 submitted successfully"
else
    echo "[ERROR] Multirun job 72 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 73/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [9] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 73/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 73 submitted successfully"
else
    echo "[ERROR] Multirun job 73 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 74/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [10] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 74/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 74 submitted successfully"
else
    echo "[ERROR] Multirun job 74 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 75/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [11] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 75/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 75 submitted successfully"
else
    echo "[ERROR] Multirun job 75 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 76/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [12] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 76/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 76 submitted successfully"
else
    echo "[ERROR] Multirun job 76 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 77/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [13] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 77/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 77 submitted successfully"
else
    echo "[ERROR] Multirun job 77 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 78/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [14] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 78/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 78 submitted successfully"
else
    echo "[ERROR] Multirun job 78 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 79/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [15] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 79/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 79 submitted successfully"
else
    echo "[ERROR] Multirun job 79 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 80/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [16] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 80/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 80 submitted successfully"
else
    echo "[ERROR] Multirun job 80 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 81/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [17] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 81/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 81 submitted successfully"
else
    echo "[ERROR] Multirun job 81 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 82/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [18] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 82/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 82 submitted successfully"
else
    echo "[ERROR] Multirun job 82 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 83/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [19] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 83/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 83 submitted successfully"
else
    echo "[ERROR] Multirun job 83 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 84/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [20] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 84/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 84 submitted successfully"
else
    echo "[ERROR] Multirun job 84 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 85/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [21] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 85/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 85 submitted successfully"
else
    echo "[ERROR] Multirun job 85 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 86/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [22] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 86/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 86 submitted successfully"
else
    echo "[ERROR] Multirun job 86 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 87/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [23] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 87/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 87 submitted successfully"
else
    echo "[ERROR] Multirun job 87 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 88/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [24] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 88/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 88 submitted successfully"
else
    echo "[ERROR] Multirun job 88 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 89/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [25] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 89/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 89 submitted successfully"
else
    echo "[ERROR] Multirun job 89 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 90/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [26] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 90/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 90 submitted successfully"
else
    echo "[ERROR] Multirun job 90 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 91/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [27] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 91/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 91 submitted successfully"
else
    echo "[ERROR] Multirun job 91 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 92/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [28] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 92/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 28 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 92 submitted successfully"
else
    echo "[ERROR] Multirun job 92 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 93/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [29] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 93/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 29 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 93 submitted successfully"
else
    echo "[ERROR] Multirun job 93 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 94/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [30] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 94/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 30 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 94 submitted successfully"
else
    echo "[ERROR] Multirun job 94 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 95/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [31] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 95/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 31 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 95 submitted successfully"
else
    echo "[ERROR] Multirun job 95 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 96/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [32] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 96/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 32 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 96 submitted successfully"
else
    echo "[ERROR] Multirun job 96 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 97/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [33] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 97/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 33 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 97 submitted successfully"
else
    echo "[ERROR] Multirun job 97 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 98/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [34] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 98/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 34 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 98 submitted successfully"
else
    echo "[ERROR] Multirun job 98 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 99/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [35] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 99/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 35 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 99 submitted successfully"
else
    echo "[ERROR] Multirun job 99 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 100/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [36] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 100/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 36 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 100 submitted successfully"
else
    echo "[ERROR] Multirun job 100 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 101/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [37] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 101/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 37 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 101 submitted successfully"
else
    echo "[ERROR] Multirun job 101 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 102/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [38] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 102/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 38 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 102 submitted successfully"
else
    echo "[ERROR] Multirun job 102 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 103/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [39] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 103/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 103 submitted successfully"
else
    echo "[ERROR] Multirun job 103 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 104/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [40] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 104/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 40 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 104 submitted successfully"
else
    echo "[ERROR] Multirun job 104 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 105/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [41] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 105/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 105 submitted successfully"
else
    echo "[ERROR] Multirun job 105 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 106/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [42] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 106/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 42 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 106 submitted successfully"
else
    echo "[ERROR] Multirun job 106 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 107/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [43] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 107/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 43 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 107 submitted successfully"
else
    echo "[ERROR] Multirun job 107 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 108/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [44] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 108/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 44 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 108 submitted successfully"
else
    echo "[ERROR] Multirun job 108 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 109/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [45] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 109/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 45 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 109 submitted successfully"
else
    echo "[ERROR] Multirun job 109 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 110/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [46] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 110/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 46 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 110 submitted successfully"
else
    echo "[ERROR] Multirun job 110 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 111/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [47] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 111/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 47 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 111 submitted successfully"
else
    echo "[ERROR] Multirun job 111 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 112/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [48] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 112/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 48 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 112 submitted successfully"
else
    echo "[ERROR] Multirun job 112 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 113/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [49] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 113/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 49 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 113 submitted successfully"
else
    echo "[ERROR] Multirun job 113 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 114/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [50] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 114/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 50 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 114 submitted successfully"
else
    echo "[ERROR] Multirun job 114 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 115/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [51] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 115/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 51 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 115 submitted successfully"
else
    echo "[ERROR] Multirun job 115 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 116/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [52] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 116/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 52 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 116 submitted successfully"
else
    echo "[ERROR] Multirun job 116 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 117/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [53] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 117/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 53 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 117 submitted successfully"
else
    echo "[ERROR] Multirun job 117 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 118/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [54] | Seed: 100 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 118/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 54 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 118 submitted successfully"
else
    echo "[ERROR] Multirun job 118 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 119/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [1] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 119/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 119 submitted successfully"
else
    echo "[ERROR] Multirun job 119 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 120/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [2] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 120/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 120 submitted successfully"
else
    echo "[ERROR] Multirun job 120 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 121/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [3] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 121/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 121 submitted successfully"
else
    echo "[ERROR] Multirun job 121 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 122/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [4] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 122/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 122 submitted successfully"
else
    echo "[ERROR] Multirun job 122 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 123/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [5] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 123/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 123 submitted successfully"
else
    echo "[ERROR] Multirun job 123 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 124/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [6] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 124/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 124 submitted successfully"
else
    echo "[ERROR] Multirun job 124 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 125/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [7] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 125/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 125 submitted successfully"
else
    echo "[ERROR] Multirun job 125 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 126/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [8] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 126/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 126 submitted successfully"
else
    echo "[ERROR] Multirun job 126 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 127/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [9] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 127/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 127 submitted successfully"
else
    echo "[ERROR] Multirun job 127 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 128/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [10] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 128/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 128 submitted successfully"
else
    echo "[ERROR] Multirun job 128 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 129/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [11] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 129/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 129 submitted successfully"
else
    echo "[ERROR] Multirun job 129 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 130/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [12] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 130/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 130 submitted successfully"
else
    echo "[ERROR] Multirun job 130 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 131/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [13] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 131/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 131 submitted successfully"
else
    echo "[ERROR] Multirun job 131 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 132/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [14] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 132/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 132 submitted successfully"
else
    echo "[ERROR] Multirun job 132 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 133/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [15] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 133/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 133 submitted successfully"
else
    echo "[ERROR] Multirun job 133 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 134/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [16] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 134/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 134 submitted successfully"
else
    echo "[ERROR] Multirun job 134 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 135/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [17] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 135/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 135 submitted successfully"
else
    echo "[ERROR] Multirun job 135 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 136/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [18] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 136/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 136 submitted successfully"
else
    echo "[ERROR] Multirun job 136 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 137/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [19] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 137/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 137 submitted successfully"
else
    echo "[ERROR] Multirun job 137 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 138/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [20] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 138/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 138 submitted successfully"
else
    echo "[ERROR] Multirun job 138 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 139/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [21] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 139/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 139 submitted successfully"
else
    echo "[ERROR] Multirun job 139 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 140/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [22] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 140/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 140 submitted successfully"
else
    echo "[ERROR] Multirun job 140 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 141/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [23] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 141/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 141 submitted successfully"
else
    echo "[ERROR] Multirun job 141 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 142/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [24] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 142/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 142 submitted successfully"
else
    echo "[ERROR] Multirun job 142 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 143/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [25] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 143/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 143 submitted successfully"
else
    echo "[ERROR] Multirun job 143 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 144/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [26] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 144/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 144 submitted successfully"
else
    echo "[ERROR] Multirun job 144 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 145/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [27] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 145/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 145 submitted successfully"
else
    echo "[ERROR] Multirun job 145 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 146/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [28] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 146/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 146 submitted successfully"
else
    echo "[ERROR] Multirun job 146 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 147/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [29] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 147/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 147 submitted successfully"
else
    echo "[ERROR] Multirun job 147 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 148/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [30] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 148/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 30 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 148 submitted successfully"
else
    echo "[ERROR] Multirun job 148 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 149/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [31] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 149/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 31 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 149 submitted successfully"
else
    echo "[ERROR] Multirun job 149 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 150/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [32] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 150/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 32 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 150 submitted successfully"
else
    echo "[ERROR] Multirun job 150 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 151/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [33] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 151/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 33 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 151 submitted successfully"
else
    echo "[ERROR] Multirun job 151 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 152/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [34] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 152/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 34 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 152 submitted successfully"
else
    echo "[ERROR] Multirun job 152 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 153/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [35] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 153/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 35 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 153 submitted successfully"
else
    echo "[ERROR] Multirun job 153 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 154/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [36] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 154/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 36 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 154 submitted successfully"
else
    echo "[ERROR] Multirun job 154 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 155/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [37] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 155/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 37 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 155 submitted successfully"
else
    echo "[ERROR] Multirun job 155 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 156/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [38] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 156/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 38 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 156 submitted successfully"
else
    echo "[ERROR] Multirun job 156 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 157/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [39] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 157/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 157 submitted successfully"
else
    echo "[ERROR] Multirun job 157 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 158/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [40] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 158/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 40 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 158 submitted successfully"
else
    echo "[ERROR] Multirun job 158 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 159/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [41] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 159/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 159 submitted successfully"
else
    echo "[ERROR] Multirun job 159 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 160/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [42] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 160/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 42 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 160 submitted successfully"
else
    echo "[ERROR] Multirun job 160 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 161/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [43] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 161/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 43 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 161 submitted successfully"
else
    echo "[ERROR] Multirun job 161 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 162/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [44] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 162/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 44 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 162 submitted successfully"
else
    echo "[ERROR] Multirun job 162 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 163/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [45] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 163/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 45 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 163 submitted successfully"
else
    echo "[ERROR] Multirun job 163 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 164/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [46] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 164/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 46 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 164 submitted successfully"
else
    echo "[ERROR] Multirun job 164 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 165/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [47] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 165/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 47 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 165 submitted successfully"
else
    echo "[ERROR] Multirun job 165 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 166/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [48] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 166/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 48 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 166 submitted successfully"
else
    echo "[ERROR] Multirun job 166 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 167/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [49] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 167/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 49 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 167 submitted successfully"
else
    echo "[ERROR] Multirun job 167 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 168/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [50] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 168/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 50 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 168 submitted successfully"
else
    echo "[ERROR] Multirun job 168 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 169/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [51] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 169/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 51 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 169 submitted successfully"
else
    echo "[ERROR] Multirun job 169 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 170/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [52] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 170/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 52 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 170 submitted successfully"
else
    echo "[ERROR] Multirun job 170 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 171/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [53] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 171/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 53 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 171 submitted successfully"
else
    echo "[ERROR] Multirun job 171 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 172/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [54] | Seed: 200
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 172/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 54 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 172 submitted successfully"
else
    echo "[ERROR] Multirun job 172 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 173/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [1] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 173/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 173 submitted successfully"
else
    echo "[ERROR] Multirun job 173 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 174/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [2] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 174/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 174 submitted successfully"
else
    echo "[ERROR] Multirun job 174 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 175/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [3] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 175/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 175 submitted successfully"
else
    echo "[ERROR] Multirun job 175 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 176/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [4] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 176/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 176 submitted successfully"
else
    echo "[ERROR] Multirun job 176 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 177/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [5] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 177/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 177 submitted successfully"
else
    echo "[ERROR] Multirun job 177 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 178/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [6] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 178/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 178 submitted successfully"
else
    echo "[ERROR] Multirun job 178 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 179/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [7] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 179/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 179 submitted successfully"
else
    echo "[ERROR] Multirun job 179 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 180/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [8] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 180/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 180 submitted successfully"
else
    echo "[ERROR] Multirun job 180 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 181/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [9] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 181/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 181 submitted successfully"
else
    echo "[ERROR] Multirun job 181 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 182/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [10] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 182/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 182 submitted successfully"
else
    echo "[ERROR] Multirun job 182 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 183/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [11] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 183/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 183 submitted successfully"
else
    echo "[ERROR] Multirun job 183 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 184/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [12] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 184/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 184 submitted successfully"
else
    echo "[ERROR] Multirun job 184 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 185/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [13] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 185/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 185 submitted successfully"
else
    echo "[ERROR] Multirun job 185 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 186/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [14] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 186/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 186 submitted successfully"
else
    echo "[ERROR] Multirun job 186 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 187/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [15] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 187/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 187 submitted successfully"
else
    echo "[ERROR] Multirun job 187 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 188/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [16] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 188/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 188 submitted successfully"
else
    echo "[ERROR] Multirun job 188 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 189/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [17] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 189/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 189 submitted successfully"
else
    echo "[ERROR] Multirun job 189 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 190/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [18] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 190/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 190 submitted successfully"
else
    echo "[ERROR] Multirun job 190 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 191/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [19] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 191/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 191 submitted successfully"
else
    echo "[ERROR] Multirun job 191 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 192/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [20] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 192/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 192 submitted successfully"
else
    echo "[ERROR] Multirun job 192 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 193/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [21] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 193/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 193 submitted successfully"
else
    echo "[ERROR] Multirun job 193 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 194/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [22] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 194/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 194 submitted successfully"
else
    echo "[ERROR] Multirun job 194 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 195/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [23] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 195/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 195 submitted successfully"
else
    echo "[ERROR] Multirun job 195 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 196/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [24] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 196/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 196 submitted successfully"
else
    echo "[ERROR] Multirun job 196 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 197/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [25] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 197/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 197 submitted successfully"
else
    echo "[ERROR] Multirun job 197 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 198/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [26] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 198/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 198 submitted successfully"
else
    echo "[ERROR] Multirun job 198 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 199/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [27] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 199/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 199 submitted successfully"
else
    echo "[ERROR] Multirun job 199 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 200/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [28] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 200/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 28 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 200 submitted successfully"
else
    echo "[ERROR] Multirun job 200 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 201/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [29] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 201/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 29 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 201 submitted successfully"
else
    echo "[ERROR] Multirun job 201 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 202/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [30] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 202/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 30 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 202 submitted successfully"
else
    echo "[ERROR] Multirun job 202 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 203/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [31] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 203/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 31 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 203 submitted successfully"
else
    echo "[ERROR] Multirun job 203 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 204/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [32] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 204/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 32 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 204 submitted successfully"
else
    echo "[ERROR] Multirun job 204 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 205/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [33] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 205/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 33 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 205 submitted successfully"
else
    echo "[ERROR] Multirun job 205 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 206/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [34] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 206/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 34 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 206 submitted successfully"
else
    echo "[ERROR] Multirun job 206 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 207/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [35] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 207/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 35 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 207 submitted successfully"
else
    echo "[ERROR] Multirun job 207 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 208/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [36] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 208/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 36 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 208 submitted successfully"
else
    echo "[ERROR] Multirun job 208 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 209/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [37] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 209/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 37 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 209 submitted successfully"
else
    echo "[ERROR] Multirun job 209 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 210/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [38] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 210/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 38 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 210 submitted successfully"
else
    echo "[ERROR] Multirun job 210 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 211/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [39] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 211/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 211 submitted successfully"
else
    echo "[ERROR] Multirun job 211 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 212/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [40] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 212/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 40 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 212 submitted successfully"
else
    echo "[ERROR] Multirun job 212 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 213/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [41] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 213/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 213 submitted successfully"
else
    echo "[ERROR] Multirun job 213 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 214/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [42] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 214/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 42 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 214 submitted successfully"
else
    echo "[ERROR] Multirun job 214 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 215/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [43] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 215/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 43 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 215 submitted successfully"
else
    echo "[ERROR] Multirun job 215 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 216/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [44] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 216/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 44 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 216 submitted successfully"
else
    echo "[ERROR] Multirun job 216 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 217/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [45] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 217/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 45 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 217 submitted successfully"
else
    echo "[ERROR] Multirun job 217 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 218/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [46] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 218/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 46 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 218 submitted successfully"
else
    echo "[ERROR] Multirun job 218 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 219/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [47] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 219/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 47 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 219 submitted successfully"
else
    echo "[ERROR] Multirun job 219 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 220/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [48] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 220/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 48 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 220 submitted successfully"
else
    echo "[ERROR] Multirun job 220 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 221/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [49] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 221/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 49 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 221 submitted successfully"
else
    echo "[ERROR] Multirun job 221 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 222/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [50] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 222/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 50 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 222 submitted successfully"
else
    echo "[ERROR] Multirun job 222 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 223/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [51] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 223/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 51 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 223 submitted successfully"
else
    echo "[ERROR] Multirun job 223 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 224/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [52] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 224/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 52 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 224 submitted successfully"
else
    echo "[ERROR] Multirun job 224 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 225/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [53] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 225/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 53 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 225 submitted successfully"
else
    echo "[ERROR] Multirun job 225 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 226/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [54] | Seed: 200 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 226/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 54 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 226 submitted successfully"
else
    echo "[ERROR] Multirun job 226 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 227/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [1] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 227/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 227 submitted successfully"
else
    echo "[ERROR] Multirun job 227 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 228/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [2] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 228/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 228 submitted successfully"
else
    echo "[ERROR] Multirun job 228 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 229/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [3] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 229/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 229 submitted successfully"
else
    echo "[ERROR] Multirun job 229 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 230/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [4] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 230/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 230 submitted successfully"
else
    echo "[ERROR] Multirun job 230 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 231/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [5] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 231/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 231 submitted successfully"
else
    echo "[ERROR] Multirun job 231 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 232/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [6] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 232/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 232 submitted successfully"
else
    echo "[ERROR] Multirun job 232 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 233/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [7] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 233/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 233 submitted successfully"
else
    echo "[ERROR] Multirun job 233 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 234/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [8] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 234/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 234 submitted successfully"
else
    echo "[ERROR] Multirun job 234 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 235/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [9] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 235/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 235 submitted successfully"
else
    echo "[ERROR] Multirun job 235 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 236/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [10] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 236/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 236 submitted successfully"
else
    echo "[ERROR] Multirun job 236 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 237/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [11] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 237/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 237 submitted successfully"
else
    echo "[ERROR] Multirun job 237 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 238/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [12] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 238/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 238 submitted successfully"
else
    echo "[ERROR] Multirun job 238 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 239/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [13] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 239/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 239 submitted successfully"
else
    echo "[ERROR] Multirun job 239 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 240/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [14] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 240/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 240 submitted successfully"
else
    echo "[ERROR] Multirun job 240 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 241/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [15] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 241/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 241 submitted successfully"
else
    echo "[ERROR] Multirun job 241 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 242/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [16] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 242/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 242 submitted successfully"
else
    echo "[ERROR] Multirun job 242 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 243/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [17] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 243/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 243 submitted successfully"
else
    echo "[ERROR] Multirun job 243 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 244/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [18] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 244/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 244 submitted successfully"
else
    echo "[ERROR] Multirun job 244 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 245/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [19] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 245/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 245 submitted successfully"
else
    echo "[ERROR] Multirun job 245 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 246/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [20] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 246/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 246 submitted successfully"
else
    echo "[ERROR] Multirun job 246 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 247/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [21] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 247/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 247 submitted successfully"
else
    echo "[ERROR] Multirun job 247 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 248/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [22] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 248/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 248 submitted successfully"
else
    echo "[ERROR] Multirun job 248 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 249/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [23] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 249/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 249 submitted successfully"
else
    echo "[ERROR] Multirun job 249 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 250/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [24] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 250/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 250 submitted successfully"
else
    echo "[ERROR] Multirun job 250 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 251/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [25] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 251/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 251 submitted successfully"
else
    echo "[ERROR] Multirun job 251 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 252/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [26] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 252/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 252 submitted successfully"
else
    echo "[ERROR] Multirun job 252 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 253/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [27] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 253/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 253 submitted successfully"
else
    echo "[ERROR] Multirun job 253 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 254/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [28] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 254/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 254 submitted successfully"
else
    echo "[ERROR] Multirun job 254 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 255/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [29] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 255/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 255 submitted successfully"
else
    echo "[ERROR] Multirun job 255 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 256/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [30] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 256/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 30 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 256 submitted successfully"
else
    echo "[ERROR] Multirun job 256 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 257/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [31] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 257/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 31 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 257 submitted successfully"
else
    echo "[ERROR] Multirun job 257 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 258/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [32] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 258/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 32 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 258 submitted successfully"
else
    echo "[ERROR] Multirun job 258 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 259/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [33] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 259/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 33 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 259 submitted successfully"
else
    echo "[ERROR] Multirun job 259 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 260/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [34] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 260/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 34 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 260 submitted successfully"
else
    echo "[ERROR] Multirun job 260 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 261/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [35] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 261/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 35 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 261 submitted successfully"
else
    echo "[ERROR] Multirun job 261 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 262/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [36] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 262/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 36 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 262 submitted successfully"
else
    echo "[ERROR] Multirun job 262 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 263/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [37] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 263/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 37 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 263 submitted successfully"
else
    echo "[ERROR] Multirun job 263 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 264/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [38] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 264/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 38 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 264 submitted successfully"
else
    echo "[ERROR] Multirun job 264 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 265/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [39] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 265/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 265 submitted successfully"
else
    echo "[ERROR] Multirun job 265 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 266/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [40] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 266/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 40 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 266 submitted successfully"
else
    echo "[ERROR] Multirun job 266 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 267/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [41] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 267/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 267 submitted successfully"
else
    echo "[ERROR] Multirun job 267 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 268/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [42] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 268/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 42 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 268 submitted successfully"
else
    echo "[ERROR] Multirun job 268 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 269/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [43] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 269/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 43 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 269 submitted successfully"
else
    echo "[ERROR] Multirun job 269 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 270/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [44] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 270/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 44 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 270 submitted successfully"
else
    echo "[ERROR] Multirun job 270 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 271/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [45] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 271/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 45 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 271 submitted successfully"
else
    echo "[ERROR] Multirun job 271 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 272/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [46] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 272/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 46 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 272 submitted successfully"
else
    echo "[ERROR] Multirun job 272 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 273/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [47] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 273/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 47 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 273 submitted successfully"
else
    echo "[ERROR] Multirun job 273 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 274/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [48] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 274/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 48 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 274 submitted successfully"
else
    echo "[ERROR] Multirun job 274 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 275/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [49] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 275/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 49 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 275 submitted successfully"
else
    echo "[ERROR] Multirun job 275 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 276/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [50] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 276/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 50 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 276 submitted successfully"
else
    echo "[ERROR] Multirun job 276 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 277/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [51] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 277/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 51 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 277 submitted successfully"
else
    echo "[ERROR] Multirun job 277 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 278/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [52] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 278/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 52 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 278 submitted successfully"
else
    echo "[ERROR] Multirun job 278 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 279/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [53] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 279/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 53 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 279 submitted successfully"
else
    echo "[ERROR] Multirun job 279 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 280/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [54] | Seed: 300
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 280/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 54 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 280 submitted successfully"
else
    echo "[ERROR] Multirun job 280 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 281/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [1] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 281/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 281 submitted successfully"
else
    echo "[ERROR] Multirun job 281 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 282/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [2] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 282/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 282 submitted successfully"
else
    echo "[ERROR] Multirun job 282 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 283/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [3] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 283/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 283 submitted successfully"
else
    echo "[ERROR] Multirun job 283 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 284/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [4] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 284/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 284 submitted successfully"
else
    echo "[ERROR] Multirun job 284 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 285/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [5] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 285/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 285 submitted successfully"
else
    echo "[ERROR] Multirun job 285 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 286/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [6] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 286/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 286 submitted successfully"
else
    echo "[ERROR] Multirun job 286 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 287/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [7] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 287/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 287 submitted successfully"
else
    echo "[ERROR] Multirun job 287 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 288/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [8] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 288/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 288 submitted successfully"
else
    echo "[ERROR] Multirun job 288 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 289/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [9] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 289/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 289 submitted successfully"
else
    echo "[ERROR] Multirun job 289 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 290/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [10] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 290/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 290 submitted successfully"
else
    echo "[ERROR] Multirun job 290 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 291/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [11] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 291/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 291 submitted successfully"
else
    echo "[ERROR] Multirun job 291 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 292/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [12] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 292/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 292 submitted successfully"
else
    echo "[ERROR] Multirun job 292 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 293/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [13] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 293/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 293 submitted successfully"
else
    echo "[ERROR] Multirun job 293 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 294/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [14] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 294/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 294 submitted successfully"
else
    echo "[ERROR] Multirun job 294 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 295/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [15] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 295/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 295 submitted successfully"
else
    echo "[ERROR] Multirun job 295 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 296/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [16] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 296/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 296 submitted successfully"
else
    echo "[ERROR] Multirun job 296 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 297/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [17] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 297/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 297 submitted successfully"
else
    echo "[ERROR] Multirun job 297 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 298/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [18] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 298/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 298 submitted successfully"
else
    echo "[ERROR] Multirun job 298 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 299/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [19] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 299/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 299 submitted successfully"
else
    echo "[ERROR] Multirun job 299 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 300/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [20] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 300/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 300 submitted successfully"
else
    echo "[ERROR] Multirun job 300 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 301/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [21] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 301/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 301 submitted successfully"
else
    echo "[ERROR] Multirun job 301 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 302/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [22] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 302/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 302 submitted successfully"
else
    echo "[ERROR] Multirun job 302 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 303/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [23] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 303/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 303 submitted successfully"
else
    echo "[ERROR] Multirun job 303 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 304/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [24] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 304/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 304 submitted successfully"
else
    echo "[ERROR] Multirun job 304 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 305/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [25] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 305/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 305 submitted successfully"
else
    echo "[ERROR] Multirun job 305 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 306/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [26] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 306/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 306 submitted successfully"
else
    echo "[ERROR] Multirun job 306 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 307/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [27] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 307/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 307 submitted successfully"
else
    echo "[ERROR] Multirun job 307 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 308/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [28] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 308/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 28 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 308 submitted successfully"
else
    echo "[ERROR] Multirun job 308 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 309/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [29] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 309/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 29 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 309 submitted successfully"
else
    echo "[ERROR] Multirun job 309 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 310/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [30] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 310/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 30 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 310 submitted successfully"
else
    echo "[ERROR] Multirun job 310 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 311/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [31] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 311/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 31 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 311 submitted successfully"
else
    echo "[ERROR] Multirun job 311 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 312/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [32] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 312/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 32 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 312 submitted successfully"
else
    echo "[ERROR] Multirun job 312 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 313/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [33] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 313/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 33 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 313 submitted successfully"
else
    echo "[ERROR] Multirun job 313 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 314/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [34] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 314/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 34 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 314 submitted successfully"
else
    echo "[ERROR] Multirun job 314 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 315/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [35] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 315/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 35 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 315 submitted successfully"
else
    echo "[ERROR] Multirun job 315 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 316/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [36] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 316/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 36 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 316 submitted successfully"
else
    echo "[ERROR] Multirun job 316 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 317/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [37] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 317/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 37 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 317 submitted successfully"
else
    echo "[ERROR] Multirun job 317 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 318/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [38] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 318/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 38 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 318 submitted successfully"
else
    echo "[ERROR] Multirun job 318 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 319/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [39] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 319/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 319 submitted successfully"
else
    echo "[ERROR] Multirun job 319 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 320/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [40] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 320/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 40 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 320 submitted successfully"
else
    echo "[ERROR] Multirun job 320 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 321/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [41] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 321/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 321 submitted successfully"
else
    echo "[ERROR] Multirun job 321 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 322/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [42] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 322/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 42 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 322 submitted successfully"
else
    echo "[ERROR] Multirun job 322 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 323/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [43] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 323/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 43 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 323 submitted successfully"
else
    echo "[ERROR] Multirun job 323 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 324/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [44] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 324/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 44 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 324 submitted successfully"
else
    echo "[ERROR] Multirun job 324 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 325/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [45] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 325/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 45 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 325 submitted successfully"
else
    echo "[ERROR] Multirun job 325 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 326/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [46] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 326/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 46 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 326 submitted successfully"
else
    echo "[ERROR] Multirun job 326 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 327/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [47] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 327/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 47 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 327 submitted successfully"
else
    echo "[ERROR] Multirun job 327 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 328/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [48] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 328/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 48 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 328 submitted successfully"
else
    echo "[ERROR] Multirun job 328 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 329/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [49] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 329/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 49 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 329 submitted successfully"
else
    echo "[ERROR] Multirun job 329 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 330/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [50] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 330/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 50 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 330 submitted successfully"
else
    echo "[ERROR] Multirun job 330 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 331/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [51] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 331/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 51 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 331 submitted successfully"
else
    echo "[ERROR] Multirun job 331 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 332/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [52] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 332/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 52 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 332 submitted successfully"
else
    echo "[ERROR] Multirun job 332 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 333/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [53] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 333/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 53 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 333 submitted successfully"
else
    echo "[ERROR] Multirun job 333 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 334/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [54] | Seed: 300 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 334/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 54 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 334 submitted successfully"
else
    echo "[ERROR] Multirun job 334 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 335/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [1] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 335/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 335 submitted successfully"
else
    echo "[ERROR] Multirun job 335 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 336/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [2] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 336/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 336 submitted successfully"
else
    echo "[ERROR] Multirun job 336 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 337/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [3] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 337/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 337 submitted successfully"
else
    echo "[ERROR] Multirun job 337 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 338/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [4] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 338/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 338 submitted successfully"
else
    echo "[ERROR] Multirun job 338 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 339/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [5] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 339/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 339 submitted successfully"
else
    echo "[ERROR] Multirun job 339 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 340/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [6] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 340/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 340 submitted successfully"
else
    echo "[ERROR] Multirun job 340 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 341/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [7] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 341/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 341 submitted successfully"
else
    echo "[ERROR] Multirun job 341 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 342/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [8] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 342/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 342 submitted successfully"
else
    echo "[ERROR] Multirun job 342 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 343/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [9] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 343/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 343 submitted successfully"
else
    echo "[ERROR] Multirun job 343 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 344/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [10] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 344/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 344 submitted successfully"
else
    echo "[ERROR] Multirun job 344 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 345/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [11] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 345/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 345 submitted successfully"
else
    echo "[ERROR] Multirun job 345 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 346/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [12] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 346/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 346 submitted successfully"
else
    echo "[ERROR] Multirun job 346 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 347/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [13] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 347/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 347 submitted successfully"
else
    echo "[ERROR] Multirun job 347 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 348/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [14] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 348/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 348 submitted successfully"
else
    echo "[ERROR] Multirun job 348 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 349/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [15] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 349/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 349 submitted successfully"
else
    echo "[ERROR] Multirun job 349 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 350/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [16] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 350/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 350 submitted successfully"
else
    echo "[ERROR] Multirun job 350 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 351/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [17] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 351/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 351 submitted successfully"
else
    echo "[ERROR] Multirun job 351 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 352/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [18] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 352/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 352 submitted successfully"
else
    echo "[ERROR] Multirun job 352 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 353/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [19] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 353/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 353 submitted successfully"
else
    echo "[ERROR] Multirun job 353 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 354/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [20] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 354/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 354 submitted successfully"
else
    echo "[ERROR] Multirun job 354 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 355/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [21] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 355/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 355 submitted successfully"
else
    echo "[ERROR] Multirun job 355 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 356/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [22] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 356/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 356 submitted successfully"
else
    echo "[ERROR] Multirun job 356 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 357/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [23] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 357/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 357 submitted successfully"
else
    echo "[ERROR] Multirun job 357 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 358/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [24] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 358/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 358 submitted successfully"
else
    echo "[ERROR] Multirun job 358 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 359/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [25] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 359/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 359 submitted successfully"
else
    echo "[ERROR] Multirun job 359 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 360/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [26] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 360/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 360 submitted successfully"
else
    echo "[ERROR] Multirun job 360 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 361/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [27] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 361/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 361 submitted successfully"
else
    echo "[ERROR] Multirun job 361 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 362/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [28] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 362/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 362 submitted successfully"
else
    echo "[ERROR] Multirun job 362 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 363/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [29] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 363/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 363 submitted successfully"
else
    echo "[ERROR] Multirun job 363 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 364/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [30] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 364/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 30 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 364 submitted successfully"
else
    echo "[ERROR] Multirun job 364 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 365/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [31] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 365/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 31 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 365 submitted successfully"
else
    echo "[ERROR] Multirun job 365 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 366/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [32] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 366/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 32 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 366 submitted successfully"
else
    echo "[ERROR] Multirun job 366 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 367/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [33] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 367/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 33 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 367 submitted successfully"
else
    echo "[ERROR] Multirun job 367 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 368/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [34] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 368/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 34 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 368 submitted successfully"
else
    echo "[ERROR] Multirun job 368 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 369/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [35] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 369/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 35 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 369 submitted successfully"
else
    echo "[ERROR] Multirun job 369 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 370/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [36] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 370/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 36 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 370 submitted successfully"
else
    echo "[ERROR] Multirun job 370 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 371/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [37] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 371/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 37 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 371 submitted successfully"
else
    echo "[ERROR] Multirun job 371 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 372/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [38] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 372/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 38 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 372 submitted successfully"
else
    echo "[ERROR] Multirun job 372 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 373/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [39] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 373/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 373 submitted successfully"
else
    echo "[ERROR] Multirun job 373 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 374/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [40] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 374/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 40 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 374 submitted successfully"
else
    echo "[ERROR] Multirun job 374 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 375/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [41] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 375/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 375 submitted successfully"
else
    echo "[ERROR] Multirun job 375 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 376/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [42] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 376/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 42 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 376 submitted successfully"
else
    echo "[ERROR] Multirun job 376 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 377/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [43] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 377/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 43 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 377 submitted successfully"
else
    echo "[ERROR] Multirun job 377 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 378/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [44] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 378/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 44 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 378 submitted successfully"
else
    echo "[ERROR] Multirun job 378 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 379/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [45] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 379/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 45 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 379 submitted successfully"
else
    echo "[ERROR] Multirun job 379 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 380/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [46] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 380/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 46 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 380 submitted successfully"
else
    echo "[ERROR] Multirun job 380 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 381/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [47] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 381/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 47 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 381 submitted successfully"
else
    echo "[ERROR] Multirun job 381 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 382/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [48] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 382/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 48 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 382 submitted successfully"
else
    echo "[ERROR] Multirun job 382 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 383/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [49] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 383/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 49 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 383 submitted successfully"
else
    echo "[ERROR] Multirun job 383 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 384/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [50] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 384/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 50 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 384 submitted successfully"
else
    echo "[ERROR] Multirun job 384 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 385/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [51] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 385/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 51 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 385 submitted successfully"
else
    echo "[ERROR] Multirun job 385 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 386/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [52] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 386/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 52 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 386 submitted successfully"
else
    echo "[ERROR] Multirun job 386 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 387/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [53] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 387/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 53 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 387 submitted successfully"
else
    echo "[ERROR] Multirun job 387 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 388/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [54] | Seed: 400
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 388/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 54 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 388 submitted successfully"
else
    echo "[ERROR] Multirun job 388 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 389/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [1] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 389/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 389 submitted successfully"
else
    echo "[ERROR] Multirun job 389 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 390/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [2] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 390/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 390 submitted successfully"
else
    echo "[ERROR] Multirun job 390 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 391/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [3] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 391/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 391 submitted successfully"
else
    echo "[ERROR] Multirun job 391 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 392/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [4] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 392/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 392 submitted successfully"
else
    echo "[ERROR] Multirun job 392 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 393/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [5] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 393/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 393 submitted successfully"
else
    echo "[ERROR] Multirun job 393 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 394/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [6] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 394/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 394 submitted successfully"
else
    echo "[ERROR] Multirun job 394 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 395/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [7] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 395/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 395 submitted successfully"
else
    echo "[ERROR] Multirun job 395 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 396/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [8] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 396/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 396 submitted successfully"
else
    echo "[ERROR] Multirun job 396 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 397/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [9] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 397/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 397 submitted successfully"
else
    echo "[ERROR] Multirun job 397 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 398/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [10] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 398/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 398 submitted successfully"
else
    echo "[ERROR] Multirun job 398 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 399/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [11] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 399/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 399 submitted successfully"
else
    echo "[ERROR] Multirun job 399 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 400/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [12] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 400/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 400 submitted successfully"
else
    echo "[ERROR] Multirun job 400 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 401/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [13] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 401/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 401 submitted successfully"
else
    echo "[ERROR] Multirun job 401 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 402/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [14] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 402/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 402 submitted successfully"
else
    echo "[ERROR] Multirun job 402 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 403/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [15] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 403/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 403 submitted successfully"
else
    echo "[ERROR] Multirun job 403 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 404/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [16] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 404/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 404 submitted successfully"
else
    echo "[ERROR] Multirun job 404 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 405/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [17] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 405/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 405 submitted successfully"
else
    echo "[ERROR] Multirun job 405 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 406/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [18] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 406/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 406 submitted successfully"
else
    echo "[ERROR] Multirun job 406 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 407/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [19] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 407/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 407 submitted successfully"
else
    echo "[ERROR] Multirun job 407 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 408/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [20] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 408/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 408 submitted successfully"
else
    echo "[ERROR] Multirun job 408 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 409/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [21] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 409/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 409 submitted successfully"
else
    echo "[ERROR] Multirun job 409 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 410/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [22] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 410/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 410 submitted successfully"
else
    echo "[ERROR] Multirun job 410 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 411/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [23] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 411/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 411 submitted successfully"
else
    echo "[ERROR] Multirun job 411 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 412/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [24] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 412/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 412 submitted successfully"
else
    echo "[ERROR] Multirun job 412 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 413/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [25] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 413/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 413 submitted successfully"
else
    echo "[ERROR] Multirun job 413 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 414/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [26] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 414/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 414 submitted successfully"
else
    echo "[ERROR] Multirun job 414 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 415/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [27] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 415/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 415 submitted successfully"
else
    echo "[ERROR] Multirun job 415 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 416/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [28] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 416/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 28 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 416 submitted successfully"
else
    echo "[ERROR] Multirun job 416 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 417/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [29] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 417/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 29 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 417 submitted successfully"
else
    echo "[ERROR] Multirun job 417 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 418/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [30] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 418/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 30 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 418 submitted successfully"
else
    echo "[ERROR] Multirun job 418 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 419/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [31] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 419/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 31 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 419 submitted successfully"
else
    echo "[ERROR] Multirun job 419 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 420/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [32] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 420/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 32 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 420 submitted successfully"
else
    echo "[ERROR] Multirun job 420 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 421/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [33] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 421/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 33 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 421 submitted successfully"
else
    echo "[ERROR] Multirun job 421 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 422/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [34] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 422/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 34 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 422 submitted successfully"
else
    echo "[ERROR] Multirun job 422 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 423/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [35] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 423/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 35 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 423 submitted successfully"
else
    echo "[ERROR] Multirun job 423 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 424/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [36] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 424/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 36 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 424 submitted successfully"
else
    echo "[ERROR] Multirun job 424 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 425/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [37] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 425/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 37 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 425 submitted successfully"
else
    echo "[ERROR] Multirun job 425 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 426/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [38] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 426/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 38 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 426 submitted successfully"
else
    echo "[ERROR] Multirun job 426 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 427/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [39] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 427/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 427 submitted successfully"
else
    echo "[ERROR] Multirun job 427 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 428/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [40] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 428/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 40 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 428 submitted successfully"
else
    echo "[ERROR] Multirun job 428 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 429/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [41] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 429/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 429 submitted successfully"
else
    echo "[ERROR] Multirun job 429 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 430/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [42] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 430/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 42 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 430 submitted successfully"
else
    echo "[ERROR] Multirun job 430 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 431/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [43] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 431/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 43 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 431 submitted successfully"
else
    echo "[ERROR] Multirun job 431 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 432/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [44] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 432/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 44 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 432 submitted successfully"
else
    echo "[ERROR] Multirun job 432 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 433/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [45] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 433/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 45 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 433 submitted successfully"
else
    echo "[ERROR] Multirun job 433 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 434/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [46] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 434/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 46 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 434 submitted successfully"
else
    echo "[ERROR] Multirun job 434 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 435/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [47] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 435/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 47 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 435 submitted successfully"
else
    echo "[ERROR] Multirun job 435 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 436/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [48] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 436/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 48 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 436 submitted successfully"
else
    echo "[ERROR] Multirun job 436 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 437/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [49] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 437/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 49 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 437 submitted successfully"
else
    echo "[ERROR] Multirun job 437 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 438/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [50] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 438/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 50 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 438 submitted successfully"
else
    echo "[ERROR] Multirun job 438 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 439/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [51] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 439/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 51 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 439 submitted successfully"
else
    echo "[ERROR] Multirun job 439 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 440/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [52] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 440/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 52 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 440 submitted successfully"
else
    echo "[ERROR] Multirun job 440 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 441/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [53] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 441/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 53 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 441 submitted successfully"
else
    echo "[ERROR] Multirun job 441 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 442/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [54] | Seed: 400 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 442/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 54 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 442 submitted successfully"
else
    echo "[ERROR] Multirun job 442 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 443/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [1] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 443/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 443 submitted successfully"
else
    echo "[ERROR] Multirun job 443 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 444/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [2] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 444/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 444 submitted successfully"
else
    echo "[ERROR] Multirun job 444 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 445/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [3] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 445/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 445 submitted successfully"
else
    echo "[ERROR] Multirun job 445 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 446/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [4] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 446/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 446 submitted successfully"
else
    echo "[ERROR] Multirun job 446 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 447/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [5] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 447/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 447 submitted successfully"
else
    echo "[ERROR] Multirun job 447 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 448/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [6] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 448/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 448 submitted successfully"
else
    echo "[ERROR] Multirun job 448 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 449/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [7] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 449/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 449 submitted successfully"
else
    echo "[ERROR] Multirun job 449 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 450/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [8] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 450/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 450 submitted successfully"
else
    echo "[ERROR] Multirun job 450 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 451/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [9] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 451/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 451 submitted successfully"
else
    echo "[ERROR] Multirun job 451 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 452/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [10] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 452/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 452 submitted successfully"
else
    echo "[ERROR] Multirun job 452 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 453/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [11] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 453/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 453 submitted successfully"
else
    echo "[ERROR] Multirun job 453 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 454/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [12] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 454/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 454 submitted successfully"
else
    echo "[ERROR] Multirun job 454 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 455/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [13] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 455/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 455 submitted successfully"
else
    echo "[ERROR] Multirun job 455 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 456/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [14] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 456/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 456 submitted successfully"
else
    echo "[ERROR] Multirun job 456 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 457/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [15] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 457/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 457 submitted successfully"
else
    echo "[ERROR] Multirun job 457 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 458/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [16] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 458/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 458 submitted successfully"
else
    echo "[ERROR] Multirun job 458 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 459/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [17] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 459/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 459 submitted successfully"
else
    echo "[ERROR] Multirun job 459 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 460/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [18] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 460/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 460 submitted successfully"
else
    echo "[ERROR] Multirun job 460 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 461/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [19] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 461/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 461 submitted successfully"
else
    echo "[ERROR] Multirun job 461 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 462/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [20] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 462/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 462 submitted successfully"
else
    echo "[ERROR] Multirun job 462 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 463/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [21] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 463/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 463 submitted successfully"
else
    echo "[ERROR] Multirun job 463 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 464/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [22] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 464/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 464 submitted successfully"
else
    echo "[ERROR] Multirun job 464 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 465/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [23] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 465/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 465 submitted successfully"
else
    echo "[ERROR] Multirun job 465 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 466/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [24] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 466/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 466 submitted successfully"
else
    echo "[ERROR] Multirun job 466 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 467/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [25] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 467/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 467 submitted successfully"
else
    echo "[ERROR] Multirun job 467 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 468/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [26] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 468/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 468 submitted successfully"
else
    echo "[ERROR] Multirun job 468 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 469/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [27] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 469/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 469 submitted successfully"
else
    echo "[ERROR] Multirun job 469 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 470/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [28] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 470/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 28 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 470 submitted successfully"
else
    echo "[ERROR] Multirun job 470 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 471/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [29] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 471/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 29 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 471 submitted successfully"
else
    echo "[ERROR] Multirun job 471 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 472/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [30] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 472/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 30 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 472 submitted successfully"
else
    echo "[ERROR] Multirun job 472 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 473/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [31] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 473/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 31 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 473 submitted successfully"
else
    echo "[ERROR] Multirun job 473 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 474/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [32] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 474/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 32 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 474 submitted successfully"
else
    echo "[ERROR] Multirun job 474 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 475/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [33] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 475/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 33 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 475 submitted successfully"
else
    echo "[ERROR] Multirun job 475 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 476/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [34] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 476/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 34 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 476 submitted successfully"
else
    echo "[ERROR] Multirun job 476 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 477/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [35] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 477/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 35 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 477 submitted successfully"
else
    echo "[ERROR] Multirun job 477 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 478/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [36] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 478/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 36 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 478 submitted successfully"
else
    echo "[ERROR] Multirun job 478 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 479/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [37] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 479/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 37 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 479 submitted successfully"
else
    echo "[ERROR] Multirun job 479 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 480/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [38] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 480/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 38 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 480 submitted successfully"
else
    echo "[ERROR] Multirun job 480 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 481/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [39] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 481/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 481 submitted successfully"
else
    echo "[ERROR] Multirun job 481 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 482/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [40] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 482/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 40 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 482 submitted successfully"
else
    echo "[ERROR] Multirun job 482 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 483/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [41] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 483/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 483 submitted successfully"
else
    echo "[ERROR] Multirun job 483 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 484/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [42] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 484/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 42 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 484 submitted successfully"
else
    echo "[ERROR] Multirun job 484 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 485/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [43] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 485/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 43 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 485 submitted successfully"
else
    echo "[ERROR] Multirun job 485 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 486/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [44] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 486/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 44 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 486 submitted successfully"
else
    echo "[ERROR] Multirun job 486 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 487/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [45] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 487/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 45 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 487 submitted successfully"
else
    echo "[ERROR] Multirun job 487 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 488/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [46] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 488/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 46 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 488 submitted successfully"
else
    echo "[ERROR] Multirun job 488 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 489/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [47] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 489/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 47 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 489 submitted successfully"
else
    echo "[ERROR] Multirun job 489 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 490/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [48] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 490/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 48 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 490 submitted successfully"
else
    echo "[ERROR] Multirun job 490 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 491/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [49] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 491/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 49 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 491 submitted successfully"
else
    echo "[ERROR] Multirun job 491 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 492/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [50] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 492/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 50 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 492 submitted successfully"
else
    echo "[ERROR] Multirun job 492 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 493/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [51] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 493/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 51 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 493 submitted successfully"
else
    echo "[ERROR] Multirun job 493 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 494/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [52] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 494/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 52 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 494 submitted successfully"
else
    echo "[ERROR] Multirun job 494 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 495/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [53] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 495/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 53 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 495 submitted successfully"
else
    echo "[ERROR] Multirun job 495 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 496/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [54] | Seed: 500
# Timeout: --time=0-02:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 496/1000..."
sbatch --time=0-02:00:00 --mem=12G unified_eval_script.sh 54 Lee2019_SSVEP CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 496 submitted successfully"
else
    echo "[ERROR] Multirun job 496 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 497/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [1] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 497/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 1 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 497 submitted successfully"
else
    echo "[ERROR] Multirun job 497 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 498/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [2] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 498/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 2 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 498 submitted successfully"
else
    echo "[ERROR] Multirun job 498 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 499/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [3] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 499/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 3 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 499 submitted successfully"
else
    echo "[ERROR] Multirun job 499 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 500/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [4] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 500/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 4 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 500 submitted successfully"
else
    echo "[ERROR] Multirun job 500 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 501/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [5] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 501/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 5 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 501 submitted successfully"
else
    echo "[ERROR] Multirun job 501 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 502/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [6] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 502/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 6 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 502 submitted successfully"
else
    echo "[ERROR] Multirun job 502 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 503/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [7] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 503/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 7 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 503 submitted successfully"
else
    echo "[ERROR] Multirun job 503 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 504/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [8] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 504/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 8 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 504 submitted successfully"
else
    echo "[ERROR] Multirun job 504 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 505/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [9] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 505/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 9 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 505 submitted successfully"
else
    echo "[ERROR] Multirun job 505 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 506/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [10] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 506/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 10 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 506 submitted successfully"
else
    echo "[ERROR] Multirun job 506 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 507/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [11] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 507/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 11 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 507 submitted successfully"
else
    echo "[ERROR] Multirun job 507 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 508/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [12] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 508/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 12 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 508 submitted successfully"
else
    echo "[ERROR] Multirun job 508 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 509/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [13] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 509/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 13 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 509 submitted successfully"
else
    echo "[ERROR] Multirun job 509 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 510/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [14] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 510/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 14 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 510 submitted successfully"
else
    echo "[ERROR] Multirun job 510 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 511/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [15] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 511/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 511 submitted successfully"
else
    echo "[ERROR] Multirun job 511 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 512/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [16] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 512/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 16 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 512 submitted successfully"
else
    echo "[ERROR] Multirun job 512 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 513/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [17] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 513/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 17 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 513 submitted successfully"
else
    echo "[ERROR] Multirun job 513 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 514/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [18] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 514/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 18 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 514 submitted successfully"
else
    echo "[ERROR] Multirun job 514 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 515/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [19] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 515/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 19 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 515 submitted successfully"
else
    echo "[ERROR] Multirun job 515 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 516/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [20] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 516/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 20 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 516 submitted successfully"
else
    echo "[ERROR] Multirun job 516 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 517/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [21] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 517/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 21 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 517 submitted successfully"
else
    echo "[ERROR] Multirun job 517 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 518/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [22] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 518/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 22 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 518 submitted successfully"
else
    echo "[ERROR] Multirun job 518 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 519/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [23] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 519/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 23 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 519 submitted successfully"
else
    echo "[ERROR] Multirun job 519 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 520/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [24] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 520/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 520 submitted successfully"
else
    echo "[ERROR] Multirun job 520 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 521/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [25] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 521/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 25 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 521 submitted successfully"
else
    echo "[ERROR] Multirun job 521 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 522/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [26] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 522/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 26 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 522 submitted successfully"
else
    echo "[ERROR] Multirun job 522 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 523/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [27] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 523/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 27 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 523 submitted successfully"
else
    echo "[ERROR] Multirun job 523 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 524/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [28] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 524/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 28 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 524 submitted successfully"
else
    echo "[ERROR] Multirun job 524 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 525/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [29] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 525/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 29 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 525 submitted successfully"
else
    echo "[ERROR] Multirun job 525 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 526/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [30] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 526/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 30 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 526 submitted successfully"
else
    echo "[ERROR] Multirun job 526 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 527/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [31] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 527/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 31 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 527 submitted successfully"
else
    echo "[ERROR] Multirun job 527 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 528/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [32] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 528/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 32 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 528 submitted successfully"
else
    echo "[ERROR] Multirun job 528 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 529/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [33] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 529/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 33 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 529 submitted successfully"
else
    echo "[ERROR] Multirun job 529 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 530/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [34] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 530/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 34 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 530 submitted successfully"
else
    echo "[ERROR] Multirun job 530 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 531/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [35] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 531/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 35 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 531 submitted successfully"
else
    echo "[ERROR] Multirun job 531 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 532/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [36] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 532/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 36 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 532 submitted successfully"
else
    echo "[ERROR] Multirun job 532 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 533/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [37] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 533/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 37 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 533 submitted successfully"
else
    echo "[ERROR] Multirun job 533 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 534/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [38] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 534/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 38 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 534 submitted successfully"
else
    echo "[ERROR] Multirun job 534 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 535/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [39] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 535/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 535 submitted successfully"
else
    echo "[ERROR] Multirun job 535 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 536/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [40] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 536/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 40 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 536 submitted successfully"
else
    echo "[ERROR] Multirun job 536 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 537/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [41] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 537/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 537 submitted successfully"
else
    echo "[ERROR] Multirun job 537 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 538/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [42] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 538/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 42 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 538 submitted successfully"
else
    echo "[ERROR] Multirun job 538 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 539/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [43] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 539/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 43 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 539 submitted successfully"
else
    echo "[ERROR] Multirun job 539 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 540/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [44] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 540/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 44 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 540 submitted successfully"
else
    echo "[ERROR] Multirun job 540 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 541/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [45] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 541/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 45 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 541 submitted successfully"
else
    echo "[ERROR] Multirun job 541 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 542/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [46] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 542/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 46 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 542 submitted successfully"
else
    echo "[ERROR] Multirun job 542 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 543/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [47] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 543/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 47 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 543 submitted successfully"
else
    echo "[ERROR] Multirun job 543 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 544/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [48] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 544/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 48 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 544 submitted successfully"
else
    echo "[ERROR] Multirun job 544 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 545/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [49] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 545/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 49 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 545 submitted successfully"
else
    echo "[ERROR] Multirun job 545 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 546/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [50] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 546/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 50 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 546 submitted successfully"
else
    echo "[ERROR] Multirun job 546 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 547/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [51] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 547/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 51 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 547 submitted successfully"
else
    echo "[ERROR] Multirun job 547 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 548/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [52] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 548/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 52 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 548 submitted successfully"
else
    echo "[ERROR] Multirun job 548 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 549/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [53] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 549/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 53 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 549 submitted successfully"
else
    echo "[ERROR] Multirun job 549 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 550/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [54] | Seed: 500 | TUNED
# Timeout: --time=3-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 550/1000..."
sbatch --time=3-12:00:00 --mem=12G unified_eval_script.sh 54 Lee2019_SSVEP CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 550 submitted successfully"
else
    echo "[ERROR] Multirun job 550 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 551/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 100
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 551/1000..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 551 submitted successfully"
else
    echo "[ERROR] Multirun job 551 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 552/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 100 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 552/1000..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 552 submitted successfully"
else
    echo "[ERROR] Multirun job 552 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 553/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 200
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 553/1000..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 553 submitted successfully"
else
    echo "[ERROR] Multirun job 553 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 554/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 200 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 554/1000..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 554 submitted successfully"
else
    echo "[ERROR] Multirun job 554 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 555/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 300
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 555/1000..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 555 submitted successfully"
else
    echo "[ERROR] Multirun job 555 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 556/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 300 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 556/1000..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 556 submitted successfully"
else
    echo "[ERROR] Multirun job 556 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 557/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 400
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 557/1000..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 557 submitted successfully"
else
    echo "[ERROR] Multirun job 557 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 558/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 400 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 558/1000..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 558 submitted successfully"
else
    echo "[ERROR] Multirun job 558 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 559/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 500
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 559/1000..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 559 submitted successfully"
else
    echo "[ERROR] Multirun job 559 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 560/1000
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 500 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 560/1000..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 560 submitted successfully"
else
    echo "[ERROR] Multirun job 560 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 561/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [1] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 561/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 1 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 561 submitted successfully"
else
    echo "[ERROR] Multirun job 561 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 562/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [2] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 562/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 2 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 562 submitted successfully"
else
    echo "[ERROR] Multirun job 562 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 563/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [3] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 563/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 3 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 563 submitted successfully"
else
    echo "[ERROR] Multirun job 563 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 564/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [4] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 564/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 4 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 564 submitted successfully"
else
    echo "[ERROR] Multirun job 564 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 565/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [5] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 565/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 5 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 565 submitted successfully"
else
    echo "[ERROR] Multirun job 565 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 566/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [6] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 566/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 6 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 566 submitted successfully"
else
    echo "[ERROR] Multirun job 566 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 567/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [7] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 567/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 7 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 567 submitted successfully"
else
    echo "[ERROR] Multirun job 567 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 568/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [8] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 568/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 8 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 568 submitted successfully"
else
    echo "[ERROR] Multirun job 568 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 569/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [9] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 569/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 9 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 569 submitted successfully"
else
    echo "[ERROR] Multirun job 569 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 570/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [10] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 570/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 10 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 570 submitted successfully"
else
    echo "[ERROR] Multirun job 570 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 571/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [11] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 571/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 11 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 571 submitted successfully"
else
    echo "[ERROR] Multirun job 571 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 572/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [12] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 572/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 12 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 572 submitted successfully"
else
    echo "[ERROR] Multirun job 572 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 573/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [13] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 573/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 13 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 573 submitted successfully"
else
    echo "[ERROR] Multirun job 573 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 574/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [14] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 574/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 14 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 574 submitted successfully"
else
    echo "[ERROR] Multirun job 574 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 575/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [15] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 575/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 15 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 575 submitted successfully"
else
    echo "[ERROR] Multirun job 575 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 576/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [16] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 576/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 16 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 576 submitted successfully"
else
    echo "[ERROR] Multirun job 576 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 577/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [17] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 577/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 17 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 577 submitted successfully"
else
    echo "[ERROR] Multirun job 577 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 578/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [18] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 578/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 18 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 578 submitted successfully"
else
    echo "[ERROR] Multirun job 578 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 579/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [19] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 579/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 19 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 579 submitted successfully"
else
    echo "[ERROR] Multirun job 579 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 580/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [20] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 580/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 20 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 580 submitted successfully"
else
    echo "[ERROR] Multirun job 580 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 581/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [21] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 581/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 21 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 581 submitted successfully"
else
    echo "[ERROR] Multirun job 581 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 582/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [22] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 582/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 22 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 582 submitted successfully"
else
    echo "[ERROR] Multirun job 582 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 583/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [23] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 583/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 23 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 583 submitted successfully"
else
    echo "[ERROR] Multirun job 583 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 584/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [24] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 584/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 24 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 584 submitted successfully"
else
    echo "[ERROR] Multirun job 584 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 585/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [25] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 585/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 25 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 585 submitted successfully"
else
    echo "[ERROR] Multirun job 585 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 586/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [26] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 586/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 26 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 586 submitted successfully"
else
    echo "[ERROR] Multirun job 586 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 587/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [27] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 587/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 27 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 587 submitted successfully"
else
    echo "[ERROR] Multirun job 587 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 588/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [28] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 588/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 28 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 588 submitted successfully"
else
    echo "[ERROR] Multirun job 588 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 589/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [29] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 589/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 29 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 589 submitted successfully"
else
    echo "[ERROR] Multirun job 589 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 590/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [30] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 590/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 30 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 590 submitted successfully"
else
    echo "[ERROR] Multirun job 590 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 591/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [31] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 591/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 31 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 591 submitted successfully"
else
    echo "[ERROR] Multirun job 591 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 592/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [32] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 592/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 32 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 592 submitted successfully"
else
    echo "[ERROR] Multirun job 592 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 593/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [33] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 593/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 33 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 593 submitted successfully"
else
    echo "[ERROR] Multirun job 593 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 594/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [34] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 594/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 34 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 594 submitted successfully"
else
    echo "[ERROR] Multirun job 594 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 595/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [35] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 595/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 35 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 595 submitted successfully"
else
    echo "[ERROR] Multirun job 595 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 596/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [36] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 596/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 36 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 596 submitted successfully"
else
    echo "[ERROR] Multirun job 596 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 597/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [37] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 597/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 37 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 597 submitted successfully"
else
    echo "[ERROR] Multirun job 597 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 598/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [38] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 598/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 38 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 598 submitted successfully"
else
    echo "[ERROR] Multirun job 598 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 599/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [39] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 599/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 39 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 599 submitted successfully"
else
    echo "[ERROR] Multirun job 599 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 600/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [40] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 600/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 40 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 600 submitted successfully"
else
    echo "[ERROR] Multirun job 600 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 601/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [41] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 601/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 41 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 601 submitted successfully"
else
    echo "[ERROR] Multirun job 601 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 602/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [42] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 602/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 42 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 602 submitted successfully"
else
    echo "[ERROR] Multirun job 602 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 603/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [43] | Seed: 100
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 603/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 43 BI2015a CrossSession false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 603 submitted successfully"
else
    echo "[ERROR] Multirun job 603 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 604/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [1] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 604/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 1 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 604 submitted successfully"
else
    echo "[ERROR] Multirun job 604 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 605/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [2] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 605/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 2 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 605 submitted successfully"
else
    echo "[ERROR] Multirun job 605 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 606/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [3] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 606/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 3 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 606 submitted successfully"
else
    echo "[ERROR] Multirun job 606 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 607/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [4] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 607/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 4 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 607 submitted successfully"
else
    echo "[ERROR] Multirun job 607 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 608/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [5] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 608/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 5 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 608 submitted successfully"
else
    echo "[ERROR] Multirun job 608 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 609/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [6] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 609/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 6 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 609 submitted successfully"
else
    echo "[ERROR] Multirun job 609 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 610/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [7] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 610/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 7 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 610 submitted successfully"
else
    echo "[ERROR] Multirun job 610 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 611/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [8] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 611/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 8 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 611 submitted successfully"
else
    echo "[ERROR] Multirun job 611 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 612/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [9] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 612/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 9 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 612 submitted successfully"
else
    echo "[ERROR] Multirun job 612 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 613/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [10] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 613/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 10 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 613 submitted successfully"
else
    echo "[ERROR] Multirun job 613 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 614/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [11] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 614/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 11 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 614 submitted successfully"
else
    echo "[ERROR] Multirun job 614 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 615/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [12] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 615/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 12 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 615 submitted successfully"
else
    echo "[ERROR] Multirun job 615 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 616/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [13] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 616/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 13 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 616 submitted successfully"
else
    echo "[ERROR] Multirun job 616 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 617/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [14] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 617/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 14 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 617 submitted successfully"
else
    echo "[ERROR] Multirun job 617 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 618/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [15] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 618/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 15 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 618 submitted successfully"
else
    echo "[ERROR] Multirun job 618 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 619/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [16] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 619/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 16 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 619 submitted successfully"
else
    echo "[ERROR] Multirun job 619 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 620/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [17] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 620/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 17 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 620 submitted successfully"
else
    echo "[ERROR] Multirun job 620 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 621/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [18] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 621/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 18 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 621 submitted successfully"
else
    echo "[ERROR] Multirun job 621 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 622/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [19] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 622/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 19 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 622 submitted successfully"
else
    echo "[ERROR] Multirun job 622 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 623/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [20] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 623/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 20 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 623 submitted successfully"
else
    echo "[ERROR] Multirun job 623 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 624/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [21] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 624/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 21 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 624 submitted successfully"
else
    echo "[ERROR] Multirun job 624 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 625/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [22] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 625/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 22 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 625 submitted successfully"
else
    echo "[ERROR] Multirun job 625 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 626/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [23] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 626/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 23 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 626 submitted successfully"
else
    echo "[ERROR] Multirun job 626 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 627/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [24] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 627/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 24 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 627 submitted successfully"
else
    echo "[ERROR] Multirun job 627 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 628/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [25] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 628/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 25 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 628 submitted successfully"
else
    echo "[ERROR] Multirun job 628 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 629/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [26] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 629/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 26 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 629 submitted successfully"
else
    echo "[ERROR] Multirun job 629 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 630/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [27] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 630/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 27 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 630 submitted successfully"
else
    echo "[ERROR] Multirun job 630 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 631/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [28] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 631/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 28 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 631 submitted successfully"
else
    echo "[ERROR] Multirun job 631 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 632/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [29] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 632/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 29 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 632 submitted successfully"
else
    echo "[ERROR] Multirun job 632 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 633/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [30] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 633/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 30 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 633 submitted successfully"
else
    echo "[ERROR] Multirun job 633 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 634/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [31] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 634/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 31 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 634 submitted successfully"
else
    echo "[ERROR] Multirun job 634 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 635/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [32] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 635/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 32 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 635 submitted successfully"
else
    echo "[ERROR] Multirun job 635 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 636/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [33] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 636/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 33 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 636 submitted successfully"
else
    echo "[ERROR] Multirun job 636 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 637/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [34] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 637/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 34 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 637 submitted successfully"
else
    echo "[ERROR] Multirun job 637 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 638/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [35] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 638/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 35 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 638 submitted successfully"
else
    echo "[ERROR] Multirun job 638 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 639/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [36] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 639/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 36 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 639 submitted successfully"
else
    echo "[ERROR] Multirun job 639 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 640/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [37] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 640/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 37 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 640 submitted successfully"
else
    echo "[ERROR] Multirun job 640 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 641/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [38] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 641/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 38 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 641 submitted successfully"
else
    echo "[ERROR] Multirun job 641 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 642/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [39] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 642/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 39 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 642 submitted successfully"
else
    echo "[ERROR] Multirun job 642 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 643/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [40] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 643/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 40 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 643 submitted successfully"
else
    echo "[ERROR] Multirun job 643 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 644/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [41] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 644/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 41 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 644 submitted successfully"
else
    echo "[ERROR] Multirun job 644 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 645/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [42] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 645/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 42 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 645 submitted successfully"
else
    echo "[ERROR] Multirun job 645 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 646/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [43] | Seed: 100 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 646/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 43 BI2015a CrossSession true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 646 submitted successfully"
else
    echo "[ERROR] Multirun job 646 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 647/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [1] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 647/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 1 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 647 submitted successfully"
else
    echo "[ERROR] Multirun job 647 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 648/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [2] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 648/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 2 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 648 submitted successfully"
else
    echo "[ERROR] Multirun job 648 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 649/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [3] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 649/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 3 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 649 submitted successfully"
else
    echo "[ERROR] Multirun job 649 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 650/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [4] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 650/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 4 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 650 submitted successfully"
else
    echo "[ERROR] Multirun job 650 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 651/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [5] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 651/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 5 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 651 submitted successfully"
else
    echo "[ERROR] Multirun job 651 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 652/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [6] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 652/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 6 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 652 submitted successfully"
else
    echo "[ERROR] Multirun job 652 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 653/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [7] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 653/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 7 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 653 submitted successfully"
else
    echo "[ERROR] Multirun job 653 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 654/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [8] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 654/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 8 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 654 submitted successfully"
else
    echo "[ERROR] Multirun job 654 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 655/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [9] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 655/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 9 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 655 submitted successfully"
else
    echo "[ERROR] Multirun job 655 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 656/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [10] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 656/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 10 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 656 submitted successfully"
else
    echo "[ERROR] Multirun job 656 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 657/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [11] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 657/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 11 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 657 submitted successfully"
else
    echo "[ERROR] Multirun job 657 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 658/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [12] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 658/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 12 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 658 submitted successfully"
else
    echo "[ERROR] Multirun job 658 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 659/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [13] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 659/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 13 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 659 submitted successfully"
else
    echo "[ERROR] Multirun job 659 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 660/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [14] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 660/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 14 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 660 submitted successfully"
else
    echo "[ERROR] Multirun job 660 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 661/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [15] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 661/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 15 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 661 submitted successfully"
else
    echo "[ERROR] Multirun job 661 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 662/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [16] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 662/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 16 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 662 submitted successfully"
else
    echo "[ERROR] Multirun job 662 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 663/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [17] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 663/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 17 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 663 submitted successfully"
else
    echo "[ERROR] Multirun job 663 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 664/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [18] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 664/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 18 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 664 submitted successfully"
else
    echo "[ERROR] Multirun job 664 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 665/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [19] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 665/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 19 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 665 submitted successfully"
else
    echo "[ERROR] Multirun job 665 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 666/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [20] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 666/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 20 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 666 submitted successfully"
else
    echo "[ERROR] Multirun job 666 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 667/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [21] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 667/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 21 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 667 submitted successfully"
else
    echo "[ERROR] Multirun job 667 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 668/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [22] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 668/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 22 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 668 submitted successfully"
else
    echo "[ERROR] Multirun job 668 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 669/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [23] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 669/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 23 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 669 submitted successfully"
else
    echo "[ERROR] Multirun job 669 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 670/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [24] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 670/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 24 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 670 submitted successfully"
else
    echo "[ERROR] Multirun job 670 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 671/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [25] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 671/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 25 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 671 submitted successfully"
else
    echo "[ERROR] Multirun job 671 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 672/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [26] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 672/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 26 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 672 submitted successfully"
else
    echo "[ERROR] Multirun job 672 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 673/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [27] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 673/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 27 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 673 submitted successfully"
else
    echo "[ERROR] Multirun job 673 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 674/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [28] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 674/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 28 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 674 submitted successfully"
else
    echo "[ERROR] Multirun job 674 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 675/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [29] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 675/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 29 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 675 submitted successfully"
else
    echo "[ERROR] Multirun job 675 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 676/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [30] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 676/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 30 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 676 submitted successfully"
else
    echo "[ERROR] Multirun job 676 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 677/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [31] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 677/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 31 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 677 submitted successfully"
else
    echo "[ERROR] Multirun job 677 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 678/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [32] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 678/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 32 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 678 submitted successfully"
else
    echo "[ERROR] Multirun job 678 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 679/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [33] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 679/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 33 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 679 submitted successfully"
else
    echo "[ERROR] Multirun job 679 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 680/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [34] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 680/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 34 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 680 submitted successfully"
else
    echo "[ERROR] Multirun job 680 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 681/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [35] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 681/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 35 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 681 submitted successfully"
else
    echo "[ERROR] Multirun job 681 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 682/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [36] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 682/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 36 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 682 submitted successfully"
else
    echo "[ERROR] Multirun job 682 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 683/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [37] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 683/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 37 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 683 submitted successfully"
else
    echo "[ERROR] Multirun job 683 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 684/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [38] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 684/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 38 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 684 submitted successfully"
else
    echo "[ERROR] Multirun job 684 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 685/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [39] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 685/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 39 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 685 submitted successfully"
else
    echo "[ERROR] Multirun job 685 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 686/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [40] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 686/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 40 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 686 submitted successfully"
else
    echo "[ERROR] Multirun job 686 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 687/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [41] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 687/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 41 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 687 submitted successfully"
else
    echo "[ERROR] Multirun job 687 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 688/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [42] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 688/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 42 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 688 submitted successfully"
else
    echo "[ERROR] Multirun job 688 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 689/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [43] | Seed: 200
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 689/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 43 BI2015a CrossSession false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 689 submitted successfully"
else
    echo "[ERROR] Multirun job 689 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 690/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [1] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 690/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 1 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 690 submitted successfully"
else
    echo "[ERROR] Multirun job 690 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 691/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [2] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 691/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 2 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 691 submitted successfully"
else
    echo "[ERROR] Multirun job 691 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 692/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [3] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 692/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 3 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 692 submitted successfully"
else
    echo "[ERROR] Multirun job 692 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 693/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [4] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 693/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 4 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 693 submitted successfully"
else
    echo "[ERROR] Multirun job 693 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 694/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [5] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 694/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 5 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 694 submitted successfully"
else
    echo "[ERROR] Multirun job 694 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 695/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [6] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 695/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 6 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 695 submitted successfully"
else
    echo "[ERROR] Multirun job 695 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 696/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [7] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 696/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 7 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 696 submitted successfully"
else
    echo "[ERROR] Multirun job 696 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 697/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [8] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 697/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 8 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 697 submitted successfully"
else
    echo "[ERROR] Multirun job 697 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 698/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [9] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 698/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 9 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 698 submitted successfully"
else
    echo "[ERROR] Multirun job 698 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 699/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [10] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 699/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 10 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 699 submitted successfully"
else
    echo "[ERROR] Multirun job 699 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 700/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [11] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 700/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 11 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 700 submitted successfully"
else
    echo "[ERROR] Multirun job 700 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 701/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [12] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 701/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 12 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 701 submitted successfully"
else
    echo "[ERROR] Multirun job 701 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 702/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [13] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 702/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 13 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 702 submitted successfully"
else
    echo "[ERROR] Multirun job 702 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 703/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [14] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 703/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 14 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 703 submitted successfully"
else
    echo "[ERROR] Multirun job 703 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 704/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [15] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 704/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 15 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 704 submitted successfully"
else
    echo "[ERROR] Multirun job 704 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 705/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [16] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 705/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 16 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 705 submitted successfully"
else
    echo "[ERROR] Multirun job 705 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 706/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [17] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 706/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 17 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 706 submitted successfully"
else
    echo "[ERROR] Multirun job 706 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 707/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [18] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 707/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 18 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 707 submitted successfully"
else
    echo "[ERROR] Multirun job 707 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 708/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [19] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 708/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 19 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 708 submitted successfully"
else
    echo "[ERROR] Multirun job 708 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 709/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [20] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 709/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 20 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 709 submitted successfully"
else
    echo "[ERROR] Multirun job 709 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 710/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [21] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 710/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 21 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 710 submitted successfully"
else
    echo "[ERROR] Multirun job 710 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 711/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [22] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 711/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 22 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 711 submitted successfully"
else
    echo "[ERROR] Multirun job 711 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 712/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [23] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 712/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 23 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 712 submitted successfully"
else
    echo "[ERROR] Multirun job 712 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 713/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [24] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 713/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 24 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 713 submitted successfully"
else
    echo "[ERROR] Multirun job 713 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 714/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [25] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 714/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 25 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 714 submitted successfully"
else
    echo "[ERROR] Multirun job 714 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 715/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [26] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 715/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 26 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 715 submitted successfully"
else
    echo "[ERROR] Multirun job 715 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 716/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [27] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 716/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 27 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 716 submitted successfully"
else
    echo "[ERROR] Multirun job 716 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 717/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [28] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 717/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 28 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 717 submitted successfully"
else
    echo "[ERROR] Multirun job 717 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 718/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [29] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 718/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 29 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 718 submitted successfully"
else
    echo "[ERROR] Multirun job 718 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 719/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [30] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 719/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 30 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 719 submitted successfully"
else
    echo "[ERROR] Multirun job 719 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 720/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [31] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 720/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 31 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 720 submitted successfully"
else
    echo "[ERROR] Multirun job 720 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 721/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [32] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 721/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 32 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 721 submitted successfully"
else
    echo "[ERROR] Multirun job 721 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 722/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [33] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 722/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 33 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 722 submitted successfully"
else
    echo "[ERROR] Multirun job 722 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 723/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [34] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 723/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 34 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 723 submitted successfully"
else
    echo "[ERROR] Multirun job 723 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 724/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [35] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 724/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 35 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 724 submitted successfully"
else
    echo "[ERROR] Multirun job 724 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 725/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [36] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 725/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 36 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 725 submitted successfully"
else
    echo "[ERROR] Multirun job 725 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 726/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [37] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 726/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 37 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 726 submitted successfully"
else
    echo "[ERROR] Multirun job 726 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 727/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [38] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 727/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 38 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 727 submitted successfully"
else
    echo "[ERROR] Multirun job 727 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 728/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [39] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 728/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 39 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 728 submitted successfully"
else
    echo "[ERROR] Multirun job 728 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 729/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [40] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 729/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 40 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 729 submitted successfully"
else
    echo "[ERROR] Multirun job 729 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 730/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [41] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 730/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 41 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 730 submitted successfully"
else
    echo "[ERROR] Multirun job 730 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 731/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [42] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 731/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 42 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 731 submitted successfully"
else
    echo "[ERROR] Multirun job 731 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 732/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [43] | Seed: 200 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 732/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 43 BI2015a CrossSession true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 732 submitted successfully"
else
    echo "[ERROR] Multirun job 732 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 733/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [1] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 733/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 1 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 733 submitted successfully"
else
    echo "[ERROR] Multirun job 733 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 734/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [2] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 734/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 2 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 734 submitted successfully"
else
    echo "[ERROR] Multirun job 734 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 735/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [3] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 735/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 3 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 735 submitted successfully"
else
    echo "[ERROR] Multirun job 735 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 736/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [4] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 736/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 4 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 736 submitted successfully"
else
    echo "[ERROR] Multirun job 736 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 737/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [5] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 737/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 5 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 737 submitted successfully"
else
    echo "[ERROR] Multirun job 737 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 738/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [6] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 738/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 6 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 738 submitted successfully"
else
    echo "[ERROR] Multirun job 738 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 739/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [7] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 739/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 7 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 739 submitted successfully"
else
    echo "[ERROR] Multirun job 739 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 740/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [8] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 740/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 8 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 740 submitted successfully"
else
    echo "[ERROR] Multirun job 740 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 741/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [9] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 741/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 9 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 741 submitted successfully"
else
    echo "[ERROR] Multirun job 741 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 742/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [10] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 742/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 10 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 742 submitted successfully"
else
    echo "[ERROR] Multirun job 742 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 743/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [11] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 743/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 11 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 743 submitted successfully"
else
    echo "[ERROR] Multirun job 743 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 744/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [12] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 744/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 12 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 744 submitted successfully"
else
    echo "[ERROR] Multirun job 744 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 745/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [13] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 745/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 13 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 745 submitted successfully"
else
    echo "[ERROR] Multirun job 745 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 746/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [14] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 746/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 14 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 746 submitted successfully"
else
    echo "[ERROR] Multirun job 746 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 747/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [15] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 747/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 15 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 747 submitted successfully"
else
    echo "[ERROR] Multirun job 747 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 748/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [16] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 748/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 16 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 748 submitted successfully"
else
    echo "[ERROR] Multirun job 748 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 749/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [17] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 749/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 17 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 749 submitted successfully"
else
    echo "[ERROR] Multirun job 749 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 750/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [18] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 750/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 18 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 750 submitted successfully"
else
    echo "[ERROR] Multirun job 750 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 751/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [19] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 751/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 19 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 751 submitted successfully"
else
    echo "[ERROR] Multirun job 751 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 752/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [20] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 752/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 20 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 752 submitted successfully"
else
    echo "[ERROR] Multirun job 752 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 753/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [21] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 753/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 21 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 753 submitted successfully"
else
    echo "[ERROR] Multirun job 753 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 754/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [22] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 754/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 22 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 754 submitted successfully"
else
    echo "[ERROR] Multirun job 754 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 755/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [23] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 755/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 23 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 755 submitted successfully"
else
    echo "[ERROR] Multirun job 755 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 756/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [24] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 756/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 24 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 756 submitted successfully"
else
    echo "[ERROR] Multirun job 756 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 757/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [25] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 757/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 25 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 757 submitted successfully"
else
    echo "[ERROR] Multirun job 757 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 758/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [26] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 758/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 26 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 758 submitted successfully"
else
    echo "[ERROR] Multirun job 758 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 759/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [27] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 759/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 27 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 759 submitted successfully"
else
    echo "[ERROR] Multirun job 759 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 760/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [28] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 760/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 28 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 760 submitted successfully"
else
    echo "[ERROR] Multirun job 760 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 761/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [29] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 761/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 29 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 761 submitted successfully"
else
    echo "[ERROR] Multirun job 761 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 762/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [30] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 762/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 30 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 762 submitted successfully"
else
    echo "[ERROR] Multirun job 762 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 763/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [31] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 763/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 31 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 763 submitted successfully"
else
    echo "[ERROR] Multirun job 763 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 764/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [32] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 764/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 32 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 764 submitted successfully"
else
    echo "[ERROR] Multirun job 764 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 765/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [33] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 765/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 33 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 765 submitted successfully"
else
    echo "[ERROR] Multirun job 765 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 766/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [34] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 766/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 34 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 766 submitted successfully"
else
    echo "[ERROR] Multirun job 766 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 767/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [35] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 767/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 35 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 767 submitted successfully"
else
    echo "[ERROR] Multirun job 767 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 768/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [36] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 768/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 36 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 768 submitted successfully"
else
    echo "[ERROR] Multirun job 768 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 769/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [37] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 769/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 37 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 769 submitted successfully"
else
    echo "[ERROR] Multirun job 769 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 770/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [38] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 770/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 38 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 770 submitted successfully"
else
    echo "[ERROR] Multirun job 770 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 771/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [39] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 771/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 39 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 771 submitted successfully"
else
    echo "[ERROR] Multirun job 771 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 772/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [40] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 772/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 40 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 772 submitted successfully"
else
    echo "[ERROR] Multirun job 772 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 773/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [41] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 773/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 41 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 773 submitted successfully"
else
    echo "[ERROR] Multirun job 773 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 774/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [42] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 774/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 42 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 774 submitted successfully"
else
    echo "[ERROR] Multirun job 774 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 775/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [43] | Seed: 300
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 775/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 43 BI2015a CrossSession false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 775 submitted successfully"
else
    echo "[ERROR] Multirun job 775 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 776/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [1] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 776/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 1 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 776 submitted successfully"
else
    echo "[ERROR] Multirun job 776 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 777/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [2] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 777/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 2 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 777 submitted successfully"
else
    echo "[ERROR] Multirun job 777 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 778/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [3] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 778/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 3 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 778 submitted successfully"
else
    echo "[ERROR] Multirun job 778 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 779/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [4] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 779/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 4 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 779 submitted successfully"
else
    echo "[ERROR] Multirun job 779 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 780/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [5] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 780/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 5 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 780 submitted successfully"
else
    echo "[ERROR] Multirun job 780 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 781/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [6] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 781/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 6 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 781 submitted successfully"
else
    echo "[ERROR] Multirun job 781 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 782/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [7] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 782/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 7 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 782 submitted successfully"
else
    echo "[ERROR] Multirun job 782 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 783/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [8] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 783/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 8 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 783 submitted successfully"
else
    echo "[ERROR] Multirun job 783 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 784/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [9] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 784/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 9 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 784 submitted successfully"
else
    echo "[ERROR] Multirun job 784 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 785/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [10] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 785/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 10 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 785 submitted successfully"
else
    echo "[ERROR] Multirun job 785 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 786/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [11] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 786/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 11 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 786 submitted successfully"
else
    echo "[ERROR] Multirun job 786 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 787/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [12] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 787/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 12 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 787 submitted successfully"
else
    echo "[ERROR] Multirun job 787 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 788/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [13] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 788/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 13 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 788 submitted successfully"
else
    echo "[ERROR] Multirun job 788 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 789/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [14] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 789/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 14 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 789 submitted successfully"
else
    echo "[ERROR] Multirun job 789 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 790/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [15] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 790/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 15 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 790 submitted successfully"
else
    echo "[ERROR] Multirun job 790 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 791/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [16] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 791/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 16 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 791 submitted successfully"
else
    echo "[ERROR] Multirun job 791 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 792/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [17] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 792/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 17 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 792 submitted successfully"
else
    echo "[ERROR] Multirun job 792 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 793/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [18] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 793/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 18 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 793 submitted successfully"
else
    echo "[ERROR] Multirun job 793 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 794/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [19] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 794/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 19 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 794 submitted successfully"
else
    echo "[ERROR] Multirun job 794 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 795/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [20] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 795/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 20 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 795 submitted successfully"
else
    echo "[ERROR] Multirun job 795 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 796/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [21] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 796/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 21 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 796 submitted successfully"
else
    echo "[ERROR] Multirun job 796 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 797/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [22] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 797/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 22 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 797 submitted successfully"
else
    echo "[ERROR] Multirun job 797 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 798/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [23] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 798/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 23 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 798 submitted successfully"
else
    echo "[ERROR] Multirun job 798 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 799/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [24] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 799/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 24 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 799 submitted successfully"
else
    echo "[ERROR] Multirun job 799 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 800/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [25] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 800/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 25 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 800 submitted successfully"
else
    echo "[ERROR] Multirun job 800 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 801/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [26] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 801/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 26 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 801 submitted successfully"
else
    echo "[ERROR] Multirun job 801 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 802/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [27] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 802/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 27 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 802 submitted successfully"
else
    echo "[ERROR] Multirun job 802 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 803/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [28] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 803/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 28 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 803 submitted successfully"
else
    echo "[ERROR] Multirun job 803 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 804/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [29] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 804/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 29 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 804 submitted successfully"
else
    echo "[ERROR] Multirun job 804 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 805/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [30] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 805/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 30 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 805 submitted successfully"
else
    echo "[ERROR] Multirun job 805 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 806/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [31] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 806/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 31 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 806 submitted successfully"
else
    echo "[ERROR] Multirun job 806 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 807/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [32] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 807/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 32 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 807 submitted successfully"
else
    echo "[ERROR] Multirun job 807 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 808/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [33] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 808/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 33 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 808 submitted successfully"
else
    echo "[ERROR] Multirun job 808 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 809/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [34] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 809/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 34 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 809 submitted successfully"
else
    echo "[ERROR] Multirun job 809 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 810/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [35] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 810/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 35 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 810 submitted successfully"
else
    echo "[ERROR] Multirun job 810 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 811/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [36] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 811/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 36 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 811 submitted successfully"
else
    echo "[ERROR] Multirun job 811 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 812/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [37] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 812/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 37 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 812 submitted successfully"
else
    echo "[ERROR] Multirun job 812 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 813/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [38] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 813/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 38 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 813 submitted successfully"
else
    echo "[ERROR] Multirun job 813 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 814/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [39] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 814/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 39 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 814 submitted successfully"
else
    echo "[ERROR] Multirun job 814 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 815/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [40] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 815/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 40 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 815 submitted successfully"
else
    echo "[ERROR] Multirun job 815 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 816/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [41] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 816/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 41 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 816 submitted successfully"
else
    echo "[ERROR] Multirun job 816 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 817/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [42] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 817/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 42 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 817 submitted successfully"
else
    echo "[ERROR] Multirun job 817 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 818/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [43] | Seed: 300 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 818/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 43 BI2015a CrossSession true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 818 submitted successfully"
else
    echo "[ERROR] Multirun job 818 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 819/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [1] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 819/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 1 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 819 submitted successfully"
else
    echo "[ERROR] Multirun job 819 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 820/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [2] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 820/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 2 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 820 submitted successfully"
else
    echo "[ERROR] Multirun job 820 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 821/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [3] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 821/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 3 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 821 submitted successfully"
else
    echo "[ERROR] Multirun job 821 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 822/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [4] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 822/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 4 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 822 submitted successfully"
else
    echo "[ERROR] Multirun job 822 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 823/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [5] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 823/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 5 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 823 submitted successfully"
else
    echo "[ERROR] Multirun job 823 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 824/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [6] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 824/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 6 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 824 submitted successfully"
else
    echo "[ERROR] Multirun job 824 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 825/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [7] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 825/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 7 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 825 submitted successfully"
else
    echo "[ERROR] Multirun job 825 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 826/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [8] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 826/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 8 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 826 submitted successfully"
else
    echo "[ERROR] Multirun job 826 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 827/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [9] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 827/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 9 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 827 submitted successfully"
else
    echo "[ERROR] Multirun job 827 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 828/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [10] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 828/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 10 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 828 submitted successfully"
else
    echo "[ERROR] Multirun job 828 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 829/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [11] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 829/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 11 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 829 submitted successfully"
else
    echo "[ERROR] Multirun job 829 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 830/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [12] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 830/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 12 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 830 submitted successfully"
else
    echo "[ERROR] Multirun job 830 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 831/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [13] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 831/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 13 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 831 submitted successfully"
else
    echo "[ERROR] Multirun job 831 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 832/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [14] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 832/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 14 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 832 submitted successfully"
else
    echo "[ERROR] Multirun job 832 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 833/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [15] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 833/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 15 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 833 submitted successfully"
else
    echo "[ERROR] Multirun job 833 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 834/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [16] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 834/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 16 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 834 submitted successfully"
else
    echo "[ERROR] Multirun job 834 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 835/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [17] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 835/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 17 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 835 submitted successfully"
else
    echo "[ERROR] Multirun job 835 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 836/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [18] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 836/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 18 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 836 submitted successfully"
else
    echo "[ERROR] Multirun job 836 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 837/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [19] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 837/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 19 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 837 submitted successfully"
else
    echo "[ERROR] Multirun job 837 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 838/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [20] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 838/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 20 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 838 submitted successfully"
else
    echo "[ERROR] Multirun job 838 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 839/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [21] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 839/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 21 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 839 submitted successfully"
else
    echo "[ERROR] Multirun job 839 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 840/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [22] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 840/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 22 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 840 submitted successfully"
else
    echo "[ERROR] Multirun job 840 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 841/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [23] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 841/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 23 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 841 submitted successfully"
else
    echo "[ERROR] Multirun job 841 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 842/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [24] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 842/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 24 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 842 submitted successfully"
else
    echo "[ERROR] Multirun job 842 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 843/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [25] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 843/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 25 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 843 submitted successfully"
else
    echo "[ERROR] Multirun job 843 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 844/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [26] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 844/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 26 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 844 submitted successfully"
else
    echo "[ERROR] Multirun job 844 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 845/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [27] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 845/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 27 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 845 submitted successfully"
else
    echo "[ERROR] Multirun job 845 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 846/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [28] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 846/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 28 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 846 submitted successfully"
else
    echo "[ERROR] Multirun job 846 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 847/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [29] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 847/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 29 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 847 submitted successfully"
else
    echo "[ERROR] Multirun job 847 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 848/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [30] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 848/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 30 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 848 submitted successfully"
else
    echo "[ERROR] Multirun job 848 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 849/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [31] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 849/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 31 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 849 submitted successfully"
else
    echo "[ERROR] Multirun job 849 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 850/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [32] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 850/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 32 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 850 submitted successfully"
else
    echo "[ERROR] Multirun job 850 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 851/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [33] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 851/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 33 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 851 submitted successfully"
else
    echo "[ERROR] Multirun job 851 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 852/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [34] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 852/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 34 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 852 submitted successfully"
else
    echo "[ERROR] Multirun job 852 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 853/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [35] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 853/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 35 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 853 submitted successfully"
else
    echo "[ERROR] Multirun job 853 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 854/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [36] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 854/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 36 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 854 submitted successfully"
else
    echo "[ERROR] Multirun job 854 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 855/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [37] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 855/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 37 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 855 submitted successfully"
else
    echo "[ERROR] Multirun job 855 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 856/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [38] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 856/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 38 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 856 submitted successfully"
else
    echo "[ERROR] Multirun job 856 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 857/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [39] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 857/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 39 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 857 submitted successfully"
else
    echo "[ERROR] Multirun job 857 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 858/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [40] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 858/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 40 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 858 submitted successfully"
else
    echo "[ERROR] Multirun job 858 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 859/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [41] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 859/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 41 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 859 submitted successfully"
else
    echo "[ERROR] Multirun job 859 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 860/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [42] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 860/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 42 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 860 submitted successfully"
else
    echo "[ERROR] Multirun job 860 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 861/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [43] | Seed: 400
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 861/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 43 BI2015a CrossSession false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 861 submitted successfully"
else
    echo "[ERROR] Multirun job 861 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 862/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [1] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 862/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 1 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 862 submitted successfully"
else
    echo "[ERROR] Multirun job 862 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 863/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [2] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 863/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 2 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 863 submitted successfully"
else
    echo "[ERROR] Multirun job 863 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 864/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [3] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 864/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 3 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 864 submitted successfully"
else
    echo "[ERROR] Multirun job 864 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 865/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [4] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 865/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 4 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 865 submitted successfully"
else
    echo "[ERROR] Multirun job 865 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 866/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [5] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 866/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 5 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 866 submitted successfully"
else
    echo "[ERROR] Multirun job 866 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 867/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [6] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 867/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 6 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 867 submitted successfully"
else
    echo "[ERROR] Multirun job 867 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 868/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [7] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 868/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 7 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 868 submitted successfully"
else
    echo "[ERROR] Multirun job 868 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 869/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [8] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 869/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 8 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 869 submitted successfully"
else
    echo "[ERROR] Multirun job 869 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 870/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [9] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 870/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 9 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 870 submitted successfully"
else
    echo "[ERROR] Multirun job 870 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 871/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [10] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 871/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 10 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 871 submitted successfully"
else
    echo "[ERROR] Multirun job 871 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 872/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [11] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 872/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 11 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 872 submitted successfully"
else
    echo "[ERROR] Multirun job 872 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 873/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [12] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 873/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 12 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 873 submitted successfully"
else
    echo "[ERROR] Multirun job 873 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 874/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [13] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 874/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 13 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 874 submitted successfully"
else
    echo "[ERROR] Multirun job 874 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 875/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [14] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 875/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 14 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 875 submitted successfully"
else
    echo "[ERROR] Multirun job 875 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 876/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [15] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 876/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 15 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 876 submitted successfully"
else
    echo "[ERROR] Multirun job 876 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 877/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [16] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 877/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 16 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 877 submitted successfully"
else
    echo "[ERROR] Multirun job 877 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 878/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [17] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 878/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 17 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 878 submitted successfully"
else
    echo "[ERROR] Multirun job 878 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 879/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [18] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 879/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 18 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 879 submitted successfully"
else
    echo "[ERROR] Multirun job 879 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 880/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [19] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 880/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 19 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 880 submitted successfully"
else
    echo "[ERROR] Multirun job 880 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 881/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [20] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 881/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 20 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 881 submitted successfully"
else
    echo "[ERROR] Multirun job 881 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 882/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [21] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 882/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 21 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 882 submitted successfully"
else
    echo "[ERROR] Multirun job 882 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 883/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [22] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 883/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 22 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 883 submitted successfully"
else
    echo "[ERROR] Multirun job 883 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 884/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [23] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 884/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 23 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 884 submitted successfully"
else
    echo "[ERROR] Multirun job 884 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 885/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [24] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 885/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 24 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 885 submitted successfully"
else
    echo "[ERROR] Multirun job 885 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 886/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [25] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 886/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 25 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 886 submitted successfully"
else
    echo "[ERROR] Multirun job 886 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 887/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [26] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 887/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 26 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 887 submitted successfully"
else
    echo "[ERROR] Multirun job 887 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 888/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [27] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 888/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 27 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 888 submitted successfully"
else
    echo "[ERROR] Multirun job 888 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 889/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [28] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 889/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 28 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 889 submitted successfully"
else
    echo "[ERROR] Multirun job 889 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 890/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [29] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 890/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 29 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 890 submitted successfully"
else
    echo "[ERROR] Multirun job 890 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 891/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [30] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 891/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 30 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 891 submitted successfully"
else
    echo "[ERROR] Multirun job 891 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 892/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [31] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 892/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 31 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 892 submitted successfully"
else
    echo "[ERROR] Multirun job 892 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 893/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [32] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 893/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 32 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 893 submitted successfully"
else
    echo "[ERROR] Multirun job 893 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 894/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [33] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 894/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 33 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 894 submitted successfully"
else
    echo "[ERROR] Multirun job 894 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 895/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [34] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 895/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 34 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 895 submitted successfully"
else
    echo "[ERROR] Multirun job 895 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 896/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [35] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 896/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 35 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 896 submitted successfully"
else
    echo "[ERROR] Multirun job 896 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 897/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [36] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 897/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 36 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 897 submitted successfully"
else
    echo "[ERROR] Multirun job 897 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 898/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [37] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 898/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 37 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 898 submitted successfully"
else
    echo "[ERROR] Multirun job 898 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 899/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [38] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 899/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 38 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 899 submitted successfully"
else
    echo "[ERROR] Multirun job 899 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 900/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [39] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 900/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 39 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 900 submitted successfully"
else
    echo "[ERROR] Multirun job 900 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 901/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [40] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 901/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 40 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 901 submitted successfully"
else
    echo "[ERROR] Multirun job 901 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 902/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [41] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 902/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 41 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 902 submitted successfully"
else
    echo "[ERROR] Multirun job 902 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 903/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [42] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 903/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 42 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 903 submitted successfully"
else
    echo "[ERROR] Multirun job 903 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 904/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [43] | Seed: 400 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 904/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 43 BI2015a CrossSession true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 904 submitted successfully"
else
    echo "[ERROR] Multirun job 904 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 905/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [1] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 905/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 1 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 905 submitted successfully"
else
    echo "[ERROR] Multirun job 905 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 906/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [2] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 906/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 2 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 906 submitted successfully"
else
    echo "[ERROR] Multirun job 906 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 907/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [3] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 907/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 3 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 907 submitted successfully"
else
    echo "[ERROR] Multirun job 907 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 908/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [4] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 908/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 4 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 908 submitted successfully"
else
    echo "[ERROR] Multirun job 908 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 909/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [5] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 909/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 5 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 909 submitted successfully"
else
    echo "[ERROR] Multirun job 909 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 910/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [6] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 910/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 6 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 910 submitted successfully"
else
    echo "[ERROR] Multirun job 910 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 911/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [7] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 911/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 7 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 911 submitted successfully"
else
    echo "[ERROR] Multirun job 911 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 912/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [8] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 912/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 8 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 912 submitted successfully"
else
    echo "[ERROR] Multirun job 912 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 913/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [9] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 913/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 9 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 913 submitted successfully"
else
    echo "[ERROR] Multirun job 913 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 914/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [10] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 914/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 10 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 914 submitted successfully"
else
    echo "[ERROR] Multirun job 914 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 915/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [11] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 915/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 11 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 915 submitted successfully"
else
    echo "[ERROR] Multirun job 915 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 916/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [12] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 916/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 12 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 916 submitted successfully"
else
    echo "[ERROR] Multirun job 916 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 917/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [13] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 917/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 13 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 917 submitted successfully"
else
    echo "[ERROR] Multirun job 917 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 918/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [14] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 918/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 14 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 918 submitted successfully"
else
    echo "[ERROR] Multirun job 918 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 919/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [15] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 919/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 15 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 919 submitted successfully"
else
    echo "[ERROR] Multirun job 919 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 920/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [16] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 920/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 16 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 920 submitted successfully"
else
    echo "[ERROR] Multirun job 920 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 921/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [17] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 921/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 17 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 921 submitted successfully"
else
    echo "[ERROR] Multirun job 921 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 922/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [18] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 922/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 18 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 922 submitted successfully"
else
    echo "[ERROR] Multirun job 922 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 923/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [19] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 923/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 19 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 923 submitted successfully"
else
    echo "[ERROR] Multirun job 923 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 924/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [20] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 924/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 20 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 924 submitted successfully"
else
    echo "[ERROR] Multirun job 924 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 925/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [21] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 925/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 21 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 925 submitted successfully"
else
    echo "[ERROR] Multirun job 925 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 926/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [22] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 926/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 22 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 926 submitted successfully"
else
    echo "[ERROR] Multirun job 926 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 927/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [23] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 927/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 23 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 927 submitted successfully"
else
    echo "[ERROR] Multirun job 927 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 928/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [24] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 928/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 24 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 928 submitted successfully"
else
    echo "[ERROR] Multirun job 928 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 929/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [25] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 929/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 25 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 929 submitted successfully"
else
    echo "[ERROR] Multirun job 929 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 930/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [26] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 930/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 26 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 930 submitted successfully"
else
    echo "[ERROR] Multirun job 930 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 931/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [27] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 931/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 27 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 931 submitted successfully"
else
    echo "[ERROR] Multirun job 931 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 932/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [28] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 932/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 28 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 932 submitted successfully"
else
    echo "[ERROR] Multirun job 932 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 933/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [29] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 933/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 29 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 933 submitted successfully"
else
    echo "[ERROR] Multirun job 933 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 934/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [30] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 934/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 30 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 934 submitted successfully"
else
    echo "[ERROR] Multirun job 934 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 935/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [31] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 935/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 31 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 935 submitted successfully"
else
    echo "[ERROR] Multirun job 935 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 936/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [32] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 936/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 32 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 936 submitted successfully"
else
    echo "[ERROR] Multirun job 936 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 937/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [33] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 937/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 33 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 937 submitted successfully"
else
    echo "[ERROR] Multirun job 937 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 938/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [34] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 938/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 34 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 938 submitted successfully"
else
    echo "[ERROR] Multirun job 938 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 939/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [35] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 939/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 35 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 939 submitted successfully"
else
    echo "[ERROR] Multirun job 939 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 940/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [36] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 940/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 36 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 940 submitted successfully"
else
    echo "[ERROR] Multirun job 940 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 941/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [37] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 941/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 37 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 941 submitted successfully"
else
    echo "[ERROR] Multirun job 941 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 942/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [38] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 942/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 38 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 942 submitted successfully"
else
    echo "[ERROR] Multirun job 942 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 943/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [39] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 943/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 39 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 943 submitted successfully"
else
    echo "[ERROR] Multirun job 943 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 944/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [40] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 944/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 40 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 944 submitted successfully"
else
    echo "[ERROR] Multirun job 944 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 945/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [41] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 945/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 41 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 945 submitted successfully"
else
    echo "[ERROR] Multirun job 945 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 946/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [42] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 946/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 42 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 946 submitted successfully"
else
    echo "[ERROR] Multirun job 946 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 947/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [43] | Seed: 500
# Timeout: --time=0-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 947/1000..."
sbatch --time=0-08:00:00 --mem=12G unified_eval_script.sh 43 BI2015a CrossSession false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 947 submitted successfully"
else
    echo "[ERROR] Multirun job 947 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 948/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [1] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 948/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 1 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 948 submitted successfully"
else
    echo "[ERROR] Multirun job 948 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 949/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [2] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 949/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 2 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 949 submitted successfully"
else
    echo "[ERROR] Multirun job 949 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 950/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [3] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 950/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 3 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 950 submitted successfully"
else
    echo "[ERROR] Multirun job 950 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 951/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [4] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 951/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 4 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 951 submitted successfully"
else
    echo "[ERROR] Multirun job 951 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 952/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [5] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 952/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 5 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 952 submitted successfully"
else
    echo "[ERROR] Multirun job 952 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 953/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [6] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 953/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 6 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 953 submitted successfully"
else
    echo "[ERROR] Multirun job 953 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 954/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [7] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 954/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 7 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 954 submitted successfully"
else
    echo "[ERROR] Multirun job 954 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 955/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [8] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 955/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 8 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 955 submitted successfully"
else
    echo "[ERROR] Multirun job 955 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 956/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [9] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 956/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 9 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 956 submitted successfully"
else
    echo "[ERROR] Multirun job 956 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 957/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [10] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 957/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 10 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 957 submitted successfully"
else
    echo "[ERROR] Multirun job 957 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 958/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [11] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 958/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 11 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 958 submitted successfully"
else
    echo "[ERROR] Multirun job 958 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 959/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [12] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 959/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 12 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 959 submitted successfully"
else
    echo "[ERROR] Multirun job 959 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 960/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [13] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 960/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 13 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 960 submitted successfully"
else
    echo "[ERROR] Multirun job 960 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 961/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [14] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 961/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 14 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 961 submitted successfully"
else
    echo "[ERROR] Multirun job 961 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 962/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [15] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 962/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 15 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 962 submitted successfully"
else
    echo "[ERROR] Multirun job 962 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 963/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [16] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 963/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 16 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 963 submitted successfully"
else
    echo "[ERROR] Multirun job 963 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 964/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [17] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 964/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 17 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 964 submitted successfully"
else
    echo "[ERROR] Multirun job 964 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 965/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [18] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 965/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 18 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 965 submitted successfully"
else
    echo "[ERROR] Multirun job 965 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 966/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [19] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 966/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 19 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 966 submitted successfully"
else
    echo "[ERROR] Multirun job 966 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 967/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [20] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 967/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 20 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 967 submitted successfully"
else
    echo "[ERROR] Multirun job 967 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 968/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [21] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 968/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 21 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 968 submitted successfully"
else
    echo "[ERROR] Multirun job 968 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 969/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [22] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 969/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 22 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 969 submitted successfully"
else
    echo "[ERROR] Multirun job 969 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 970/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [23] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 970/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 23 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 970 submitted successfully"
else
    echo "[ERROR] Multirun job 970 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 971/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [24] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 971/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 24 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 971 submitted successfully"
else
    echo "[ERROR] Multirun job 971 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 972/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [25] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 972/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 25 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 972 submitted successfully"
else
    echo "[ERROR] Multirun job 972 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 973/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [26] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 973/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 26 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 973 submitted successfully"
else
    echo "[ERROR] Multirun job 973 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 974/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [27] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 974/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 27 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 974 submitted successfully"
else
    echo "[ERROR] Multirun job 974 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 975/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [28] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 975/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 28 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 975 submitted successfully"
else
    echo "[ERROR] Multirun job 975 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 976/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [29] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 976/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 29 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 976 submitted successfully"
else
    echo "[ERROR] Multirun job 976 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 977/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [30] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 977/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 30 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 977 submitted successfully"
else
    echo "[ERROR] Multirun job 977 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 978/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [31] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 978/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 31 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 978 submitted successfully"
else
    echo "[ERROR] Multirun job 978 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 979/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [32] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 979/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 32 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 979 submitted successfully"
else
    echo "[ERROR] Multirun job 979 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 980/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [33] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 980/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 33 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 980 submitted successfully"
else
    echo "[ERROR] Multirun job 980 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 981/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [34] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 981/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 34 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 981 submitted successfully"
else
    echo "[ERROR] Multirun job 981 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 982/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [35] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 982/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 35 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 982 submitted successfully"
else
    echo "[ERROR] Multirun job 982 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 983/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [36] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 983/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 36 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 983 submitted successfully"
else
    echo "[ERROR] Multirun job 983 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 984/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [37] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 984/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 37 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 984 submitted successfully"
else
    echo "[ERROR] Multirun job 984 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 985/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [38] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 985/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 38 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 985 submitted successfully"
else
    echo "[ERROR] Multirun job 985 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 986/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [39] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 986/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 39 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 986 submitted successfully"
else
    echo "[ERROR] Multirun job 986 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 987/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [40] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 987/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 40 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 987 submitted successfully"
else
    echo "[ERROR] Multirun job 987 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 988/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [41] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 988/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 41 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 988 submitted successfully"
else
    echo "[ERROR] Multirun job 988 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 989/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [42] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 989/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 42 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 989 submitted successfully"
else
    echo "[ERROR] Multirun job 989 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 990/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSession | Subjects: [43] | Seed: 500 | TUNED
# Timeout: --time=3-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 990/1000..."
sbatch --time=3-00:00:00 --mem=12G unified_eval_script.sh 43 BI2015a CrossSession true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 990 submitted successfully"
else
    echo "[ERROR] Multirun job 990 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 991/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] | Seed: 100
# Timeout: --time=1-08:00:00 --mem=16G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 991/1000..."
sbatch --time=1-08:00:00 --mem=16G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 BI2015a CrossSubject false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 991 submitted successfully"
else
    echo "[ERROR] Multirun job 991 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 992/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] | Seed: 100 | TUNED
# Timeout: --time=1-08:00:00 --mem=16G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 992/1000..."
sbatch --time=1-08:00:00 --mem=16G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 BI2015a CrossSubject true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 992 submitted successfully"
else
    echo "[ERROR] Multirun job 992 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 993/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] | Seed: 200
# Timeout: --time=1-08:00:00 --mem=16G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 993/1000..."
sbatch --time=1-08:00:00 --mem=16G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 BI2015a CrossSubject false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 993 submitted successfully"
else
    echo "[ERROR] Multirun job 993 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 994/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] | Seed: 200 | TUNED
# Timeout: --time=1-08:00:00 --mem=16G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 994/1000..."
sbatch --time=1-08:00:00 --mem=16G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 BI2015a CrossSubject true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 994 submitted successfully"
else
    echo "[ERROR] Multirun job 994 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 995/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] | Seed: 300
# Timeout: --time=1-08:00:00 --mem=16G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 995/1000..."
sbatch --time=1-08:00:00 --mem=16G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 BI2015a CrossSubject false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 995 submitted successfully"
else
    echo "[ERROR] Multirun job 995 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 996/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] | Seed: 300 | TUNED
# Timeout: --time=1-08:00:00 --mem=16G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 996/1000..."
sbatch --time=1-08:00:00 --mem=16G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 BI2015a CrossSubject true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 996 submitted successfully"
else
    echo "[ERROR] Multirun job 996 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 997/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] | Seed: 400
# Timeout: --time=1-08:00:00 --mem=16G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 997/1000..."
sbatch --time=1-08:00:00 --mem=16G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 BI2015a CrossSubject false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 997 submitted successfully"
else
    echo "[ERROR] Multirun job 997 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 998/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] | Seed: 400 | TUNED
# Timeout: --time=1-08:00:00 --mem=16G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 998/1000..."
sbatch --time=1-08:00:00 --mem=16G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 BI2015a CrossSubject true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 998 submitted successfully"
else
    echo "[ERROR] Multirun job 998 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 999/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] | Seed: 500
# Timeout: --time=1-08:00:00 --mem=16G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 999/1000..."
sbatch --time=1-08:00:00 --mem=16G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 BI2015a CrossSubject false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 999 submitted successfully"
else
    echo "[ERROR] Multirun job 999 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 1000/1000
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] | Seed: 500 | TUNED
# Timeout: --time=1-08:00:00 --mem=16G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 1000/1000..."
sbatch --time=1-08:00:00 --mem=16G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 BI2015a CrossSubject true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 1000 submitted successfully"
else
    echo "[ERROR] Multirun job 1000 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

