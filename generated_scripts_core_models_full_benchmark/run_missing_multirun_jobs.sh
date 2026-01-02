#!/bin/bash
# Generated multirun sbatch automation script
# Generated on: 2026-01-02 15:23:01
# Total missing multirun jobs: 42

set -e  # Exit on any error

echo "Starting multirun experiment automation..."
echo "Total multirun jobs to submit: 42"

# Multirun Job 1/42
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [15] | Seed: 100
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 1/42..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP WithinSession false eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 1 submitted successfully"
else
    echo "[ERROR] Multirun job 1 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 2/42
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [15] | Seed: 100 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 2/42..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP WithinSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 2 submitted successfully"
else
    echo "[ERROR] Multirun job 2 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 3/42
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [24] | Seed: 200
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 3/42..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP WithinSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 3 submitted successfully"
else
    echo "[ERROR] Multirun job 3 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 4/42
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [39] | Seed: 200
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 4/42..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP WithinSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 4 submitted successfully"
else
    echo "[ERROR] Multirun job 4 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 5/42
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [24] | Seed: 200 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 5/42..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP WithinSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 5 submitted successfully"
else
    echo "[ERROR] Multirun job 5 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 6/42
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [39] | Seed: 200 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 6/42..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP WithinSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 6 submitted successfully"
else
    echo "[ERROR] Multirun job 6 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 7/42
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [41] | Seed: 400
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 7/42..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP WithinSession false eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 7 submitted successfully"
else
    echo "[ERROR] Multirun job 7 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 8/42
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [41] | Seed: 400 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 8/42..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP WithinSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 8 submitted successfully"
else
    echo "[ERROR] Multirun job 8 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 9/42
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 100 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 9/42..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 9 submitted successfully"
else
    echo "[ERROR] Multirun job 9 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 10/42
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 200 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 10/42..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 10 submitted successfully"
else
    echo "[ERROR] Multirun job 10 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 11/42
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 300 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 11/42..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 11 submitted successfully"
else
    echo "[ERROR] Multirun job 11 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 12/42
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 400 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 12/42..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 12 submitted successfully"
else
    echo "[ERROR] Multirun job 12 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 13/42
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 500 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 13/42..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 13 submitted successfully"
else
    echo "[ERROR] Multirun job 13 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 14/42
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [15] | Seed: 100
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 14/42..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP WithinSession false reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 14 submitted successfully"
else
    echo "[ERROR] Multirun job 14 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 15/42
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [15] | Seed: 100 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 15/42..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP WithinSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 15 submitted successfully"
else
    echo "[ERROR] Multirun job 15 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 16/42
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [24] | Seed: 200
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 16/42..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP WithinSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 16 submitted successfully"
else
    echo "[ERROR] Multirun job 16 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 17/42
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [39] | Seed: 200
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 17/42..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP WithinSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 17 submitted successfully"
else
    echo "[ERROR] Multirun job 17 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 18/42
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [24] | Seed: 200 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 18/42..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP WithinSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 18 submitted successfully"
else
    echo "[ERROR] Multirun job 18 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 19/42
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [39] | Seed: 200 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 19/42..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP WithinSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 19 submitted successfully"
else
    echo "[ERROR] Multirun job 19 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 20/42
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [41] | Seed: 400
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 20/42..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP WithinSession false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 20 submitted successfully"
else
    echo "[ERROR] Multirun job 20 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 21/42
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [41] | Seed: 400 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 21/42..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP WithinSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 21 submitted successfully"
else
    echo "[ERROR] Multirun job 21 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 22/42
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 100 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 22/42..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 22 submitted successfully"
else
    echo "[ERROR] Multirun job 22 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 23/42
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 200 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 23/42..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 23 submitted successfully"
else
    echo "[ERROR] Multirun job 23 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 24/42
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 300 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 24/42..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 24 submitted successfully"
else
    echo "[ERROR] Multirun job 24 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 25/42
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 400
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 25/42..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 25 submitted successfully"
else
    echo "[ERROR] Multirun job 25 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 26/42
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 400 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 26/42..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 26 submitted successfully"
else
    echo "[ERROR] Multirun job 26 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 27/42
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 500
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 27/42..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 27 submitted successfully"
else
    echo "[ERROR] Multirun job 27 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 28/42
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 500 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 28/42..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 28 submitted successfully"
else
    echo "[ERROR] Multirun job 28 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 29/42
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [15] | Seed: 100
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 29/42..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP WithinSession false cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 29 submitted successfully"
else
    echo "[ERROR] Multirun job 29 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 30/42
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [15] | Seed: 100 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 30/42..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP WithinSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 30 submitted successfully"
else
    echo "[ERROR] Multirun job 30 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 31/42
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [24] | Seed: 200
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 31/42..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP WithinSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 31 submitted successfully"
else
    echo "[ERROR] Multirun job 31 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 32/42
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [39] | Seed: 200
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 32/42..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP WithinSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 32 submitted successfully"
else
    echo "[ERROR] Multirun job 32 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 33/42
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [24] | Seed: 200 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 33/42..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP WithinSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 33 submitted successfully"
else
    echo "[ERROR] Multirun job 33 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 34/42
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [39] | Seed: 200 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 34/42..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP WithinSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 34 submitted successfully"
else
    echo "[ERROR] Multirun job 34 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 35/42
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [41] | Seed: 400
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 35/42..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP WithinSession false cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 35 submitted successfully"
else
    echo "[ERROR] Multirun job 35 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 36/42
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [41] | Seed: 400 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 36/42..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP WithinSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 36 submitted successfully"
else
    echo "[ERROR] Multirun job 36 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 37/42
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 100 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 37/42..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 37 submitted successfully"
else
    echo "[ERROR] Multirun job 37 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 38/42
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 200 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 38/42..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 38 submitted successfully"
else
    echo "[ERROR] Multirun job 38 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 39/42
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 300 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 39/42..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 39 submitted successfully"
else
    echo "[ERROR] Multirun job 39 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 40/42
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 400 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 40/42..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 40 submitted successfully"
else
    echo "[ERROR] Multirun job 40 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 41/42
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 500 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 41/42..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 41 submitted successfully"
else
    echo "[ERROR] Multirun job 41 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 42/42
# Dataset: BI2015a | Model: cnn_ncp | Eval: WithinSession | Subjects: [1] | Seed: 500 | TUNED
# Timeout: --time=5-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 42/42..."
sbatch --time=5-00:00:00 --mem=12G unified_eval_script.sh 1 BI2015a WithinSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 42 submitted successfully"
else
    echo "[ERROR] Multirun job 42 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

