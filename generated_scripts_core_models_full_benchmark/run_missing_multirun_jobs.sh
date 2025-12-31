#!/bin/bash
# Generated multirun sbatch automation script
# Generated on: 2025-12-31 08:58:50
# Total missing multirun jobs: 51

set -e  # Exit on any error

echo "Starting multirun experiment automation..."
echo "Total multirun jobs to submit: 51"

# Multirun Job 1/51
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [15] | Seed: 100
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 1/51..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP WithinSession false eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 1 submitted successfully"
else
    echo "[ERROR] Multirun job 1 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 2/51
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [15] | Seed: 100 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 2/51..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP WithinSession true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 2 submitted successfully"
else
    echo "[ERROR] Multirun job 2 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 3/51
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [24] | Seed: 200
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 3/51..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP WithinSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 3 submitted successfully"
else
    echo "[ERROR] Multirun job 3 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 4/51
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [39] | Seed: 200
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 4/51..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP WithinSession false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 4 submitted successfully"
else
    echo "[ERROR] Multirun job 4 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 5/51
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [24] | Seed: 200 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 5/51..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP WithinSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 5 submitted successfully"
else
    echo "[ERROR] Multirun job 5 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 6/51
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [39] | Seed: 200 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 6/51..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP WithinSession true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 6 submitted successfully"
else
    echo "[ERROR] Multirun job 6 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 7/51
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [41] | Seed: 400
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 7/51..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP WithinSession false eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 7 submitted successfully"
else
    echo "[ERROR] Multirun job 7 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 8/51
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: WithinSession | Subjects: [41] | Seed: 400 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 8/51..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP WithinSession true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 8 submitted successfully"
else
    echo "[ERROR] Multirun job 8 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 9/51
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 100
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 9/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 9 submitted successfully"
else
    echo "[ERROR] Multirun job 9 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 10/51
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 100 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 10/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 10 submitted successfully"
else
    echo "[ERROR] Multirun job 10 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 11/51
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 200
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 11/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 11 submitted successfully"
else
    echo "[ERROR] Multirun job 11 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 12/51
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 200 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 12/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 12 submitted successfully"
else
    echo "[ERROR] Multirun job 12 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 13/51
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 300
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 13/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 13 submitted successfully"
else
    echo "[ERROR] Multirun job 13 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 14/51
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 300 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 14/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 14 submitted successfully"
else
    echo "[ERROR] Multirun job 14 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 15/51
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 400
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 15/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 15 submitted successfully"
else
    echo "[ERROR] Multirun job 15 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 16/51
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 400 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 16/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 16 submitted successfully"
else
    echo "[ERROR] Multirun job 16 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 17/51
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 500
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 17/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 17 submitted successfully"
else
    echo "[ERROR] Multirun job 17 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 18/51
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 500 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 18/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 18 submitted successfully"
else
    echo "[ERROR] Multirun job 18 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 19/51
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [15] | Seed: 100
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 19/51..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP WithinSession false reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 19 submitted successfully"
else
    echo "[ERROR] Multirun job 19 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 20/51
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [15] | Seed: 100 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 20/51..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP WithinSession true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 20 submitted successfully"
else
    echo "[ERROR] Multirun job 20 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 21/51
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [24] | Seed: 200
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 21/51..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP WithinSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 21 submitted successfully"
else
    echo "[ERROR] Multirun job 21 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 22/51
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [39] | Seed: 200
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 22/51..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP WithinSession false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 22 submitted successfully"
else
    echo "[ERROR] Multirun job 22 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 23/51
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [24] | Seed: 200 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 23/51..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP WithinSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 23 submitted successfully"
else
    echo "[ERROR] Multirun job 23 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 24/51
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [39] | Seed: 200 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 24/51..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP WithinSession true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 24 submitted successfully"
else
    echo "[ERROR] Multirun job 24 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 25/51
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [41] | Seed: 400
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 25/51..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP WithinSession false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 25 submitted successfully"
else
    echo "[ERROR] Multirun job 25 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 26/51
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: WithinSession | Subjects: [41] | Seed: 400 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 26/51..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP WithinSession true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 26 submitted successfully"
else
    echo "[ERROR] Multirun job 26 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 27/51
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 100
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 27/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 27 submitted successfully"
else
    echo "[ERROR] Multirun job 27 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 28/51
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 100 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 28/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 28 submitted successfully"
else
    echo "[ERROR] Multirun job 28 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 29/51
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 200
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 29/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 29 submitted successfully"
else
    echo "[ERROR] Multirun job 29 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 30/51
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 200 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 30/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 30 submitted successfully"
else
    echo "[ERROR] Multirun job 30 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 31/51
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 300
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 31/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 31 submitted successfully"
else
    echo "[ERROR] Multirun job 31 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 32/51
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 300 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 32/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 32 submitted successfully"
else
    echo "[ERROR] Multirun job 32 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 33/51
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 400
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 33/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 33 submitted successfully"
else
    echo "[ERROR] Multirun job 33 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 34/51
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 400 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 34/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 34 submitted successfully"
else
    echo "[ERROR] Multirun job 34 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 35/51
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 500
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 35/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 35 submitted successfully"
else
    echo "[ERROR] Multirun job 35 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 36/51
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 500 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 36/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 36 submitted successfully"
else
    echo "[ERROR] Multirun job 36 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 37/51
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [15] | Seed: 100
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 37/51..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP WithinSession false cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 37 submitted successfully"
else
    echo "[ERROR] Multirun job 37 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 38/51
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [15] | Seed: 100 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 38/51..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 15 Lee2019_SSVEP WithinSession true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 38 submitted successfully"
else
    echo "[ERROR] Multirun job 38 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 39/51
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [24] | Seed: 200
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 39/51..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP WithinSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 39 submitted successfully"
else
    echo "[ERROR] Multirun job 39 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 40/51
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [39] | Seed: 200
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 40/51..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP WithinSession false cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 40 submitted successfully"
else
    echo "[ERROR] Multirun job 40 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 41/51
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [24] | Seed: 200 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 41/51..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 24 Lee2019_SSVEP WithinSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 41 submitted successfully"
else
    echo "[ERROR] Multirun job 41 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 42/51
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [39] | Seed: 200 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 42/51..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 39 Lee2019_SSVEP WithinSession true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 42 submitted successfully"
else
    echo "[ERROR] Multirun job 42 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 43/51
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [41] | Seed: 400
# Timeout: --time=0-10:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 43/51..."
sbatch --time=0-10:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP WithinSession false cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 43 submitted successfully"
else
    echo "[ERROR] Multirun job 43 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 44/51
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: WithinSession | Subjects: [41] | Seed: 400 | TUNED
# Timeout: --time=5-12:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 44/51..."
sbatch --time=5-12:00:00 --mem=12G unified_eval_script.sh 41 Lee2019_SSVEP WithinSession true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 44 submitted successfully"
else
    echo "[ERROR] Multirun job 44 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 45/51
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 100 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 45/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 45 submitted successfully"
else
    echo "[ERROR] Multirun job 45 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 46/51
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 200 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 46/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 46 submitted successfully"
else
    echo "[ERROR] Multirun job 46 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 47/51
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 300 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 47/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 47 submitted successfully"
else
    echo "[ERROR] Multirun job 47 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 48/51
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 400 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 48/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 48 submitted successfully"
else
    echo "[ERROR] Multirun job 48 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 49/51
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 500
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 49/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 49 submitted successfully"
else
    echo "[ERROR] Multirun job 49 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 50/51
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 500 | TUNED
# Timeout: --time=1-08:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 50/51..."
sbatch --time=1-08:00:00 --mem=12G unified_eval_script.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 50 submitted successfully"
else
    echo "[ERROR] Multirun job 50 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 51/51
# Dataset: BI2015a | Model: cnn_ncp | Eval: WithinSession | Subjects: [1] | Seed: 500 | TUNED
# Timeout: --time=5-00:00:00 --mem=12G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 51/51..."
sbatch --time=5-00:00:00 --mem=12G unified_eval_script.sh 1 BI2015a WithinSession true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 51 submitted successfully"
else
    echo "[ERROR] Multirun job 51 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

