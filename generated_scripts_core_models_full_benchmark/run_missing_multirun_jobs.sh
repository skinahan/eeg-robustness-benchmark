#!/bin/bash
# Generated multirun sbatch automation script
# Generated on: 2026-01-10 12:04:49
# Total missing multirun jobs: 15

set -e  # Exit on any error

echo "Starting multirun experiment automation..."
echo "Total multirun jobs to submit: 15"

# Multirun Job 1/15
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 1/15..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true eegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 1 submitted successfully"
else
    echo "[ERROR] Multirun job 1 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 2/15
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 2/15..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true eegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 2 submitted successfully"
else
    echo "[ERROR] Multirun job 2 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 3/15
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 3/15..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true eegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 3 submitted successfully"
else
    echo "[ERROR] Multirun job 3 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 4/15
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 4/15..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true eegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 4 submitted successfully"
else
    echo "[ERROR] Multirun job 4 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 5/15
# Dataset: Lee2019_SSVEP | Model: eegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: eegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 5/15..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true eegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 5 submitted successfully"
else
    echo "[ERROR] Multirun job 5 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 6/15
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 6/15..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true reegnet 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 6 submitted successfully"
else
    echo "[ERROR] Multirun job 6 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 7/15
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 7/15..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true reegnet 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 7 submitted successfully"
else
    echo "[ERROR] Multirun job 7 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 8/15
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 8/15..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true reegnet 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 8 submitted successfully"
else
    echo "[ERROR] Multirun job 8 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 9/15
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 9/15..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true reegnet 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 9 submitted successfully"
else
    echo "[ERROR] Multirun job 9 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 10/15
# Dataset: Lee2019_SSVEP | Model: reegnet | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: reegnet
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 10/15..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true reegnet 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 10 submitted successfully"
else
    echo "[ERROR] Multirun job 10 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 11/15
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 11/15..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true cnn_ncp 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 11 submitted successfully"
else
    echo "[ERROR] Multirun job 11 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 12/15
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 12/15..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true cnn_ncp 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 12 submitted successfully"
else
    echo "[ERROR] Multirun job 12 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 13/15
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 13/15..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true cnn_ncp 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 13 submitted successfully"
else
    echo "[ERROR] Multirun job 13 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 14/15
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 14/15..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true cnn_ncp 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 14 submitted successfully"
else
    echo "[ERROR] Multirun job 14 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 15/15
# Dataset: Lee2019_SSVEP | Model: cnn_ncp | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: cnn_ncp
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 15/15..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true cnn_ncp 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 15 submitted successfully"
else
    echo "[ERROR] Multirun job 15 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

