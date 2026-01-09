#!/bin/bash
# Generated multirun sbatch automation script
# Generated on: 2026-01-09 14:37:38
# Total missing multirun jobs: 30

set -e  # Exit on any error

echo "Starting multirun experiment automation..."
echo "Total multirun jobs to submit: 30"

# Multirun Job 1/30
# Dataset: BNCI2014_001 | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | Seed: 100
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 1/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 1 submitted successfully"
else
    echo "[ERROR] Multirun job 1 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 2/30
# Dataset: BNCI2014_001 | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 2/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 2 submitted successfully"
else
    echo "[ERROR] Multirun job 2 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 3/30
# Dataset: BNCI2014_001 | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | Seed: 200
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 3/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 3 submitted successfully"
else
    echo "[ERROR] Multirun job 3 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 4/30
# Dataset: BNCI2014_001 | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 4/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 4 submitted successfully"
else
    echo "[ERROR] Multirun job 4 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 5/30
# Dataset: BNCI2014_001 | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | Seed: 300
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 5/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 5 submitted successfully"
else
    echo "[ERROR] Multirun job 5 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 6/30
# Dataset: BNCI2014_001 | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 6/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 6 submitted successfully"
else
    echo "[ERROR] Multirun job 6 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 7/30
# Dataset: BNCI2014_001 | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | Seed: 400
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 7/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 7 submitted successfully"
else
    echo "[ERROR] Multirun job 7 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 8/30
# Dataset: BNCI2014_001 | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 8/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 8 submitted successfully"
else
    echo "[ERROR] Multirun job 8 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 9/30
# Dataset: BNCI2014_001 | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | Seed: 500
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 9/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 9 submitted successfully"
else
    echo "[ERROR] Multirun job 9 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 10/30
# Dataset: BNCI2014_001 | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 10/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 BNCI2014_001 CrossSubject true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 10 submitted successfully"
else
    echo "[ERROR] Multirun job 10 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 11/30
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 100
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 11/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 11 submitted successfully"
else
    echo "[ERROR] Multirun job 11 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 12/30
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 12/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 12 submitted successfully"
else
    echo "[ERROR] Multirun job 12 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 13/30
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 200
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 13/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 13 submitted successfully"
else
    echo "[ERROR] Multirun job 13 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 14/30
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 14/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 14 submitted successfully"
else
    echo "[ERROR] Multirun job 14 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 15/30
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 300
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 15/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 15 submitted successfully"
else
    echo "[ERROR] Multirun job 15 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 16/30
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 16/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 16 submitted successfully"
else
    echo "[ERROR] Multirun job 16 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 17/30
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 400
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 17/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 17 submitted successfully"
else
    echo "[ERROR] Multirun job 17 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 18/30
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 18/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 18 submitted successfully"
else
    echo "[ERROR] Multirun job 18 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 19/30
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 500
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 19/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 19 submitted successfully"
else
    echo "[ERROR] Multirun job 19 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 20/30
# Dataset: Lee2019_SSVEP | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 20/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 Lee2019_SSVEP CrossSubject true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 20 submitted successfully"
else
    echo "[ERROR] Multirun job 20 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 21/30
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] | Seed: 100
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 21/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 BI2015a CrossSubject false branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 21 submitted successfully"
else
    echo "[ERROR] Multirun job 21 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 22/30
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] | Seed: 100 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 100
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 22/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 BI2015a CrossSubject true branched_wiredcfc_arch4 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 22 submitted successfully"
else
    echo "[ERROR] Multirun job 22 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 23/30
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] | Seed: 200
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 23/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 BI2015a CrossSubject false branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 23 submitted successfully"
else
    echo "[ERROR] Multirun job 23 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 24/30
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] | Seed: 200 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 200
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 24/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 BI2015a CrossSubject true branched_wiredcfc_arch4 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 24 submitted successfully"
else
    echo "[ERROR] Multirun job 24 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 25/30
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] | Seed: 300
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 25/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 BI2015a CrossSubject false branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 25 submitted successfully"
else
    echo "[ERROR] Multirun job 25 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 26/30
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] | Seed: 300 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 300
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 26/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 BI2015a CrossSubject true branched_wiredcfc_arch4 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 26 submitted successfully"
else
    echo "[ERROR] Multirun job 26 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 27/30
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] | Seed: 400
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 27/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 BI2015a CrossSubject false branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 27 submitted successfully"
else
    echo "[ERROR] Multirun job 27 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 28/30
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] | Seed: 400 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 400
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 28/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 BI2015a CrossSubject true branched_wiredcfc_arch4 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 28 submitted successfully"
else
    echo "[ERROR] Multirun job 28 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 29/30
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] | Seed: 500
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 29/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 BI2015a CrossSubject false branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 29 submitted successfully"
else
    echo "[ERROR] Multirun job 29 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Multirun Job 30/30
# Dataset: BI2015a | Model: branched_wiredcfc_arch4 | Eval: CrossSubject | Subjects: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] | Seed: 500 | TUNED
# Timeout: --time=1-12:00:00 --mem=64G
# This multirun will generate test_perturb results for model: branched_wiredcfc_arch4
# This multirun will generate test_perturb results for seed: 500
# This multirun will generate test_perturb results for all noise types and intensities
echo "Submitting multirun job 30/30..."
sbatch --time=1-12:00:00 --mem=64G unified_eval_script_crosssubject_foldbyfold.sh 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 BI2015a CrossSubject true branched_wiredcfc_arch4 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Multirun job 30 submitted successfully"
else
    echo "[ERROR] Multirun job 30 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

