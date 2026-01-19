#!/bin/bash
# Generated ablation sbatch automation script
# This script submits SLURM jobs for all ablation experiments
# Total jobs: 720 (3 datasets × 3 eval_modes × 16 ablations × 5 seeds)

set -e  # Exit on any error

# Configuration
DATASETS=("BNCI2014_001" "Lee2019_SSVEP" "BI2015a")
EVAL_MODES=("CrossSubject" "CrossSession" "WithinSession")
ABLATIONS=("baseline" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15")
SEEDS=(100 200 300 400 500)

# Calculate total jobs
TOTAL_JOBS=$((${#DATASETS[@]} * ${#EVAL_MODES[@]} * ${#ABLATIONS[@]} * ${#SEEDS[@]}))

echo "Starting ablation experiment automation..."
echo "Total ablation jobs to submit: $TOTAL_JOBS"
echo "Datasets: ${DATASETS[@]}"
echo "Eval Modes: ${EVAL_MODES[@]}"
echo "Ablations: ${ABLATIONS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo ""
echo "Note: This script should be run from the project root directory"
echo ""

# Create logs directory if it doesn't exist
mkdir -p ablations/logs

# Counter for job numbering
JOB_NUM=0

# Loop through all combinations
for dataset in "${DATASETS[@]}"; do
    for eval_mode in "${EVAL_MODES[@]}"; do
        for ablation in "${ABLATIONS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                JOB_NUM=$((JOB_NUM + 1))
                
                # Create job name (sanitize for SLURM)
                job_name="ablation_${ablation}_${dataset}_${eval_mode}_${seed}"
                # Replace special characters that might cause issues
                job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]/_/g')
                
                # Create log file names
                log_file="ablations/logs/${job_name}_%j.out"
                err_file="ablations/logs/${job_name}_%j.err"
                
                echo "Submitting ablation job $JOB_NUM/$TOTAL_JOBS..."
                echo "  Dataset: $dataset"
                echo "  Eval Mode: $eval_mode"
                echo "  Ablation: $ablation"
                echo "  Seed: $seed"
                
                sbatch --time=3-00:00:00 --mem=32G --job-name="$job_name" \
                    --output="$log_file" \
                    --error="$err_file" \
                    --mail-type=ALL --mail-user=skinahan@asu.edu \
                    ablations/run_ablation_sbatch.sh "$ablation" "$seed" "$dataset" "$eval_mode"
                
                if [ $? -eq 0 ]; then
                    echo "[SUCCESS] Ablation job $JOB_NUM submitted successfully"
                else
                    echo "[ERROR] Ablation job $JOB_NUM submission failed"
                    exit 1
                fi
                sleep 1  # Brief pause between submissions
            done
        done
    done
done

echo ""
echo "=========================================="
echo "All ablation jobs submitted successfully!"
echo "=========================================="
echo "Total jobs submitted: $JOB_NUM"
echo "Check job status with: squeue -u \$USER"
echo "Check logs in: ablations/logs/"
