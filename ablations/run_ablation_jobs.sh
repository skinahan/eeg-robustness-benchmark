#!/bin/bash
# Generated ablation sbatch automation script
# This script submits SLURM jobs for all ablation experiments
# Total jobs: 20 (4 ablations × 5 seeds: baseline, 1, 2, 3 × seeds 100, 200, 300, 400, 500)

set -e  # Exit on any error

echo "Starting ablation experiment automation..."
echo "Total ablation jobs to submit: 20"
echo "Ablations: baseline, 1, 2, 3"
echo "Seeds: 100, 200, 300, 400, 500"
echo ""

# Create logs directory if it doesn't exist
mkdir -p ablations/logs

# Ablation Job 1/20
# Ablation: Baseline (Full HYDRA) | Seed: 100
# Timeout: --time=3-00:00:00 --mem=32G
echo "Submitting ablation job 1/20..."
sbatch --time=3-00:00:00 --mem=32G --job-name="ablation_baseline_100" \
    --output="ablations/logs/ablation_baseline_100_%j.out" \
    --error="ablations/logs/ablation_baseline_100_%j.err" \
    --mail-type=ALL --mail-user=skinahan@asu.edu \
    ablations/run_ablation_sbatch.sh baseline 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Ablation job 1 submitted successfully"
else
    echo "[ERROR] Ablation job 1 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Ablation Job 2/20
# Ablation: Baseline (Full HYDRA) | Seed: 200
# Timeout: --time=3-00:00:00 --mem=32G
echo "Submitting ablation job 2/20..."
sbatch --time=3-00:00:00 --mem=32G --job-name="ablation_baseline_200" \
    --output="ablations/logs/ablation_baseline_200_%j.out" \
    --error="ablations/logs/ablation_baseline_200_%j.err" \
    --mail-type=ALL --mail-user=skinahan@asu.edu \
    ablations/run_ablation_sbatch.sh baseline 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Ablation job 2 submitted successfully"
else
    echo "[ERROR] Ablation job 2 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Ablation Job 3/20
# Ablation: Baseline (Full HYDRA) | Seed: 300
# Timeout: --time=3-00:00:00 --mem=32G
echo "Submitting ablation job 3/20..."
sbatch --time=3-00:00:00 --mem=32G --job-name="ablation_baseline_300" \
    --output="ablations/logs/ablation_baseline_300_%j.out" \
    --error="ablations/logs/ablation_baseline_300_%j.err" \
    --mail-type=ALL --mail-user=skinahan@asu.edu \
    ablations/run_ablation_sbatch.sh baseline 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Ablation job 3 submitted successfully"
else
    echo "[ERROR] Ablation job 3 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Ablation Job 4/20
# Ablation: Baseline (Full HYDRA) | Seed: 400
# Timeout: --time=3-00:00:00 --mem=32G
echo "Submitting ablation job 4/20..."
sbatch --time=3-00:00:00 --mem=32G --job-name="ablation_baseline_400" \
    --output="ablations/logs/ablation_baseline_400_%j.out" \
    --error="ablations/logs/ablation_baseline_400_%j.err" \
    --mail-type=ALL --mail-user=skinahan@asu.edu \
    ablations/run_ablation_sbatch.sh baseline 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Ablation job 4 submitted successfully"
else
    echo "[ERROR] Ablation job 4 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Ablation Job 5/20
# Ablation: Baseline (Full HYDRA) | Seed: 500
# Timeout: --time=3-00:00:00 --mem=32G
echo "Submitting ablation job 5/20..."
sbatch --time=3-00:00:00 --mem=32G --job-name="ablation_baseline_500" \
    --output="ablations/logs/ablation_baseline_500_%j.out" \
    --error="ablations/logs/ablation_baseline_500_%j.err" \
    --mail-type=ALL --mail-user=skinahan@asu.edu \
    ablations/run_ablation_sbatch.sh baseline 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Ablation job 5 submitted successfully"
else
    echo "[ERROR] Ablation job 5 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Ablation Job 6/20
# Ablation: Ablation 1 (No Carry Gate) | Seed: 100
# Timeout: --time=3-00:00:00 --mem=32G
echo "Submitting ablation job 6/20..."
sbatch --time=3-00:00:00 --mem=32G --job-name="ablation_1_100" \
    --output="ablations/logs/ablation_1_100_%j.out" \
    --error="ablations/logs/ablation_1_100_%j.err" \
    --mail-type=ALL --mail-user=skinahan@asu.edu \
    ablations/run_ablation_sbatch.sh 1 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Ablation job 6 submitted successfully"
else
    echo "[ERROR] Ablation job 6 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Ablation Job 7/20
# Ablation: Ablation 1 (No Carry Gate) | Seed: 200
# Timeout: --time=3-00:00:00 --mem=32G
echo "Submitting ablation job 7/20..."
sbatch --time=3-00:00:00 --mem=32G --job-name="ablation_1_200" \
    --output="ablations/logs/ablation_1_200_%j.out" \
    --error="ablations/logs/ablation_1_200_%j.err" \
    --mail-type=ALL --mail-user=skinahan@asu.edu \
    ablations/run_ablation_sbatch.sh 1 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Ablation job 7 submitted successfully"
else
    echo "[ERROR] Ablation job 7 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Ablation Job 8/20
# Ablation: Ablation 1 (No Carry Gate) | Seed: 300
# Timeout: --time=3-00:00:00 --mem=32G
echo "Submitting ablation job 8/20..."
sbatch --time=3-00:00:00 --mem=32G --job-name="ablation_1_300" \
    --output="ablations/logs/ablation_1_300_%j.out" \
    --error="ablations/logs/ablation_1_300_%j.err" \
    --mail-type=ALL --mail-user=skinahan@asu.edu \
    ablations/run_ablation_sbatch.sh 1 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Ablation job 8 submitted successfully"
else
    echo "[ERROR] Ablation job 8 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Ablation Job 9/20
# Ablation: Ablation 1 (No Carry Gate) | Seed: 400
# Timeout: --time=3-00:00:00 --mem=32G
echo "Submitting ablation job 9/20..."
sbatch --time=3-00:00:00 --mem=32G --job-name="ablation_1_400" \
    --output="ablations/logs/ablation_1_400_%j.out" \
    --error="ablations/logs/ablation_1_400_%j.err" \
    --mail-type=ALL --mail-user=skinahan@asu.edu \
    ablations/run_ablation_sbatch.sh 1 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Ablation job 9 submitted successfully"
else
    echo "[ERROR] Ablation job 9 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Ablation Job 10/20
# Ablation: Ablation 1 (No Carry Gate) | Seed: 500
# Timeout: --time=3-00:00:00 --mem=32G
echo "Submitting ablation job 10/20..."
sbatch --time=3-00:00:00 --mem=32G --job-name="ablation_1_500" \
    --output="ablations/logs/ablation_1_500_%j.out" \
    --error="ablations/logs/ablation_1_500_%j.err" \
    --mail-type=ALL --mail-user=skinahan@asu.edu \
    ablations/run_ablation_sbatch.sh 1 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Ablation job 10 submitted successfully"
else
    echo "[ERROR] Ablation job 10 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Ablation Job 11/20
# Ablation: Ablation 2 (No Branching) | Seed: 100
# Timeout: --time=3-00:00:00 --mem=32G
echo "Submitting ablation job 11/20..."
sbatch --time=3-00:00:00 --mem=32G --job-name="ablation_2_100" \
    --output="ablations/logs/ablation_2_100_%j.out" \
    --error="ablations/logs/ablation_2_100_%j.err" \
    --mail-type=ALL --mail-user=skinahan@asu.edu \
    ablations/run_ablation_sbatch.sh 2 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Ablation job 11 submitted successfully"
else
    echo "[ERROR] Ablation job 11 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Ablation Job 12/20
# Ablation: Ablation 2 (No Branching) | Seed: 200
# Timeout: --time=3-00:00:00 --mem=32G
echo "Submitting ablation job 12/20..."
sbatch --time=3-00:00:00 --mem=32G --job-name="ablation_2_200" \
    --output="ablations/logs/ablation_2_200_%j.out" \
    --error="ablations/logs/ablation_2_200_%j.err" \
    --mail-type=ALL --mail-user=skinahan@asu.edu \
    ablations/run_ablation_sbatch.sh 2 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Ablation job 12 submitted successfully"
else
    echo "[ERROR] Ablation job 12 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Ablation Job 13/20
# Ablation: Ablation 2 (No Branching) | Seed: 300
# Timeout: --time=3-00:00:00 --mem=32G
echo "Submitting ablation job 13/20..."
sbatch --time=3-00:00:00 --mem=32G --job-name="ablation_2_300" \
    --output="ablations/logs/ablation_2_300_%j.out" \
    --error="ablations/logs/ablation_2_300_%j.err" \
    --mail-type=ALL --mail-user=skinahan@asu.edu \
    ablations/run_ablation_sbatch.sh 2 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Ablation job 13 submitted successfully"
else
    echo "[ERROR] Ablation job 13 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Ablation Job 14/20
# Ablation: Ablation 2 (No Branching) | Seed: 400
# Timeout: --time=3-00:00:00 --mem=32G
echo "Submitting ablation job 14/20..."
sbatch --time=3-00:00:00 --mem=32G --job-name="ablation_2_400" \
    --output="ablations/logs/ablation_2_400_%j.out" \
    --error="ablations/logs/ablation_2_400_%j.err" \
    --mail-type=ALL --mail-user=skinahan@asu.edu \
    ablations/run_ablation_sbatch.sh 2 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Ablation job 14 submitted successfully"
else
    echo "[ERROR] Ablation job 14 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Ablation Job 15/20
# Ablation: Ablation 2 (No Branching) | Seed: 500
# Timeout: --time=3-00:00:00 --mem=32G
echo "Submitting ablation job 15/20..."
sbatch --time=3-00:00:00 --mem=32G --job-name="ablation_2_500" \
    --output="ablations/logs/ablation_2_500_%j.out" \
    --error="ablations/logs/ablation_2_500_%j.err" \
    --mail-type=ALL --mail-user=skinahan@asu.edu \
    ablations/run_ablation_sbatch.sh 2 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Ablation job 15 submitted successfully"
else
    echo "[ERROR] Ablation job 15 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Ablation Job 16/20
# Ablation: Ablation 3 (LSTM Replacement) | Seed: 100
# Timeout: --time=3-00:00:00 --mem=32G
echo "Submitting ablation job 16/20..."
sbatch --time=3-00:00:00 --mem=32G --job-name="ablation_3_100" \
    --output="ablations/logs/ablation_3_100_%j.out" \
    --error="ablations/logs/ablation_3_100_%j.err" \
    --mail-type=ALL --mail-user=skinahan@asu.edu \
    ablations/run_ablation_sbatch.sh 3 100
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Ablation job 16 submitted successfully"
else
    echo "[ERROR] Ablation job 16 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Ablation Job 17/20
# Ablation: Ablation 3 (LSTM Replacement) | Seed: 200
# Timeout: --time=3-00:00:00 --mem=32G
echo "Submitting ablation job 17/20..."
sbatch --time=3-00:00:00 --mem=32G --job-name="ablation_3_200" \
    --output="ablations/logs/ablation_3_200_%j.out" \
    --error="ablations/logs/ablation_3_200_%j.err" \
    --mail-type=ALL --mail-user=skinahan@asu.edu \
    ablations/run_ablation_sbatch.sh 3 200
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Ablation job 17 submitted successfully"
else
    echo "[ERROR] Ablation job 17 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Ablation Job 18/20
# Ablation: Ablation 3 (LSTM Replacement) | Seed: 300
# Timeout: --time=3-00:00:00 --mem=32G
echo "Submitting ablation job 18/20..."
sbatch --time=3-00:00:00 --mem=32G --job-name="ablation_3_300" \
    --output="ablations/logs/ablation_3_300_%j.out" \
    --error="ablations/logs/ablation_3_300_%j.err" \
    --mail-type=ALL --mail-user=skinahan@asu.edu \
    ablations/run_ablation_sbatch.sh 3 300
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Ablation job 18 submitted successfully"
else
    echo "[ERROR] Ablation job 18 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Ablation Job 19/20
# Ablation: Ablation 3 (LSTM Replacement) | Seed: 400
# Timeout: --time=3-00:00:00 --mem=32G
echo "Submitting ablation job 19/20..."
sbatch --time=3-00:00:00 --mem=32G --job-name="ablation_3_400" \
    --output="ablations/logs/ablation_3_400_%j.out" \
    --error="ablations/logs/ablation_3_400_%j.err" \
    --mail-type=ALL --mail-user=skinahan@asu.edu \
    ablations/run_ablation_sbatch.sh 3 400
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Ablation job 19 submitted successfully"
else
    echo "[ERROR] Ablation job 19 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

# Ablation Job 20/20
# Ablation: Ablation 3 (LSTM Replacement) | Seed: 500
# Timeout: --time=3-00:00:00 --mem=32G
echo "Submitting ablation job 20/20..."
sbatch --time=3-00:00:00 --mem=32G --job-name="ablation_3_500" \
    --output="ablations/logs/ablation_3_500_%j.out" \
    --error="ablations/logs/ablation_3_500_%j.err" \
    --mail-type=ALL --mail-user=skinahan@asu.edu \
    ablations/run_ablation_sbatch.sh 3 500
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Ablation job 20 submitted successfully"
else
    echo "[ERROR] Ablation job 20 submission failed"
    exit 1
fi
sleep 1  # Brief pause between submissions

echo ""
echo "=========================================="
echo "All ablation jobs submitted successfully!"
echo "=========================================="
echo "Total jobs submitted: 20"
echo "Check job status with: squeue -u \$USER"
echo "Check logs in: ablations/logs/"
