#!/bin/bash
#SBATCH --export=NONE
# Wrapper script for running ablation experiments via SLURM
# This script is called by run_ablation_jobs.sh with ablation number and seed as arguments

# Parse command line arguments
ABLATION=$1
SEED=$2
DATASET=$3
EVAL_MODE=$4

# Validate required arguments
if [ -z "$ABLATION" ] || [ -z "$SEED" ] || [ -z "$DATASET" ] || [ -z "$EVAL_MODE" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch run_ablation_sbatch.sh <ablation> <seed> <dataset> <eval_mode>"
    echo "  ablation: ablation number ('baseline', '1', '2', or '3')"
    echo "  seed: random seed (e.g., '100', '200', '300', '400', '500')"
    echo "  dataset: dataset name ('BNCI2014_001', 'Lee2019_SSVEP', or 'BI2015a')"
    echo "  eval_mode: evaluation mode ('CrossSubject', 'CrossSession', or 'WithinSession')"
    exit 1
fi

# CRITICAL: Limit all threading to prevent memory bloat on clusters
# These MUST be set BEFORE loading any Python libraries
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAX_THREADS=1
export NUMEXPR_MAX_THREADS=1
export TORCH_NUM_THREADS=1

echo "[MEMORY] Threading limited to 1 thread per library to prevent OOM"
echo "[MEMORY] OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "[MEMORY] MKL_NUM_THREADS=${MKL_NUM_THREADS}"

# Load environment
module load mamba/latest
source activate ncp_env

# Note: This script assumes it is run from the project root directory
# Paths are relative to project root, just like unified_eval_script.sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root to ensure relative paths work
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Starting Ablation Experiment"
echo "=========================================="
echo "Ablation: $ABLATION"
echo "Dataset: $DATASET"
echo "Eval Mode: $EVAL_MODE"
echo "Seed: $SEED"
echo "Project Root: $PROJECT_ROOT"
echo "Started at: $(date)"
echo "=========================================="

# Run the ablation experiment
python ablations/run_ablations.py --ablation "$ABLATION" --seed "$SEED" --dataset "$DATASET" --eval_mode "$EVAL_MODE"

EXIT_CODE=$?

echo "=========================================="
echo "Ablation Experiment Complete"
echo "=========================================="
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
