#!/bin/bash
#SBATCH -p public 
#SBATCH -q public
#SBATCH -G 1
#SBATCH -c 1
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skinahan@asu.edu
#SBATCH --export=NONE

# Parse command line arguments for CrossSubject mode
# Format: unified_eval_script_crosssubject.sh <subject1> <subject2> ... <subjectN> <dataset> <eval_mode> <tune_flag> <model> <seed>
# The script collects all numeric arguments until it finds a non-numeric argument (the dataset)

SUBJECTS=()
CURRENT_ARG=1

# Collect all numeric arguments (subjects) until we find a non-numeric argument
while [ $CURRENT_ARG -le $# ]; do
    ARG_VALUE="${!CURRENT_ARG}"
    
    # Check if the argument is numeric (a subject ID)
    if [[ "$ARG_VALUE" =~ ^[0-9]+$ ]]; then
        SUBJECTS+=("$ARG_VALUE")
        CURRENT_ARG=$((CURRENT_ARG + 1))
    else
        # Found non-numeric argument - this should be the dataset
        break
    fi
done

# Now parse the remaining arguments
DATASET="${!CURRENT_ARG}"
CURRENT_ARG=$((CURRENT_ARG + 1))

EVAL_MODE="${!CURRENT_ARG}"
CURRENT_ARG=$((CURRENT_ARG + 1))

TUNE_FLAG="${!CURRENT_ARG}"
CURRENT_ARG=$((CURRENT_ARG + 1))

MODEL="${!CURRENT_ARG}"
CURRENT_ARG=$((CURRENT_ARG + 1))

SEED="${!CURRENT_ARG}"

# Validate required arguments
if [ ${#SUBJECTS[@]} -eq 0 ] || [ -z "$DATASET" ] || [ -z "$EVAL_MODE" ] || [ -z "$TUNE_FLAG" ] || [ -z "$MODEL" ] || [ -z "$SEED" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch unified_eval_script_crosssubject.sh <subject1> <subject2> ... <subjectN> <dataset> <eval_mode> <tune_flag> <model> <seed>"
    echo "  subjects: one or more subject IDs (e.g., '1 2 3 4 5' or '1 2 3 ... 54')"
    echo "  dataset: dataset name (e.g., 'Lee2019_SSVEP')"
    echo "  eval_mode: evaluation mode (should be 'CrossSubject')"
    echo "  tune_flag: tuning flag ('true' or 'false')"
    echo "  model: model name (e.g., 'eegnet', 'reegnet', 'cnn_ncp')"
    echo "  seed: random seed (e.g., '100', '200', '300', '400', '500')"
    exit 1
fi

# Verify eval_mode is CrossSubject
if [ "$EVAL_MODE" != "CrossSubject" ]; then
    echo "Error: This script is designed for CrossSubject evaluation mode only"
    echo "Received eval_mode: $EVAL_MODE"
    exit 1
fi

# Convert subjects array to space-separated string
SUBJECTS_STR="${SUBJECTS[*]}"

# Load environment
module load mamba/latest
source activate ncp_env

# Set memory limits to prevent OOM errors
# Note: ulimit may not work on all systems, but it's worth trying
# Limit virtual memory to 60GB (slightly less than SLURM's 64GB to allow for overhead)
# This will cause the process to be killed if it tries to allocate more than 60GB
if ulimit -v 62914560 2>/dev/null; then  # 60GB in KB (60 * 1024 * 1024)
    echo "[MEMORY] Virtual memory limit set: 60GB"
else
    echo "[MEMORY] Could not set virtual memory limit (may not be supported on this system)"
fi

# Also set a soft limit on RSS (resident set size) to 58GB
# This is a soft limit, so it will warn but not kill the process
if ulimit -m 60817408 2>/dev/null; then  # 58GB in KB (58 * 1024 * 1024)
    echo "[MEMORY] RSS soft limit set: 58GB"
else
    echo "[MEMORY] Could not set RSS limit (may not be supported on this system)"
fi

# Set environment variable for Python to use (in MB, converted to GB for display)
# This allows Python code to check memory usage and warn before exceeding limits
export PYTHON_MAX_MEMORY_GB=60
echo "[MEMORY] Python memory monitoring enabled: PYTHON_MAX_MEMORY_GB=${PYTHON_MAX_MEMORY_GB}"

# Convert tune flag to command line argument
if [ "$TUNE_FLAG" = "true" ]; then
    TUNE_ARG="--tune"
else
    TUNE_ARG=""
fi

# Build the subjects argument string
SUBJECTS_ARG="--subjects $SUBJECTS_STR"

echo "=========================================="
echo "Starting Multirun Experiment (CrossSubject)"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Evaluation Mode: $EVAL_MODE"
echo "Subjects: $SUBJECTS_STR"
echo "Number of Subjects: ${#SUBJECTS[@]}"
echo "Tuning: $TUNE_FLAG"
echo "Seed: $SEED"
echo "=========================================="

# Run the multirun experiment
# Note: multirun mode now processes:
# - Single model: specified by MODEL parameter
# - Single seed: specified by SEED parameter
# - All noise types: gaussian, dropout, eog
# - All intensity levels: 20 steps from 1.0 to 50.0
python evaluation/unified_experiment_runner.py \
    --model $MODEL \
    --dataset $DATASET \
    --mode multirun \
    --seed $SEED \
    $SUBJECTS_ARG \
    --noise_type gaussian \
    --intensity 10.0 \
    --eval_mode $EVAL_MODE \
    $TUNE_ARG \
    --overwrite

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Multirun experiment completed successfully"
    echo "=========================================="
else
    echo "=========================================="
    echo "Multirun experiment failed"
    echo "=========================================="
    exit 1
fi
