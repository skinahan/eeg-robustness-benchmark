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

# Parse command line arguments
SUBJECTS=$1
DATASET=$2
EVAL_MODE=$3
TUNE_FLAG=$4
MODEL=$5
SEED=$6

# Note: Time and memory limits are passed as sbatch command line arguments
# when this script is submitted (e.g., sbatch --time=4-12:00:00 --mem=12G ...)

# Validate required arguments
if [ -z "$SUBJECTS" ] || [ -z "$DATASET" ] || [ -z "$EVAL_MODE" ] || [ -z "$TUNE_FLAG" ] || [ -z "$MODEL" ] || [ -z "$SEED" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch unified_eval_script.sh <subjects> <dataset> <eval_mode> <tune_flag> <model> <seed>"
    echo "  subjects: space-separated list of subject IDs (e.g., '1' or '1 2 3 4 5 6 7 8 9')"
    echo "  dataset: dataset name (e.g., 'BNCI2014_001')"
    echo "  eval_mode: evaluation mode ('CrossSession' or 'WithinSession')"
    echo "  tune_flag: tuning flag ('true' or 'false')"
    echo "  model: model name (e.g., 'eegnet', 'reegnet', 'cnn_ncp')"
    echo "  seed: random seed (e.g., '100', '200', '300', '400', '500')"
    exit 1
fi

# Load environment
module load mamba/latest
source activate ncp_env

# Convert tune flag to command line argument
if [ "$TUNE_FLAG" = "true" ]; then
    TUNE_ARG="--tune"
else
    TUNE_ARG=""
fi

# Handle CrossSubject evaluation mode (for future use)
if [ "$EVAL_MODE" = "CrossSubject" ]; then
    echo "Warning: CrossSubject evaluation mode requires special handling - not yet implemented"
    exit 1
fi

# Build the subjects argument string
SUBJECTS_ARG="--subjects $SUBJECTS"

echo "=========================================="
echo "Starting Multirun Experiment"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Evaluation Mode: $EVAL_MODE"
echo "Subjects: $SUBJECTS"
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
