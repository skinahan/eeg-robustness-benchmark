#!/bin/bash
#SBATCH --job-name=EEG_Experiments_Array
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute
#SBATCH --array=0-999%50  # Process 1000 experiments, max 50 concurrent
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@institution.edu

# EEG Experiment Cluster Submission Script using SLURM Job Arrays
# This script efficiently processes experiments from a manifest file
# using SLURM's job array feature for better resource management.

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MANIFEST_FILE="${1:-experiment_manifest.json}"
ARRAY_INDEX=${SLURM_ARRAY_TASK_ID:-0}

# Create necessary directories
mkdir -p logs
mkdir -p job_status
mkdir -p completed_experiments

# Function to log messages with timestamps
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Array[$ARRAY_INDEX] $1" | tee -a logs/array_${SLURM_ARRAY_JOB_ID}_${ARRAY_INDEX}.log
}

# Function to extract experiment by array index
get_experiment_by_index() {
    local index="$1"
    
    python3 -c "
import json
import sys

try:
    with open('$MANIFEST_FILE', 'r') as f:
        manifest = json.load(f)
    
    if $index < len(manifest['experiments']):
        experiment = manifest['experiments'][$index]
        print(json.dumps(experiment))
    else:
        print('INDEX_OUT_OF_RANGE', file=sys.stderr)
        sys.exit(1)
        
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
"
}

# Function to extract experiment parameters
extract_experiment_params() {
    local experiment_json="$1"
    
    python3 -c "
import json
import sys

try:
    exp = json.loads('$experiment_json')
    
    # Extract all parameters
    params = {
        'model': exp['model'],
        'dataset': exp['dataset'],
        'subjects': exp['subjects'],
        'mode': exp['mode'],
        'eval_mode': exp['eval_mode'],
        'seed': exp['seed'],
        'tune': exp['tune'],
        'overwrite': exp['overwrite']
    }
    
    # Add noise parameters if present
    if exp.get('noise_type'):
        params['noise_type'] = exp['noise_type']
        params['intensity'] = exp['intensity']
    
    # Build command line arguments
    args = []
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                args.append(f'--{key}')
        elif isinstance(value, list):
            args.append(f'--{key}')
            args.extend(map(str, value))
        elif value is not None:
            args.append(f'--{key}')
            args.append(str(value))
    
    print(' '.join(args))
    
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
"
}

# Function to update experiment status
update_experiment_status() {
    local experiment_id="$1"
    local status="$2"
    local additional_info="$3"
    
    python3 -c "
import json
from datetime import datetime

try:
    with open('$MANIFEST_FILE', 'r') as f:
        manifest = json.load(f)
    
    for exp in manifest['experiments']:
        if exp['experiment_id'] == '$experiment_id':
            exp['status'] = '$status'
            exp['last_updated'] = datetime.now().isoformat()
            exp['slurm_job_id'] = ${SLURM_ARRAY_JOB_ID:-0}
            exp['array_index'] = ${ARRAY_INDEX:-0}
            
            if '$additional_info':
                exp['additional_info'] = '$additional_info'
            break
    
    with open('$MANIFEST_FILE', 'w') as f:
        json.dump(manifest, f, indent=2)
        
except Exception as e:
    print(f'ERROR updating status: {e}', file=sys.stderr)
"
}

# Main execution
main() {
    log_message "Starting experiment processing for array index $ARRAY_INDEX"
    
    # Check if manifest file exists
    if [[ ! -f "$MANIFEST_FILE" ]]; then
        log_message "ERROR: Manifest file '$MANIFEST_FILE' not found!"
        exit 1
    fi
    
    # Get experiment for this array index
    log_message "Retrieving experiment for index $ARRAY_INDEX"
    local experiment_json=$(get_experiment_by_index $ARRAY_INDEX)
    
    if [[ $? -ne 0 ]]; then
        log_message "ERROR: Failed to retrieve experiment for index $ARRAY_INDEX"
        exit 1
    fi
    
    # Extract experiment ID and parameters
    local experiment_id=$(echo "$experiment_json" | python3 -c "import json, sys; print(json.load(sys.stdin)['experiment_id'])")
    local experiment_params=$(extract_experiment_params "$experiment_json")
    
    if [[ $? -ne 0 ]]; then
        log_message "ERROR: Failed to extract parameters for experiment $experiment_id"
        update_experiment_status "$experiment_id" "failed" "Parameter extraction failed"
        exit 1
    fi
    
    log_message "Processing experiment: $experiment_id"
    log_message "Parameters: $experiment_params"
    
    # Update status to running
    update_experiment_status "$experiment_id" "running" "Started processing"
    
    # Create output directory
    local output_dir="outputs/experiments/${experiment_id}"
    mkdir -p "$output_dir/logs"
    mkdir -p "$output_dir/completed_experiments"
    
    # Activate conda environment
    source ~/.bashrc
    conda activate ncp_robustness_proj
    
    # Set working directory
    cd "$PROJECT_ROOT"
    
    # Run the experiment
    log_message "Starting experiment execution"
    local start_time=$(date +%s)
    
    if python evaluation/unified_experiment_runner.py $experiment_params; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log_message "Experiment completed successfully in ${duration} seconds"
        update_experiment_status "$experiment_id" "completed" "Completed successfully in ${duration}s"
        
        # Mark as completed
        echo "$(date)" > "${output_dir}/completed_experiments/${experiment_id}.completed"
        
        # Copy logs
        cp "logs/array_${SLURM_ARRAY_JOB_ID}_${ARRAY_INDEX}.log" "${output_dir}/logs/"
        
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log_message "Experiment failed after ${duration} seconds"
        update_experiment_status "$experiment_id" "failed" "Failed after ${duration}s"
        
        # Copy error logs
        cp "logs/array_${SLURM_ARRAY_JOB_ID}_${ARRAY_INDEX}.log" "${output_dir}/logs/"
        
        exit 1
    fi
    
    log_message "Experiment processing complete for $experiment_id"
}

# Run main function
main "$@"


