#!/bin/bash
# Master script for submitting EEG experiments using SLURM job arrays
# This script automatically determines the optimal job array configuration
# and submits experiments efficiently to the cluster.

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MANIFEST_FILE="${1:-experiment_manifest.json}"
MAX_CONCURRENT_JOBS=50  # Maximum concurrent jobs in the array
PARTITION="compute"      # SLURM partition to use
TIME_LIMIT="24:00:00"   # Time limit for each job
MEMORY="12G"            # Memory per job
CPUS=1                 # CPUs per job

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $message"
}

# Function to check if manifest file exists and is valid
check_manifest() {
    if [[ ! -f "$MANIFEST_FILE" ]]; then
        print_status $RED "ERROR: Manifest file '$MANIFEST_FILE' not found!"
        print_status $RED "Usage: $0 <manifest_file.json>"
        exit 1
    fi
    
    print_status $BLUE "Using manifest file: $MANIFEST_FILE"
    
    # Validate JSON format
    if ! python3 -c "import json; json.load(open('$MANIFEST_FILE'))" 2>/dev/null; then
        print_status $RED "ERROR: Invalid JSON format in manifest file!"
        exit 1
    fi
    
    # Count total experiments
    local total_experiments=$(python3 -c "
import json
with open('$MANIFEST_FILE', 'r') as f:
    manifest = json.load(f)
print(len(manifest['experiments']))
")
    
    print_status $GREEN "Total experiments in manifest: $total_experiments"
    
    if [[ $total_experiments -eq 0 ]]; then
        print_status $RED "ERROR: No experiments found in manifest!"
        exit 1
    fi
    
    echo $total_experiments
}

# Function to analyze manifest and determine optimal configuration
analyze_manifest() {
    local total_experiments=$1
    
    print_status $BLUE "Analyzing manifest for optimal submission configuration..."
    
    # Calculate optimal array size
    local array_size=$((total_experiments - 1))  # SLURM arrays are 0-indexed
    
    # Ensure we don't exceed SLURM limits (typically 1000)
    if [[ $array_size -gt 999 ]]; then
        print_status $YELLOW "Warning: Array size $array_size exceeds SLURM limit of 999"
        print_status $YELLOW "Will need to submit multiple array jobs"
        array_size=999
    fi
    
    # Calculate number of array jobs needed
    local num_array_jobs=1
    if [[ $total_experiments -gt 1000 ]]; then
        num_array_jobs=$(( (total_experiments + 999) / 1000 ))
        print_status $YELLOW "Need to submit $num_array_jobs array jobs"
    fi
    
    # Determine optimal concurrency
    local optimal_concurrency=$MAX_CONCURRENT_JOBS
    if [[ $total_experiments -lt $MAX_CONCURRENT_JOBS ]]; then
        optimal_concurrency=$total_experiments
    fi
    
    print_status $GREEN "Optimal configuration:"
    print_status $GREEN "  Array size: $array_size"
    print_status $GREEN "  Max concurrent: $optimal_concurrency"
    print_status $GREEN "  Array jobs needed: $num_array_jobs"
    
    # Return configuration
    echo "$array_size $optimal_concurrency $num_array_jobs"
}

# Function to create job array submission script
create_array_script() {
    local array_size=$1
    local max_concurrent=$2
    local array_job_num=$3
    local start_index=$4
    
    local script_name="job_status/array_job_${array_job_num}.sbatch"
    
    cat > "$script_name" << EOF
#!/bin/bash
#SBATCH --job-name=EEG_Experiments_Array_${array_job_num}
#SBATCH --output=logs/array_${array_job_num}_%A_%a.out
#SBATCH --error=logs/array_${array_job_num}_%A_%a.err
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEMORY}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=${PARTITION}
#SBATCH --array=${start_index}-${array_size}%${max_concurrent}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@institution.edu

# EEG Experiment Array Job ${array_job_num}
# Processing experiments ${start_index} to ${array_size}

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "\$SCRIPT_DIR")"
MANIFEST_FILE="${MANIFEST_FILE}"
ARRAY_INDEX=\${SLURM_ARRAY_TASK_ID:-0}
GLOBAL_INDEX=\$((ARRAY_INDEX + ${start_index}))

# Create necessary directories
mkdir -p logs
mkdir -p job_status
mkdir -p completed_experiments

# Function to log messages with timestamps
log_message() {
    echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Array[${array_job_num}:\$ARRAY_INDEX] \$1" | tee -a logs/array_${array_job_num}_\${SLURM_ARRAY_JOB_ID}_\${ARRAY_INDEX}.log
}

# Function to extract experiment by global index
get_experiment_by_index() {
    local index="\$1"
    
    python3 -c "
import json
import sys

try:
    with open('\$MANIFEST_FILE', 'r') as f:
        manifest = json.load(f)
    
    if \$index < len(manifest['experiments']):
        experiment = manifest['experiments'][\$index]
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
    local experiment_json="\$1"
    
    python3 -c "
import json
import sys

try:
    exp = json.loads('\$experiment_json')
    
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
    local experiment_id="\$1"
    local status="\$2"
    local additional_info="\$3"
    
    python3 -c "
import json
from datetime import datetime

try:
    with open('\$MANIFEST_FILE', 'r') as f:
        manifest = json.load(f)
    
    for exp in manifest['experiments']:
        if exp['experiment_id'] == '\$experiment_id':
            exp['status'] = '\$status'
            exp['last_updated'] = datetime.now().isoformat()
            exp['slurm_job_id'] = \${SLURM_ARRAY_JOB_ID:-0}
            exp['array_job_num'] = ${array_job_num}
            exp['array_index'] = \${ARRAY_INDEX:-0}
            exp['global_index'] = \${GLOBAL_INDEX:-0}
            
            if '\$additional_info':
                exp['additional_info'] = '\$additional_info'
            break
    
    with open('\$MANIFEST_FILE', 'w') as f:
        json.dump(manifest, f, indent=2)
        
except Exception as e:
    print(f'ERROR updating status: {e}', file=sys.stderr)
"
}

# Main execution
main() {
    log_message "Starting experiment processing for global index \$GLOBAL_INDEX (array \$ARRAY_INDEX)"
    
    # Check if manifest file exists
    if [[ ! -f "\$MANIFEST_FILE" ]]; then
        log_message "ERROR: Manifest file '\$MANIFEST_FILE' not found!"
        exit 1
    fi
    
    # Get experiment for this global index
    log_message "Retrieving experiment for global index \$GLOBAL_INDEX"
    local experiment_json=\$(get_experiment_by_index \$GLOBAL_INDEX)
    
    if [[ \$? -ne 0 ]]; then
        log_message "ERROR: Failed to retrieve experiment for global index \$GLOBAL_INDEX"
        exit 1
    fi
    
    # Extract experiment ID and parameters
    local experiment_id=\$(echo "\$experiment_json" | python3 -c "import json, sys; print(json.load(sys.stdin)['experiment_id'])")
    local experiment_params=\$(extract_experiment_params "\$experiment_json")
    
    if [[ \$? -ne 0 ]]; then
        log_message "ERROR: Failed to extract parameters for experiment \$experiment_id"
        update_experiment_status "\$experiment_id" "failed" "Parameter extraction failed"
        exit 1
    fi
    
    log_message "Processing experiment: \$experiment_id"
    log_message "Parameters: \$experiment_params"
    
    # Update status to running
    update_experiment_status "\$experiment_id" "running" "Started processing"
    
    # Create output directory
    local output_dir="outputs/experiments/\${experiment_id}"
    mkdir -p "\$output_dir/logs"
    mkdir -p "\$output_dir/completed_experiments"
    
    # Activate conda environment
    source ~/.bashrc

    mamba load module/latest
    source activate ncp_env
    
    # Set working directory
    cd "\$PROJECT_ROOT"
    
    # Run the experiment
    log_message "Starting experiment execution"
    local start_time=\$(date +%s)
    
    if python evaluation/unified_experiment_runner.py \$experiment_params; then
        local end_time=\$(date +%s)
        local duration=\$((end_time - start_time))
        
        log_message "Experiment completed successfully in \${duration} seconds"
        update_experiment_status "\$experiment_id" "completed" "Completed successfully in \${duration}s"
        
        # Mark as completed
        echo "\$(date)" > "\${output_dir}/completed_experiments/\${experiment_id}.completed"
        
        # Copy logs
        cp "logs/array_${array_job_num}_\${SLURM_ARRAY_JOB_ID}_\${ARRAY_INDEX}.log" "\${output_dir}/logs/"
        
    else
        local end_time=\$(date +%s)
        local duration=\$((end_time - start_time))
        
        log_message "Experiment failed after \${duration} seconds"
        update_experiment_status "\$experiment_id" "failed" "Failed after \${duration}s"
        
        # Copy error logs
        cp "logs/array_${array_job_num}_\${SLURM_ARRAY_JOB_ID}_\${ARRAY_INDEX}.log" "\${output_dir}/logs/"
        
        exit 1
    fi
    
    log_message "Experiment processing complete for \$experiment_id"
}

# Run main function
main "\$@"
EOF

    chmod +x "$script_name"
    print_status $GREEN "Created array script: $script_name"
}

# Function to submit array jobs
submit_array_jobs() {
    local total_experiments=$1
    local array_size=$2
    local max_concurrent=$3
    local num_array_jobs=$4
    
    print_status $BLUE "Submitting array jobs..."
    
    # Create job_status directory
    mkdir -p job_status
    
    local submitted_jobs=()
    local start_index=0
    
    for ((i=1; i<=num_array_jobs; i++)); do
        local end_index=$((start_index + array_size))
        if [[ $end_index -gt $total_experiments ]]; then
            end_index=$total_experiments
        fi
        
        local actual_array_size=$((end_index - start_index - 1))
        
        print_status $BLUE "Creating array job $i (indices $start_index to $((end_index-1)))"
        create_array_script $actual_array_size $max_concurrent $i $start_index
        
        # Submit the job
        local job_id=$(sbatch "job_status/array_job_${i}.sbatch" | grep -o '[0-9]\+')
        
        if [[ -n "$job_id" ]]; then
            submitted_jobs+=("$job_id")
            print_status $GREEN "Submitted array job $i as SLURM job $job_id"
            
            # Update manifest with job information
            python3 -c "
import json
with open('$MANIFEST_FILE', 'r') as f:
    manifest = json.load(f)

# Update experiments in this array job range
for exp in manifest['experiments'][$start_index:$end_index]:
    exp['array_job_num'] = $i
    exp['slurm_job_id'] = $job_id

with open('$MANIFEST_FILE', 'w') as f:
    json.dump(manifest, f, indent=2)
"
        else
            print_status $RED "ERROR: Failed to submit array job $i"
        fi
        
        start_index=$end_index
        
        # Small delay between submissions
        sleep 2
    done
    
    print_status $GREEN "Successfully submitted $num_array_jobs array jobs:"
    for job_id in "${submitted_jobs[@]}"; do
        print_status $GREEN "  SLURM Job ID: $job_id"
    done
    
    return ${#submitted_jobs[@]}
}

# Function to create monitoring script
create_monitoring_script() {
    local script_name="monitor_experiments.sh"
    
    cat > "$script_name" << 'EOF'
#!/bin/bash
# Monitoring script for EEG experiments
# Run this script to monitor the progress of submitted experiments

set -e

MANIFEST_FILE="${1:-experiment_manifest.json}"
CHECK_INTERVAL=60  # Check every 60 seconds

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $message"
}

if [[ ! -f "$MANIFEST_FILE" ]]; then
    print_status $RED "ERROR: Manifest file '$MANIFEST_FILE' not found!"
    exit 1
fi

print_status $BLUE "Starting experiment monitoring..."
print_status $BLUE "Press Ctrl+C to stop monitoring"

while true; do
    # Get current status
    local total_experiments=$(python3 -c "
import json
with open('$MANIFEST_FILE', 'r') as f:
    manifest = json.load(f)
print(len(manifest['experiments']))
")
    
    local completed_jobs=$(python3 -c "
import json
with open('$MANIFEST_FILE', 'r') as f:
    manifest = json.load(f)
completed = sum(1 for exp in manifest['experiments'] if exp.get('status') == 'completed')
print(completed)
")
    
    local running_jobs=$(python3 -c "
import json
with open('$MANIFEST_FILE', 'r') as f:
    manifest = json.load(f)
running = sum(1 for exp in manifest['experiments'] if exp.get('status') == 'running')
print(running)
")
    
    local failed_jobs=$(python3 -c "
import json
with open('$MANIFEST_FILE', 'r') as f:
    manifest = json.load(f)
failed = sum(1 for exp in manifest['experiments'] if exp.get('status') == 'failed')
print(failed)
")
    
    local pending_jobs=$(python3 -c "
import json
with open('$MANIFEST_FILE', 'r') as f:
    manifest = json.load(f)
pending = sum(1 for exp in manifest['experiments'] if exp.get('status') == 'pending')
print(pending)
")
    
    # Clear screen and show status
    clear
    print_status $BLUE "=== EEG Experiment Status Monitor ==="
    echo
    print_status $GREEN "Total experiments: $total_experiments"
    print_status $GREEN "Completed: $completed_jobs"
    print_status $YELLOW "Running: $running_jobs"
    print_status $RED "Failed: $failed_jobs"
    print_status $BLUE "Pending: $pending_jobs"
    
    if [[ $total_experiments -gt 0 ]]; then
        local progress=$((completed_jobs * 100 / total_experiments))
        echo
        print_status $BLUE "Progress: $progress% complete"
        
        # Progress bar
        local bar_length=50
        local filled=$((progress * bar_length / 100))
        local empty=$((bar_length - filled))
        printf "Progress: ["
        printf "%${filled}s" | tr ' ' '#'
        printf "%${empty}s" | tr ' ' '-'
        printf "] $progress%%\n"
    fi
    
    echo
    print_status $BLUE "Next update in $CHECK_INTERVAL seconds..."
    
    sleep $CHECK_INTERVAL
done
EOF

    chmod +x "$script_name"
    print_status $GREEN "Created monitoring script: $script_name"
}

# Main execution
main() {
    print_status $BLUE "=== EEG Experiment Cluster Submission ==="
    print_status $BLUE "Using manifest: $MANIFEST_FILE"
    
    # Check manifest
    local total_experiments=$(check_manifest)
    
    # Analyze manifest
    local config=$(analyze_manifest $total_experiments)
    local array_size=$(echo $config | cut -d' ' -f1)
    local max_concurrent=$(echo $config | cut -d' ' -f2)
    local num_array_jobs=$(echo $config | cut -d' ' -f3)
    
    # Create monitoring script
    create_monitoring_script
    
    # Submit array jobs
    local submitted_count=$(submit_array_jobs $total_experiments $array_size $max_concurrent $num_array_jobs)
    
    print_status $GREEN "=== Submission Complete ==="
    print_status $GREEN "Successfully submitted $submitted_count array jobs"
    print_status $GREEN "Total experiments: $total_experiments"
    print_status $GREEN "Max concurrent jobs: $max_concurrent"
    echo
    print_status $BLUE "To monitor progress, run:"
    print_status $BLUE "  ./monitor_experiments.sh $MANIFEST_FILE"
    echo
    print_status $BLUE "To check SLURM job status:"
    print_status $BLUE "  squeue -u \$USER"
}

# Run main function
main "$@"
