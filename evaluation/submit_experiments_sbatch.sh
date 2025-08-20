#!/bin/bash
#SBATCH --job-name=EEG_Experiments
#SBATCH --output=logs/sbatch_%j.out
#SBATCH --error=logs/sbatch_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@institution.edu

# EEG Experiment Cluster Submission Script
# This script processes an experiment manifest and submits jobs efficiently
# to the SLURM cluster using job arrays and dependency management.

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MANIFEST_FILE="${1:-experiment_manifest.json}"
MAX_CONCURRENT_JOBS=50  # Maximum jobs to submit simultaneously
BATCH_SIZE=100          # Submit jobs in batches
DELAY_BETWEEN_BATCHES=30  # Seconds to wait between batches

# Create necessary directories
mkdir -p logs
mkdir -p job_status
mkdir -p completed_experiments

# Function to log messages with timestamps
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a logs/submission.log
}

# Function to check if manifest file exists
check_manifest() {
    if [[ ! -f "$MANIFEST_FILE" ]]; then
        log_message "ERROR: Manifest file '$MANIFEST_FILE' not found!"
        log_message "Usage: sbatch $0 <manifest_file.json>"
        exit 1
    fi
    
    log_message "Using manifest file: $MANIFEST_FILE"
    
    # Validate JSON format
    if ! python3 -c "import json; json.load(open('$MANIFEST_FILE'))" 2>/dev/null; then
        log_message "ERROR: Invalid JSON format in manifest file!"
        exit 1
    fi
}

# Function to extract experiment parameters from manifest
extract_experiment_params() {
    local experiment_json="$1"
    
    # Use Python to safely extract parameters
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

# Function to create individual experiment sbatch script
create_experiment_sbatch() {
    local experiment_id="$1"
    local experiment_params="$2"
    local output_dir="$3"
    
    cat > "job_status/${experiment_id}.sbatch" << EOF
#!/bin/bash
#SBATCH --job-name=${experiment_id}
#SBATCH --output=${output_dir}/logs/${experiment_id}_%j.out
#SBATCH --error=${output_dir}/logs/${experiment_id}_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@institution.edu

# Individual experiment job script
set -e

# Activate conda environment
source ~/.bashrc
conda activate ncp_robustness_proj

# Set working directory
cd "$PROJECT_ROOT"

# Create output directory
mkdir -p "${output_dir}/logs"

# Run the experiment
echo "Starting experiment: ${experiment_id}"
echo "Parameters: ${experiment_params}"
echo "Started at: \$(date)"

# Use the user's preferred Python executable
python evaluation/unified_experiment_runner.py ${experiment_params}

echo "Completed experiment: ${experiment_id}"
echo "Finished at: \$(date)"

# Mark as completed
echo "\$(date)" > "${output_dir}/completed_experiments/${experiment_id}.completed"
EOF

    chmod +x "job_status/${experiment_id}.sbatch"
}

# Function to submit a batch of jobs
submit_batch() {
    local batch_experiments=("$@")
    local batch_size=${#batch_experiments[@]}
    
    log_message "Submitting batch of $batch_size experiments..."
    
    local submitted_jobs=()
    
    for experiment in "${batch_experiments[@]}"; do
        local experiment_id=$(echo "$experiment" | python3 -c "import json, sys; print(json.load(sys.stdin)['experiment_id'])")
        local experiment_params=$(extract_experiment_params "$experiment")
        
        # Create output directory for this experiment
        local output_dir="outputs/experiments/${experiment_id}"
        mkdir -p "$output_dir/logs"
        mkdir -p "$output_dir/completed_experiments"
        
        # Create individual sbatch script
        create_experiment_sbatch "$experiment_id" "$experiment_params" "$output_dir"
        
        # Submit the job
        local job_id=$(sbatch "job_status/${experiment_id}.sbatch" | grep -o '[0-9]\+')
        
        if [[ -n "$job_id" ]]; then
            submitted_jobs+=("$job_id")
            log_message "Submitted experiment $experiment_id as job $job_id"
            
            # Update manifest with job ID
            python3 -c "
import json
with open('$MANIFEST_FILE', 'r') as f:
    manifest = json.load(f)

for exp in manifest['experiments']:
    if exp['experiment_id'] == '$experiment_id':
        exp['slurm_job_id'] = $job_id
        exp['status'] = 'submitted'
        exp['submitted_at'] = '$(date -Iseconds)'
        break

with open('$MANIFEST_FILE', 'w') as f:
    json.dump(manifest, f, indent=2)
"
        else
            log_message "ERROR: Failed to submit experiment $experiment_id"
        fi
        
        # Small delay between submissions to avoid overwhelming the scheduler
        sleep 1
    done
    
    log_message "Batch submission complete. Submitted $batch_size jobs."
    return ${#submitted_jobs[@]}
}

# Function to monitor job progress
monitor_progress() {
    log_message "Starting job monitoring..."
    
    while true; do
        local total_jobs=$(python3 -c "
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
        
        log_message "Progress: $completed_jobs/$total_jobs completed, $running_jobs running, $failed_jobs failed"
        
        # Check for completed jobs
        python3 -c "
import json
import os
import subprocess

with open('$MANIFEST_FILE', 'r') as f:
    manifest = json.load(f)

for exp in manifest['experiments']:
    if exp.get('status') == 'submitted':
        job_id = exp.get('slurm_job_id')
        if job_id:
            # Check job status
            try:
                result = subprocess.run(['squeue', '-j', str(job_id)], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode != 0 or 'slurm_load_jobs error' in result.stderr:
                    # Job completed or failed
                    exp['status'] = 'completed'
                    exp['completed_at'] = '$(date -Iseconds)'
            except:
                # Job completed
                exp['status'] = 'completed'
                exp['completed_at'] = '$(date -Iseconds)'

with open('$MANIFEST_FILE', 'w') as f:
    json.dump(manifest, f, indent=2)
"
        
        # If all jobs are completed, break
        if [[ $completed_jobs -eq $total_jobs ]]; then
            log_message "All experiments completed!"
            break
        fi
        
        # Wait before next check
        sleep 60
    done
}

# Function to generate summary report
generate_summary() {
    log_message "Generating final summary report..."
    
    python3 -c "
import json
from datetime import datetime

with open('$MANIFEST_FILE', 'r') as f:
    manifest = json.load(f)

# Count by status
status_counts = {}
for exp in manifest['experiments']:
    status = exp.get('status', 'unknown')
    status_counts[status] = status_counts.get(status, 0) + 1

# Count by mode
mode_counts = {}
for exp in manifest['experiments']:
    mode = exp.get('mode', 'unknown')
    mode_counts[mode] = mode_counts.get(mode, 0) + 1

# Generate report
report = f'''# Experiment Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Statistics
Total experiments: {len(manifest['experiments'])}

## Status Summary
'''
for status, count in sorted(status_counts.items()):
    report += f'{status}: {count}\n'

report += '\n## Mode Summary\n'
for mode, count in sorted(mode_counts.items()):
    report += f'{mode}: {count}\n'

# Save report
with open('experiment_summary_report.md', 'w') as f:
    f.write(report)

print('Summary report generated: experiment_summary_report.md')
"
}

# Main execution
main() {
    log_message "Starting EEG experiment cluster submission"
    
    # Check manifest file
    check_manifest
    
    # Load and parse manifest
    log_message "Loading experiment manifest..."
    local total_experiments=$(python3 -c "
import json
with open('$MANIFEST_FILE', 'r') as f:
    manifest = json.load(f)
print(len(manifest['experiments']))
")
    
    log_message "Total experiments to submit: $total_experiments"
    
    # Sort experiments by priority
    log_message "Sorting experiments by priority..."
    python3 -c "
import json
with open('$MANIFEST_FILE', 'r') as f:
    manifest = json.load(f)

# Sort by priority (lower number = higher priority)
manifest['experiments'].sort(key=lambda x: x.get('priority', 999))

with open('$MANIFEST_FILE', 'w') as f:
    json.dump(manifest, f, indent=2)
"
    
    # Submit experiments in batches
    local batch_start=0
    local batch_end=0
    
    while [[ $batch_end -lt $total_experiments ]]; do
        batch_end=$((batch_start + BATCH_SIZE))
        if [[ $batch_end -gt $total_experiments ]]; then
            batch_end=$total_experiments
        fi
        
        # Extract batch of experiments
        local batch_experiments=$(python3 -c "
import json
with open('$MANIFEST_FILE', 'r') as f:
    manifest = json.load(f)

batch = manifest['experiments'][$batch_start:$batch_end]
for exp in batch:
    print(json.dumps(exp))
")
        
        # Submit batch
        submit_batch $batch_experiments
        
        # Update batch start
        batch_start=$batch_end
        
        # Wait before next batch (unless this is the last batch)
        if [[ $batch_end -lt $total_experiments ]]; then
            log_message "Waiting $DELAY_BETWEEN_BATCHES seconds before next batch..."
            sleep $DELAY_BETWEEN_BATCHES
        fi
    done
    
    log_message "All experiment batches submitted successfully!"
    
    # Start monitoring (optional - can be run separately)
    if [[ "${2:-}" == "--monitor" ]]; then
        monitor_progress
        generate_summary
    else
        log_message "Use 'sbatch $0 $MANIFEST_FILE --monitor' to monitor progress"
    fi
    
    log_message "Submission script completed successfully!"
}

# Run main function with all arguments
main "$@"


