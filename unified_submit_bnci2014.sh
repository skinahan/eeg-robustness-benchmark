#!/bin/bash
#SBATCH -p general 
#SBATCH -q public
#SBATCH -G 1
#SBATCH -c 1
#SBATCH -t 0-17:30:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skinahan@asu.edu
#SBATCH --export=NONE

SUBJECT=$1
EVAL_MODE=$2

module load mamba/latest
source activate ncp_env

python evaluation/unified_experiment_runner.py --model eegnet --dataset BNCI2014_001 --mode multirun --seed 100 --subjects $SUBJECT --noise_type gaussian --intensity 10.0 --eval_mode $EVAL_MODE --overwrite
