#!/bin/bash
#SBATCH --job-name=BenchEval                # Job name
#SBATCH --output=logs/BenchEval.%j.out      # logs (%j = job ID)
#SBATCH --error=logs/BenchEval.%j.err       # logs (%j = job ID)
#SBATCH --hint=nomultithread                # no hyperthreading
#SBATCH --account=ggl@v100                  # V100 account
#SBATCH --constraint=v100-16g               # demander des GPU a 16 Go de RAM
#SBATCH --nodes=1                           # allocate 1 node (one machine)
#SBATCH --ntasks-per-node=1                 # allocate 1 task (one processus)
#SBATCH --gres=gpu:1                        # allocate 1 GPU
#SBATCH --cpus-per-task=10                  # allocate 10 CPU per task
#SBATCH --qos=qos_gpu-t3                    # Default QoS, allows 20h
#SBATCH --time=20:00:00                     # Max job duration


# Cleanup environment
module purge
conda deactivate

# Load necessary modules
# module load git/2.39.1
# module load pytorch-gpu/py3/2.2.0

# Load necessary modules
module load git/2.39.1
# module load pytorch-gpu/py3/2.2.0
module load python/3.11.5
module load cuda/12.1.0

# Print out all the commands
set -x

# Read script arguments (benchmark name and dataset)
benchmark=${1}
dataset=${2}

# Setup directories
GENERATED_DIR=$SCRATCH/Maurice/generated/${benchmark}/
MODELS_DIR=$SCRATCH/Maurice/models/${benchmark}/
DATA_DIR=$SCRATCH/Maurice/data/

# Prevent huggingface from trying to reach the network
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# SETUP Huggingface model hub cache directory
# export HF_HUB_CACHE=$MODELS_DIR
export HF_HOME=$MODELS_DIR

# Setup the folders config passed to the script
config="--models-dir=$MODELS_DIR --data-dir=$DATA_DIR --generated-dir=$GENERATED_DIR --batch-size=128"

# Evaluate the models 
# srun python -u main.py $config accuracy --benchmark $benchmark --dataset $dataset

# Evaluate the distance between models
srun python main.py $config oracle-distance --benchmark $benchmark --dataset $dataset --split "test"
srun python main.py $config oracle-distance --benchmark $benchmark --dataset $dataset --split "train"