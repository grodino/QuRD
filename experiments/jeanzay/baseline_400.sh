#!/bin/bash
#SBATCH --job-name=Baselines400             # Job name
#SBATCH --output=logs/BaselinesEval.%j.out  # logs (%j = job ID)
#SBATCH --error=logs/BaselinesEval.%j.err   # logs (%j = job ID)
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
module load git/2.39.1
# module load pytorch-gpu/py3/2.2.0
module load python/3.11.5
module load cuda/12.1.0

# Print out all the commands
set -x

# Read script arguments (benchmark name and dataset)
benchmark=${1}
dataset=${2}
split=${3}

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
config="--models-dir=$MODELS_DIR --data-dir=$DATA_DIR --generated-dir=$GENERATED_DIR --batch-size=64"

# for budget in 10 20 50 100 200 400 600; do
for budget in 400; do
    for SAMPLER in "Random" "Negative" "NegativeAugment" "Adversarial" "NegativeAdversarial" "Boundary" "RandomSubsample10" "RandomSubsample100" "NegativeSubsample10" "NegativeSubsample100"; do  
        # Skip invalid configurations
        if [ "$SAMPLER" = "RandomSubsample100" ] && [ 100 -gt $budget ]; then continue; fi
        if [ "$SAMPLER" = "NegativeSubsample100" ] && [ 100 -gt $budget ]; then continue; fi

        # Sample the queries
        python main.py ${config} sample --benchmark=${benchmark} --dataset=${dataset} --split=${split} --budget=${budget} --sampler=$SAMPLER

        # Baseline distance on hard labels
        python main.py ${config} generate --benchmark=${benchmark} --dataset=${dataset} --split=${split} --budget=${budget} --sampler=$SAMPLER --representation="Labels"
        python main.py ${config} pair-distance --benchmark=${benchmark} --dataset=${dataset} --split=${split} --budget=${budget} --sampler=$SAMPLER --representation="Labels" --distance="hamming"

        # Baseline distance on soft labels
        python main.py ${config} generate --benchmark=${benchmark} --dataset=${dataset} --split=${split} --budget=${budget} --sampler=$SAMPLER --representation="Logits"
        python main.py ${config} pair-distance --benchmark=${benchmark} --dataset=${dataset} --split=${split} --budget=${budget} --sampler=$SAMPLER --representation="Logits" --distance="cosine"
    done
done
