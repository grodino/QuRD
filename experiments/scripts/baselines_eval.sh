#!/bin/bash
# Generate the queries for each fignerprint
# Author(s): Anonymous
#

# Print out all the commands
set -x

# NOTE: change this to point to the directory where you want the data, models
# and generated results to be.
SCRATCH=./

# Read script arguments (benchmark name and dataset)
benchmark=${1}
dataset=${2}
split=${3}
seed=${4:-123456789}

# Setup directories
GENERATED_DIR=$SCRATCH/Maurice/generated/$seed/$benchmark/
MODELS_DIR=$SCRATCH/Maurice/models/$benchmark/
DATA_DIR=$SCRATCH/Maurice/data/

mkdir -p $GENERATED_DIR

# Prevent huggingface from trying to reach the network
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# SETUP Huggingface model hub cache directory
export HF_HOME=$MODELS_DIR

# Setup the folders config passed to the script
config="--models-dir=$MODELS_DIR --data-dir=$DATA_DIR --generated-dir=$GENERATED_DIR --batch-size=64 --seed=$seed"

# for budget in 10 20 50 100 200 400 600; do
for budget in 10 20 50 100 200 400; do
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
