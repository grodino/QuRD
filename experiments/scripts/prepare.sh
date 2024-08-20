#!/bin/bash
# Computes metrics used to compare model fingerprinting benchmarks
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

# Setup directories
GENERATED_DIR=$SCRATCH/Maurice/generated/${benchmark}/
MODELS_DIR=$SCRATCH/Maurice/models/${benchmark}/
DATA_DIR=$SCRATCH/Maurice/data/

# Prevent huggingface from trying to reach the network
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# SETUP Huggingface model hub cache directory
export HF_HOME=$MODELS_DIR

# Setup the folders config passed to the script
config="--models-dir=$MODELS_DIR --data-dir=$DATA_DIR --generated-dir=$GENERATED_DIR --batch-size=128"

# Evaluate the models 
# srun python -u main.py $config accuracy --benchmark $benchmark --dataset $dataset

# Evaluate the distance between models
srun python main.py $config oracle-distance --benchmark $benchmark --dataset $dataset --split "test"
srun python main.py $config oracle-distance --benchmark $benchmark --dataset $dataset --split "train"