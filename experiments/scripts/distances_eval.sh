#!/bin/bash
# Once all the fingerprint queries have been generated, use them with the
# different representations
# Author(s): Anonymous
#

# Print out all the commands
set -x

# NOTE: change this to point to the directory where you want the data, models
# and generated results to be.
SCRATCH=./

# Read script arguments (benchmark name, dataset, split and seed)
benchmark=${1}
dataset=${2}
split=${3}
seed=${4:-123456789}

# Setup directories
GENERATED_DIR=$SCRATCH/generated/$seed/$benchmark/
MODELS_DIR=$SCRATCH/models/${benchmark}/
DATA_DIR=$SCRATCH/data/

# Prevent huggingface from trying to reach the network
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# SETUP Huggingface model hub cache directory
export HF_HOME=$MODELS_DIR

# Setup the folders config passed to the script
config="--models-dir=$MODELS_DIR --data-dir=$DATA_DIR --generated-dir=$GENERATED_DIR --batch-size=64 --seed=$seed"


for budget in 10 20 50 100 200 400
do
    for SAMPLER in "Random" "Negative" "Boundary" "NegativeAugment" "Adversarial" "NegativeAdversarial" "RandomSubsample10" "RandomSubsample100" "NegativeSubsample10" "NegativeSubsample100"
    do  
        # Skip invalid configurations
        if [ "$SAMPLER" = "RandomSubsample100" ] && [ 100 -gt $budget ]; then continue; fi
        if [ "$SAMPLER" = "NegativeSubsample100" ] && [ 100 -gt $budget ]; then continue; fi

        # SAC on hard labels
        python main.py ${config} generate --benchmark $benchmark --dataset=${dataset} --split=${split} --budget=${budget} --sampler=$SAMPLER --representation="HardSAC"
        python main.py ${config} pair-distance --benchmark $benchmark --dataset=${dataset} --split=${split} --budget=${budget} --sampler=$SAMPLER --representation="HardSAC" --distance="l2"

        # SAC on soft labels
        python main.py ${config} generate --benchmark $benchmark --dataset=${dataset} --split=${split} --budget=${budget} --sampler=$SAMPLER --representation="SAC"
        python main.py ${config} pair-distance --benchmark $benchmark --dataset=${dataset} --split=${split} --budget=${budget} --sampler=$SAMPLER --representation="SAC" --distance="l2"
    done

    # DDV on hard labels
    python main.py ${config} generate --benchmark $benchmark --dataset=${dataset} --split=${split} --budget=${budget} --sampler="Adversarial" --representation="HardDDV"
    python main.py ${config} pair-distance --benchmark $benchmark --dataset=${dataset} --split=${split} --budget=${budget} --sampler="Adversarial" --representation="HardDDV" --distance="hamming"
    python main.py ${config} generate --benchmark $benchmark --dataset=${dataset} --split=${split} --budget=${budget} --sampler="NegativeAdversarial" --representation="HardDDV"
    python main.py ${config} pair-distance --benchmark $benchmark --dataset=${dataset} --split=${split} --budget=${budget} --sampler="NegativeAdversarial" --representation="HardDDV" --distance="hamming"

    # DDV on soft labels
    python main.py ${config} generate --benchmark $benchmark --dataset=${dataset} --split=${split} --budget=${budget} --sampler="Adversarial" --representation="DDV"
    python main.py ${config} pair-distance --benchmark $benchmark --dataset=${dataset} --split=${split} --budget=${budget} --sampler="Adversarial" --representation="DDV" --distance="cosine"
    python main.py ${config} generate --benchmark $benchmark --dataset=${dataset} --split=${split} --budget=${budget} --sampler="NegativeAdversarial" --representation="DDV"
    python main.py ${config} pair-distance --benchmark $benchmark --dataset=${dataset} --split=${split} --budget=${budget} --sampler="NegativeAdversarial" --representation="DDV" --distance="cosine"

done


# ZestOfLIME on large budgets
for budget in 100 200 400
do
    # ZLIME on random data
    python main.py ${config} generate --benchmark $benchmark --dataset=${dataset} --split=${split} --budget=${budget} --sampler="RandomSubsample100" --representation="ZLIME"
    python main.py ${config} pair-distance --benchmark $benchmark --dataset=${dataset} --split=${split} --budget=${budget} --sampler="RandomSubsample100" --representation="ZLIME" --distance="l2"

    # ZLIME on negative data
    python main.py ${config} generate --benchmark $benchmark --dataset=${dataset} --split=${split} --budget=${budget} --sampler="NegativeSubsample100" --representation="ZLIME"
    python main.py ${config} pair-distance --benchmark $benchmark --dataset=${dataset} --split=${split} --budget=${budget} --sampler="NegativeSubsample100" --representation="ZLIME" --distance="l2"
done

# ZestOfLIME on small budgets
for budget in 10 20 50 100 200 400
do
    # ZLIME on random data
    python main.py ${config} generate --benchmark $benchmark --dataset=${dataset} --split=${split} --budget=${budget} --sampler="RandomSubsample10" --representation="ZLIME"
    python main.py ${config} pair-distance --benchmark $benchmark --dataset=${dataset} --split=${split} --budget=${budget} --sampler="RandomSubsample10" --representation="ZLIME" --distance="l2"

    # ZLIME on negative data
    python main.py ${config} generate --benchmark $benchmark --dataset=${dataset} --split=${split} --budget=${budget} --sampler="NegativeSubsample10" --representation="ZLIME"
    python main.py ${config} pair-distance --benchmark $benchmark --dataset=${dataset} --split=${split} --budget=${budget} --sampler="NegativeSubsample10" --representation="ZLIME" --distance="l2"
done
