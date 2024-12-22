#!/bin/bash
#
# Download datasets (and models if possible) for the corresponding benchmark and
# setup the directories. 
#
# These scripts are meant to be run on Jean-Zay GPU cluster using SLURM. To run
# this locally, you must
#   - set the $SCRATCH variable to the directory you want all the experiments to
#     be saved to
#   - replace all the `srun python ...` instructions by `python ...` 
#
#
#SBATCH --job-name=BenchPrepare             # Job name
#SBATCH --output=logs/BenchPrepare.%j.out   # logs (%j = job ID)
#SBATCH --error=logs/BenchPrepare.%j.err    # logs (%j = job ID)
#SBATCH --partition=prepost                 # preprocessing partition with internet access
#SBATCH --hint=nomultithread                # no hyperthreading
#SBATCH --account=ggl@v100                  # V100 account

source ~/.bashrc

# Read script arguments (benchmark name and dataset)
benchmark=${1}

# Setup directories
export QURD=$SCRATCH/QuRD
export GENERATED_DIR=$QURD/generated/
export MODELS_DIR=$QURD/models/${benchmark}/
export DATA_DIR=$QURD/data/

# SETUP Huggingface model hub cache directory
# export HF_HUB_CACHE=$MODELS_DIR
export HF_HOME=$MODELS_DIR
# export HF_HUB_OFFLINE=1

# Print out all the commands
set -x

# Cleanup environment
module purge
conda deactivate

# Create the pixi cache folder in $SCRATCH and link it here (to bypass $HOME
# size limitations)
mkdir -p $QURD/.pixi
ln -s $QURD/.pixi .pixi

# Create the cache folder in $SCRATCH and link it here (to bypass $HOME size
# limitations)
mkdir -p $SCRATCH/.cache/
ln -s $SCRATCH/.cache/ ~/.cache

# Install pixi and the environment
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc
pixi install -e cuda

# Make the directories (Ok if they already exists)
mkdir -pv $GENERATED_DIR
mkdir -pv $MODELS_DIR
mkdir -pv $DATA_DIR

# Make sure that the script sees the GPU
# CUDA_VISIBLE_DEVICES=0

# Prepare the benchmark
srun python main.py prepare --benchmark $benchmark