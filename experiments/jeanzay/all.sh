# srun --pty -p prepost --hint=nomultithread -A ggl@v100 bash
# srun --pty --gres=gpu:1 --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --constraint=v100-16g --hint=nomultithread --time=1:00:00 -A ggl@v100 bash
# sinfo -p gpu_p13 -o"%P %.16F"

# Download data and verify that models are here. 
# You need to provide the Stranford-dogs dataset and the model weights in the
# right folder
sbatch experiments/jeanzay/prepare.sh ModelReuse SDog120
sbatch experiments/jeanzay/prepare.sh ModelReuse Flower102
sbatch experiments/jeanzay/prepare.sh SACBenchmark CIFAR10
sbatch experiments/jeanzay/prepare.sh FBIBenchmark imagenet-1k

# Evaluate the benchmarks
sbatch experiments/jeanzay/bench_eval.sh ModelReuse SDog120
sbatch experiments/jeanzay/bench_eval.sh ModelReuse Flower102
sbatch experiments/jeanzay/bench_eval.sh SACBenchmark CIFAR10
sbatch experiments/jeanzay/bench_eval.sh FBIBenchmark imagenet-1k


# Evaluate the baseline methods
sbatch experiments/jeanzay/baselines_eval.sh ModelReuse SDog120 test
sbatch experiments/jeanzay/baselines_eval.sh ModelReuse Flower102 test
sbatch experiments/jeanzay/baselines_eval.sh SACBenchmark CIFAR10 test
sbatch experiments/jeanzay/baselines_eval.sh FBIBenchmark imagenet-1k test

sbatch experiments/jeanzay/baselines_eval.sh ModelReuse SDog120 train
sbatch experiments/jeanzay/baselines_eval.sh ModelReuse Flower102 train
sbatch experiments/jeanzay/baselines_eval.sh SACBenchmark CIFAR10 train
sbatch experiments/jeanzay/baselines_eval.sh FBIBenchmark imagenet-1k train


# Evaluate all the other methods
sbatch experiments/jeanzay/distances_eval.sh ModelReuse SDog120 test
sbatch experiments/jeanzay/distances_eval.sh ModelReuse Flower102 test
sbatch experiments/jeanzay/distances_eval.sh SACBenchmark CIFAR10 test
sbatch experiments/jeanzay/distances_eval.sh FBIBenchmark imagenet-1k test

sbatch experiments/jeanzay/distances_eval.sh ModelReuse SDog120 train
sbatch experiments/jeanzay/distances_eval.sh ModelReuse Flower102 train
sbatch experiments/jeanzay/distances_eval.sh SACBenchmark CIFAR10 train
sbatch experiments/jeanzay/distances_eval.sh FBIBenchmark imagenet-1k train