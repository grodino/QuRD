
for seed in 1315320158 1391606125 150704254 2614738243 1968368282; do
    # Evaluate the baseline methods
    sbatch experiments/jeanzay/baselines_eval.sh ModelReuse SDog120 test $seed
    sbatch experiments/jeanzay/baselines_eval.sh ModelReuse Flower102 test $seed
    sbatch experiments/jeanzay/baselines_eval.sh SACBenchmark CIFAR10 test $seed
    # sbatch experiments/jeanzay/baselines_eval.sh FBIBenchmark imagenet-1k test $seed

    sbatch experiments/jeanzay/baselines_eval.sh ModelReuse SDog120 train $seed
    sbatch experiments/jeanzay/baselines_eval.sh ModelReuse Flower102 train $seed
    sbatch experiments/jeanzay/baselines_eval.sh SACBenchmark CIFAR10 train $seed
    # sbatch experiments/jeanzay/baselines_eval.sh FBIBenchmark imagenet-1k train $seed


    # # Evaluate all the other methods
    # sbatch experiments/jeanzay/distances_eval.sh ModelReuse SDog120 test $seed
    # sbatch experiments/jeanzay/distances_eval.sh ModelReuse Flower102 test $seed
    # sbatch experiments/jeanzay/distances_eval.sh SACBenchmark CIFAR10 test $seed
    # # sbatch experiments/jeanzay/distances_eval.sh FBIBenchmark imagenet-1k test $seed

    # sbatch experiments/jeanzay/distances_eval.sh ModelReuse SDog120 train $seed
    # sbatch experiments/jeanzay/distances_eval.sh ModelReuse Flower102 train $seed
    # sbatch experiments/jeanzay/distances_eval.sh SACBenchmark CIFAR10 train $seed
    # # sbatch experiments/jeanzay/distances_eval.sh FBIBenchmark imagenet-1k train $seed
done