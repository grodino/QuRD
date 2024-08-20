"""Generate a bash file with all the commands to run to reproduce the paper

The resulting file is save in LOGS_DIR/commands.sh
"""

from itertools import product
from pathlib import Path
from datetime import datetime

import numpy as np

ENTROPY = 123456789
BENCHMARKS = {
    "ModelReuse": ["SDog120", "Flower102"],
    "SACBenchmark": ["CIFAR10"],
}
SPLITS = ["test", "train"]

# NOTE: change this to the dir where you want to save your logs
LOGS_DIR = Path("./")


if __name__ == "__main__":
    seeds = list(map(str, np.random.SeedSequence(ENTROPY).generate_state(5)))

    now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    exp_dir = LOGS_DIR / now
    exp_dir.mkdir(parents=True, exist_ok=False)

    commands = []

    # Run the commands to download models/data if possible or else verify that
    # the required files are here.
    for benchmark, datasets in BENCHMARKS.items():
        for dataset in datasets:
            command = [
                "experiments/scripts/prepare.sh",
                benchmark,
                dataset,
                "&>",
                str(LOGS_DIR / "prepare" / f"{benchmark}-{dataset}.log"),
            ]

    # Compute the benchmark comparison metrics
    for benchmark, datasets in BENCHMARKS.items():
        for dataset in datasets:
            command = [
                "experiments/scripts/bench_eval.sh",
                benchmark,
                dataset,
                "&>",
                str(LOGS_DIR / "bench_eval" / f"{benchmark}-{dataset}.log"),
            ]

    # Run the baselines
    for benchmark, datasets in BENCHMARKS.items():
        for dataset in datasets:

            for split, seed in product(SPLITS, seeds):
                command = [
                    "experiments/scripts/baselines_eval.sh",
                    benchmark,
                    dataset,
                    split,
                    seed,
                    "&>",
                    str(
                        LOGS_DIR
                        / "baselines"
                        / f"{benchmark}-{dataset}-{split}-{seed}.log"
                    ),
                ]

                commands.append(" ".join(command))

            commands.append("")

    commands.append("\n")

    # Run all the distances
    for benchmark, datasets in BENCHMARKS.items():
        for dataset in datasets:
            for split, seed in product(SPLITS, seeds):
                command = [
                    "experiments/scripts/distances_eval.sh",
                    benchmark,
                    dataset,
                    split,
                    seed,
                    "&>",
                    str(
                        LOGS_DIR
                        / "distances"
                        / f"{benchmark}-{dataset}-{split}-{seed}.log"
                    ),
                ]
                commands.append(" ".join(command))

            commands.append("")

    with open(exp_dir / "seeds.csv", "w") as file:
        file.write("\n".join(seeds))

    with open(exp_dir / "commands.sh", "w") as file:
        file.write("#!/bin/bash\n")
        file.write("\n".join(commands))
