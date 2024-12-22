from itertools import product
from subprocess import run
import json
from pathlib import Path
from datetime import datetime

import numpy as np

ENTROPY = 123456789
BENCHMARKS = {
    # "ModelReuse": ["SDog120", "Flower102"],
    "ModelReuse": ["Flower102"],
    # "SACBenchmark": ["CIFAR10"],
}
SPLITS = ["test", "train"]


def hold(command: list[str]) -> list[str]:
    # command.insert(1, "--hold")
    # command.insert(1, "--time")
    # command.insert(2, "10:00:00")
    return command


def dependencies(command: list[str], job_ids: list[str] | str) -> list[str]:
    if not isinstance(job_ids, list):
        job_ids = [job_ids]
    elif len(job_ids) == 0:
        return command

    deps_str = f"afterany:{','.join(job_ids)}"
    # deps_str = f"afterok:{','.join(job_ids)}"
    command.insert(1, "--dependency")
    command.insert(2, deps_str)

    return command


if __name__ == "__main__":
    seeds = list(map(str, np.random.SeedSequence(ENTROPY).generate_state(5)))
    print("SEEDS", seeds)

    now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    exp_dir = Path("./experiments/jeanzay/runs/") / now
    exp_dir.mkdir(parents=True, exist_ok=False)
    log_dir = exp_dir / "logs"
    log_dir.mkdir()

    job_ids = {}
    commands = []

    # Make sure all the benchmarks are prepared
    logs = [
        "--output",
        str(log_dir / "BenchPrepare.%j.out"),
        "--error",
        str(log_dir / "BenchPrepare.%j.err"),
    ]

    for benchmark, datasets in BENCHMARKS.items():
        job_ids[benchmark] = {}

        for dataset in datasets:
            command = hold(
                [
                    "sbatch",
                    "--parsable",
                    *logs,
                    "experiments/jeanzay/prepare.sh",
                    benchmark,
                    dataset,
                ]
            )
            commands.append(" ".join(command))

            job_ids[benchmark][dataset] = {
                "prepare": run(command, capture_output=True, check=True)
                .stdout.decode("ascii")
                .strip()
            }

    # Run the baselines
    logs = [
        "--output",
        str(log_dir / "BaselinesEval.%j.out"),
        "--error",
        str(log_dir / "BaselinesEval.%j.err"),
    ]

    for benchmark, datasets in BENCHMARKS.items():
        for dataset in datasets:
            job_ids[benchmark][dataset]["baselines"] = {split: {} for split in SPLITS}

            for split, seed in product(SPLITS, seeds):
                command = hold(
                    dependencies(
                        [
                            "sbatch",
                            "--parsable",
                            *logs,
                            "experiments/jeanzay/baselines_eval.sh",
                            benchmark,
                            dataset,
                            split,
                            seed,
                        ],
                        job_ids[benchmark][dataset]["prepare"],
                    )
                )
                commands.append(" ".join(command))

                job_ids[benchmark][dataset]["baselines"][split][seed] = (
                    run(command, capture_output=True, check=True)
                    .stdout.decode("ascii")
                    .strip()
                )

    # Run all the distances
    logs = [
        "--output",
        str(log_dir / "DistancesEval.%j.out"),
        "--error",
        str(log_dir / "DistancesEval.%j.err"),
    ]

    for benchmark, datasets in BENCHMARKS.items():
        for dataset in datasets:
            job_ids[benchmark][dataset]["distances"] = {split: {} for split in SPLITS}

            for split, seed in product(SPLITS, seeds):
                command = hold(
                    dependencies(
                        [
                            "sbatch",
                            "--parsable",
                            *logs,
                            "experiments/jeanzay/distances_eval.sh",
                            benchmark,
                            dataset,
                            split,
                            seed,
                        ],
                        job_ids[benchmark][dataset]["baselines"][split][seed],
                    )
                )
                commands.append(" ".join(command))

                job_ids[benchmark][dataset]["distances"][split][seed] = (
                    run(command, capture_output=True, check=True)
                    .stdout.decode("ascii")
                    .strip()
                )

    with open(exp_dir / "job_ids.json", "w") as file:
        json.dump(job_ids, file)

    with open(exp_dir / "seeds.csv", "w") as file:
        file.write("\n".join(seeds))

    with open(exp_dir / "commands.csv", "w") as file:
        file.write("\n".join(commands))
