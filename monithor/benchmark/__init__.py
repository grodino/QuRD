from pathlib import Path

from .base import Benchmark
from .sac import SACBenchmark
from .model_reuse import ModelReuse


def get_benchmark(
    name: str, data_dir: Path, models_dir: Path, device: str
) -> Benchmark:
    match name:
        case "SACBenchmark":
            bench = SACBenchmark(
                data_dir=data_dir,
                models_dir=models_dir,
                device=device,
            )

        case "ModelReuse":
            bench = ModelReuse(
                data_dir=data_dir,
                models_dir=models_dir,
                device=device,
            )

        case _:
            raise NotImplementedError()

    return bench
