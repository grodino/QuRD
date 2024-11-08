from pathlib import Path

from .base import Benchmark
from .sac import SACBenchmark
from .model_reuse import ModelReuse


def get_benchmark(
    name: str,
    data_dir: Path,
    models_dir: Path,
    device: str,
    with_description: bool = False,
) -> Benchmark:
    match name:
        case "SACBenchmark":
            bench = SACBenchmark(
                data_dir=data_dir,
                models_dir=models_dir,
                device=device,
            )
            desc = "Original paper: Are You Stealing My Model? Sample Correlation for Fingerprinting Deep Neural Networks"

        case "ModelReuse":
            bench = ModelReuse(
                data_dir=data_dir,
                models_dir=models_dir,
                device=device,
            )
            desc = "Original paper: ModelDiff: Testing-Based DNN Similarity Comparison for Model Reuse Detection"

        case _:
            raise NotImplementedError()

    if with_description:
        return bench, desc

    return bench
