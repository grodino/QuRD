from itertools import islice
from pathlib import Path
import shutil
from time import perf_counter

from huggingface_hub import create_collection, add_collection_item, scan_cache_dir
from timm.models import push_to_hf_hub
import torch

from monithor.benchmark.base import Benchmark
from monithor.benchmark.sac import SACBenchmark
from monithor.utils import to_hf_name
from monithor.benchmark.model_reuse import ModelReuse


def upload_benchmark(name: str, description: str, benchmark: Benchmark):
    """Assuming the models were downloaded to the right folder, upload them to
    huggingface"""

    collection = create_collection(
        name,
        namespace="maurice-fp",
        description=description,
        private=True,
        exists_ok=True,
    )

    print(collection.slug)

    for dataset in benchmark.base_models.keys():
        for i, model_name in enumerate(benchmark.list_models(dataset)):
            print(i, model_name)

            # Quantized models cannot be sent to the hf hub
            if "quantize" in model_name:
                continue

            model = benchmark.torch_model(model_name, from_disk=True)
            model.num_classes = model.pretrained_cfg["num_classes"]
            model.num_features = model.pretrained_cfg["num_features"]

            model_id = f"maurice-fp/{name}-{to_hf_name(model_name)}"

            # delete_repo(model_id, missing_ok=True)
            print(model_id)
            push_to_hf_hub(model, model_id, private=True)
            add_collection_item(
                collection_slug=collection.slug,
                item_id=model_id,
                item_type="model",
                exists_ok=True,
            )


def validate_upload(benchmark: Benchmark, delete_after_download: bool = False):
    for dataset in benchmark.base_models.keys():
        for i, model_name in enumerate(benchmark.list_models(dataset)):
            print(i, model_name, end="")
            model = benchmark.torch_model(model_name, jit=True)
            print(model.pretrained_cfg["hf_hub_id"])

            data = torch.randn((10, *model.pretrained_cfg["input_size"]))
            model(data)

            if delete_after_download:
                cache_dir = (
                    Path.home()
                    / ".cache/huggingface/hub/"
                    / f"models--{model.pretrained_cfg['hf_hub_id'].replace('/', '--')}"
                )
                shutil.rmtree(cache_dir)


if __name__ == "__main__":
    model_reuse = ModelReuse(device="cpu", models_dir=Path("models/ModelDiff"))
    sac_benchmark = SACBenchmark(
        device="cpu",
        models_dir=Path(
            "/lustre/fswork/projects/rech/ggl/uvm54nl/Maurice/SACBenchmark/"
        ),
    )

    # upload_benchmark(
    #     name="ModelReuse",
    #     description="Original paper: ModelDiff: Testing-Based DNN Similarity Comparison for Model Reuse Detection",
    #     benchmark=model_reuse,
    # )
    # upload_benchmark(
    #     name="SACBenchmark",
    #     description="Original paper: Are You Stealing My Model? Sample Correlation for Fingerprinting Deep Neural Networks",
    #     benchmark=sac_benchmark,
    # )

    # validate_upload(model_reuse)

    # model_reuse.prepare("Flower102")
    total = perf_counter()
    for model_name in islice(model_reuse.list_models(dataset="Flower102"), 5):
        print(model_name)
        this = perf_counter()
        model_reuse.test(model_name, "Flower102")
        this = perf_counter() - this
        print(f"Time: {this}")

    total = perf_counter() - total
    print(f"Total: {total}")

    # validate_upload(sac_benchmark, delete_after_download=True)
