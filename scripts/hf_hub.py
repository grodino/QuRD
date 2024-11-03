from itertools import islice
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any

from huggingface_hub import create_collection, add_collection_item
from datasets import load_dataset, Features, Image, ClassLabel
from timm.models import push_to_hf_hub
from timm.data import create_dataset
import torch
from scipy.io import loadmat
import polars as pl

from monithor.benchmark.base import Benchmark
from monithor.benchmark.sac import SACBenchmark
from monithor.benchmark.utils import to_hf_name
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


def sdogs_to_hfds(data_dir: Path, to: Path, hub_id: str):
    """Converts the StanfordDogs dataset to the hugginface dataset format.

    Assumes that the data is downloaded `data_dir`. The original data can be
    found at https://vision.stanford.edu/aditya86/ImageNetDogs/.

    The folder pointed by `data_dir` should contain
    - The images (http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar) extracted in the "Images" folder.
    - `train_list.mat` and `test_list.mat` (http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar) at its root.

    An example of `hub_id` is `myname/my-dataset`
    """

    splits = {
        "train": loadmat(data_dir / "train_list.mat"),
        "test": loadmat(data_dir / "test_list.mat"),
    }

    classes: dict[str, int] = {}
    ordered_classnames: list[str] = [None] * 120
    metadata: dict[str, list[dict]] = {"train": [], "test": []}

    for split_name, split in splits.items():
        for (file_name,), label in zip(split["file_list"][:, 0], split["labels"][:, 0]):
            file_name = Path(file_name)
            label = int(label.item()) - 1

            # read the class label
            class_name = file_name.parent.name
            if class_name in classes:
                assert classes[class_name] == label
            else:
                classes[class_name] = label
                ordered_classnames[label] = class_name

            # copy the files to the right folder
            dest = to / split_name / file_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(data_dir / "Images" / file_name, dest)

            metadata[split_name].append({"file_name": str(file_name), "label": label})

        pl.from_records(metadata[split_name]).write_csv(
            to / split_name / "metadata.csv"
        )

    # Print the class mapping to include it in the readme
    for class_name, class_idx in classes.items():
        print(f"'{class_idx}': {class_name}")

    # Declare the features of the dataset
    features = Features(
        {
            "image": Image(),
            "label": ClassLabel(
                num_classes=120,
                names=ordered_classnames,
            ),
        }
    )

    dataset = load_dataset("imagefolder", data_dir=to, features=features)
    dataset.push_to_hub(hub_id)


if __name__ == "__main__":
    # Export the Stanford Dogs Dataset to the huggingface hub
    sdogs_to_hfds(
        Path("data/SDog120"), Path("data/hf-SDog120"), "maurice-fp/stanford-dogs"
    )

    # Upload the ModelReuse and SAC benchmarks to the huggingface hub
    model_reuse = ModelReuse(device="cpu", models_dir=Path("models/ModelDiff"))
    sac_benchmark = SACBenchmark(
        device="cpu",
        models_dir=Path(
            "/lustre/fswork/projects/rech/ggl/uvm54nl/Maurice/SACBenchmark/"
        ),
    )

    upload_benchmark(
        name="ModelReuse",
        description="Original paper: ModelDiff: Testing-Based DNN Similarity Comparison for Model Reuse Detection",
        benchmark=model_reuse,
    )
    upload_benchmark(
        name="SACBenchmark",
        description="Original paper: Are You Stealing My Model? Sample Correlation for Fingerprinting Deep Neural Networks",
        benchmark=sac_benchmark,
    )
    validate_upload(model_reuse)
    validate_upload(sac_benchmark, delete_after_download=True)
