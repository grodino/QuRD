from multiprocessing import cpu_count
from pathlib import Path
import pickle
from typing import Any, Callable, Iterable, TypeVar
import json

from timm.data import create_transform, resolve_data_config
import torch
from torchmetrics import Accuracy
from torch.utils.data import DataLoader

from maurice.benchmark.base import Benchmark
from maurice.benchmark.model_reuse import ModelReuse
from maurice.fingerprint.base import OutputRepresentation, QueriesSampler
from maurice.fingerprint.fingerprints import make_fingerprint


class Experiment:
    """Run a fingerprint evaluation benchmark

    - Cache the queries and representations if asked to
    - Manage model loading
    - Compute the desired metrics
    """

    def __init__(
        self, benchmark: Benchmark, dir: Path, batch_size: int, device: str
    ) -> None:
        self.benchmark = benchmark
        self.batch_size = batch_size
        self.device = device
        self.dir = dir

    def compute_accuracy(
        self, datasets: Iterable[str] | str | None = None, jit: bool = False
    ):
        """Compute the accuracy of the models in the benchmark.

        You can specify a subset of datasets with the `datasets` argument
        """
        if datasets is None:
            datasets = self.benchmark.base_models.keys()
        elif isinstance(datasets, str):
            datasets = [datasets]

        datasets_accuracies = {}

        for dataset_name in datasets:
            # Prepare where the results will be stored
            save_path = (
                self.dir
                / self.benchmark.__class__.__name__
                / dataset_name
                / "accuracy.json"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if save_path.is_file():
                models_accuracy = json.loads(save_path.read_text())

            else:
                models_accuracy = {}

            for model_name in self.benchmark.list_models(dataset_name):
                # Check that the accuracy was not already computed
                if "model_name" in models_accuracy:
                    continue

                model = self.benchmark.torch_model(model_name, jit=jit)

                # Get the dataset with the right transform
                transform = create_transform(
                    **resolve_data_config(model.pretrained_cfg)
                )
                dataset = self.benchmark.dataset(dataset_name, transform=transform)
                test_loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    num_workers=cpu_count() // 4,
                    pin_memory=True,
                )

                # Send model to device
                model.eval()
                model.to(self.device)

                # Prepare the metric
                accuracy = Accuracy(
                    task="multiclass",
                    num_classes=model.pretrained_cfg["num_classes"],
                    top_k=1,
                ).to(self.device)

                # Run the inference
                for images, labels in test_loader:
                    images.to(self.device)
                    preds: torch.Tensor = model(images).argmax(dim=-1)

                    acc = accuracy(preds, labels.to(self.device))

                    labels.cpu()
                    preds.cpu()
                    images.cpu()

                # Unload model and save everything in case of crash
                model.cpu()
                models_accuracy[model_name] = accuracy.compute().cpu().item()
                save_path.write_text(json.dumps(models_accuracy))

            datasets_accuracies[dataset_name] = models_accuracy

        return datasets_accuracies

    def evaluate(
        self,
        fingerprints: dict[str, tuple[QueriesSampler, OutputRepresentation]],
        budget: int,
    ):
        """Run the fingerprints on the benchmark and evaluate them.

        - It caches the queries that are sampled by the `QuerySampler` of
          each fingerprint. The cache is shared accross fingerprints. That is,
          if two fingerprints have the same queries sampler (with the same
          parameters) then they will have the same queries. This allows to make
          sure that two competing fingerprints that share the same sampler also
          share the randomness.
        - It caches the representations similarly.
        """

        dataset_name = "SDog120"
        dataset = self.benchmark.dataset(dataset_name)

        for fingerprint, (sampler, representation, distance) in fingerprints.items():
            for source_name, target_name in self.benchmark.pairs(dataset_name):
                print(source_name, target_name)
                source_model = self.benchmark.torch_model(source_name)
                target_model = self.benchmark.torch_model(target_name)
                source_transform = create_transform(
                    **resolve_data_config(source_model.pretrained_cfg)
                )
                target_transform = create_transform(
                    **resolve_data_config(target_model.pretrained_cfg)
                )

                # Compute the queries
                queries_path: Path = (
                    self.dir
                    / self.benchmark.__class__.__name__
                    / dataset_name
                    / "queries"
                    / str(sampler)
                    / (source_name + ".pickle")
                )
                queries = _load_or_compute(
                    lambda: sampler.sample(
                        dataset=dataset,
                        budget=budget,
                        source_model=source_model,
                        source_transform=source_transform,
                    ),
                    queries_path,
                )

                # Compute the source representation
                source_repr_path: Path = (
                    self.dir
                    / self.benchmark.__class__.__name__
                    / dataset_name
                    / "representation"
                    / str(representation)
                    / source_name
                    / "source.pickle"
                )
                source_repr = _load_or_compute(
                    lambda: representation(
                        queries=queries, model=source_model, transform=source_transform
                    ),
                    source_repr_path,
                )

                # Compute the target representation
                target_repr_path: Path = (
                    self.dir
                    / self.benchmark.__class__.__name__
                    / dataset_name
                    / "representation"
                    / str(representation)
                    / source_name
                    / (target_name + ".pickle")
                )
                target_repr = _load_or_compute(
                    lambda: representation(
                        queries=queries, model=target_model, transform=target_transform
                    ),
                    target_repr_path,
                )

                # Compute the distance
                score = distance(source_repr, target_repr)
                print(score)


T = TypeVar("T")


def _cache(obj: T, to: Path):
    to.parent.mkdir(parents=True, exist_ok=True)

    with open(to, "wb") as file:
        pickle.dump(obj, file)


def _load(from_path: Path) -> T:
    with open(from_path, "rb") as file:
        return pickle.load(file)


def _load_or_compute(func: Callable[[], T], path: Path) -> T:
    if path.is_file():
        return _load(path)

    result = func()
    _cache(result, path)

    return result


DATA_DIR = Path("data/")
MODELS_DIR = Path("models/")
GENERATED_DIR = Path("generated/")
DEVICE = "cpu"

print("start")
benchmark = ModelReuse(data_dir=DATA_DIR, models_dir=MODELS_DIR, device=DEVICE)
# benchmark.prepare("SDog120")

runner = Experiment(benchmark, GENERATED_DIR, 8, "cpu")
# runner.compute_accuracy("SDog120")

runner.evaluate(
    {
        "AKH": make_fingerprint("AKH", batch_size=8, device=DEVICE),
        "ModelDiff": make_fingerprint("ModelDiff", batch_size=8, device=DEVICE),
        "ZestOfLIME": make_fingerprint("ZestOfLIME", batch_size=8, device=DEVICE),
    },
    budget=10,
)

# model_name = next(benchmark.list_models("SDog120"))
# model = benchmark.torch_model(model_name)
# transform = create_transform(**resolve_data_config(model.pretrained_cfg))

# print(transform)

# dataset = benchmark.dataset("SDog120")

# sampler, representation = make_fingerprint(
#     "AKH", source_model=model, source_transform=transform, batch_size=64, device=DEVICE
# )

# queries = sampler.sample(dataset, budget=10)
# rpz = representation(queries, model, transform=transform)

# print("OK")
