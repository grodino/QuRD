from pathlib import Path

from abc import ABC, abstractmethod
from logging import debug
from multiprocessing import cpu_count
from pathlib import Path
from typing import Iterator

from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Compose
from torchvision.transforms import v2
from torchmetrics.classification import Accuracy
import polars as pl

from monithor.utils import get_dataset


class Benchmark(ABC):
    # list of pretrained models from the TIMM/HF/Torch Hub
    base_models = {}

    input_variations = []
    output_variations = []
    model_variations = []

    def __init__(
        self,
        data_dir: Path = Path("data/"),
        models_dir: Path = Path("generated/models/"),
        device: str = "cuda",
    ) -> None:
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.device = device

    def prepare(self, dataset: str):
        """Download the datasets and test if they load properly"""

        _ = get_dataset(
            dataset,
            data_dir=self.data_dir,
            split="train",
            transform=None,
            download=True,
        )
        _ = get_dataset(
            dataset,
            data_dir=self.data_dir,
            split="test",
            transform=None,
            download=True,
        )

        # for model in self.list_models(dataset, only_source=False):
        for model in self.list_models(dataset, only_source=False):
            _, data_config = self.torch_model(model)
            print(f"{model:<40} {data_config['input_size']}")

    def dataset(
        self,
        name: str,
        transform: Compose | None = None,
        split: str = "test",
        download: bool = False,
    ) -> Dataset:
        dataset, _ = get_dataset(
            name,
            data_dir=self.data_dir,
            split=split,
            transform=transform,
            download=download,
        )
        return dataset

    def list_models(
        self, dataset: str = "CIFAR10", only_source: bool = False
    ) -> Iterator[str]:
        for base_model_name in self.base_models[dataset]:
            # Raw model
            yield base_model_name

            if not only_source:
                # Raw model + input variation
                for variation_name in self.input_variations:
                    yield base_model_name + "->" + variation_name

                # Raw model + output variations
                for variation_name in self.output_variations:
                    yield base_model_name + "->" + variation_name

                # Modification of the model weights/architecture
                for variation_name in self.model_variations:
                    yield base_model_name + "->" + variation_name

    @abstractmethod
    def torch_model(
        self, model_name: str, no_variation: bool = False
    ) -> tuple[nn.Module, dict]:
        """Returns the torch Module given the model name.

        If `no_variation` is True, only return the base torch Module, without
        the model variations.
        """
        ...

    def test(self, model_name: str, dataset_name: str, batch_size: int = 64):
        """Evaluates the performance of the given model on the specified dataset"""

        model, config = self.torch_model(model_name)

        # Transform needed by the source model
        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(size=config["input_size"][-2:], antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=config["mean"], std=config["std"]),
            ]
        )

        model.eval()

        # Send the model on the right device
        if config.get("force_cpu", False):
            device = "cpu"
            debug(f"forcing device=cpu as requested ({model_name = })")
        else:
            device = self.device
        model = model.to(device)

        dataset, n_classes = get_dataset(
            dataset=dataset_name,
            split="test",
            data_dir=self.data_dir,
            transform=transform,
        )
        accuracy = Accuracy(task="multiclass", num_classes=n_classes, top_k=1).to(
            self.device
        )

        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=cpu_count() // 4,
            pin_memory=True,
        )

        for images, labels in test_loader:
            images = images.to(device)
            preds: torch.Tensor = model(images).argmax(dim=-1)

            acc = accuracy(preds, labels.to(device))

            labels.cpu()
            preds.cpu()
            images.cpu()

        model.cpu()

        return accuracy.compute().cpu().item()

    @abstractmethod
    def from_records(self, generated_dir: Path) -> pl.DataFrame:
        """Creates a dataframe with columns [dataset, distance, representation,
        sampler, budget, variation_name, task, source_model, target_model,
        value, unrelated mean/min/max] from records of the experiments.
        """
        raise NotImplementedError
