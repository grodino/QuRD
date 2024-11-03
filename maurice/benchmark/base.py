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

from .utils import get_dataset


class Benchmark(ABC):
    base_models = {}

    input_variations = []
    output_variations = []
    model_variations = []

    def __init__(
        self,
        data_dir: Path = Path("data/"),
        models_dir: Path = Path("models/"),
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

        for model in self.list_models(dataset):
            model = self.torch_model(model)

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

    @abstractmethod
    def list_models(self, dataset: str = "CIFAR10") -> Iterator[str]:
        """Lists all the models used in the benchmark, by their name"""
        ...

    @abstractmethod
    def torch_model(
        self, model_name: str, from_disk: bool = False, jit: bool = False
    ) -> nn.Module:
        """Returns the torch Module given the model name.

        If `no_variation` is True, only return the base torch Module, without
        the model variations.
        """
        ...

    def test(
        self,
        model_name: str,
        dataset_name: str,
        batch_size: int = 64,
        jit: bool = False,
    ) -> float:
        """Evaluates the performance of the given model on the specified dataset"""

        model = self.torch_model(model_name, jit=jit)

        # Transform needed by the source model
        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(size=model.pretrained_cfg["input_size"][-2:], antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=model.pretrained_cfg["mean"],
                    std=model.pretrained_cfg["std"],
                ),
            ]
        )

        model.eval()

        # Send the model on the right device
        if model.pretrained_cfg.get("force_cpu", False):
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
