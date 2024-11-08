from collections.abc import Iterable
from itertools import permutations
from pathlib import Path

import torch
from torch import nn
from timm import create_model
from polars.dataframe import DataFrame
from huggingface_hub import get_collection

from maurice.benchmark.base import Benchmark


class TimmCollection(Benchmark):
    """Create a benchmark from a timm collection of models hosted on
    huggingface

    NOTE: if this is the first time you use the benchmark, you will need access
    to the internet. On init, the list of models is downloaded/loaded to/from
    the given models dir.
    """

    base_models = {}

    def __init__(
        self,
        collection_slug: str,
        dataset_name: str,
        data_dir: Path = ...,
        models_dir: Path = ...,
        device: str = "cuda",
    ) -> None:
        super().__init__(data_dir, models_dir, device)

        self.dataset_name = dataset_name
        self.collection_slug = collection_slug
        self.cache_dir = (
            models_dir
            / f"{self.__class__.__name__}({collection_slug.replace('/','~')}) /"
        )
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        self.base_models[dataset_name] = self._get_models_list(
            collection_slug, self.cache_dir / "model_names.csv"
        )

    @staticmethod
    def _get_models_list(collection_slug: str, cache_file: Path) -> Iterable[str]:
        if cache_file.exists():
            return cache_file.read_text().split("\n")[:-1]

        collection = get_collection(collection_slug)
        models = []
        models_str = ""

        for model in collection.items:
            models.append(model.item_id)
            models_str += model.item_id + "\n"

        cache_file.write_text(models_str)

        return models

    def list_models(self, dataset: str = "imagenet-1k") -> Iterable[str]:
        for model in self.base_models[dataset]:
            yield model

    def pairs(self, dataset: str = "imagenet-1k") -> Iterable[tuple[str, str]]:
        yield from permutations(self.list_models(dataset), 2)

    def torch_model(
        self, model_name: str, from_disk: bool = False, jit: bool = False
    ) -> nn.Module:
        model = create_model(model_name, pretrained=True, scriptable=jit)

        if jit:
            model = torch.compile(model, mode="reduce-overhead")

        return model

    def from_records(self, generated_dir: Path) -> DataFrame:
        raise NotImplementedError()
