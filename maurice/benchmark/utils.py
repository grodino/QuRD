from pathlib import Path
import re
from typing import Iterable

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, Flowers102
from timm.data import create_dataset


def get_dataset(
    dataset: str, split: str, data_dir: Path, transform, download: bool = False
) -> tuple[Dataset, int]:
    """Returns the dataset instance with thee number of classes"""

    if download:
        (data_dir / dataset).mkdir(exist_ok=True)

    if dataset == "CIFAR10":
        return (
            CIFAR10(
                str(data_dir / dataset),
                train=(split == "train"),
                transform=transform,
                download=download,
            ),
            10,
        )
    elif dataset == "CIFAR100":
        return (
            CIFAR100(
                str(data_dir / dataset),
                train=(split == "train"),
                transform=transform,
                download=download,
            ),
            100,
        )
    elif dataset == "Flower102":
        return (
            Flowers102(
                str(data_dir / dataset),
                split=split,
                transform=transform,
                download=download,
            ),
            102,
        )
    elif dataset == "SDog120":
        sdogs = create_dataset(
            "hfds/maurice-fp/stanford-dogs",
            root=str(data_dir / dataset),
            download=download,
            transform=transform,
            split="train" if split == "train" else "test",
        )
        return sdogs, 120

    elif dataset == "imagenet-1k":
        imagenet = create_dataset(
            "hfds/imagenet-1k",
            root=str(data_dir / dataset),
            download=download,
            transform=transform,
            split="train" if split == "train" else "validation",
            trust_remote_code=True,
        )

        return imagenet, 1_000
    else:
        raise NotImplementedError(f"The given dataset {dataset} is not supported yet")


def try_convert(x: str) -> str | int | float:
    """Try to convert a string to an integer or a float"""

    x = x.strip()

    if x.isnumeric():
        return int(x)
    elif x.replace(".", "", 1).isnumeric():
        return float(x)

    return x


def decompose_name(model_str: str) -> Iterable[tuple[str, list[str | int | float]]]:
    """Turns strings of the form `fn1(arg1, arg2,..)->fn2(arg1, arg2, ...)->...`
    into an interator of the form `[("fn1", [arg1, arg2, ...]), ("fn2, [arg1,
    arg2, ...]), ...]`"""

    matches = re.findall(r"(\w+)\(([^()]*)\)", model_str)

    for match in matches:
        function_name = match[0]
        arguments = list(map(try_convert, match[1].split(",")))
        yield function_name, arguments


def to_hf_name(model_name: str) -> str:
    """Converts a model name `fn1(arg1, arg2,..)->fn2(arg1, arg2, ...)->...` to
    a model string that can be used as a huggingface model id"""

    return (
        model_name.replace(">", "")
        .replace(".", "_")
        .replace("(", ".")
        .replace(",", ".")
        .replace(")", "")
        .strip("-.")
    )
