from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    Flowers102,
    SVHN,
)
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, v2
from timm.data import create_dataset


def batch_predict(
    model: nn.Module,
    images: torch.Tensor,
    transform: v2.Transform | None = None,
    batch_size: int = 64,
    device: str = "auto",
) -> torch.Tensor:
    """Batch the input images, run the model and return the predictions.

    Puts the images on the specified device then back to their previous devices

    WARNING: The model should already be on the right device

    Parameters
    ----------
    model : nn.Module images : torch.Tensor (n_images, C, H, W) batch_size :
    int, optional
        by default 64

    Returns
    -------
    torch.Tensor (n_images, n_classes)
    """

    if device == "auto":
        device = images.device  # type: ignore

    previous_device = images.device

    # Batch the images
    to_label = images.split(batch_size, dim=0)

    # Put model in eval mode
    model.eval()

    probas = []

    # Compute the class probabilities for all the images
    with torch.no_grad():
        for batch in to_label:
            if transform:
                batch = transform(batch)

            batch = batch.to(device)
            probas.append(model(batch).to(previous_device))
            batch.to(previous_device)

    return torch.cat(probas)


def sample_batch(dataset: Dataset, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of images with the core"""

    loader = DataLoader(dataset, batch_size=n, shuffle=True)

    return next(iter(loader))


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
        from external.stanford_dog import SDog120

        return (
            SDog120(
                Path(data_dir) / "SDog120",
                is_train=(split == "train"),
                transform=transform,
                shots=-1,
            ),
            120,
        )
    elif dataset == "SVHN":
        return (
            SVHN(
                str(data_dir / dataset),
                split=split,
                transform=transform,
                download=download,
            ),
            10,
        )
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


def to_v2_transform(transform: Compose) -> v2.Compose:
    resize, center, _, normalize = transform.transforms

    if (
        (not isinstance(resize, Resize))
        or (not isinstance(center, CenterCrop))
        or (not isinstance(normalize, Normalize))
    ):
        raise NotImplementedError("Can only convert typical transforms created by timm")

    resize = transform.transforms[0]
    center = transform.transforms[1]
    normalize = transform.transforms[3]

    return v2.Compose(
        [
            v2.Resize(
                size=resize.size,
                interpolation=resize.interpolation,
                max_size=resize.max_size,
                antialias=resize.antialias,
            ),
            v2.CenterCrop(size=center.size),
            v2.Normalize(mean=normalize.mean, std=normalize.std),
        ]
    )


def try_convert(x: str) -> str | int | float:
    x = x.strip()

    if x.isnumeric():
        return int(x)
    elif x.replace(".", "", 1).isnumeric():
        return float(x)

    return x
