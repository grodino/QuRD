from typing import OrderedDict
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, Flowers102
from timm.data import create_dataset
from datasets import get_dataset_infos


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

    elif dataset == "mini-imagenet":
        mini_imagenet = create_dataset(
            "hfds/timm/mini-imagenet",
            root=str(data_dir / dataset),
            download=download,
            transform=transform,
            split="train" if split == "train" else "validation",
            trust_remote_code=True,
            # Get the indexes of mini-imagenet labels subset in the original
            # imagenet-1k dataset
            class_map=MINI_IMAGENET_CLASSMAP,
        )

        return mini_imagenet, 1_000
    else:
        raise NotImplementedError(f"The given dataset {dataset} is not supported yet")


# Computed using the following code. Hardcoded for efficiency. NOTE: the
# IMAGENET2012_CLASSES can be found at
# https://huggingface.co/datasets/ILSVRC/imagenet-1k/blob/main/classes.py
#
# mini_info = get_dataset_infos("timm/mini-imagenet")["default"]
# imagenet_info = get_dataset_infos("imagenet-1k")["default"]

# class_map = {}
# for synset_id in mini_info.features["label"].names:
#     class_map[mini_info.features["label"].str2int(synset_id)] = (
#         imagenet_info.features["label"].str2int(IMAGENET2012_CLASSES[synset_id])
#     )

MINI_IMAGENET_CLASSMAP = {
    0: 12,
    1: 15,
    2: 51,
    3: 64,
    4: 70,
    5: 96,
    6: 99,
    7: 107,
    8: 111,
    9: 121,
    10: 149,
    11: 166,
    12: 173,
    13: 176,
    14: 207,
    15: 214,
    16: 228,
    17: 242,
    18: 244,
    19: 245,
    20: 249,
    21: 251,
    22: 256,
    23: 266,
    24: 270,
    25: 275,
    26: 279,
    27: 291,
    28: 299,
    29: 301,
    30: 306,
    31: 310,
    32: 359,
    33: 364,
    34: 392,
    35: 403,
    36: 412,
    37: 427,
    38: 440,
    39: 454,
    40: 471,
    41: 476,
    42: 478,
    43: 484,
    44: 494,
    45: 502,
    46: 503,
    47: 507,
    48: 519,
    49: 524,
    50: 533,
    51: 538,
    52: 546,
    53: 553,
    54: 556,
    55: 567,
    56: 569,
    57: 584,
    58: 597,
    59: 602,
    60: 604,
    61: 605,
    62: 629,
    63: 655,
    64: 657,
    65: 659,
    66: 683,
    67: 687,
    68: 702,
    69: 709,
    70: 713,
    71: 735,
    72: 741,
    73: 758,
    74: 779,
    75: 781,
    76: 800,
    77: 801,
    78: 807,
    79: 815,
    80: 819,
    81: 847,
    82: 854,
    83: 858,
    84: 860,
    85: 880,
    86: 881,
    87: 883,
    88: 909,
    89: 912,
    90: 914,
    91: 919,
    92: 925,
    93: 927,
    94: 934,
    95: 950,
    96: 972,
    97: 973,
    98: 997,
    99: 998,
}
