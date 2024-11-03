from enum import Enum
from typing import Literal

from maurice.fingerprint.distance import cosine, hamming

from .base import OutputRepresentation, QueriesSampler
from .queries import AdversarialQueries, RandomQueries, RandomNegativeQueries
from .representation import ZLIME, DecisionDistanceVector, HardLabels


Fingerprint = Literal["AKH", "ModelDiff", "ZestOfLIME"]


def make_fingerprint(
    name: Fingerprint, batch_size: int, device: str, **kwargs
) -> tuple[QueriesSampler, OutputRepresentation]:
    if name == "ZestOfLIME":
        subsample = kwargs.get("subsample", 1_000)
        sampler = RandomQueries(subsample)
        representation = ZLIME(batch_size, device)
        distance = cosine

    elif name == "AKH":
        sampler = RandomNegativeQueries(device=device, batch_size=batch_size)
        representation = HardLabels(batch_size=batch_size, device=device)
        distance = hamming

    elif name == "ModelDiff":
        sampler = AdversarialQueries(
            subsample=False, batch_size=batch_size, device=device
        )
        representation = DecisionDistanceVector(
            hard_labels=False, batch_size=batch_size, device=device
        )
        distance = cosine

    else:
        raise NotImplementedError(f"Fingerprint {name} not implemented")

    return sampler, representation, distance
