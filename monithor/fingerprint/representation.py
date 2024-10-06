from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.v2 import Transform

from .base import OutputRepresentation
from .modeldiff import decision_distance_vector
from .zlime import zlime_vector
from .utils import batch_predict


class HardLabels(OutputRepresentation):
    def __init__(self, batch_size: int = 64, device: str = "cpu") -> None:
        self.batch_size = batch_size
        self.device = device

    def __call__(
        self, queries: Any, model: nn.Module, transform: Transform | None = None
    ) -> torch.Tensor:
        logits = batch_predict(
            model,
            queries,
            transform=transform,
            batch_size=self.batch_size,
            device=self.device,
        )

        labels = torch.argmax(logits, -1)

        return labels


class Logits(OutputRepresentation):
    def __init__(self, batch_size: int = 64, device: str = "cpu") -> None:
        self.batch_size = batch_size
        self.device = device

    def __call__(
        self, queries: Any, model: nn.Module, transform: Transform | None = None
    ) -> torch.Tensor:
        logits = batch_predict(
            model,
            queries,
            transform=transform,
            batch_size=self.batch_size,
            device=self.device,
        )
        return logits


class DecisionDistanceVector(OutputRepresentation):
    def __init__(
        self, hard_labels: bool = False, batch_size: int = 64, device: str = "cpu"
    ) -> None:
        self.hard_labels = hard_labels
        self.batch_size = batch_size
        self.device = device

    def __call__(
        self, queries: Any, model: nn.Module, transform: Transform | None = None
    ) -> torch.Tensor:
        seed_inputs, adv_inputs = queries

        ddv = decision_distance_vector(
            model,
            seed_inputs,
            adv_inputs,
            hard_labels=self.hard_labels,
            transform=transform,
            device=self.device,
        )

        return ddv


class SAC(OutputRepresentation):
    def __init__(
        self,
        hard_labels: bool = False,
        batch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        self.hard_labels = hard_labels
        self.batch_size = batch_size
        self.device = device

    def __call__(
        self,
        queries: torch.Tensor,
        model: nn.Module,
        transform: Transform | None = None,
    ) -> torch.Tensor:
        # Get the predictions of the model
        logits = batch_predict(
            model,
            queries,
            transform=transform,
            batch_size=self.batch_size,
            device=self.device,
        )

        if self.hard_labels:
            labels = logits.argmax(-1)
            # Computing the distance matrix using hamming distance
            correlation_matrix = labels[:, None] == labels[None, :]

        else:
            # Compute the output correlation matrix using cosine kernel
            normalized_logits = F.normalize(logits, dim=-1)
            correlation_matrix = torch.mm(
                normalized_logits, normalized_logits.transpose(0, 1)
            )

        return correlation_matrix


class ZLIME(OutputRepresentation):
    def __init__(self, batch_size: int = 64, device: str = "cpu") -> None:
        self.batch_size = batch_size
        self.device = device

    def __call__(
        self,
        queries: tuple[torch.Tensor, list[torch.Tensor]],
        model: nn.Module,
        transform: Transform | None = None,
    ) -> torch.Tensor:
        lime_images, lime_features = queries

        return zlime_vector(
            model,
            lime_images,
            lime_features,
            transform=transform,
            batch_size=self.batch_size,
            device=self.device,
        )
