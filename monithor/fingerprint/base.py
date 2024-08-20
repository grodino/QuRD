from abc import ABC, abstractmethod
import inspect
from pathlib import Path
import pickle
from typing import Any
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform


class QueriesSampler(ABC):
    @abstractmethod
    def sample(self, dataset: Dataset, budget: int) -> Any:
        pass

    def from_file(self, path: Path, *args) -> Any:
        with open(path, "rb") as file:
            source_queries: torch.Tensor = pickle.load(file)

        return source_queries

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator

        NOTE: copied from the numpy estimator base class
        """
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "Distances should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def __str__(self):
        out = ""
        for key in self._get_param_names():
            value = getattr(self, key)

            if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
                value = value.shape

            out += f"{key}={value},"

        return f"{self.__class__.__name__}({out[:-1]})"


class OutputRepresentation(ABC):
    device: str

    @abstractmethod
    def __call__(
        self,
        queries: Any,
        model: nn.Module,
        transform: Transform | None = None,
        *args: Any,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Takes a set of queries and a torch model and computes a
        representation of the outputs of the model"""
        ...
