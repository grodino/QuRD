import numpy as np
import torch
from numba import njit


def hamming(source_repr: torch.Tensor, target_repr: torch.Tensor) -> float:
    return (target_repr != source_repr).mean(dtype=float).item()


def cosine(source_repr: torch.Tensor, target_repr: torch.Tensor) -> float:
    return (
        (1 - torch.nn.functional.cosine_similarity(source_repr, target_repr, dim=-1))
        .abs()
        .mean()
        .item()
    )


def l2(source_repr: torch.Tensor, target_repr: torch.Tensor) -> float:
    return ((target_repr.to(float) - source_repr.to(float)) ** 2).mean().sqrt().item()


@njit
def mi_dist(a: np.ndarray, b: np.ndarray) -> float:
    n_labels = max(int(np.max(a)), int(np.max(b))) + 1
    n_points = a.shape[0]

    coocurences = np.zeros((n_labels, n_labels))

    # Compute the joint probability of the labels from a and b
    for label_a in range(n_labels):
        for label_b in range(n_labels):
            coocurences[label_a, label_b] = (
                (a == label_a) & (b == label_b)
            ).sum() / n_points

    # Compute the marginal probabilities
    p_a = np.sum(coocurences, axis=1)
    p_b = np.sum(coocurences, axis=0)

    # Compute the product of marginals
    marginals_product = p_a[:, None] * p_b[None, :]

    # Compute the mutual information (nans issued by a zero-divison or log of 0
    # are counted as zeros) and entropies
    mi = np.nansum(coocurences * np.log(coocurences / marginals_product))
    entropy_a = -np.nansum(p_a * np.log(p_a))
    entropy_b = -np.nansum(p_b * np.log(p_b))

    if mi == 0.0:
        return 1.0

    return 1 - mi / min(entropy_a, entropy_b)


def mutual_information(source_repr: torch.Tensor, target_repr: torch.Tensor) -> float:
    return mi_dist(source_repr.numpy(), target_repr.numpy())
