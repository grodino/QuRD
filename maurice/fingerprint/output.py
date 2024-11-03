import numpy as np
import torch


def hard_class_align(
    source: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    labels_source = source.unique()
    labels_target = target.unique()
    n_labels_source, n_labels_target = (
        labels_source.max().item(),
        labels_target.max().item(),
    )

    # Swap the source and target to ensure that the target has the least classes
    if n_labels_source < n_labels_target:
        source, target = target, source
        labels_source, labels_target = labels_target, labels_source

    # For each (label source, label target) pair, compute the number of
    # images that are classified as label_source by source and label_target
    # by target
    cooccurences = torch.zeros(
        (labels_source.max().item() + 1, labels_target.max().item() + 1),
        dtype=torch.int,
    )
    cooccurences.index_put_(
        (source, target),
        torch.Tensor([1]).type(torch.int),
        accumulate=True,
    )

    # For each class of the target, find the closest class in the source (by
    # looking at the number of label cooccurences)
    matching = cooccurences.argmax(dim=1)
    source = matching[source]

    # Swap back source and target
    if n_labels_source < n_labels_target:
        source, target = target, source

    return source, target


def score_mutual_information(u, v, to_distance=False):
    """
    Args:
        u (_type_): first vector
        v (_type_): second vector
        to_distance (bool, optional): if True return the mutual distance else return the mutual information

    NOTE : taken from https://github.com/t-maho/FBI_fingerprinting
    """

    assert u.shape == v.shape
    m = max(u.max(), v.max()) + 1
    mat = torch.zeros((m, m, m, m))
    for i in range(m):
        for j in range(m):
            mat[i, j, i, j] = 1

    t = torch.cat([u.unsqueeze(2), v.unsqueeze(2)], dim=2)

    e = mat[tuple(t.reshape(-1, 2).transpose(1, 0))]
    counts = e.reshape((t.shape[0], t.shape[1], m, m)).transpose(2, 3).sum(1)
    counts /= u.shape[1]
    p_u = counts.sum(1)
    p_v = counts.sum(2)

    h_u = -(p_u * torch.log2(p_u)).nan_to_num().sum(1)
    h_v = -(p_v * torch.log2(p_v)).nan_to_num().sum(1)
    h_u_v = -(counts * torch.log2(counts)).nan_to_num().sum([1, 2])

    mutual_information = h_u + h_v - h_u_v
    if to_distance:
        m, _ = torch.min(torch.cat((h_u.unsqueeze(1), h_v.unsqueeze(1)), dim=1), 1)
        mutual_information /= m
        return (1 - mutual_information.clip(0, 1)).nan_to_num(np.inf)
    else:
        return mutual_information
