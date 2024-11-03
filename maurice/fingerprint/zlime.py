from typing import Literal

import numpy as np
from skimage.segmentation import quickshift
import torch
from torch import nn
from torchvision.transforms.v2 import Transform
from numba import njit
from scipy.optimize import linear_sum_assignment

from .utils import batch_predict


def subsample(images: np.ndarray, n: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
    # Step 1: segment images
    #
    # NOTE: The segmentation function needs (W, H, C) images and returns a
    # (W, H) integer array where each integer indicates the segment label of
    # the pixel.
    segmented_beacons = np.array(
        [
            quickshift(np.moveaxis(image, 0, 2), kernel_size=4, ratio=0.2, max_dist=200)
            for image in images
        ]
    )

    # Step 2: create new images by assembling random segments from the seed
    # image. Also remember the segments that were chosen for each seed image
    lime_images, lime_features = sample_segments(
        torch.Tensor(images).cpu(),
        torch.Tensor(segmented_beacons).cpu(),
        n_samples=n,
    )

    return lime_images, lime_features


def sample_segments(
    images: torch.Tensor, images_segments: torch.Tensor, n_samples: int
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Create new images by assembling random segments from a source image.

    NOTE: this function seems to be much faster when the images are on the CPU.

    Parameters
    ----------
    images : torch.Tensor (n_images, C, H, W)
        The base images whose segments will be used to create new images
    images_segments : torch.Tensor (n_images, H, W)
        The same images, after segmentation. The value of each pixel is the
        segment it belongs to.
    n_samples : int
        The number of new images to create from each image

    Returns
    -------
    tuple[torch.Tensor (n_images, n_samples, C, H, W), list[torch.Tensor
    (n_samples, n_segments)]]
        The created images and a list of segment masks. For each image i,
        created image c and segment s, we have segment_mask[i, c, s] = segment s
        of image i is present in created image c.
    """

    # Number of images to generate around one image. NOTE: this is 1_000 in the Zest
    # of LIME paper
    lime_blobs = []
    lime_masks = []

    for image, image_segments in zip(images, images_segments):
        # Compute the number of segments in the image
        (n_segments,) = torch.unique(image_segments).shape

        # Select random combinations of image segments
        segment_masks = torch.randint(0, 2, [n_samples, n_segments])

        # Add the full image as the first sampled image
        segment_masks[0, :] = 1

        # Generate the image by combining the segments
        sampled_images = []

        for segment_mask in segment_masks:
            assembled_image = image.clone()
            # print(f"{assembled_image.shape = }")
            (segments_to_mask,) = torch.nonzero(segment_mask == 0, as_tuple=True)

            for segment_label in segments_to_mask:
                assembled_image[:, image_segments == segment_label] = 0

            sampled_images.append(assembled_image.unsqueeze(0))

        lime_blobs.append(torch.cat(sampled_images).unsqueeze(0))
        lime_masks.append(segment_masks)

    return torch.cat(lime_blobs), lime_masks


def zlime_vector(
    model: nn.Module,
    images: torch.Tensor,
    lime_features: list[torch.Tensor],
    transform: Transform | None = None,
    batch_size: int = 64,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute the ZestOfLIME signature of the given model

     images:
    lime_features: [n_neighbors, n_segments in beacon] for each beacon

    return: [n_beacons, max(n_segments), n_classes]

    Parameters
    ----------
    model : nn.Module

    images : torch.Tensor (n_beacons, n_neigbors, C, H, W)
        The images created by drawing segments from a set of seed images.
        n_beacons is the number of original images, n_neighbors is the number of
        images created from each original image.
    lime_features : list[torch.Tensor (n_neighbors, n_segments)]
        The segment masks used to create each image in images. They will be used
        as the input features for the linear models trained in as part of
        ZestOfLime.
    batch_size : int, optional
        The batch size to use when requesting labels to the model, by default 64

    Returns
    -------
    torch.Tensor (n_beacons, max(n_segments), n_classes)
        The ZestOfLime signature of the model
    """
    n_beacons, n_neigbors, *img_shape = images.shape

    # Ask the model to label these images
    batched_images = images.reshape(-1, *img_shape)
    batched_preds = (
        batch_predict(
            model,
            batched_images,
            transform=transform,
            batch_size=batch_size,
            device=device,
        )
        .view(n_beacons, n_neigbors, -1)
        .to(device)
    )
    # preds: [n_beacons, n_neigbors, n_classes]
    preds = batched_preds.reshape(n_beacons, n_neigbors, -1).to(torch.float32)

    # Compute the local linear approximations
    weights = []

    with torch.no_grad():
        # Iter over each beacon image
        for lime_mask, pred in zip(lime_features, preds):
            lime_mask = torch.Tensor(lime_mask).to(device).type(torch.float32)

            w = torch.linalg.multi_dot(
                (
                    torch.pinverse(torch.matmul(lime_mask.T, lime_mask)),
                    lime_mask.T,
                    pred,
                )
            )
            weights.append(w)

    # Concat the weights, adding padding where necessary
    weights = torch.nn.utils.rnn.pad_sequence(weights)

    # Put the batch dimension first
    weights = weights.permute(1, 0, 2)

    return weights


@njit()
def greedy_assignement(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_cols = distances.shape
    assignement = np.zeros(n_rows, dtype=np.int64)
    assigned_mask = np.zeros(n_cols, dtype=np.int8)

    assert n_rows <= n_cols

    for i in range(n_rows):
        assignement[i] = np.argmin(distances[i, ~assigned_mask])
        assigned_mask[assignement[i]] = 1

    return np.arange(n_rows), assignement


def class_align(
    source: torch.Tensor,
    target: torch.Tensor,
    mode: Literal["greedy", "optimal", "many2many"] = "greedy",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assigns a class of source to each class of target using the procedure
    described in the ZestOfLime paper.

    If the source model has less classes than the target model, invert the role
    of source and target (but return the aligned representations in the same
    order as passed).

    Parameters
    ----------
    source : torch.Tensor (n_beacons, n_features, n_classes_s)
        The source model lime vector
    target : torch.Tensor (n_beacons, n_features, n_classes_t)
        The target model lime vector

    Returns
    -------
    tuple[torch.Tensor (n_beacons, n_features, n_classes), torch.Tensor (n_beacons, n_features, n_classes)]
    """
    n_beacons_source, n_features_source, n_classes_source = source.shape
    n_beacons_target, n_features_target, n_classes_target = target.shape

    assert (n_beacons_source, n_features_source) == (
        n_beacons_target,
        n_features_target,
    )

    # Swap the source and target to ensure that the target has the least classes
    if n_classes_source <= n_classes_target:
        source, target = target, source

    # Compute the distance between the lime_weights of each tuple (source_class,
    # target_class), batched over thee images. Need to permute the dims of
    # source and target to make sure that the weights of each class are in the
    # last dimension.
    #
    # distances: [n_beacons, n_classes_target, n_classes_source]
    distances = torch.cdist(
        source.permute(0, 2, 1), target.permute(0, 2, 1), p=2
    ).permute(0, 2, 1)

    # For each class of the target, find the closest class in the source (by
    # looking at the similarity in their LIME weights)
    #
    # NOTE: Some classes in source might be assigned to multiple classes in
    # target
    if mode == "many2many":
        class_matching = distances.argmin(dim=2)
        source = torch.take_along_dim(source, class_matching.unsqueeze(1), dim=2)

    # For each class of the target, find the closest class in the source. This
    # time, make sure that the relations are one to one.
    else:
        # class_matching: [n_beacons, n_classes in target]
        class_matching = np.zeros((target.shape[0], target.shape[2]), dtype=int)

        for i, beacon_distances in enumerate(distances):
            n_classes_target, n_classes_source = beacon_distances.shape

            # NOTE: Performance bottleneck.
            if mode == "optimal":
                row, col = linear_sum_assignment(beacon_distances)
            elif mode == "greedy":
                row, col = greedy_assignement(beacon_distances.numpy())
            else:
                raise NotImplementedError()

            class_matching[i, row] = col

        source = torch.take_along_dim(
            source, torch.Tensor(class_matching).unsqueeze(1).type(torch.long), dim=2
        )

    # Swap the source and target back if we swapped them at the beginning
    if n_classes_source <= n_classes_target:
        source, target = target, source

    return source, target


def zlime_distance(source_zlime: torch.Tensor, target_zlime: torch.Tensor) -> float:
    """Computes the ZestOfLime distance between the ZLIME representations.

    Parameters
    ----------
    source_zlime : torch.Tensor (n_beacons, n_features, n_classes)
        The aligned ZLIME representations of the source model
    target_zlime : torch.Tensor
        The aligned ZLIME representations of the target model

    Returns
    -------
    float
    """

    # Compute the distance between the zlime representations of the two models for
    # each (beacon, class) pair
    class_zlime_distance = nn.functional.cosine_similarity(
        source_zlime, target_zlime, dim=1
    )

    return 1 - class_zlime_distance.mean().item()
