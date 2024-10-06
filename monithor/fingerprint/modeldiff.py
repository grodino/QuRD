import torch
from torch import nn
from torchvision.transforms.v2 import Transform
import foolbox as fb

from .utils import batch_predict


def find_adversarial(
    model: nn.Module,
    images: torch.Tensor,
    preprocessing: dict = dict(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3
    ),
    batch_size: int = 64,
    device: str = "cpu",
) -> torch.Tensor:
    """Computes one adversarial image from each image in images.

    NOTE: automatically moves the inputs to the specified device. However, the
    model must be on the right device when this function is called.

    Parameters
    ----------
    model : nn.Module
        The model to attack
    images : torch.Tensor (n_images, C, H, W)
        The base images to compute the adversarial samples from
    batch_size : int, optional
        The batch size to compute the model's labels, by default 64

    Returns
    -------
    torch.Tensor (n_images, C, H, W)
        The generated adversarial images
    """

    prev_device = images.device

    # Construct the foolbox object to generate the adversarial examples
    model = model.eval()
    fmodel = fb.models.pytorch.PyTorchModel(
        model, bounds=(0, 1), preprocessing=preprocessing, device=device
    )

    # Select the attack used to generate the adversarial samples
    attack = fb.attacks.PGD()

    # Compute the labels predicted by the model on the images
    class_probas = batch_predict(model, images, batch_size=batch_size, device=device)
    _, labels = class_probas.max(dim=1)

    # foolbox requires images values to be in (0, 1) for the attack.
    im_min, im_max = images.min().item(), images.max().item()
    normed_images = (images - im_min) / (im_max - im_min)

    # Run the attack
    normed_images = normed_images.split(batch_size, dim=0)
    labels = labels.split(batch_size, dim=0)
    raw_adversarial = []

    for image, label in zip(normed_images, labels):
        image = image.to(device)
        label = label.to(device)

        raw, _, _ = attack(fmodel, image, label, epsilons=0.03)
        raw_adversarial.append(raw.cpu())

        image.to(prev_device)
        label.to(prev_device)

    raw = torch.cat(raw_adversarial)

    # Map the images back to their original values space
    raw = raw * (im_max - im_min) + im_min

    return raw


def decision_distance_vector(
    model: nn.Module,
    images: torch.Tensor,
    adversarial_images: torch.Tensor,
    hard_labels: bool = False,
    transform: Transform | None = None,
    batch_size: int = 64,
    device: str = "auto",
):
    """Compute the Decision Distance Vector as defined in the ModelDiff article.

    Parameters
    ----------
    model : nn.Module
        The model to represent as the DDV
    images : torch.Tensor (n_images, C, H, W)
        The seed images
    adversarial_images : torch.Tensor (n_images, C, H, W)
        One adversarial image crafted from each image in images
    batch_size : int, optional
        The batch size used to get the labels from the model, by default 64
    device : str, optional
        The device to run the computations on

    Returns
    -------
    torch.Tensor (n_images, )
        The DDV
    """
    # outputs and adv_outputs: [n_images, n_classes]
    outputs = batch_predict(
        model, images, transform=transform, batch_size=batch_size, device=device
    )
    adv_outputs = batch_predict(
        model,
        adversarial_images,
        transform=transform,
        batch_size=batch_size,
        device=device,
    )
    _, n_classes = outputs.shape

    if hard_labels:
        labels = outputs.argmax(-1)
        adv_labels = adv_outputs.argmax(-1)
        ddv = (adv_labels - labels) % n_classes

    else:
        # dist: [n_images]
        ddv = 1 - nn.functional.cosine_similarity(outputs, adv_outputs, dim=1)

    return ddv
