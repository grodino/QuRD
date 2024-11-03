from pathlib import Path
from typing import Iterator

from torch import nn
import torch
import polars as pl
from timm.models import load_model_config_from_hf, load_state_dict_from_hf

from maurice.external import fe_mobilenet, fe_resnet
from .utils import decompose_name, to_hf_name
from .base import Benchmark


class ModelReuse(Benchmark):
    """ModelReuse fingerprinting benchmark from ModelDiff: Testing-based DNN
    Similarity Comparison for Model Reuse Detection"""

    base_models = {
        "imagenet-1k": ["mbnetv2", "resnet18"],
        "Flower102": ["mbnetv2", "resnet18"],
        "SDog120": ["mbnetv2", "resnet18"],
        # NOTE: MIT67 modesl are present in the code but not mentionned in the
        # paper
        # "MIT67": ["mbnetv2", "resnet18"],
    }

    # Models transfered from pretrained models on imagenet-1k
    transfers = {
        target_dataset: [
            f"transfer({target_dataset},{layers_ratio})"
            for layers_ratio in [0.1, 0.5, 1.0]
        ]
        for target_dataset in ["Flower102", "SDog120"]
    }
    variations = (
        []
        # Pruning a given percentage of weights in the network
        + [f"prune({weights_ratio})" for weights_ratio in [0.2, 0.5, 0.8]]
        # Quantize the weights
        + [f"quantize({precision})" for precision in ["float16", "qint8"]]
        # Train a model on the labels of an other one
        + [f"steal({architecture})" for architecture in ["mbnetv2", "resnet18"]]
        # Train a model on the logits of an other one
        + ["distill()"]
    )

    n_classes = {"Flower102": 102, "SDog120": 120, "imagenet-1k": 1_000}

    def list_models(
        self, dataset: str = "Flower102", jit: bool = False
    ) -> Iterator[str]:
        if dataset == "imagenet-1k":
            for base_model in self.base_models["imagenet-1k"]:
                yield f"pretrain({base_model},imagenet-1k)"

            return

        # Enumerate the models transferred from pretrained imagenet models
        for base_model in self.base_models["imagenet-1k"]:
            for transfer in self.transfers[dataset]:
                source_model = f"pretrain({base_model},imagenet-1k)->" + transfer
                yield source_model

                # Enumerate the variations of this source_model
                for variation in self.variations:
                    yield source_model + "->" + variation

        # Enumerate the models retrained from scratch
        for base_model in self.base_models[dataset]:
            source_model = f"train({base_model},{dataset})"
            yield source_model

            # Enumerate the variations of this source_model
            for variation in self.variations:
                # The quantize() variation is not available for the retrained
                # models
                if "quantize" in variation:
                    continue
                yield source_model + "->" + variation

    def torch_model(
        self, model_name: str, from_disk: bool = False, jit: bool = False
    ) -> nn.Module:

        # Get the dataset used, the model to instanciate and detect if it is a
        # quantized model
        quantized = None
        for variation, params in decompose_name(model_name):
            if variation in ("pretrain", "train"):
                architecture, dataset = params
            elif variation == "transfer":
                dataset, _ = params
            elif variation == "steal":
                (architecture,) = params
            elif variation == "quantize":
                (quantized,) = params

        # Instanciate the model architecture
        if architecture == "mbnetv2":
            model = fe_mobilenet.mbnetv2_dropout(
                pretrained=False, num_classes=self.n_classes[str(dataset)]
            )
            first_conv = "layer1[0][0]"
            classifier = "classifier"
            num_features = 1_280
        elif architecture == "resnet18":
            model = fe_resnet.resnet18_dropout(
                pretrained=False, num_classes=self.n_classes[str(dataset)]
            )
            first_conv = "fc"
            classifier = "conv1"
            num_features = 512
        else:
            raise NotImplementedError(f"Unknown architecture {architecture}")

        # Load the weights
        if from_disk:
            model = load_from_disk(
                model_name=model_name,
                architecture=model,
                models_dir=self.models_dir,
                device=self.device,
            )
            model.pretrained_cfg = {  # type: ignore
                "architecture": architecture,
                "crop_mode": "center",
                "first_conv": first_conv,
                "classifier": classifier,
                "num_classes": 10,
                "num_features": num_features,
                # Mean and std are taken from ModelDiff's code and seem to be
                # independent of the dataset.
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
                "input_size": (3, 224, 224),
            }
        else:
            model_id = "maurice-fp/ModelReuse-" + to_hf_name(
                model_name.replace(f"->quantize({quantized})", "")
            )
            model.load_state_dict(load_state_dict_from_hf(model_id))
            model.pretrained_cfg, _, _ = load_model_config_from_hf(model_id)

        # Apply quantization if requested
        if quantized == "float16":
            model = torch.quantization.quantize_dynamic(model, dtype=torch.float16)
        elif quantized == "qint8":
            model = torch.quantization.quantize_dynamic(model, dtype=torch.qint8)
        elif quantized is not None:
            raise NotImplementedError(f"quantization {quantized} is not supported")

        if jit:
            model: nn.Module = torch.compile(model, mode="reduce-overhead")  # type: ignore

        return model

    def from_records(self, generated_dir: Path) -> pl.DataFrame:
        """Creates a dataframe with columns [dataset, distance, representation,
        sampler, budget, variation_name, task, source_model, target_model,
        value, unrelated mean/min/max] from records of the experiments.
        """

        source_and_dist_key = [
            "dataset",
            "split",
            "sampler",
            "representation",
            "distance",
            "budget",
            "source_model",
        ]
        tasks = ["same", "prune", "quantize", "steal", "distill"]

        records = (
            pl.scan_csv(generated_dir / "*" / "*" / "*.csv")
            .with_columns(
                # Get the variation name
                variation_name=(
                    # Source = target model
                    pl.when(pl.col("target_model") == pl.col("source_model"))
                    .then(pl.lit("same"))
                    # Source_model is source or target_model
                    .when(
                        pl.col("target_model").str.starts_with(pl.col("source_model"))
                    )
                    .then(
                        pl.col("target_model")
                        .str.split("->")
                        .list.last()
                        .str.split("(")
                        .list.first()
                    )
                    # source_model is not source of target_model, thus unrelated
                    .otherwise(pl.lit("unrelated"))
                )
            )
            .with_columns(
                # Repeat the negatives for all the tasks
                task=pl.when(pl.col("variation_name") == "unrelated")
                .then(pl.lit(tasks))
                .otherwise(pl.col("variation_name").cast(pl.List(pl.String))),
            )
            .explode("task")
            # Then filter the unrelated pairs so that they correspond to the
            # criteria specified in the ModelDiff paper: 1) models trained with
            # the same dataset of f from scratch or 2) built upon other
            # pretrained models
            .filter(
                # Models unrelated to the source model of the pair
                (
                    # 1) We know that the pairs only contain models that are on the same
                    #    dataset so we just need to find the train(arch, dataset) models
                    (
                        (pl.col("variation_name") == "unrelated")
                        & pl.col("target_model").str.starts_with("train")
                    )
                    # 2) Have to find models which are based on a different pretrain(arch,
                    #    imagenet), no need to check that the transferred dataset is the
                    #    right one.
                    | (
                        (pl.col("variation_name") == "unrelated")
                        & pl.col("target_model").str.starts_with("pretrain")
                        & (
                            pl.col("target_model").str.split("->").list.get(0)
                            != pl.col("source_model").str.split("->").list.get(0)
                        )
                    )
                )
                # Keep the positive pairs
                | (pl.col("variation_name") != "unrelated")
            )
        )

        calibration = records.group_by(source_and_dist_key).agg(
            unrelated_mean=pl.col("value")
            .filter(pl.col("variation_name") == "unrelated")
            .mean(),
            unrelated_max=pl.col("value")
            .filter(pl.col("variation_name") == "unrelated")
            .max(),
            unrelated_min=pl.col("value")
            .filter(pl.col("variation_name") == "unrelated")
            .min(),
        )

        results = records.join(calibration, on=source_and_dist_key).collect()
        return results


def load_from_disk(
    model_name: str, architecture: nn.Module, models_dir: Path, device
) -> nn.Module:
    model_str = (
        model_name.replace("->", "-")
        .replace("imagenet-1k", "ImageNet")
        .replace(".0", "")
        + "-"
    )

    # Find the model checkpoint path
    model_path = models_dir / model_str / "final_ckpt.pth"
    state_dict = torch.load(model_path, map_location=device, weights_only=False)[
        "state_dict"
    ]

    # Adapt the quantized models
    if "quantize" in model_name:
        if "float16" in model_name:
            model = torch.quantization.quantize_dynamic(
                architecture, dtype=torch.float16
            )
        elif "qint8" in model_name:
            model = torch.quantization.quantize_dynamic(architecture, dtype=torch.qint8)
        else:
            raise ValueError()

    else:
        model = architecture

    # Load the weights
    model.load_state_dict(state_dict)

    return model
