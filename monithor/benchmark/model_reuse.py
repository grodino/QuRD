from pathlib import Path
from typing import Any, Iterator

from torch import nn
import torch
import polars as pl

from monithor.external import fe_mobilenet, fe_resnet
from .base import Benchmark


class ModelReuse(Benchmark):
    """From the ModelDiff paper

    NOTE: here we do not use the base imagenet-pretrained models as sources
    because we restrict ourselves to comparisons between models that have the
    same output space.
    """

    base_models = {
        "imagenet-1k": ["mbnetv2", "resnet18"],
        "Flower102": ["mbnetv2", "resnet18"],
        "SDog120": ["mbnetv2", "resnet18"],
        # NOTE: MIT67 is present in the code but not mentionned in the paper
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

    n_classes = {"Flower102": 102, "SDog120": 120}

    def list_models(
        self, dataset: str = "Flower102", only_source: bool = False
    ) -> Iterator[str]:

        # FIXME: restrict to specific datasets

        # Enumerate the models transferred from pretrained imagenet models
        for base_model in self.base_models["imagenet-1k"]:
            for transfer in self.transfers[dataset]:
                source_model = f"pretrain({base_model},imagenet-1k)->" + transfer
                yield source_model

                if only_source:
                    continue

                # Enumerate the variations of this source_model
                for variation in self.variations:
                    yield source_model + "->" + variation

        # Enumerate the models retrained from scratch
        for base_model in self.base_models[dataset]:
            source_model = f"train({base_model},{dataset})"
            yield source_model

            if only_source:
                continue

            # Enumerate the variations of this source_model
            for variation in self.variations:
                # The quantize() variation is not available for the retrained
                # models
                if "quantize" in variation:
                    continue
                yield source_model + "->" + variation

    def torch_model(
        self, model_name: str, no_variation: bool = False
    ) -> tuple[nn.Module, dict[str, Any]]:

        if no_variation is True:
            raise NotImplementedError()

        # Convert the model name to the names used by the ModelReuse folder
        model_str = (
            model_name.replace("->", "-")
            .replace("imagenet-1k", "ImageNet")
            .replace(".0", "")
            + "-"
        )

        # Find the model checkpoint path
        model_path = self.models_dir / model_str / "final_ckpt.pth"
        assert model_path.resolve().exists(), f"{model_str = }, {model_path = }"

        # Get the dataset used
        if "SDog120" in model_name:
            dataset = "SDog120"
        else:
            dataset = "Flower102"

        # Get which architecture to use
        if "->" in model_name:
            source, variation = model_name.split("->", 1)
        else:
            source, variation = model_name, ""

        if "mbnetv2" in variation:
            model = fe_mobilenet.mbnetv2_dropout(
                pretrained=False, num_classes=self.n_classes[dataset]
            )
        elif "resnet18" in variation:
            model = fe_resnet.resnet18_dropout(
                pretrained=False, num_classes=self.n_classes[dataset]
            )

        elif "mbnetv2" in source:
            model = fe_mobilenet.mbnetv2_dropout(
                pretrained=False, num_classes=self.n_classes[dataset]
            )
        elif "resnet18" in source:
            model = fe_resnet.resnet18_dropout(
                pretrained=False, num_classes=self.n_classes[dataset]
            )

        state_dict = torch.load(
            model_path,
            map_location="cpu",
        )["state_dict"]

        # Data config taken from ModelDiff's code
        data_config = dict(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            input_size=[3, 224, 224],
        )

        # Adapt the quantized models
        if "quantize" in model_name:
            data_config["force_cpu"] = True  # type: ignore

            if "float16" in model_name:
                model = torch.quantization.quantize_dynamic(model, dtype=torch.float16)

            else:
                model = torch.quantization.quantize_dynamic(model, dtype=torch.qint8)

        # Load the weights
        model.load_state_dict(state_dict)
        model.num_classes = self.n_classes[dataset]  # type: ignore

        return model, data_config

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

        return records.join(calibration, on=source_and_dist_key).collect()
