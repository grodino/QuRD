from pathlib import Path
from typing import Iterator

from torch import nn
import torch
import torchvision
import polars as pl

from monithor.utils import try_convert
from .base import Benchmark


class SACBenchmark(Benchmark):
    base_models = dict(CIFAR10=["vgg_model"])

    irrelevant_models = (
        # Irrelevant models trained on CIFAR10
        [f"vgg13({seed})" for seed in range(5)]
        + [f"resnet18({5 + seed})" for seed in range(5)]
        + [f"densenet121({10 + seed})" for seed in range(5)]
        + [f"mobilenet_v2({15 + seed})" for seed in range(5)]
        # Irrelevant models trained on CIFAR10C
        + [f"vgg16_bn_cifar10C({seed})" for seed in range(5)]
        + [f"resnet18_cifar10C({5 + seed})" for seed in range(5)]
    )

    # fmt: off
    model_variations = dict(vgg_model=([]      
        # Finetuning the model on the same dataset but different split
        +[f"finetune({seed})" for seed in range(10)]

        # Transfer the model on a different dataset 
        # +[f"transfer(CIFAR100,{seed})" for seed in range(10)]
        + [f"transfer(CIFAR10C, {seed})" for seed in range(10)]
        # + [f"transfer(TinyImageNet100, {seed})" for seed in range(10)]

        # Pruning the model to remove watermarks
        + [f"fineprune({seed})" for seed in range(10)]

        # Extracting the model from its labels
        + [f"label_extraction(vgg13, {seed})" for seed in range(5)]
        + [f"label_extraction(resnet18, {5 + seed})" for seed in range(5)]
        + [f"label_extraction(densenet121, {10 + seed})" for seed in range(5)]
        + [f"label_extraction(mobilenet_v2, {15 + seed})"for seed in range(5)]
        
        # Aversarial model extraction from labels
        + [f"adv_label_extraction(vgg13, {seed})" for seed in range(5)]
        + [f"adv_label_extraction(resnet18, {5 + seed})" for seed in range(5)]
        + [f"adv_label_extraction(densenet121, {10 + seed})" for seed in range(5)]
        + [f"adv_label_extraction(mobilenet_v2, {15 + seed})"for seed in range(5)]

        # Extracting the model from its logits
        + [f"probit_extraction(vgg13, {seed})" for seed in range(5)]
        + [f"probit_extraction(resnet18, {5 + seed})" for seed in range(5)]
        + [f"probit_extraction(densenet121, {10 + seed})" for seed in range(5)]
        + [f"probit_extraction(mobilenet_v2, {15 + seed})" for seed in range(5)]
    ))
    # fmt: on

    def list_models(
        self, dataset: str = "CIFAR10", only_source: bool = False
    ) -> Iterator[str]:
        for base_model_name in self.base_models[dataset]:
            # Raw model
            yield base_model_name

            if not only_source:
                # Irrelevant models (yielded with the same name structure as a
                # base model since they are not derived from a source model).
                for variation_name in self.irrelevant_models:
                    yield variation_name

                # Raw model + input variation
                for variation_name in self.input_variations:
                    yield base_model_name + "->" + variation_name

                # Raw model + output variations
                for variation_name in self.output_variations:
                    yield base_model_name + "->" + variation_name

                # Modification of the model weights/architecture
                if base_model_name in self.model_variations:
                    for variation_name in self.model_variations[base_model_name]:
                        yield base_model_name + "->" + variation_name

    def torch_model(
        self, model_name: str, no_variation: bool = False
    ) -> tuple[nn.Module, dict]:

        # Get the SAC variation name and params
        source_model, variations = None, None
        SAC_variation, SAC_variation_name, SAC_params = None, None, None

        if "->" in model_name:
            source_model, variations = model_name.split("->", 1)

            if "->" in variations:
                SAC_variation, variations = variations.split("->", 1)
            else:
                SAC_variation = variations
                variations = None

            SAC_variation_name, params_str = SAC_variation.split("(")
            SAC_params = list(map(try_convert, params_str[:-1].split(",")))
        else:
            source_model = model_name

        # Base model
        if SAC_variation_name is None and source_model == "vgg_model":
            model = torchvision.models.vgg16_bn(weights=None)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)

            model.load_state_dict(
                torch.load(
                    self.models_dir / "model" / f"{source_model}.pth",
                    map_location=self.device,
                )
            )

            data_config = dict(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            )

        # Irrelevant model trained on CIFAR10C
        elif SAC_variation_name is None and "cifar10C" in source_model:
            seed = int(source_model.split("(", 1)[1][:-1])

            if "resnet18" in source_model:
                model = torchvision.models.resnet18(weights=None)
                in_feature = model.fc.in_features
                model.fc = torch.nn.Linear(in_feature, 10)

            elif "vgg16_bn" in source_model:
                model = torchvision.models.vgg16_bn(weights=None)
                in_feature = model.classifier[-1].in_features
                model.classifier[-1] = torch.nn.Linear(in_feature, 10)

            else:
                raise NotImplementedError()

            model.load_state_dict(
                torch.load(
                    self.models_dir / "finetune_model" / f"CIFAR10C_{seed}.pth",
                    map_location=self.device,
                )
            )

            data_config = dict(
                mean=(0.4645897160947712, 0.6514782475490196, 0.5637088950163399),
                std=(0.18422159112571024, 0.3151505122530825, 0.26127269383599344),
            )

        # Irrelevant model trained on CIFAR10
        elif SAC_variation_name is None:
            seed = int(source_model.split("(", 1)[1][:-1])

            if "vgg13" in source_model:
                model = torchvision.models.vgg13(weights=None)
                in_feature = model.classifier[-1].in_features
                model.classifier[-1] = torch.nn.Linear(in_feature, 10)

            elif "resnet18" in source_model:
                model = torchvision.models.resnet18(weights=None)
                in_feature = model.fc.in_features
                model.fc = torch.nn.Linear(in_feature, 10)

            elif "densenet121" in source_model:
                model = torchvision.models.densenet121(weights=None)
                in_feature = model.classifier.in_features
                model.classifier = torch.nn.Linear(in_feature, 10)

            elif "mobilenet_v2" in source_model:
                model = torchvision.models.mobilenet_v2(weights=None)
                in_feature = model.classifier[-1].in_features
                model.classifier[-1] = torch.nn.Linear(in_feature, 10)

            else:
                raise NotImplementedError()

            model.load_state_dict(
                torch.load(
                    self.models_dir / "model" / f"clean_model_{seed}.pth",
                    map_location=self.device,
                )
            )

            data_config = dict(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            )

        # Model extraction (label, label adversarially, probits)
        elif "extraction" in SAC_variation_name:
            student, seed = SAC_params  # type: ignore

            if student == "vgg13":
                model = torchvision.models.vgg13(weights=None)
                in_feature = model.classifier[-1].in_features
                model.classifier[-1] = torch.nn.Linear(in_feature, 10)

            elif student == "resnet18":
                model = torchvision.models.resnet18(weights=None)
                in_feature = model.fc.in_features
                model.fc = torch.nn.Linear(in_feature, 10)

            elif student == "densenet121":
                model = torchvision.models.densenet121(weights=None)
                in_feature = model.classifier.in_features
                model.classifier = torch.nn.Linear(in_feature, 10)

            elif student == "mobilenet_v2":
                model = torchvision.models.mobilenet_v2(weights=None)
                in_feature = model.classifier[-1].in_features
                model.classifier[-1] = torch.nn.Linear(in_feature, 10)

            else:
                raise NotImplementedError()

            if SAC_variation_name == "label_extraction":  # from victim's labels
                model.load_state_dict(
                    torch.load(
                        self.models_dir / "model" / f"student_model_1_{seed}.pth",
                        map_location=self.device,
                    )
                )

            elif (
                SAC_variation_name == "adv_label_extraction"
            ):  # adversarially from victim's labels
                model.load_state_dict(
                    torch.load(
                        self.models_dir / "adv_train" / f"adv_{seed}.pth",
                        map_location=self.device,
                    )
                )

            else:  # From victim's logits
                model.load_state_dict(
                    torch.load(
                        self.models_dir / "model" / f"student_model_kd_{seed}.pth",
                        map_location=self.device,
                    )
                )

            data_config = dict(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            )

        # Model pruning
        elif SAC_variation_name == "fineprune":

            # When creating the finepruned models, the SAC paper authors likely
            # saved the models using torch.save(). This resulted in the pickling
            # of the FeatureHook object as a link to its declaration in the
            # module __main__. Therefore, we need to add this object in the
            # __main__ module in order to unpickle the model.
            class FeatureHook:
                def __init__(self, module):
                    self.hook = module.register_forward_hook(self.hook_fn)

                def hook_fn(self, module, input, output):
                    self.output = output

                def close(self):
                    self.hook.remove()

            main = __import__("__main__")
            main.FeatureHook = FeatureHook

            (seed,) = SAC_params  # type: ignore
            model = torch.load(
                self.models_dir / "Fine-Pruning" / f"prune_model_{seed}.pth",
                map_location=self.device,
            )

            data_config = dict(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            )

        # Model transfer
        elif "transfer" in SAC_variation_name:
            (dataset, seed) = SAC_params  # type: ignore

            if dataset != "CIFAR10C":
                raise NotImplementedError()

            model = torchvision.models.vgg16_bn(weights=None)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)
            model.load_state_dict(
                torch.load(
                    self.models_dir
                    / "finetune_model"
                    / f"finetune_{dataset.lower()}_{seed}.pth",  # type: ignore
                    map_location=self.device,
                )
            )

            data_config = dict(
                mean=(0.4645897160947712, 0.6514782475490196, 0.5637088950163399),
                std=(0.18422159112571024, 0.3151505122530825, 0.26127269383599344),
            )

        # Finetune on different split of the dataset
        elif SAC_variation_name == "finetune":
            (seed,) = SAC_params  # type: ignore

            model = torchvision.models.vgg16_bn(weights=None)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)
            model.load_state_dict(
                torch.load(
                    self.models_dir / "finetune_10" / f"finetune{seed}.pth",
                    map_location=self.device,
                )
            )

            data_config = dict(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            )

        else:
            raise NotImplementedError()

        # For now, the input size is common to all the models
        data_config["input_size"] = [3, 32, 32]  # type: ignore

        model.num_classes = 10  # type: ignore

        return model, data_config

    def from_records(self, generated_dir: Path) -> pl.DataFrame:
        """Creates a dataframe with columns [dataset, distance, representation,
        sampler, budget, variation_name, task, source_model, target_model,
        value, unrelated mean/min/max] from records of the experiments.
        """

        tasks_no_transfer = [
            "adv_label_extraction",
            "fineprune",
            "finetune",
            "label_extraction",
            "probit_extraction",
            "same",
        ]
        source_and_dist_key = [
            "dataset",
            "split",
            "sampler",
            "representation",
            "distance",
            "budget",
            "source_model",
        ]

        records = (
            pl.scan_csv(generated_dir / "*" / "*" / "*.csv")
            .with_columns(
                # Get the variation name. Target models that have no variation
                # (i.e. that do not have a ->) are the unrelated models used for
                # each task.
                variation_name=pl.when(pl.col("source_model") != pl.col("target_model"))
                .then(
                    pl.col("target_model")
                    .str.split("->")
                    .list.slice(1)
                    .list.join("->")
                    # Remove the parameters and parentheses
                    .str.replace_all(r"\((.*?)\)", "")
                    .replace("", "unrelated")
                )
                .otherwise(pl.lit("same")),
            )
            .with_columns(
                # Repeat the unrelated models for all the tasks except the
                # CIFAR10c transfer task.
                task=pl.when(  # Unrelated models on CIFAR10
                    pl.col("variation_name") == "unrelated",
                    ~pl.col("target_model").str.contains("cifar10C"),
                )
                .then(pl.lit(tasks_no_transfer))
                # Repeat the unrelated models scpecific to the CIFAR10c transfer
                # task
                .when(  # Unrelated models on cifar10C
                    pl.col("variation_name") == "unrelated",
                    pl.col("target_model").str.contains("cifar10C"),
                )
                .then(pl.lit(["transfer"]))
                # If it is not an unrelated model, just pass the variation name
                # (and cast it into a singleton)
                .otherwise(pl.col("variation_name").cast(pl.List(pl.String)))
            )
            .explode("task")
            .collect()
        )

        # Create a column with the constants (per source) that will be used for
        # calibration
        results = records.join(
            records.group_by(source_and_dist_key)
            .agg(
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
            .sort(source_and_dist_key),
            on=source_and_dist_key,
        )

        return results
