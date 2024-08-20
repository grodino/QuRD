import logging
from logging import info, debug
from multiprocessing import cpu_count
from pathlib import Path
import pickle
from time import perf_counter
from typing import Annotated, Callable, TypedDict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import typer
from typer import Option
import polars as pl
import numpy as np
from numba import njit

from monithor.benchmark import get_benchmark

from monithor.fingerprint.base import OutputRepresentation, QueriesSampler
from monithor.fingerprint.queries import (
    AdversarialNegativeQueries,
    AdversarialQueries,
    BoundaryQueries,
    RandomNegativeQueries,
    RandomQueries,
)
from monithor.fingerprint.representation import (
    ZLIME,
    DecisionDistanceVector,
    HardLabels,
    Logits,
    SAC,
)


class State(TypedDict):
    data_dir: Path
    models_dir: Path
    generated_dir: Path
    device: str
    batch_size: int
    seed: int


DEFAULT_DATA_DIR = Path("data/")
DEFAULT_MODELS_DIR = Path("generated/models/")
DEFAULT_GENERATED_DIR = Path("generated/")
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    help="Benchmark on image data based on the ActiveDiff benchmark (temporary name)",
)
state = State(
    data_dir=DEFAULT_DATA_DIR,
    models_dir=DEFAULT_MODELS_DIR,
    generated_dir=DEFAULT_GENERATED_DIR,
    batch_size=128,
    device=DEFAULT_DEVICE,
    seed=123456789,
)

DatasetName = Annotated[str, Option(help="The dataset to get the queries from.")]
DistanceName = Annotated[
    str, Option(help="Which distance to use to compare the representations.")
]
RepresentationName = Annotated[
    str, Option(help="Which representation of the model's outputs to use.")
]
SamplerName = Annotated[str, Option(help="which sampler to use for the queries.")]
Budget = Annotated[
    int,
    Option(help="The maximum number of queries that can be sent to the remote model."),
]


def get_query_sampler(
    sampler: SamplerName,
    source_model: nn.Module,
    source_transform: v2.Transform,
    device: str,
    batch_size: int,
) -> QueriesSampler:
    # Get the queries sampler
    match sampler:
        case "Random":
            query_sampler = RandomQueries()

        case "RandomSubsample10":
            query_sampler = RandomQueries(subsample=10)

        case "RandomSubsample100":
            query_sampler = RandomQueries(subsample=100)

        case "Negative":
            query_sampler = RandomNegativeQueries(
                source_model=source_model,
                source_transform=source_transform,
                num_classes=source_model.num_classes,
                augment=False,
                device=device,
                batch_size=batch_size,
            )

        case "NegativeSubsample10":
            query_sampler = RandomNegativeQueries(
                source_model=source_model,
                source_transform=source_transform,
                num_classes=source_model.num_classes,
                augment=False,
                subsample=10,
                device=device,
                batch_size=batch_size,
            )

        case "NegativeSubsample100":
            query_sampler = RandomNegativeQueries(
                source_model=source_model,
                source_transform=source_transform,
                num_classes=source_model.num_classes,
                augment=False,
                subsample=100,
                device=device,
                batch_size=batch_size,
            )

        case "NegativeAugment":
            query_sampler = RandomNegativeQueries(
                source_model,
                source_transform=source_transform,
                num_classes=source_model.num_classes,
                augment=True,
                device=device,
                batch_size=batch_size,
            )

        case "Adversarial":
            query_sampler = AdversarialQueries(
                source_model=source_model,
                source_transform=source_transform,
                batch_size=batch_size,
                device=device,
            )

        case "NegativeAdversarial":
            query_sampler = AdversarialNegativeQueries(
                source_model=source_model,
                source_transform=source_transform,
                num_classes=source_model.num_classes,
                augment=False,
                batch_size=batch_size,
                device=device,
            )

        case "Boundary":
            query_sampler = BoundaryQueries(
                source_model=source_model,
                source_transform=source_transform,
                batch_size=batch_size,
                device=device,
            )

        case _:
            raise NotImplementedError()

    return query_sampler


def get_query_representation(
    representation: RepresentationName, batch_size: int, device: str
) -> tuple[OutputRepresentation, bool, bool | None]:
    flatten = True
    return_zlime_features = False

    match representation:
        case "Labels":
            representation_fn = HardLabels(batch_size=batch_size, device=device)

        case "Logits":
            representation_fn = Logits(batch_size=batch_size, device=device)

        case "DDV":
            flatten = False
            representation_fn = DecisionDistanceVector(
                batch_size=batch_size, device=device
            )

        case "HardDDV":
            flatten = False
            representation_fn = DecisionDistanceVector(
                hard_labels=True,
                batch_size=batch_size,
                device=device,
            )

        case "SAC":
            flatten = True
            representation_fn = SAC(
                hard_labels=False,
                batch_size=batch_size,
                device=device,
            )

        case "HardSAC":
            flatten = True
            representation_fn = SAC(
                hard_labels=True,
                batch_size=batch_size,
                device=device,
            )

        case "ZLIME":
            flatten = False
            return_zlime_features = True
            representation_fn = ZLIME(
                batch_size=batch_size,
                device=device,
            )

        case _:
            raise NotImplementedError()

    return representation_fn, flatten, return_zlime_features


def get_distance_fn(
    distance: DistanceName,
) -> Callable[[torch.Tensor, torch.Tensor], float]:
    match distance:
        case "hamming":

            def dist_fn(source_repr: torch.Tensor, target_repr: torch.Tensor) -> float:
                return (target_repr != source_repr).mean(dtype=torch.float32).item()

        case "cosine":

            def dist_fn(source_repr: torch.Tensor, target_repr: torch.Tensor) -> float:
                return (
                    (
                        1
                        - torch.nn.functional.cosine_similarity(
                            source_repr, target_repr, dim=-1
                        )
                    )
                    .abs()
                    .mean()
                    .item()
                )

        case "l2":

            def dist_fn(source_repr: torch.Tensor, target_repr: torch.Tensor) -> float:
                return (
                    (
                        (target_repr.to(torch.float32) - source_repr.to(torch.float32))
                        ** 2
                    )
                    .mean()
                    .sqrt()
                    .item()
                )

        case "MI":

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

            def dist_fn(source_repr: torch.Tensor, target_repr: torch.Tensor) -> float:
                return mi_dist(source_repr.numpy(), target_repr.numpy())

        case _:
            raise NotImplementedError(
                f"Provided distance {distance} is not implemented"
            )

    return dist_fn


@app.command()
def list_models(
    benchmark: str = "DefendedBenchmark",
    dataset: DatasetName = "CIFAR10",
    only_source: bool = False,
):

    bench = get_benchmark(
        benchmark, state["data_dir"], state["models_dir"], state["device"]
    )

    for i, model_name in enumerate(bench.list_models(dataset, only_source=only_source)):
        print(f"{i+1}) {model_name}")
        _ = bench.torch_model(model_name)


@app.command()
def prepare(benchmark: str = "DefendedBenchmark", dataset: DatasetName = "CIFAR10"):
    """Download the dataset and the corresponding models"""

    info("Preparing benchmark")
    info(f"{benchmark = }, {dataset = }")

    bench = get_benchmark(
        benchmark, state["data_dir"], state["models_dir"], state["device"]
    )
    bench.prepare(dataset)


@app.command()
def accuracy(benchmark: str = "DefendedBenchmark", dataset: DatasetName = "CIFAR10"):
    """Print the accuracy of all the models and their variations"""

    bench = get_benchmark(
        benchmark, state["data_dir"], state["models_dir"], state["device"]
    )
    records = []

    for model_name in bench.list_models(dataset=dataset):
        info(model_name)
        records.append(
            {
                "model": model_name,
                "accuracy": bench.test(
                    model_name, dataset, batch_size=state["batch_size"]
                ),
                "dataset": dataset,
            }
        )
        info(records[-1])

    (state["generated_dir"] / dataset).mkdir(exist_ok=True, parents=True)
    pl.from_records(records).write_csv(
        state["generated_dir"] / dataset / "accuracy.csv"
    )


@app.command()
def oracle_distance(
    benchmark: str = "DefendedBenchmark",
    dataset: DatasetName = "CIFAR10",
    split: str = "test",
):
    """Compute the l2 distance between the the logits (resp. hamming distance
    between the labels) of each source model pair"""

    info(state)

    # Get the benchmark
    bench = get_benchmark(
        benchmark, state["data_dir"], state["models_dir"], state["device"]
    )
    # Get the size of images
    _, config = bench.torch_model(
        next(bench.list_models(dataset=dataset, only_source=True))
    )
    generic_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=config["input_size"][-2:], antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    # Init the dataloaded
    test_data = bench.dataset(dataset, transform=generic_transform, split=split)
    test_loader = DataLoader(
        test_data,
        batch_size=state["batch_size"],
        num_workers=cpu_count() // 4,
        pin_memory=True,
        shuffle=False,
    )

    # Save the outputs of the models
    for model_name in bench.list_models(dataset=dataset, only_source=False):
        info(f"{model_name = }")

        model_str = model_name.replace("/", "_")
        save_path = (
            state["generated_dir"] / dataset / "logits" / f"{split}-{model_str}.pickle"
        )
        if save_path.exists():
            info(f"logits for {model_name} already exist, skipping generation")
            continue
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)

        model, config = bench.torch_model(model_name)
        transform = v2.Normalize(mean=config["mean"], std=config["std"])
        logits = []

        model.eval()

        # Send the target model on the right device
        if config.get("force_cpu", False):
            device = "cpu"
            debug(f"forcing device=cpu as requested ({model_name = })")
        else:
            device = state["device"]
        model = model.to(device)

        with torch.no_grad():
            for images, _ in test_loader:
                images = transform(images).to(device)
                logits.append(model(images).cpu())

                del images

        model.cpu()

        with open(save_path, "wb") as file:
            pickle.dump(torch.cat(logits), file)

        del model

    records = []

    for source_model_name in bench.list_models(dataset=dataset, only_source=True):
        source_model_str = source_model_name.replace("/", "_")
        with open(
            state["generated_dir"]
            / dataset
            / "logits"
            / f"{split}-{source_model_str}.pickle",
            "rb",
        ) as file:
            source_logits: torch.Tensor = pickle.load(file)

        for target_model_name in bench.list_models(dataset=dataset):
            target_model_str = target_model_name.replace("/", "_")
            with open(
                state["generated_dir"]
                / dataset
                / "logits"
                / f"{split}-{target_model_str}.pickle",
                "rb",
            ) as file:
                target_logits: torch.Tensor = pickle.load(file)

            # [
            #   [
            #       [h(x) != h'(x) and h(x) != y and h'(x) != y], [h(x) != h'(x) and h(x) != y and h'(x) == y],
            #       [h(x) != h'(x) and h(x) == y and h'(x) != y], [h(x) != h'(x) and h(x) == y and h'(x) == y],
            #   ],
            #   [
            #       [h(x) == h'(x) and h(x) != y and h'(x) != y], [h(x) == h'(x) and h(x) != y and h'(x) == y],
            #       [h(x) == h'(x) and h(x) == y and h'(x) != y], [h(x) == h'(x) and h(x) == y and h'(x) == y],
            #   ],
            # ]
            counts_matrix = np.zeros((2, 2, 2), dtype=int)

            for (_, labels), sources, targets in zip(
                test_loader,
                torch.split(source_logits.argmax(-1), state["batch_size"]),
                torch.split(target_logits.argmax(-1), state["batch_size"]),
            ):
                for source, target, label in zip(sources, targets, labels):
                    counts_matrix[
                        int(source.item() == target.item()),
                        int(source.item() == label.item()),
                        int(target.item() == label.item()),
                    ] += 1

            records.append(
                dict(
                    dataset=dataset,
                    split=split,
                    source_model=source_model_name,
                    target_model=target_model_name,
                    l2=((source_logits - target_logits) ** 2).mean().sqrt(),
                    hamming=(source_logits.argmax(-1) != target_logits.argmax(-1)).mean(
                        dtype=torch.float32
                    ),
                    counts_matrix=counts_matrix.tolist(),
                )
            )

            info(records[-1])

    pl.from_records(records).write_parquet(
        state["generated_dir"] / dataset / f"{split}-oracle_distance.pq"
    )
    info("DONE")


@app.command()
def sample(
    benchmark: str = "DefendedBenchmark",
    dataset: DatasetName = "CIFAR10",
    split: str = "test",
    sampler: SamplerName = "Random",
    budget: Budget = 300,
    force: bool = False,
):
    """Sample the queries for all the models in the benchmark"""

    bench = get_benchmark(
        benchmark, state["data_dir"], state["models_dir"], state["device"]
    )

    info("Sampling audit queries")
    info(f"{dataset = }, {split = }, {sampler = }, {budget = }, {force = }")

    timings = []

    for model_name in bench.list_models(dataset, only_source=True):
        info(f"{model_name = }")

        # Setup the directory where the queries will be saved
        model_str = model_name.replace("/", "_")
        filename = state["generated_dir"].joinpath(
            f"{dataset}/{sampler}-{split}-{budget}/{model_str}/queries.pickle"
        )

        # If the queries already exist and we are not asked to re-generate them,
        # do no sample them again
        if filename.exists() and (force is False):
            continue

        # Get the source model associated to this variation
        source_model, source_config = bench.torch_model(model_name)

        # Move the model to the right device
        if source_config.get("force_cpu", False):
            device = "cpu"
        else:
            device = state["device"]
        source_model = source_model.to(device)

        # Create a transform that only deals with PIL->torch conversion
        generic_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(size=source_config["input_size"][-2:], antialias=True),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        # Transform needed by the source model
        source_transform = v2.Compose(
            [
                v2.Normalize(mean=source_config["mean"], std=source_config["std"]),
            ]
        )

        # Get the initial pool of images that can be used to contruct
        # the query set.
        image_pool = bench.dataset(dataset, transform=generic_transform, split=split)

        # Get the query sampler
        query_sampler = get_query_sampler(
            sampler,
            source_model=source_model,
            source_transform=source_transform,
            device=device,
            batch_size=state["batch_size"],
        )

        # Sample the queries
        t_start = perf_counter()
        queries = query_sampler.sample(image_pool, budget=budget)
        walltime = perf_counter() - t_start

        # Log the walltime
        timings.append(
            dict(
                dataset=dataset,
                sampler=sampler,
                split=split,
                budget=budget,
                model=model_name,
                walltime=walltime,
            )
        )

        if isinstance(queries, torch.Tensor):
            info(f"{queries.shape = }")

        # Save the queries
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "wb") as file:
            pickle.dump(queries, file)

        # Cleanup variables
        del query_sampler, source_model

    # Save the timings
    timings_filename = (
        state["generated_dir"]
        / dataset
        / f"{sampler}-{split}-{budget}"
        / "timings"
        / "query-sampling-timings.csv"
    )
    timings_filename.parent.mkdir(exist_ok=True, parents=True)
    pl.from_records(timings).write_csv(timings_filename)

    info("DONE")


@app.command()
def generate(
    benchmark: str = "DefendedBenchmark",
    dataset: DatasetName = "CIFAR10",
    split: str = "test",
    sampler: SamplerName = "Random",
    representation: RepresentationName = "HardLabels",
    budget: Budget = 300,
    force: bool = False,
):
    """Generate the representations of all the models in the benchmark

    The queries should already have been sampled using the `sample` command.
    """

    info("Generating the representations")
    info(f"{dataset = }, {split = }, {budget = }, {sampler = }, {representation = }")

    bench = get_benchmark(
        benchmark, state["data_dir"], state["models_dir"], state["device"]
    )
    timings = []

    for source_model_name in bench.list_models(dataset, only_source=True):
        info(f"{source_model_name = }")

        # Setup the queries path and representations directory
        src_model_str = source_model_name.replace("/", "_")
        queries_filename = state["generated_dir"].joinpath(
            f"{dataset}/{sampler}-{split}-{budget}/{src_model_str}/queries.pickle"
        )
        repr_filename = state["generated_dir"].joinpath(
            f"{dataset}/{sampler}-{split}-{budget}/{src_model_str}/{representation}/source.pickle"
        )

        # Get the torch model
        source_model, source_config = bench.torch_model(source_model_name)

        # Send the source model on the right device
        if source_config.get("force_cpu", False):
            source_device = "cpu"
            debug("forcing device=cpu as requested")
        else:
            source_device = state["device"]
        source_model = source_model.to(source_device)

        # Transform needed by the source model
        source_transform = v2.Normalize(
            mean=source_config["mean"], std=source_config["std"]
        )

        # Get the representation function
        representation_fn, flatten, return_zlime_features = get_query_representation(
            representation, state["batch_size"], state["device"]
        )

        # Get the queries
        queries_sampler = get_query_sampler(
            sampler=sampler,
            source_model=source_model,
            source_transform=source_transform,
            device=source_device,
            batch_size=state["batch_size"],
        )
        if return_zlime_features is not False:
            source_queries = queries_sampler.from_file(
                queries_filename, flatten, return_zlime_features
            )
        else:
            source_queries = queries_sampler.from_file(queries_filename, flatten)

        if isinstance(source_queries, torch.Tensor):
            info(f"{source_queries.shape = }")

        # Generate the representation of the source model with the queries built
        # from this source model.
        #
        # If the representation already exists and we are not asked to re-generate it,
        # do no generate it again
        if force or not repr_filename.exists():
            t_start = perf_counter()
            source_representation = representation_fn(
                queries=source_queries, model=source_model, transform=source_transform
            ).cpu()
            walltime = perf_counter() - t_start

            # Log the walltime
            timings.append(
                dict(
                    dataset=dataset,
                    sampler=sampler,
                    representation=representation,
                    split=split,
                    budget=budget,
                    source_model=source_model_name,
                    walltime=walltime,
                )
            )

            # Save the source representation
            repr_filename.parent.mkdir(parents=True, exist_ok=True)
            with open(repr_filename, "wb") as file:
                pickle.dump(source_representation, file)

            # Cleanup variables to avoid OOM
            del source_representation

        else:
            debug("Source representation already exists, skipping generation")

        del source_model

        # Generate the representations of the target models with the queries
        # built from the source model
        for target_model_name in bench.list_models(dataset):
            # Setup the directory where the target's representatino will be saved
            target_model_str = target_model_name.replace("/", "_")
            target_repr_filename = state["generated_dir"].joinpath(
                f"{dataset}/{sampler}-{split}-{budget}/{src_model_str}/{representation}/{target_model_str}.pickle"
            )

            info(f"{target_model_name = }")

            # If the representation already exists and we are not asked to re-generate it,
            # do no generate it again
            if target_repr_filename.exists() and not force:
                debug("Target representation already exists, skipping generation")
                continue

            # Get the model and load it on the device
            target_model, target_config = bench.torch_model(target_model_name)

            # Send the target model on the right device
            if target_config.get("force_cpu", False):
                target_device = "cpu"
                representation_fn.device = "cpu"
                debug(f"forcing device=cpu as requested ({target_model_name = })")
            else:
                target_device = state["device"]
            target_model = target_model.to(target_device)

            if source_config != target_config:
                debug(f"{source_config = }, {target_config = }")

            # Transform needed by the source model
            target_transform = v2.Normalize(
                mean=target_config["mean"], std=target_config["std"]
            )

            # Generate the representation
            t_start = perf_counter()
            target_representation = representation_fn(
                queries=source_queries, model=target_model, transform=target_transform
            ).cpu()
            walltime = perf_counter() - t_start

            # Log the walltime
            timings.append(
                dict(
                    dataset=dataset,
                    sampler=sampler,
                    representation=representation,
                    split=split,
                    budget=budget,
                    source_model=source_model_name,
                    target_model=target_model_name,
                    walltime=walltime,
                )
            )

            # Save the target representation
            target_repr_filename.parent.mkdir(parents=True, exist_ok=True)
            with open(target_repr_filename, "wb") as file:
                pickle.dump(target_representation, file)

            # Cleanup variables to avoid OOM
            del target_model, target_representation

            # Revert the representation device change if necessary
            representation_fn.device = state["device"]

    # Save the timings
    timings_filename = (
        state["generated_dir"]
        / dataset
        / f"{sampler}-{split}-{budget}"
        / "timings"
        / f"representation-{representation}-timings.csv"
    )
    timings_filename.parent.mkdir(exist_ok=True, parents=True)
    pl.from_records(timings).write_csv(timings_filename)

    info("DONE")


@app.command()
def pair_distance(
    benchmark: str = "DefendedBenchmark",
    dataset: DatasetName = "CIFAR10",
    split: str = "test",
    sampler: SamplerName = "Random",
    representation: RepresentationName = "HardLabels",
    distance: DistanceName = "hamming",
    budget: Budget = 300,
):
    """Compute the distances between all model pairs in the benchmark"""

    info("Computing the distances")
    info(
        f"{dataset = }, {split = }, {budget = }, {sampler = }, {representation = }, {distance = }"
    )

    bench = get_benchmark(
        benchmark, state["data_dir"], state["models_dir"], state["device"]
    )
    dist_fn = get_distance_fn(distance)

    records = []

    for source_model_name in bench.list_models(dataset, only_source=True):
        info(f"{source_model_name = }")

        # Get the torch model
        source_model, _ = bench.torch_model(source_model_name)

        # Get the source representation
        src_model_str = source_model_name.replace("/", "_")
        source_filename = state["generated_dir"].joinpath(
            f"{dataset}/{sampler}-{split}-{budget}/{src_model_str}/{representation}/{src_model_str}.pickle"
        )
        with open(source_filename, "rb") as file:
            source_repr = pickle.load(file)

        for target_model_name in bench.list_models(dataset):
            # Get the target representation
            target_model_str = target_model_name.replace("/", "_")
            target_filename = state["generated_dir"].joinpath(
                f"{dataset}/{sampler}-{split}-{budget}/{src_model_str}/{representation}/{target_model_str}.pickle"
            )
            with open(target_filename, "rb") as file:
                target_repr = pickle.load(file)

            # Compute the distance
            t_start = perf_counter()
            distance_value = dist_fn(target_repr, source_repr)  # type: ignore
            walltime = perf_counter() - t_start

            # Save the results
            records.append(
                {
                    "value": distance_value,
                    "walltime": walltime,
                    "source_model": source_model_name,
                    "target_model": target_model_name,
                    "dataset": dataset,
                    "distance": distance,
                    "representation": representation,
                    "sampler": sampler,
                    "split": split,
                    "budget": budget,
                }
            )

            # Cleanup inner loop
            del target_repr

        # Cleanup variables to avoid OOM
        del source_model, source_repr

    # Save the distance values
    filename = state["generated_dir"].joinpath(
        f"{dataset}/{sampler}-{split}-{budget}/{representation}-{distance}.csv"
    )
    pl.from_records(records).write_csv(filename)

    info("DONE")


@app.callback()
def main(
    data_dir: Path = DEFAULT_DATA_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    generated_dir: Path = DEFAULT_GENERATED_DIR,
    device: str = DEFAULT_DEVICE,
    seed: int = 123456789,
    batch_size: int = 128,
    verbose: bool = False,
):
    if data_dir:
        assert data_dir.exists(), f"The provided data_dir {data_dir} does not exist"
        state["data_dir"] = data_dir

    if models_dir:
        assert (
            models_dir.exists()
        ), f"The provided models_dir {models_dir.resolve()} does not exist"
        state["models_dir"] = models_dir

    if generated_dir:
        generated_dir.mkdir(parents=True, exist_ok=True)
        state["generated_dir"] = generated_dir

    state["batch_size"] = batch_size
    state["device"] = device

    torch.manual_seed(seed)
    np.random.seed(seed)
    state["seed"] = seed

    logging.basicConfig(
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )
    logging.getLogger("timm").setLevel(logging.WARNING)

    info(f"{state = }")
