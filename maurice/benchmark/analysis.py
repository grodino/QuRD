from logging import warning
import numpy as np
import polars as pl
from sklearn import metrics


DISTANCE_KEY = ["dataset", "split", "sampler", "representation", "distance", "budget"]
COL_WIDTH = 400
RATIO = 1.5 / 1.61803398875
SCALE = 103


def benchmark_stats(
    results: pl.DataFrame,
    accuracy: pl.DataFrame,
    oracle_distance: pl.DataFrame,
    split: str = "test",
):
    """Create plots and tables to compare benchmarks

    - Compute the number of positive/negative pairs per task and the number of
      negative target model per source model.
    - Create plot of the l2/hamming distance between source/target models on the
      entire test/train set.
    - Create plot of the accuracy drop between source/target models.
    """

    # Number of positive/negative pairs per task
    task_stats = (
        results.group_by(*DISTANCE_KEY, "task")
        .agg(
            n_negative=(pl.col("variation_name") == "unrelated").sum(),
            n_positive=(pl.col("variation_name") != "unrelated").sum(),
        )
        .select("task", "n_positive", "n_negative")
        .unique()
        .sort("task")
    )

    # Number of positive/negative pairs per source model
    source_stats = (
        results.group_by(*DISTANCE_KEY, "source_model")
        .agg(
            n_negative=(pl.col("variation_name") == "unrelated").sum(),
            n_positive=(pl.col("variation_name") != "unrelated").sum(),
        )
        .select("source_model", "n_positive", "n_negative")
        .unique()
        .sort("source_model")
    )

    # Compute the accuracy drop
    accuracy_drop = (
        # Only keep dataset/source/target triplets
        results.filter(pl.col("split") == split)
        .select(
            pl.col(
                "dataset",
                "split",
                "source_model",
                "target_model",
                "task",
                "variation_name",
            )
        )
        .unique()
        # Get the accuracy of each source model
        .join(
            accuracy,
            left_on=["dataset", "source_model"],
            right_on=["dataset", "model"],
            how="left",
        )
        .rename({"accuracy": "source_accuracy"})
        # Get the accuracy of each target model
        .join(
            accuracy,
            left_on=["dataset", "target_model"],
            right_on=["dataset", "model"],
            how="left",
        )
        .rename({"accuracy": "target_accuracy"})
        # Remove model pairs for which the accuracy was not computed
        .drop_nulls()
        # Compute the accuracy drop
        .with_columns(
            accuracy_diff=pl.col("target_accuracy") - pl.col("source_accuracy")
        )
        .sort("dataset", "task", descending=True)
    )

    # Compute the l2/hamming distance between pairs of source/target models on
    # the entire test/train set
    oracle_results = (
        results.filter(
            pl.col("split") == split,
        )
        .select(
            pl.col(
                "dataset",
                "split",
                "source_model",
                "target_model",
                "task",
                "variation_name",
            )
        )
        .unique()
        .join(
            oracle_distance,
            on=["dataset", "split", "source_model", "target_model"],
        )
    )

    bench_stats = (
        accuracy_drop.join(
            oracle_results,
            on=["dataset", "split", "source_model", "target_model", "task"],
            how="left",
        )
        .group_by(pl.all().exclude("counts_matrix"))
        .agg(
            pl.col("counts_matrix").flatten(),
            total=pl.col("counts_matrix")
            .list.explode()
            .list.explode()
            .list.explode()
            .sum(),
        )
        .with_columns(
            diff_but_accurate=pl.col("counts_matrix").list.get(0).list.get(1).list.sum()
            / pl.col("total")
        )
        .with_columns(
            conditioned_hamming=(
                (pl.col("hamming") - pl.col("diff_but_accurate"))
                / pl.col("source_accuracy")
            ),
            corrected_hamming=(
                (pl.col("hamming") - (1 - pl.col("target_accuracy")))
                / (1 - pl.col("source_accuracy"))
            ),
        )
        .select(pl.all().exclude("counts_matrix", "total"))
        .sort("dataset", "task", descending=True)
    )

    return bench_stats, source_stats, task_stats


def roc_curve(
    data: pl.DataFrame, key: list[str], score: pl.Expr, true_value: pl.Expr
) -> pl.DataFrame:
    """Compute the accuracy, true/false positive ratios and other statistics
    based on the provided `data`.

    - the `score` column (or expression) of `data` contains the score estimated
      by the model that generated the data, identified by the provided `key`.
    - the `true_value` column (or expression) of `data` contains the true
      decision that should have been made

    By convention, decision is positive if value >= threshold
    """
    perfs = []

    for idx, model_pairs in data.partition_by(key, as_dict=True).items():
        fprs, tprs, _ = metrics.roc_curve(
            y_true=model_pairs.select(true_value).to_numpy(),
            y_score=model_pairs.select(score).to_numpy(),
        )

        named_idx = {name: value for name, value in zip(key, idx)}
        for fpr, tpr in zip(fprs, tprs):
            roc = named_idx.copy()
            roc["tpr"] = tpr
            roc["fpr"] = fpr
            perfs.append(roc)

    return pl.from_records(perfs)


def roc_metrics(
    perfs: pl.DataFrame,
    key: list[str],
    score: pl.Expr,
    true_value: pl.Expr,
    fpr_threshold: float = 0.05,
) -> pl.DataFrame:
    """Computes the auc  and TPR@(FPR < fpr_threshold)"""

    perfs_metrics = []

    for idx, model_pairs in perfs.partition_by(key, as_dict=True).items():

        if np.unique(model_pairs.select(true_value).to_numpy()).size == 1:
            warning(
                f"the source model {idx} does not have variations for the specified task"
            )
            named_idx = {name: value for name, value in zip(key, idx)}
            named_idx["auc"] = None
            named_idx[f"tpr@{fpr_threshold}"] = None
            perfs_metrics.append(named_idx)
            continue

        # Compute the AUC
        auc = metrics.roc_auc_score(
            y_true=model_pairs.select(true_value).to_numpy(),
            y_score=model_pairs.select(score).to_numpy(),
        )

        # Compute the TPR@thresh
        fprs, tprs, _ = metrics.roc_curve(
            y_true=model_pairs.select(true_value).to_numpy(),
            y_score=model_pairs.select(score).to_numpy(),
        )
        tpr = tprs[fprs <= fpr_threshold][-1]

        # tpr = metrics.confusion_matrix()
        named_idx = {name: value for name, value in zip(key, idx)}
        named_idx["auc"] = auc
        named_idx[f"tpr@{fpr_threshold}"] = tpr

        perfs_metrics.append(named_idx)

    return pl.from_records(perfs_metrics)
