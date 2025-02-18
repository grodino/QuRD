from pathlib import Path


import polars as pl
import plotly.express as px
from typer import Typer

from qurd.benchmark.analysis import roc_curve, roc_metrics

app = Typer(
    help="Analyze results obtained with the bench command",
    no_args_is_help=True,
    add_completion=False,
)
GENERATED_DIR = Path("generated/")


@app.command()
def main():
    scores = pl.read_csv(GENERATED_DIR / "scores.csv").with_columns(
        true_value=pl.col("source") == pl.col("target")
    )
    metrics = roc_metrics(
        scores,
        ["dataset", "fingerprint", "source"],
        score=1 - pl.col("score"),
        true_value=pl.col("source") == pl.col("target"),
        fpr_threshold=0.05,
    )
    print(metrics)

    roc = roc_curve(
        scores,
        ["dataset", "fingerprint", "source"],
        score=1 - pl.col("score"),
        true_value=pl.col("source") == pl.col("target"),
    )

    print(roc)
    fig = px.histogram(scores, x="score", facet_row="fingerprint", color="true_value")
    fig.show()
