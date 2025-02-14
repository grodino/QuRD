from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
import typer

from maurice.benchmark.timm_collection import TimmCollection
from maurice.experiments import Experiment
from maurice.benchmark.model_reuse import ModelReuse
from maurice.fingerprint.fingerprints import make_fingerprint
from maurice.benchmark.analysis import roc_curve, roc_metrics

import polars as pl
import plotly.express as px

load_dotenv()
app = typer.Typer()


@app.command()
def main(
    data_dir: Annotated[Path, typer.Option(envvar="DATA_DIR")],
    models_dir: Annotated[Path, typer.Option(envvar="MODELS_DIR")],
    generated_dir: Annotated[Path, typer.Option(envvar="GENERATED_DIR")],
):
    DEVICE = "cuda:0"
    DEVICE = "cpu"
    BATCH_SIZE = 128

    # print("PREPARE")
    # benchmark = ModelReuse(data_dir=data_dir, models_dir=models_dir, device="cpu")
    # benchmark.prepare()

    # print("ACCURACY")
    # benchmark = ModelReuse(data_dir=data_dir, models_dir=models_dir, device=DEVICE)
    # runner = Experiment(benchmark, dir=generated_dir, batch_size=BATCH_SIZE, device=DEVICE)
    # runner.compute_accuracy()

    # print("FINGERPRINTS")
    # runner.scores(
    #     {
    #         "AKH": make_fingerprint("AKH", batch_size=BATCH_SIZE, device=DEVICE),
    #         "ModelDiff": make_fingerprint(
    #             "ModelDiff", batch_size=BATCH_SIZE, device=DEVICE
    #         ),
    #         "ZestOfLIME": make_fingerprint(
    #             "ZestOfLIME", batch_size=BATCH_SIZE, device=DEVICE
    #         ),
    #     },
    #     budget=10,
    # )

    # data_dir = Path("data/")
    # models_dir = Path("models/")
    # generated_dir = Path("generated/")
    # DEVICE = "cpu"
    # BATCH_SIZE = 8
    # data_dir.mkdir(exist_ok=True, parents=True)
    # models_dir.mkdir(exist_ok=True, parents=True)
    # generated_dir.mkdir(exist_ok=True, parents=True)

    # print("PREPARE")
    # benchmark = TimmCollection(
    #     "timm/timm-tiny-test-models-66f18bd70518277591a86cef",
    #     "imagenet-1k",
    #     data_dir=data_dir,
    #     models_dir=models_dir,
    #     device="cpu",
    # )
    # benchmark.prepare()

    benchmark = TimmCollection(
        "timm/timm-tiny-test-models-66f18bd70518277591a86cef",
        "mini-imagenet",
        data_dir=data_dir,
        models_dir=models_dir,
        device=DEVICE,
    )
    runner = Experiment(
        benchmark, dir=generated_dir, batch_size=BATCH_SIZE, device=DEVICE
    )

    print("ACCURACY")
    runner.eval_models()

    print("FINGERPRINTS")
    score_records = runner.scores(
        {
            # "Random": make_fingerprint("Random", batch_size=BATCH_SIZE, device=DEVICE),
            "AKH": make_fingerprint("AKH", batch_size=BATCH_SIZE, device=DEVICE),
            # "ModelDiff": make_fingerprint(
            #     "ModelDiff", batch_size=BATCH_SIZE, device=DEVICE
            # ),
            # "ZestOfLIME": make_fingerprint(
            #     "ZestOfLIME", batch_size=BATCH_SIZE, device=DEVICE
            # ),
        },
        budget=10,
    )
    scores = pl.from_records(score_records)
    scores.write_csv(generated_dir / "scores.csv")

    # scores = pl.read_csv(generated_dir / "scores.csv").with_columns(
    #     true_value=pl.col("source") == pl.col("target")
    # )
    # metrics = roc_metrics(
    #     scores,
    #     ["dataset", "fingerprint", "source"],
    #     score=1 - pl.col("score"),
    #     true_value=pl.col("source") == pl.col("target"),
    #     fpr_threshold=0.05,
    # )

    # roc = roc_curve(
    #     scores,
    #     ["dataset", "fingerprint", "source"],
    #     score=1 - pl.col("score"),
    #     true_value=pl.col("source") == pl.col("target"),
    # )

    # print(roc)
    # fig = px.histogram(scores, x="score", facet_row="fingerprint", color="true_value")
    # fig.show()


if __name__ == "__main__":
    app()
