from pathlib import Path

from maurice.benchmark.timm_collection import TimmCollection
from maurice.experiments import Experiment
from maurice.benchmark.model_reuse import ModelReuse
from maurice.fingerprint.fingerprints import make_fingerprint
from maurice.benchmark.analysis import roc_curve, roc_metrics

import polars as pl
import plotly.express as px

# DATA_DIR = Path("/lustre/fsn1/projects/rech/ggl/uvm54nl/maurice/data/")
# MODELS_DIR = Path("/lustre/fsn1/projects/rech/ggl/uvm54nl/maurice/models/")
# GENERATED_DIR = Path("/lustre/fsn1/projects/rech/ggl/uvm54nl/maurice/generated/")
# DEVICE = "cuda"
# BATCH_SIZE = 128

# print("PREPARE")
# benchmark = ModelReuse(data_dir=DATA_DIR, models_dir=MODELS_DIR, device="cpu")
# benchmark.prepare()

# print("ACCURACY")
# benchmark = ModelReuse(data_dir=DATA_DIR, models_dir=MODELS_DIR, device=DEVICE)
# runner = Experiment(benchmark, dir=GENERATED_DIR, batch_size=BATCH_SIZE, device=DEVICE)
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


DATA_DIR = Path("data/")
MODELS_DIR = Path("models/")
GENERATED_DIR = Path("generated/")
DEVICE = "cpu"
BATCH_SIZE = 8
DATA_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)
GENERATED_DIR.mkdir(exist_ok=True, parents=True)

# print("PREPARE")
# benchmark = TimmCollection(
#     "timm/timm-tiny-test-models-66f18bd70518277591a86cef",
#     "imagenet-1k",
#     data_dir=DATA_DIR,
#     models_dir=MODELS_DIR,
#     device="cpu",
# )
# benchmark.prepare()

# benchmark = TimmCollection(
#     "timm/timm-tiny-test-models-66f18bd70518277591a86cef",
#     "imagenet-1k",
#     data_dir=DATA_DIR,
#     models_dir=MODELS_DIR,
#     device=DEVICE,
# )
# runner = Experiment(benchmark, dir=GENERATED_DIR, batch_size=BATCH_SIZE, device=DEVICE)

# # # print("ACCURACY")
# # # runner.compute_accuracy()

print("FINGERPRINTS")
# score_records = runner.scores(
#     {
#         "Random": make_fingerprint("Random", batch_size=BATCH_SIZE, device=DEVICE),
#         "AKH": make_fingerprint("AKH", batch_size=BATCH_SIZE, device=DEVICE),
#         "ModelDiff": make_fingerprint(
#             "ModelDiff", batch_size=BATCH_SIZE, device=DEVICE
#         ),
#         # "ZestOfLIME": make_fingerprint(
#         #     "ZestOfLIME", batch_size=BATCH_SIZE, device=DEVICE
#         # ),
#     },
#     budget=10,
# )
# scores = pl.from_records(score_records)
# scores.write_csv(GENERATED_DIR / "scores.csv")


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
