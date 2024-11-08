from pathlib import Path

from maurice.benchmark.timm_collection import TimmCollection
from maurice.experiments import Experiment
from maurice.benchmark.model_reuse import ModelReuse
from maurice.fingerprint.fingerprints import make_fingerprint


DATA_DIR = Path("/lustre/fsn1/projects/rech/ggl/uvm54nl/maurice/data/")
MODELS_DIR = Path("/lustre/fsn1/projects/rech/ggl/uvm54nl/maurice/models/")
GENERATED_DIR = Path("/lustre/fsn1/projects/rech/ggl/uvm54nl/maurice/generated/")
DEVICE = "cuda"
BATCH_SIZE = 128

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


# DATA_DIR = Path("data/")
# MODELS_DIR = Path("models/")
# GENERATED_DIR = Path("generated/")
# DATA_DIR.mkdir(exist_ok=True, parents=True)
# MODELS_DIR.mkdir(exist_ok=True, parents=True)
# GENERATED_DIR.mkdir(exist_ok=True, parents=True)

# print("PREPARE")
# benchmark = TimmCollection(
#     "timm/timm-tiny-test-models-66f18bd70518277591a86cef",
#     "imagenet-1k",
#     data_dir=DATA_DIR,
#     models_dir=MODELS_DIR,
#     device="cpu",
# )
# benchmark.prepare()

print("ACCURACY")
benchmark = TimmCollection(
    "timm/timm-tiny-test-models-66f18bd70518277591a86cef",
    "imagenet-1k",
    data_dir=DATA_DIR,
    models_dir=MODELS_DIR,
    device=DEVICE,
)
runner = Experiment(benchmark, dir=GENERATED_DIR, batch_size=BATCH_SIZE, device=DEVICE)
runner.compute_accuracy()

print("FINGERPRINTS")
runner.scores(
    {
        "AKH": make_fingerprint("AKH", batch_size=BATCH_SIZE, device=DEVICE),
        "ModelDiff": make_fingerprint(
            "ModelDiff", batch_size=BATCH_SIZE, device=DEVICE
        ),
        "ZestOfLIME": make_fingerprint(
            "ZestOfLIME", batch_size=BATCH_SIZE, device=DEVICE
        ),
    },
    budget=10,
)
