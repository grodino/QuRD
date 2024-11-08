from pathlib import Path

from maurice.experiments import Experiment
from maurice.benchmark.model_reuse import ModelReuse
from maurice.fingerprint.fingerprints import make_fingerprint


DATA_DIR = Path("/lustre/fsn1/projects/rech/ggl/uvm54nl/maurice/data/")
MODELS_DIR = Path("/lustre/fsn1/projects/rech/ggl/uvm54nl/maurice/models/")
GENERATED_DIR = Path("/lustre/fsn1/projects/rech/ggl/uvm54nl/maurice/generated/")
DEVICE = "cuda"
BATCH_SIZE = 256

print("PREPARE")
benchmark = ModelReuse(data_dir=DATA_DIR, models_dir=MODELS_DIR, device="cpu")
benchmark.prepare()

print("ACCURACY")
benchmark = ModelReuse(data_dir=DATA_DIR, models_dir=MODELS_DIR, device=DEVICE)
runner = Experiment(benchmark, GENERATED_DIR, batch_size=BATCH_SIZE, device=DEVICE)
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
