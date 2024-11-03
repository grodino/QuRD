from pathlib import Path
from maurice.benchmark.model_reuse import ModelReuse


class BenchmarkRunner:
    """Run a fingerprint evaluation benchmark

    - Cache the queries and representations if asked to
    - Manage model loading
    - Compute the desired metrics
    """

    pass


DATA_DIR = Path("data/")
MODELS_DIR = Path("models/")
DEVICE = "cuda"

benchmark = ModelReuse(data_dir=DATA_DIR, models_dir=MODELS_DIR, device=DEVICE)
