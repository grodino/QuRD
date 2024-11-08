from pathlib import Path
from maurice.benchmark.base import Benchmark


class TimmCollection(Benchmark):
    """Create a benchmark from a timm collection of models hosted on
    huggingface"""

    def __init__(
        self,
        collection_slug: str,
        data_dir: Path = ...,
        models_dir: Path = ...,
        device: str = "cuda",
    ) -> None:
        self.collection_slug = collection_slug
