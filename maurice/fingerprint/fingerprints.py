from .queries import AdversarialQueries, RandomQueries
from .representation import ZLIME, DecisionDistanceVector


def make_fingerprint(name: str, batch_size: int, device: str, **kwargs):
    if name == "ZestOfLIME":
        subsample = kwargs.get("subsample", 1_000)
        sampler = RandomQueries(subsample)
        representation = ZLIME(batch_size, device)

        return sampler, representation

    elif name == "ModelDiff":
        assert (
            "source_model" in kwargs
        ), "The ModelDiff fingerprint requires the source_model"
        assert (
            "source_transform" in kwargs
        ), "The ModelDiff fingerprint requires the source_transform"
        source_model, source_transform = (
            kwargs["source_model"],
            kwargs["source_transform"],
        )
        sampler = AdversarialQueries(
            source_model,
            source_transform,
            subsample=False,
            batch_size=batch_size,
            device=device,
        )
        representation = DecisionDistanceVector(
            hard_labels=False, batch_size=batch_size, device=device
        )

        return sampler, representation
