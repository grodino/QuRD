<p align="center">
  <img align="center" src="assets/qurd_logo.svg" width="200px" />
</p>
<p align="left">

# Queries, Representations and Detection: The next 100 Model Fingerprinting Schemes

QuRD is the framework for implementing, and benchmarking model fingerprinting schemes.

**Do you need to use a fingerprinting schemes?** You might be interested in knowing that...\
... QuRD makes it easy to 

**Are you working on fingerprinting schemes?** You might be interested in knowing that...\
.... QuRD makes it easy to *create new fingerprints*.\
.... QuRD makes it easy to *create new benchmarks*.\
.... QuRD makes it easy to *run large scale experiments*.

## Installation

If you have not installed it yet, install pixi `curl -fsSL https://pixi.sh/install.sh | bash`.
Then, clone this repository and run the install command.
```bash
git clone git@github.com:grodino/QuRD.git && cd QuRD
pixi install # -e cuda if you have a GPU on your machine
```

## Quick example
Here is a simple example that should run a small fingerprinting benchmark on a laptop, without any
GPU. 
Using the default configuration, 
- the models from the [timm tiny test models](https://huggingface.co/collections/timm/timm-tiny-test-models-66f18bd70518277591a86cef) collection will be downloaded to the `./models` folder.
- The [mini-imagenet](https://huggingface.co/datasets/timm/mini-imagenet) dataset will be downloaded to the `./data` folder.
- The AKH [\[1\]] fingerprint will be used to compare the models.

All the folders will be created if they do not exist. On a laptop, downloading the data and models
should not take more than ~10min. Running the fingerprint on the benchmark takes ~1min.

### Using the CLI
If you do not have acces to a GPU, remove the `-e gpu` option.
```bash
pixi r -e cuda bench scores TinyImageNetModels "AKH"
```

### In a python script

```python
from qurd.benchmark import get_benchmark
 
smol_bench = get_benchmark("TinyImageNetModels")
runner = Experiment(smol_bench)
akh = make_fingerprint("AKH")

print(runner.scores(akh, budget=10))
```


## References

[\[1\]]: #ref-qurd
<a id="ref-qurd"></a>
[1] **Queries, Representation & Detection: The Next 100 Model Fingerprinting Schemes**, *Augustin
  Godinot, Erwan Le Merrer, Camilla Penzo, François Taïani, Gilles Trédan* [arXiv](https://arxiv.org/abs/2412.13021)



## Logo
Augustin Godinot made the logo from the following SVGs. No AI involved, just plain [Inkscape](https://inkscape.org) gym.
- https://www.svgrepo.com/svg/220887/fingerprint
- https://www.svgrepo.com/svg/530364/lemon