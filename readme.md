<p align="center">
  <img align="center" src="assets/qurd_logo.svg" width="200px" />
</p>
<p align="left">

# Queries, Representations and Detection: The next 100 Model Fingerprinting Schemes

QuRD is the framework for implementing, and benchmarking model fingerprinting schemes.

**Do you need to use a fingerprinting schemes?** You might be interested in knowing that...\
... QuRD implements SoTA fingerprinting schemes such as IPGuard&nbsp;[\[2\]], ModelDiff&nbsp;[\[3\]], ZestOfLIME&nbsp;[\[4\]], SAC&nbsp;[\[5\]], FBI&nbsp;[\[6\]] and AKH&nbsp;[\[1\]]. \
... QuRD implements common benchmarks such as ModelReuse&nbsp;[\[3\]] and SACBench&nbsp;[\[5\]]. \
... QuRD can be used as a CLI or in your own python script.

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
Here is a simple example that will run a small fingerprinting benchmark.
It should run on any laptop, even without GPU. 
Using the default configuration, 
- the models from the [timm tiny test models](https://huggingface.co/collections/timm/timm-tiny-test-models-66f18bd70518277591a86cef) collection will be downloaded to the `./models` folder.
- The [mini-imagenet](https://huggingface.co/datasets/timm/mini-imagenet) dataset will be downloaded to the `./data` folder.
- The AKH [\[1\]] fingerprint scores will be stored in the `./generated` folder.

All the folders will be created if they do not exist. On a laptop, downloading the data and models
should not take more than ~10min. Running the fingerprint on the benchmark takes ~1min.

### Using the CLI to run an existing benchmark
If you do not have access to a GPU, remove the `-e cuda` option.
```bash
pixi r -e cuda bench scores TinyImageNetModels "AKH"
```

### Using the experiment runner a python script

```python
from qurd.experiments import Experiment
from qurd.benchmark import get_benchmark
from qurd.fingerprint import make_fingerprint
 
smol_bench = get_benchmark("TinyImageNetModels")
runner = Experiment(smol_bench)
akh = make_fingerprint("AKH")

print(runner.scores(akh, budget=10))
```

### Calling the fingerprint by hand in a python script
```python
from qurd.benchmark import get_benchmark
from qurd.fingerprint import make_fingerprint
 
smol_bench = get_benchmark("TinyImageNetModels")
sampler, representation, distance = make_fingerprint("AKH")

# Get a pair of models to compare
model_1, model_2 = next(smol_bench.pairs())
model_1, model_2 = smol_bench.torch_model(model_1), smol_bench.torch_model(model_2)

# Sample the queries
queries = sampler(
    dataset=smol_bench.dataset,
    budget=10,
    source_model=model_1
)

# Generate the representations
repr_1 = representation(queries=queries,model=model_1)
repr_2 = representation(queries=queries,model=model_2)

# Compute the score
print(distance(repr_1, repr_2))
```

## 🚧 Documentation (WIP) 🚧
There is no documentation for now, work in progress! If you need more examples, here are few useful files to look at.

The fingerprints are constructed in [`make_fingerprint`](qurd/fingerprint/fingerprints.py#L18).
Each fingerprint is a tuple made of a [`QuerySampler`](qurd/fingerprint/base.py#L14), an [`OutputRepresentation`](qurd/fingerprint/base.py#L82) and a [distance function](qurd/fingerprint/distance.py).

The benchmarks are all subclasses of [`Benchmark`](qurd/benchmark/base.py#L14). The benchmarks are constructed in [`get_benchmark`](qurd/benchmark/__init__.py#L10). 
If you want to create your own, you need to define the following methods:
- [`list_models()`](qurd/benchmark/base.py#L85): all the models used in the benchmark, by their name.
- [`pairs()`](qurd/benchmark/base.py#L90): all the possible (source_model, target_model) pairs.
- [`torch_model(model)`](qurd/benchmark/base.py#L95): the torch Module of `model`.

## Citing
If you use this code, I would be glad to hear from you!
If you need any help, you can contact me at augustin.godinot@inria.fr.

If you publish your results, please cite the original paper.
You can find reference to the implemented fingerprints and benchmarks in the [References](#references) section.
```bibtex
@inproceedings{qurd, 
    title={Queries, Representations and Detection: The next 100 Model Fingerprinting Schemes}, 
    author={Godinot, Augustin and {Le Merrer}, Erwan and Penzo, Camilla and Taïani, François and Trédan, Gilles},  
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence}, 
    year={2025}
}
```

## References

[\[1\]]: #ref-qurd "Queries, Representation & Detection: The Next 100 Model Fingerprinting Schemes, Augustin
  Godinot, Erwan Le Merrer, Camilla Penzo, François Taïani, Gilles Trédan"
<a id="ref-qurd"></a>
[1] **Queries, Representation & Detection: The Next 100 Model Fingerprinting Schemes**, *Augustin
  Godinot, Erwan Le Merrer, Camilla Penzo, François Taïani, Gilles Trédan* [arXiv](https://arxiv.org/abs/2412.13021)


[\[2\]]: #ref-ipguard
<a id="ref-ipguard"></a>
[2] **IPGuard: Protecting Intellectual Property of Deep Neural Networks via Fingerprinting the Classification Boundary**, *Xiaoyu Cao, Jinyuan Jia, Neil Zhenqiang Gong*, [arXiv](https://arxiv.org/abs/1910.12903), [paper](https://dl.acm.org/doi/10.1145/3433210.3437526)

[\[3\]]: #ref-modeldiff
<a id="ref-modeldiff"></a>
[3] **ModelDiff: testing-based DNN similarity comparison for model reuse detection**, *Yuanchun Li, Ziqi Zhang,  Bingyan Liu, Ziyue Yang, Yunxin Liu*, [arXiv](https://arxiv.org/abs/2106.08890), [paper](https://dl.acm.org/doi/10.1145/3460319.3464816)

[\[4\]]: #ref-zestoflime
<a id="ref-zestoflime"></a>
[4] **A Zest of LIME: Towards  Architecture-Independent Model Distances**, *Hengrui Jia, Hongyu Chen, Jonas Guan,  Ali Shahin Shamsabadi, Nicolas Papernot*, [paper](https://openreview.net/forum?id=OUz_9TiTv9j)

[\[5\]]: #ref-SAC
<a id="ref-SAC"></a>
[5] **Are You Stealing My Model? Sample Correlation for Fingerprinting Deep Neural Networks**, *Jiyang Guan, Jian Liang, Ran He*, [arXiv](https://arxiv.org/abs/2210.15427), [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ed189de2611f200bd4c2ab30c576e99e-Abstract-Conference.html)

[\[6\]]: #ref-fbi
<a id="ref-fbi"></a>
[6] **Fingerprinting Classifiers With Benign Inputs**, *Thibault Maho, Teddy Furon, Erwan Le Merrer*, [arXiv](https://arxiv.org/abs/2208.03169), [paper](https://ieeexplore.ieee.org/abstract/document/10201933)


## Logo
Augustin Godinot made the logo from the following SVGs. No AI involved, just plain [Inkscape](https://inkscape.org) gym.
- https://www.svgrepo.com/svg/220887/fingerprint
- https://www.svgrepo.com/svg/530364/lemon