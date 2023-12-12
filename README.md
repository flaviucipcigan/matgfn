[![DOI](https://zenodo.org/badge/719047804.svg)](https://zenodo.org/doi/10.5281/zenodo.10246464)

# Introduction
`matgfn` is a Python package to train and sample GFlowNets. The `gflow` module implements [Generative Flow Networks](https://milayb.notion.site/The-GFlowNet-Tutorial-95434ef0e2d94c24aab90e69b30be9b30) using [PyTorch](https://pytorch.org/) and [Gymnasium](https://gymnasium.farama.org/). 

The `reticular` module implements utility functions to generate reticular materials using Secondary Building Units and predict their properties based on features or atom positions and graphs. The `notebooks` folder shows example notebooks for both use-cases.

Results are reported in [Discovery of New Reticular Materials for Carbon Dioxide Capture using GFlowNets](https://arxiv.org/abs/2310.07671).

# Citation

```
@inproceedings{
    cipcigan2023discovery,
    title={Discovery of Novel Reticular Materials for Carbon Dioxide Capture using {GF}lowNets},
    author={Flaviu Cipcigan and Jonathan Booth and Rodrigo Neumann Barros Ferreira and Carine Ribeiro Dos Santos and Mathias B Steiner},
    booktitle={AI for Accelerated Materials Design - NeurIPS 2023 Workshop},
    year={2023},
    url={https://openreview.net/forum?id=cq2MJtq9iA}
}
```

# Installation

```shell
# Ensure you have Python>=3.11
pyenv local 3.11

# Clone repository via either SSH or HTTPS
git clone $REPOSITORY_PATH

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip if current pip version is an old version
pip install --upgrade pip

# This pulls all dependencies
pip install -e matgfn
```

