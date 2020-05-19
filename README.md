# Neural hierarchical models of ecological populations

[![DOI](https://zenodo.org/badge/206636551.svg)](https://zenodo.org/badge/latestdoi/206636551)

Paper: https://onlinelibrary.wiley.com/doi/full/10.1111/ele.13462  
Preprint: https://www.biorxiv.org/content/10.1101/759944v3

### Key idea

Parameterize a hierarchical model (an observation + process + parameter model) with a neural network, creating a **neural hierarchical model**.

![Alt text](./fig/fig2.svg)

Here, (a) shows linear regression, mapping input x to an output y. 
In (b) a neural network inserts hidden layers between x and y.
Analogously, an ecological model (c) maps an input x to parameters of a hierarchical model.
A neural version of model (d) would similarly involve hidden layers between x and these parameters. 
Deep models (e) can also be constructed that use more complex neural architectures, especially when data are structured in time, space, and/or over networks. 

A variety of neural network components can be readily used in neural hierarchical models. 
For example, you might parameterize a hidden Markov model of animal movement using a convolutional neural network that takes remotely sensed imagery as input (see Appendix S2 for details). 

![convHMM](fig/conv_hmm_edited.png)

## Hardware requirements

- 20+ GB of RAM
- 4 or more CPU cores
- GPU recommended

## Setting up the environment

This project uses [conda](https://docs.conda.io/en/latest/) to install python dependencies.

```
conda env create -f environment.yml
```

Once installed, activate the environment via:

```
conda activate neural-ecology
```

To install R dependencies:

```
R -e "devtools::install_deps(dependencies = TRUE)"
```

## Running the toy models

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mbjoseph/neuralecology/master)

The `notebooks/` subdirectory contains toy models in Jupyter notebooks:

- [A neural occupancy model](notebooks/simple-occupancy.ipynb)
- [A neural dynamic occupancy model](notebooks/dynamic-occupancy.ipynb)
- [A neural N-mixture model](notebooks/n-mixture-model.ipynb)
- [A deep Markov model for capture-recapture data](notebooks/hidden-markov-model.ipynb)

## Building the paper

The workflow for building the paper is handled with GNU Make. 
To build the paper (including running the models for the case study) takes ~ 5 hours with 6 CPU cores and a GPU. 

```
make
```

