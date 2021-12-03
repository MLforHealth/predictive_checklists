# Learning Optimal Predictive Checklists

A Python package to learn simple predictive checklists from data subject to customizable constraints. For more details please see our [NeurIPS 2021 paper](https://arxiv.org/abs/2112.01020).

## Contents
- [Installation](#installation)
  * [1. Installing the Package](#1-installing-the-package)
  * [2. Installing a MIP Solver](#2-installing-a-mip-solver)
    + [2.1. CPLEX (Recommended)](#21-cplex-recommended)
    + [2.2. Python-MIP (Not Recommended)](#22-python-mip-not-recommended)
- [Usage](#usage)
- [Reproducing the Paper](#reproducing-the-paper)
- [Citation](#citation)

## Installation

### 1. Installing the Package
Our package is available on [PyPI](https://pypi.org/project/predictive-checklists/). Simply run the following with Python >= 3.7:

```
pip install predictive-checklists
```


### 2. Installing a MIP Solver

#### 2.1. CPLEX (Recommended)  

CPLEX is a proprietary optimization software package from IBM. All of the experiments in our paper were ran with CPLEX. To install CPLEX, download and install [CPLEX Optimization Studio](https://www.ibm.com/ca-en/products/ilog-cplex-optimization-studio) (we use version 20.1.0). If you are affiliated with an academic institution, you can obtain a free academic version. 

After installing CPLEX Optimization Studio, install the `cplex` Python package by following the instructions [here](https://www.ibm.com/docs/en/icos/12.8.0.0?topic=cplex-setting-up-python-api). Note that we create our MIP in this project using `cplex`, not `docplex`.


#### 2.2. Python-MIP (Not Recommended) 

If you are not able to obtain CPLEX, we provide the same formulation using [Python-MIP](https://github.com/coin-or/python-mip), which allows for the use of [CBC](https://python-mip.readthedocs.io/en/latest/intro.html), a free and open source MIP solver. You will not have to install any additional packages if you choose to use Python-MIP with CBC. 

However, note that all of the experiments in our paper were conducted using CPLEX. In limited tests, Python-MIP with CBC seems to perform markedly worse than CPLEX for the same solution time, and so we provide **no guarantees** on the performance of Python-MIP.

## Usage


We provide the following examples as Jupyter Notebooks:
1. [Getting Started](examples/getting_started.ipynb) 
2. [Creating Fair Checklists](examples/fair_checklists.ipynb) 


## Reproducing the Paper

See [reproducing_paper.md](reproducing_paper.md).


## Citation

If you use this code or package in your research, please cite the following publication:
```
@article{zhang2021learning,
  title={Learning Optimal Predictive Checklists},
  author={Zhang, Haoran and Morris, Quaid and Ustun, Berk and Ghassemi, Marzyeh},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
