# Scalable Quantile Regression via Stochastic Optimization for Uncertainty Quantification

This repository contains code for the numerical experiments in the manuscript:

> Colin Furey and Farhad Pourkamali Anaraki,  
> **Scalable Quantile Regression via Stochastic Optimization for Uncertainty Quantification**.

The paper investigates proximal stochastic optimization as a scalable alternative for fitting linear quantile regression models. Quantile regression estimates conditional quantiles of a response distribution and can be used to construct prediction intervals for uncertainty quantification. Classical solvers based on linear programming (LP) and iteratively reweighted least squares (IRLS) can become computationally expensive because they rely on global linear algebra operations involving the full dataset.

The proposed method, referred to as **proxSGD**, replaces these global operations with mini-batch stochastic updates. The main estimator implemented in this repository is `SGDQuantileRegressor`, a `scikit-learn`-compatible quantile regression estimator based on proximal stochastic subgradient descent.

## Overview

The experiments compare three approaches for fitting linear quantile regression models:

1. **LP**: `scikit-learn`'s `QuantileRegressor` with the HiGHS backend.
2. **IRLS**: `statsmodels`' `QuantReg` implementation.
3. **proxSGD**: the proposed proximal stochastic gradient method implemented in `SGDQuantileRegressor`.

The experiments evaluate:

- test pinball loss,
- empirical prediction interval coverage,
- mean prediction interval width,
- runtime and empirical scalability,
- robustness to learning rate and mini-batch size,
- behavior under optional `l1` regularization.

The empirical study includes synthetic experiments, standard benchmark datasets, and large-scale experiments using data derived from the Current Population Survey (CPS).

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── notebooks/
│   ├── 01_SGD_Quantile_Regression_Utils.ipynb
│   └── 02_SGD_Quantile_Results.ipynb
├── data/
└── results/
    ├── figures/
```

The notebooks are organized as follows:

- `01_SGD_Quantile_Regression_Utils.ipynb` contains the estimator implementation, helper functions, data-loading utilities, evaluation metrics, and plotting functions.
- `02_SGD_Quantile_Results.ipynb` runs the experiments and produces the numerical results, tables, and figures reported in the manuscript.

The `results/` directory is intended to store generated figures and tables. It may be empty when the repository is first cloned; running the results notebook will populate the relevant output folders.

## Installation

Clone the repository:

```bash
git clone https://github.com/fureyc/proxsgd-quantile-regression.git
cd proxsgd-quantile-regression
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, use:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The main dependencies are:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `statsmodels`
- `matplotlib`
- `seaborn`
- `jupyter`
- `openpyxl`

## Reproducing the experiments

The notebooks should be run in the following order:

1. `notebooks/01_SGD_Quantile_Regression_Utils.ipynb`
2. `notebooks/02_SGD_Quantile_Results.ipynb`

The first notebook defines the estimator and utility functions. The second notebook runs the numerical experiments and generates the figures and tables.

To start Jupyter locally, run:

```bash
jupyter notebook
```

Then open the notebooks in the order listed above.

For Google Colab, upload or open the notebooks in the same order. If running in Colab, make sure that any required local data paths are updated before running the CPS experiments.

## Data

The repository uses a combination of synthetic data, publicly available benchmark datasets, and CPS data.

### Synthetic data

Synthetic datasets are generated directly within the notebooks. No external files are required.

### Benchmark datasets

The benchmark experiments use datasets available through standard Python packages or public repositories, including:

- Engel data from `statsmodels`,
- California Housing data from `scikit-learn`,
- Abalone data from the UCI Machine Learning Repository,
- Concrete Compressive Strength data from the UCI Machine Learning Repository.

The notebook contains the relevant loading and preprocessing routines.

### CPS data

The CPS experiments use data derived from the IPUMS Current Population Survey database. Raw CPS data are **not included** in this repository because users must obtain them directly from IPUMS CPS subject to the relevant data-use terms.

To reproduce the CPS experiments, download the appropriate CPS extract from IPUMS and place the cleaned data files in the expected local data directory. For example:

```text
data/ipums_clean_chunks/
```

The results notebook expects cleaned CPS chunks with filenames of the form:

```text
clean_part_*.csv
```

If you use a different local directory, update the CPS path variables in the results notebook before running the CPS experiments.

## Computational requirements

The synthetic and benchmark experiments can be run on a standard laptop or cloud notebook environment.

The full CPS scaling experiments are more computationally demanding. In the manuscript, these experiments were run on a Google Colab virtual machine with an x86_64 CPU, 8 logical cores, and approximately 55 GB of RAM. Full reproduction of the largest CPS experiments may require substantial memory and runtime.

For a faster smoke test, reduce the CPS subsample sizes in the results notebook before running the scaling experiments.

## Manuscript outputs

The results notebook generates the main experimental outputs used in the manuscript, including:

- synthetic robustness summaries,
- benchmark performance tables,
- benchmark convergence figures,
- regularized quantile regression tables,
- CPS runtime scaling figures,
- CPS coverage and mean interval width figures.

Generated outputs should be saved under:

```text
results/figures/
results/tables/
```

## Method summary

The proposed estimator minimizes the empirical quantile regression objective using proximal stochastic subgradient descent. For a quantile level `tau`, the method uses mini-batches to compute stochastic subgradients of the pinball loss and applies a coordinate-wise proximal update for optional `l1` regularization.

The implementation supports:

- mini-batch stochastic subgradient updates,
- AdaGrad step sizes,
- diminishing step-size schedules,
- Polyak--Ruppert iterate averaging,
- optional `l1` regularization,
- optional validation-based early stopping,
- `scikit-learn`-style `fit` and `predict` methods.

## Citation

If you use this code, please cite the associated manuscript:

```text
Furey, C. and Pourkamali Anaraki, F.
Scalable Quantile Regression via Stochastic Optimization for Uncertainty Quantification.
```

A `CITATION.cff` file is included for citation metadata.

## License

This project is released under the BSD 3-Clause License. See `LICENSE` for details.

## Notes

This repository is intended to support reproducibility for the numerical experiments in the manuscript and to provide a usable implementation of `SGDQuantileRegressor` for other researchers and practitioners. Users are encouraged to experiment with, adapt, and extend the class for their own quantile regression and uncertainty quantification workflows.
