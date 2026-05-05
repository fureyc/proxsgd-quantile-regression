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

And here is an updated repository structure block reflecting the changes:

```markdown
## Repository structure

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── LICENSE
├── CITATION.cff
├── src/
│   └── proxsgd_quantile/
│       ├── __init__.py
│       └── sgd_quantile.py
├── notebooks/
│   ├── 01_SGD_Quantile_Regression_Utils.ipynb
│   └── 02_SGD_Quantile_Results.ipynb
├── data/
│   └── README.md
└── results/
    ├── README.md
    ├── figures/
    └── tables/

## Quick start: using `SGDQuantileRegressor`

After cloning the repository, install the package locally in editable mode:

```bash
pip install -e .

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

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

Install the package locally in editable mode:

```bash
pip install -e .
```

This makes the estimator available as a standard Python import:

```python
from proxsgd_quantile import SGDQuantileRegressor
```

To install the additional dependencies needed to run the notebooks and reproduce the experiments, use:

```bash
pip install -r requirements.txt
```

For closer reproduction of the submitted manuscript results, use the pinned environment file:

```bash
pip install -r requirements-repro.txt
pip install -e .
```

The submitted experiments were run with Python 3.12.13, NumPy 2.0.2, pandas 2.2.2, SciPy 1.16.3, scikit-learn 1.6.1, and statsmodels 0.14.6.

## Quick start: using `SGDQuantileRegressor`

The main estimator in this repository is `SGDQuantileRegressor`, a `scikit-learn`-compatible linear quantile regression estimator trained using proximal stochastic subgradient descent.

The estimator follows the usual `scikit-learn` `fit`/`predict` interface. The example below fits a linear quantile regression model at the 0.9 quantile.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_pinball_loss

from proxsgd_quantile import SGDQuantileRegressor

# Load example regression data
X, y = fetch_california_housing(return_X_y=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit an upper conditional quantile model
model = SGDQuantileRegressor(
    quantile=0.9,
    learning_rate=0.5,
    max_iter=5000,
    batch_size=256,
    use_adagrad=True,
    average_iterates=True,
    random_state=42,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

loss = mean_pinball_loss(y_test, y_pred, alpha=0.9)
print(f"Test pinball loss: {loss:.4f}")
```

Prediction intervals can be constructed by fitting separate models at lower and upper quantile levels.

```python
lower = SGDQuantileRegressor(
    quantile=0.1,
    learning_rate=0.5,
    max_iter=5000,
    batch_size=256,
    use_adagrad=True,
    average_iterates=True,
    random_state=42,
)

upper = SGDQuantileRegressor(
    quantile=0.9,
    learning_rate=0.5,
    max_iter=5000,
    batch_size=256,
    use_adagrad=True,
    average_iterates=True,
    random_state=42,
)

lower.fit(X_train, y_train)
upper.fit(X_train, y_train)

q_lower = lower.predict(X_test)
q_upper = upper.predict(X_test)

coverage = ((y_test >= q_lower) & (y_test <= q_upper)).mean()
mean_width = (q_upper - q_lower).mean()

print(f"Coverage: {coverage:.3f}")
print(f"Mean interval width: {mean_width:.3f}")
```

The resulting interval `[q_lower, q_upper]` is a nominal 80% prediction interval because it is constructed from the 0.1 and 0.9 conditional quantile estimates.

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

## Data notes

The synthetic and benchmark experiments can be run directly from the notebooks. Public benchmark datasets are either loaded through standard Python packages or downloaded from their public sources as needed.

The large-scale CPS experiments require an IPUMS CPS extract. Due to data-use restrictions and file size, the CPS data are not included in this repository. To reproduce the CPS experiments, obtain the corresponding IPUMS CPS extract, clean it using the preprocessing steps described in the notebook, and place the cleaned chunks in:

```text
data/ipums_clean_chunks/
```

The CSV files in `results/tables/` are included so that the manuscript tables and figures can be regenerated without rerunning every large-scale experiment from scratch.

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
