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

## Method summary

For a quantile level `tau` in `(0, 1)`, linear quantile regression estimates a conditional quantile function using a model of the form

```text
f(x) = x^T theta + b,
```

where `x` is a feature vector, `theta` is the coefficient vector, and `b` is the intercept. Given observations `(x_i, y_i)` for `i = 1, ..., n`, the residual for observation `i` is

```text
r_i = y_i - x_i^T theta - b.
```

The pinball loss is

```text
rho_tau(r) = r * (tau - 1{r < 0}),
```

which assigns asymmetric penalties to positive and negative residuals. The main optimization problem considered in this repository is the `l1`-regularized quantile regression objective

```text
min_{theta, b} sum_{i=1}^n rho_tau(y_i - x_i^T theta - b) + lambda_1 ||theta||_1,
```

where the intercept `b` is left unpenalized. The implementation also allows an optional `l2` stabilization term for numerical robustness.

### Classical solvers

The classical linear programming formulation introduces nonnegative slack variables `xi_i^+` and `xi_i^-` for the positive and negative parts of each residual, together with a decomposition

```text
theta = theta^+ - theta^-
```

for `l1` regularization. This yields a linear program with variables

```text
z in R^(2d + 2n + 1),
```

and constraints whose number grows with the sample size `n`. In this repository, the LP baseline is implemented using `scikit-learn`'s `QuantileRegressor` with the HiGHS backend.

As a second classical baseline, we use the `statsmodels` implementation of quantile regression, `QuantReg`, which is based on iteratively reweighted least squares (IRLS). IRLS solves a sequence of weighted least squares problems of the form

```text
min_{theta, b} || W^(1/2) (y - X theta - b 1_n) ||_2^2,
```

where the diagonal weight matrix `W` is updated using residuals from the previous iterate. This approach often performs well in moderate-scale settings, but each iteration requires dense linear algebra involving the full design matrix.

### Proximal stochastic optimization

The proposed estimator, `SGDQuantileRegressor`, avoids repeated global linear system solves by using mini-batch stochastic subgradient updates. At iteration `t`, a mini-batch `B_t` of size `m` is sampled. For each `i` in `B_t`, the residual subgradient is

```text
s_i = tau - 1{r_i < 0}.
```

Let `s_Bt` denote the vector of mini-batch residual subgradients. The corresponding stochastic subgradients of the data-fit term are

```text
grad_theta f_Bt = -(1/m) X_Bt^T s_Bt,

grad_b f_Bt = -(1/m) 1_m^T s_Bt.
```

The coefficient vector is first updated by a stochastic subgradient step,

```text
v_t = theta_t - eta_t grad_theta f_Bt(theta_t, b_t),
```

and the intercept is updated by

```text
b_{t+1} = b_t - eta_t grad_b f_Bt(theta_t, b_t).
```

The `l1` penalty is handled using the proximal operator, which reduces to coordinate-wise soft-thresholding:

```text
theta_{t+1} = S_{eta_t lambda_1}(v_t),
```

where

```text
S_kappa(v_j) = sign(v_j) max(|v_j| - kappa, 0).
```

Thus, each proxSGD iteration replaces the full-dataset linear algebra required by LP and IRLS with mini-batch matrix-vector multiplications and coordinate-wise proximal updates. The implementation supports square-root learning-rate decay, AdaGrad step sizes, Polyak--Ruppert iterate averaging, optional early stopping, and the standard `scikit-learn` `fit`/`predict` interface.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── requirements-repro.txt
├── pyproject.toml
├── LICENSE
├── CITATION.cff
├── src/
│   └── proxsgd_quantile/
│       ├── __init__.py
│       └── sgd_quantile.py
├── examples/
│   ├── quick_start.py
│   └── prediction_interval_demo.py
├── notebooks/
│   ├── 01_SGD_Quantile_Regression_Utils.ipynb
│   └── 02_SGD_Quantile_Results.ipynb
├── data/
│   └── README.md
└── results/
    ├── README.md
    ├── figures/
    └── tables/
```

The submitted experiments were run with Python 3.12.13, NumPy 2.0.2, pandas 2.2.2, SciPy 1.16.3, scikit-learn 1.6.1, and statsmodels 0.14.6.

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

As a quick import check, run:

```bash
python -c "from proxsgd_quantile import SGDQuantileRegressor; print(SGDQuantileRegressor)"
```

A successful import confirms that the editable installation is working.

## Quick start: using `SGDQuantileRegressor`

The main estimator in this repository is `SGDQuantileRegressor`, a `scikit-learn`-compatible linear quantile regression estimator trained using proximal stochastic subgradient descent.

After installation, the estimator can be imported with:

```python
from proxsgd_quantile import SGDQuantileRegressor
```

The estimator follows the usual `scikit-learn` `fit`/`predict` interface. The example below fits a linear quantile regression model at the 0.9 quantile on the California Housing dataset.

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
    random_state=33,
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit an upper conditional quantile model
model = SGDQuantileRegressor(
    quantile=0.9,
    base_lr=0.5,
    max_iter=1000,
    batch_size=256,
    use_adagrad=True,
    use_averaging=True,
    random_state=33,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

loss = mean_pinball_loss(y_test, y_pred, alpha=0.9)
print(f"Test pinball loss: {loss:.4f}")
```

Prediction intervals can be constructed by fitting separate models at lower and upper quantile levels. The example below fits models at the 0.1 and 0.9 quantiles to form a nominal 80% prediction interval.

```python
lower = SGDQuantileRegressor(
    quantile=0.1,
    base_lr=0.5,
    max_iter=1000,
    batch_size=256,
    use_adagrad=True,
    use_averaging=True,
    random_state=33,
)

upper = SGDQuantileRegressor(
    quantile=0.9,
    base_lr=0.5,
    max_iter=1000,
    batch_size=256,
    use_adagrad=True,
    use_averaging=True,
    random_state=33,
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


## Estimator and package structure

The reusable estimator code is separated from the experiment notebooks. The main estimator is implemented in:

```text
src/proxsgd_quantile/sgd_quantile.py
```

and can be imported with:

```python
from proxsgd_quantile import SGDQuantileRegressor
```

The notebooks are used to reproduce the numerical experiments reported in the manuscript:

- `notebooks/01_SGD_Quantile_Regression_Utils.ipynb` contains helper functions, data-loading utilities, evaluation metrics, and plotting routines used by the experiments.
- `notebooks/02_SGD_Quantile_Results.ipynb` runs the synthetic, benchmark, and CPS experiments and generates the manuscript figures and tables.

Generated outputs should be written to:

```text
results/figures/
results/tables/
```

## Estimator API

`SGDQuantileRegressor` follows the standard `scikit-learn` estimator interface.

Main methods:

- `fit(X, y)`: fit a linear quantile regression model.
- `predict(X)`: return fitted conditional quantile predictions.
- `pinball_loss(X, y_true)`: compute the mean pinball loss of a fitted model.

Common initialization parameters include:

- `quantile`: target quantile level in `(0, 1)`.
- `max_iter`: maximum number of proximal stochastic gradient iterations.
- `base_lr`: base learning rate used in the step-size schedule.
- `batch_size`: mini-batch size for stochastic updates.
- `l1`: optional L1 regularization strength.
- `l2`: optional L2 regularization strength.
- `use_adagrad`: whether to use AdaGrad adaptive step sizes.
- `use_averaging`: whether to use Polyak--Ruppert iterate averaging.
- `eval_every`: frequency at which training diagnostics are recorded.
- `early_stopping`: whether to use validation-based early stopping.
- `random_state`: random seed for reproducibility.

After fitting, the learned parameters are available as:

- `coef_`: fitted coefficient vector.
- `intercept_`: fitted intercept.
- `n_iter_`: number of stochastic iterations performed.
- `history_`: dictionary containing recorded optimization diagnostics.

## Examples

Small standalone examples are provided in the `examples/` directory.

```text
examples/
├── quick_start.py
└── prediction_interval_demo.py
```

These scripts demonstrate common uses of `SGDQuantileRegressor`:

- `quick_start.py`: fits a single conditional quantile model and reports test pinball loss.
- `prediction_interval_demo.py`: fits lower and upper quantile models, computes empirical coverage and mean interval width, and displays a simple prediction interval plot.

After installing the package, the examples can be run from the repository root:

```bash
python examples/quick_start.py
python examples/prediction_interval_demo.py
```

## Reproducing the experiments

The numerical experiments are organized across two notebooks and should be run in the following order:

1. `notebooks/01_SGD_Quantile_Regression_Utils.ipynb`
2. `notebooks/02_SGD_Quantile_Results.ipynb`

The first notebook defines helper functions, data-loading utilities, evaluation metrics, and plotting routines used by the experiments. The second notebook runs the numerical experiments and generates the figures and tables reported in the manuscript.

To start Jupyter locally, run:

```bash
jupyter notebook
```

Then open the notebooks in the order listed above.

For Google Colab, upload or open the notebooks in the same order. If running in Colab, make sure that the repository has been installed and that any required local data paths are updated before running the CPS experiments.

The synthetic and benchmark experiments can be run directly from the notebooks. The large-scale CPS experiments require separately obtained IPUMS CPS data; see the Data notes section below for details.

## Data notes

The synthetic and benchmark experiments can be run directly from the notebooks. Public benchmark datasets are either loaded through standard Python packages or downloaded from their public sources as needed.

The large-scale CPS experiments require an IPUMS CPS extract. Due to data-use restrictions and file size, the raw CPS data are not included in this repository. Details for reproducing the CPS experiments are provided below.

The CSV files in `results/tables/` are included so that the manuscript tables and figures can be regenerated without rerunning every large-scale experiment from scratch.

### Synthetic data

Synthetic datasets are generated directly within the notebooks. No external files are required.

### Benchmark datasets

The benchmark experiments use datasets available through standard Python packages or public repositories, including:

- Engel data from `statsmodels`,
- California Housing data from `scikit-learn`,
- Abalone data from the UCI Machine Learning Repository,
- Concrete Compressive Strength data from the UCI Machine Learning Repository.

The notebooks contain the relevant loading and preprocessing routines.

### CPS data

The CPS experiments use data derived from the IPUMS Current Population Survey database. Raw CPS data are **not included** in this repository because users must obtain them directly from IPUMS CPS subject to the relevant data-use terms.

To reproduce the CPS experiments, download the appropriate CPS extract from IPUMS and apply the preprocessing steps described in the notebooks. The results notebook expects cleaned CPS chunks with filenames of the form:

```text
clean_part_*.csv
```

placed in:

```text
data/ipums_clean_chunks/
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
