# Scalable Quantile Regression via Stochastic Optimization for Uncertainty Quantification

This repository contains code for the numerical experiments in:

> Colin Furey,  
> *Scalable Quantile Regression via Stochastic Optimization for Uncertainty Quantification*.

The paper investigates proximal stochastic optimization as a scalable alternative for fitting linear quantile regression models. Quantile regression is used to estimate conditional quantiles and construct prediction intervals for uncertainty quantification. The proposed method, referred to as **proxSGD**, replaces the global linear algebra operations required by classical linear programming (LP) and iteratively reweighted least squares (IRLS) solvers with mini-batch stochastic updates.

The main estimator implemented in this repository is `SGDQuantileRegressor`, a scikit-learn-compatible quantile regression estimator based on proximal stochastic subgradient descent.

## Overview

The repository supports experiments comparing three approaches to linear quantile regression:

1. **LP**: scikit-learn's `QuantileRegressor` with the HiGHS backend.
2. **IRLS**: statsmodels' `QuantReg` implementation.
3. **proxSGD**: the proposed stochastic optimization method implemented in `SGDQuantileRegressor`.

The experiments evaluate:

- test pinball loss,
- empirical prediction interval coverage,
- mean prediction interval width,
- runtime and scalability,
- robustness to learning rate and mini-batch size,
- behavior under optional `l1` regularization.

## Repository structure
.
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── notebooks/
│   ├── 01_SGD_Quantile_Regression_Utils.ipynb
│   └── 02_SGD_Quantile_Results.ipynb
├── data/
│   └── README.md
└── results/
    ├── figures/
    └── tables/
