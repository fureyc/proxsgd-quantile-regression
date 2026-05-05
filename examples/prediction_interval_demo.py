"""
Prediction interval demo for SGDQuantileRegressor.

This script fits lower and upper quantile regression models using the
proximal stochastic gradient estimator, computes empirical coverage and
mean interval width on a test set, and displays a simple plot of the
resulting prediction intervals.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from proxsgd_quantile import SGDQuantileRegressor


def main():
    # Load example regression data
    X, y = fetch_california_housing(return_X_y=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=33,
    )

    # Standardize features before fitting linear models
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Lower and upper quantile levels for a nominal 80% interval
    q_low = 0.1
    q_high = 0.9

    # Fit lower quantile model
    lower_model = SGDQuantileRegressor(
        quantile=q_low,
        base_lr=0.5,
        max_iter=1000,
        batch_size=256,
        use_adagrad=True,
        use_averaging=True,
        random_state=42,
    )

    # Fit upper quantile model
    upper_model = SGDQuantileRegressor(
        quantile=q_high,
        base_lr=0.5,
        max_iter=1000,
        batch_size=256,
        use_adagrad=True,
        use_averaging=True,
        random_state=42,
    )

    lower_model.fit(X_train, y_train)
    upper_model.fit(X_train, y_train)

    # Predict lower and upper conditional quantiles
    q_lower = lower_model.predict(X_test)
    q_upper = upper_model.predict(X_test)

    # Compute interval statistics
    coverage = ((y_test >= q_lower) & (y_test <= q_upper)).mean()
    mean_width = (q_upper - q_lower).mean()

    print("SGDQuantileRegressor prediction interval demo")
    print("---------------------------------------------")
    print(f"Lower quantile: {q_low}")
    print(f"Upper quantile: {q_high}")
    print(f"Nominal coverage: {q_high - q_low:.1%}")
    print(f"Empirical coverage: {coverage:.3f}")
    print(f"Mean interval width: {mean_width:.3f}")

    # ---- Simple visualization ----
    # For plotting, display a subset of test points sorted by interval midpoint
    # so the interval band is easier to read.
    n_plot = min(100, len(y_test))
    midpoint = 0.5 * (q_lower + q_upper)

    order = np.argsort(midpoint[:n_plot])

    x_plot = np.arange(n_plot)
    y_plot = y_test[:n_plot][order]
    lower_plot = q_lower[:n_plot][order]
    upper_plot = q_upper[:n_plot][order]

    plt.figure(figsize=(10, 6))
    plt.fill_between(x_plot, lower_plot, upper_plot, alpha=0.3, label="80% prediction interval")
    plt.plot(x_plot, lower_plot, label="Lower quantile (0.1)")
    plt.plot(x_plot, upper_plot, label="Upper quantile (0.9)")
    plt.scatter(x_plot, y_plot, s=20, label="Observed response")

    plt.xlabel("Test samples (sorted by interval midpoint)")
    plt.ylabel("Median house value")
    plt.title("Prediction intervals from SGDQuantileRegressor")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
