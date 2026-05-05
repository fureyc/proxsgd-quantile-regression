"""
Quick start example for SGDQuantileRegressor.

This script fits a single linear quantile regression model using the
proximal stochastic gradient estimator and reports test pinball loss.
"""

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from proxsgd_quantile import SGDQuantileRegressor


def main():
    # Load example regression data
    X, y = fetch_california_housing(return_X_y=True)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=33,
    )

    # Standardize features before fitting a linear model
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit an upper conditional quantile model
    quantile = 0.9

    model = SGDQuantileRegressor(
        quantile=quantile,
        base_lr=0.5,
        max_iter=1000,
        batch_size=256,
        use_adagrad=True,
        use_averaging=True,
        random_state=33,
    )

    model.fit(X_train, y_train)

    # Predict the fitted conditional quantile on test data
    y_pred = model.predict(X_test)

    # Evaluate using pinball loss
    sklearn_loss = mean_pinball_loss(y_test, y_pred, alpha=quantile)
    estimator_loss = model.pinball_loss(X_test, y_test)

    print("SGDQuantileRegressor quick start")
    print("--------------------------------")
    print(f"Quantile level: {quantile}")
    print(f"Test pinball loss, sklearn:   {sklearn_loss:.4f}")
    print(f"Test pinball loss, estimator: {estimator_loss:.4f}")
    print(f"Number of iterations: {model.n_iter_}")
    print(f"Coefficient shape: {model.coef_.shape}")
    print(f"Intercept: {model.intercept_:.4f}")


if __name__ == "__main__":
    main()
