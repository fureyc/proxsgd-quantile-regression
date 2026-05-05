"""Proximal stochastic gradient quantile regression estimator."""

import time

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

class SGDQuantileRegressor(RegressorMixin, BaseEstimator):
    '''
    Linear quantile regression estimator trained via proximal stochastic gradient descent.

    This estimator fits a linear model of the form
        y ≈ X beta + b
    by minimizing a regularized quantile regression objective based on the pinball loss:

        min_{beta, b}
            (1/n) * sum_{i=1}^n pinball_loss(y_i - x_i^T beta - b)
            + l1 * ||beta||_1 + (l2/2) * ||beta||_2^2,

    The intercept b is left unpenalized, while the coefficient vector beta is regularized via l1 (l2 regularization
    with a small default coefficient of 10^{-6} is included for numerical stability).

    Optimization is performed using proximal stochastic subgradient descent with
    mini-batches. At each iteration, a stochastic subgradient of the pinball loss is
    computed, followed by a proximal update that applies soft-thresholding (L1) and
    ridge shrinkage (L2) to the coefficients. Two step-size regimes are supported:
    square-root decay and AdaGrad with per-coordinate adaptive learning rates.
    Computations are performed in a user-specified floating-point precision (dtype),
    allowing a trade-off between numerical stability and memory efficiency.

    Optionally, Polyak/Ruppert iterate averaging is used to stabilize optimization and
    reduce variance, particularly in the presence of stochastic updates and nonsmooth
    loss functions. The final model parameters are taken as either the averaged
    iterates or the last raw iterate, depending on the `use_averaging` setting.

    For compatibility with scikit-learn, sample weights are supported via exact sample
    replication for nonnegative integer weights, ensuring that the weighted objective
    is equivalent to a replicated empirical risk minimization problem.

    Parameters
    ----------
    quantile : float, default=0.5
        Target quantile level tau in (0, 1). For example, tau=0.5 corresponds to median
        regression.

    max_iter : int, default=500
        Maximum number of proximal SGD iterations.

    base_lr : float, default=0.2
        Base learning rate used in the step-size schedule.

    batch_size : int, default=32
        Mini-batch size used for stochastic gradient updates.

    l1 : float, default=0.0
        L1 regularization strength (lasso penalty) applied to beta.

    l2 : float, default=0.0
        L2 regularization strength (ridge penalty) applied to beta (can add numerical stability).

    eval_every : int, default=10
        Frequency (in iterations) at which training loss is evaluated and recorded.

    random_state : int or None, default=None
        Seed for the random number generator used in mini-batch sampling.

    verbose : bool, default=False
        If True, print training diagnostics during optimization.

    use_adagrad : bool, default=False
        If True, use AdaGrad adaptive step sizes instead of square-root decay.

    adagrad_eps : float, default=1e-8
        Numerical stabilizer used in AdaGrad step-size computation.

    use_averaging : bool, default=True
        If True, use Polyak/Ruppert averaging of iterates; otherwise, return the last
        raw proximal SGD iterate.

    dtype : numpy.dtype, default=numpy.float64
        Floating-point precision used for internal computations and model parameters.
        Using float32 may reduce memory usage and improve speed, but can reduce numerical
        stability, especially for nonsmooth optimization and small regularization values.

    early_stopping : bool, default=False
        If True, hold out a validation set and apply early stopping based on validation
        pinball loss.

    validation_fraction : float, default=0.1
        Fraction of the training data to set aside as validation when early_stopping=True.

    tol : float, default=1e-4
        Minimum decrease in validation loss required to qualify as an improvement for
        early stopping.

    n_iter_no_change : int, default=10
        Stop training if validation loss does not improve by at least tol for this many
        consecutive evaluation checks (evaluations occur every eval_every iterations).

    restore_best_weights : bool, default=True
        If True and early_stopping=True, restore the model parameters corresponding to
        the best validation loss before returning.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficient vector beta.
    intercept_ : float
        Estimated intercept b.
    n_iter_ : int
        Number of iterations performed.
    history_ : dict
        Training diagnostics recorded during optimization.
        Contains:
        - 'iter': list of iteration indices at which evaluation occurred,
        - 'train_pinball_loss': list of mean training pinball losses at those iterations,
        - 'val_pinball_loss': list of mean validation pinball losses (only recorded when
            early_stopping=True; otherwise this list remains empty).

    best_val_loss_ : float or None
        Best (lowest) validation pinball loss observed during training when early stopping
        is enabled. Set to None if early_stopping=False.

    best_train_loss_ : float or None
        Training pinball loss at the iteration where best_val_loss_ was achieved
        (only recorded when early_stopping=True).

    best_iter_ : int or None
        Iteration index at which best_val_loss_ was achieved when early stopping is enabled.
        Set to None if early_stopping=False.

    Notes
    -----
    This estimator implements a proximal stochastic optimization scheme suitable for
    convex and nonsmooth objectives. It is designed to be compatible with scikit-learn's
    estimator API, including input validation, reproducibility via random_state, and
    sklearn-style sample weighting.
    '''

    def __init__(self, quantile=0.5, max_iter=1000, base_lr=0.5, batch_size=256,
                 l1=0.0, l2=0.0, eval_every=10, random_state=None, verbose=False,
                 use_adagrad=False, adagrad_eps=1e-8, use_averaging=True, dtype=np.float64,
                 early_stopping=False, validation_fraction=0.1, tol=1e-4, n_iter_no_change=10,
                 restore_best_weights=True):
        self.quantile = quantile
        self.max_iter = max_iter
        self.base_lr = base_lr
        self.batch_size = batch_size
        self.l1 = l1
        self.l2 = l2
        self.eval_every = eval_every
        self.random_state = random_state
        self.verbose = verbose
        self.use_adagrad = use_adagrad
        self.adagrad_eps = adagrad_eps
        self.use_averaging = use_averaging
        self.dtype=dtype
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.restore_best_weights = restore_best_weights

    def _train_val_split(self, X, y, rng, validation_fraction):
        """Split X,y into train/val subsets for early stopping."""
        if not (0.0 < validation_fraction < 1.0):
            raise ValueError("validation_fraction must be in (0, 1).")
        n = X.shape[0]
        n_val = max(1, int(np.floor(validation_fraction * n)))
        if n_val >= n:
            raise ValueError("validation_fraction is too large; validation set would be empty train set.")

        perm = rng.permutation(n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    def _get_effective_params(self, beta, intercept, beta_avg, intercept_avg):
        """Return the parameters that are currently considered the model parameters."""
        if self.use_averaging:
            return beta_avg, intercept_avg
        return beta, intercept

    def _copy_params(self, beta, intercept):
        """Deep-copy params for snapshotting best model."""
        return beta.copy(), self.dtype(intercept)

    def _is_improvement(self, current, best):
        """Return True if current is better (lower) than best by at least tol."""
        return current < (best - self.tol)

    def _early_stopping_step(self, loss_value, best_loss, no_improve_count,
                             best_beta, best_intercept, current_beta,
                             current_intercept):
        """
        One update step for early stopping bookkeeping.
        """
        if best_loss is None or self._is_improvement(loss_value, best_loss):
            best_loss = loss_value
            no_improve_count = 0
            best_beta, best_intercept = self._copy_params(current_beta, current_intercept)
            should_stop = False
        else:
            no_improve_count += 1
            should_stop = (no_improve_count >= self.n_iter_no_change)

        return best_loss, no_improve_count, best_beta, best_intercept, should_stop


    def _sqrt_decay_lr(self, base_lr, t):
        '''
        Helper method for .fit(). Square-root learning-rate decay schedule.

        Returns the step size
             base_lr / sqrt(t)
        at iterate t. This is commonly used for stochastic (sub)gradient methods in convex / nonsmooth settings.
        The 1/sqrt(t) decay reduces gradient noise over time and is compatible with
        standard convergence guarantees when using averaged iterates.

        Parameters
        ----------
        base_lr : float
            Initial step size η_0.
        t : int
            Iteration counter (starts at 1 to avoid division by zero).

        Returns
        -------
        step_size : float
            Learning rate at iteration t.
        '''

        return base_lr / np.sqrt(t)

    def _update_running_average(self, avg, current, t):
        '''
        Update a running average of an iterate sequence.
        Since this applies to both the model coeficcient beta and intercept term b,
        we will let theta denote an arbitrary parameter(s).

        Given the previous running average avg = \\bar{theta}_{t-1} and the current iterate
        current = theta_t, this returns
            \\bar{theta}_t = \\bar{theta}_{t-1} + (theta_t - \\bar{theta}_{t-1}) / t,
        which is exactly the arithmetic mean of {\theta_1, ..., theta_t}.

        This type of iterate averaging (Polyak/Ruppert averaging) is frequently used with SGD /
        stochastic subgradient methods to reduce variance and improve stability, especially
        for convex nonsmooth objectives.

        Parameters
        ----------
        avg : float or ndarray
            Previous running average \\bar{θ}_{t-1}.
        current : float or ndarray
            Current iterate theta_t.
        t : int
            Iteration counter (t >= 1).

        Returns
        -------
        avg_new : float or ndarray
            Updated running average \\bar{θ}_t.
        '''

        return avg + (current - avg) / t

    def _pinball_loss_and_subgrad(self, y, y_pred):
        '''
        Helper method used in .fit(). Computes the mean pinball (check) loss for quantile
        tau and a subgradient w.r.t. residuals r = y-y_pred. For residual r_i = y_i - y_pred_i,
        the pinball loss is loss(r_i) = tau * max(r_i, 0) + (1-tau) * max(-r_i, 0).

        For a given quantile level tau, the pinball loss for residual r_i = y_i - y_pred_i is
            pinball_loss(r_i) = tau max(r_i, 0) + (1 - tau) max(-r_i, 0).

        This method returns:
        (i) the mean pinball loss over the samples, and
        (ii) a vector g of subgradients of the pinball loss with respect to the residuals r,
            where each component is chosen as
                g_i = tau - 1{r_i < 0}.
        At the nondifferentiable point r_i = 0, this corresponds to selecting the
        subgradient g_i = tau, which lies in the valid subdifferential interval [tau - 1, tau].

        Note that g represents subgradients with respect to residuals r = y - y_pred.
        Gradients with respect to model parameters are obtained later via the chain rule,
        which introduces an additional minus sign.

        Parameters
        ----------
        y : ndarray of shape (m,)
            True target values for the mini-batch.
        y_pred : ndarray of shape (m,)
            Predicted values for the mini-batch.

        Returns
        -------
        loss : float
            Mean pinball loss over the mini-batch.
        g : ndarray of shape (m,)
            Subgradient of the pinball loss with respect to the residuals r = y - y_pred.
        '''
        r = y - y_pred
        g = self.quantile - (r < 0).astype(float)
        per_sample = (
            self.quantile * np.maximum(r, 0.0)
            + (1.0 - self.quantile) * np.maximum(-r, 0.0)
        )
        loss = float(np.mean(per_sample))
        return loss, g

    def _batch_gradient(self, X, y, beta, intercept):
        '''
        Helper method used in .fit(). Computes the mini-batch subgradient of the mean pinball loss
        w.r.t. (beta, intercept). Our mini-batch objective is:
            L(beta, b) = (1/m) * sum_{i=1}^m pinball_loss(r_i),
        where r_i = y_i - y_pred_i and y_pred_i = x_i^T beta + b.

        In this implementation, g is computed by `_pinball_loss_and_subgrad` as
            g_i = tau - 1{r_i < 0},
        which selects g_i = tau at the nondifferentiable point r_i = 0. See _pinball_loss_and_subgrad
        documentation for more detail.

        By the chain rule, since r_i = y_i - x_i^T beta - b, we have the parial of r_i w.r.t. beta
            = -x_i
        and the partial of r_i w.r.t. the intercept b
         = -1.
        So the subgradients of the mean loss with respect to the parameters are
            ∇_beta L = -(1/m) * X^T g,
            ∇_b    L = -(1/m) * 1^T g = -mean(g),

        where X \\in R^{m×d} is the mini-batch design matrix and g \\in R^m stacks the per-sample
        residual subgradients.

        Notes
        -----
        - This method returns gradients for the *data-fit* term only (pinball loss). L1/L2
        regularization is handled separately via the proximal operator in the update step.
        - The negative signs appear because residuals are defined as r = y - y_pred.

        Parameters
        ----------
        X : ndarray of shape (m, d)
            Mini-batch feature matrix.
        y : ndarray of shape (m,)
            Mini-batch targets.
        beta : ndarray of shape (d,)
            Current coefficient vector.
        intercept : float
            Current intercept.

        Returns
        -------
        grad_beta : ndarray of shape (d,)
            Mini-batch (sub)gradient of the mean pinball loss with respect to beta.
        grad_intercept : float
            Mini-batch (sub)gradient of the mean pinball loss with respect to the intercept.
        '''
        y_pred = X @ beta + intercept
        _, g = self._pinball_loss_and_subgrad(y, y_pred)
        grad_beta = -(X.T @ g) / X.shape[0]
        grad_intercept = -g.mean()

        return grad_beta, grad_intercept

    def _prox_elastic_net(self, beta, l1, l2, step_size):
        '''
        Helper method for .fit(). Applies the proximal operator of the elastic-net penalty to the vector beta.

        This method computes the proximal map of the function
            g(beta) = l1 * ||beta||_1 + (l2/2) * ||beta||_2^2,
        evaluated at the point beta with step_size. That is, it returns

            prox_{step_size, g}(beta)
            = argmin_z { (1/2)||z - beta||_2^2 + step_size * l1 ||z||_1 + step_size * (l2/2)||z||_2^2 }.

        The elastic-net proximal operator is separable across coordinates and admits the
        closed-form expression
            prox_{step_size, g}(beta) = (1 / (1 + step_size * l2)) * S_{step_size, l1}(beta),

        where S_{k}(·) is the soft-thresholding operator defined componentwise by
            S_{k}(beta_j) = sign(beta_j) * max(|beta_j| - k, 0).

        Parameters
        ----------
        beta : ndarray of shape (d,)
            Input vector to be prox-mapped (typically the post-gradient iterate).
        l1 : float
            L1 regularization strength (promotes sparsity).
        l2 : float
            L2 regularization strength (ridge shrinkage). Used for stability.
        step_size : float or ndarray of shape (d,)
            Step size. If an array is provided (e.g., in AdaGrad), the proximal map is
            applied coordinate-wise using per-feature step sizes.

        Returns
        -------
        beta_prox : ndarray of shape (d,)
            Coefficient vector after applying the elastic-net proximal operator.
        '''

        # step_size can be float or shape (n_features,)
        thresh = step_size * l1
        beta_soft = np.sign(beta) * np.maximum(np.abs(beta) - thresh, 0.0)

        return beta_soft / (1.0 + step_size * l2)
        # old implementation
        # beta = beta / (1.0 + step_size * l2)
        # thresh = step_size * l1
        # return np.sign(beta) * np.maximum(np.abs(beta) - thresh, 0.0)

    def _sgd_prox_step(self, X, y, beta, intercept, step_size, *, l1=0.0, l2=0.0):
        '''
        Helper method for .fit(). Performs one proximal SGD (proximal stochastic subgradient) update on a mini-batch.

        We minimize an objective of the form
            f(beta, b) + g(beta),
        where f is the mean pinball loss (quantile regression data-fit term) and
            g(beta) = l1 * ||beta||_1 + (l2/2) * ||beta||_2^2
        is an elastic-net penalty applied to the coefficients only (the intercept is unpenalized).
        Primary purpose of the l2 penalty is numerical stability.

        Given a mini-batch (X, y), this method computes a stochastic subgradient of f and applies:
            v_beta = beta - step_size * \\hat{grad}_beta f(beta, b)   (forward / gradient step)
            b_next = b - step_size * \\hat{grad}_b    f(beta, b)   (SGD step on intercept)
            beta_next = prox_{step_size, g}(v_beta)    (backward / proximal step)

        The proximal operator for the elastic-net penalty is applied coordinate-wise via
        `_prox_elastic_net`, which combines soft-thresholding (L1) and ridge shrinkage (L2).

        Parameters
        ----------
        X : ndarray of shape (m, d)
            Mini-batch feature matrix.
        y : ndarray of shape (m,)
            Mini-batch targets.
        beta : ndarray of shape (d,)
            Current coefficient vector.
        intercept : float
            Current intercept.
        step_size : float or ndarray of shape (d,)
            Step size. Typically a scalar for standard SGD, or a vector of per-feature
            step sizes when using AdaGrad.
        l1 : float, default=0.0
            L1 regularization strength.
        l2 : float, default=0.0
            L2 regularization strength.

        Returns
        -------
        beta_next : ndarray of shape (d,)
            Updated coefficient vector after the gradient and proximal steps.
        intercept_next : float
            Updated intercept after the SGD step.
        '''
        grad_beta, grad_intercept = self._batch_gradient(X, y, beta, intercept)

        beta = beta - step_size * grad_beta
        intercept = intercept - step_size * grad_intercept

        if l1 > 0 or l2 > 0:
            beta = self._prox_elastic_net(beta, l1, l2, step_size)

        return beta, intercept

    def _init_model_params(self, n_features, y):
        '''
        Helper method for .fit(). Initializes model parameters (beta, intercept) for quantile
        regression training.

        We initialize the coefficient vector to zero and choose the intercept as an
        empirical tau-quantile of the targets. With beta = 0, the model predicts a constant
        value b, and the minimizer of the empirical pinball loss
            (1/n) * sum_i pinball_loss(y_i - b)
        over b is any empirical tau-quantile of {y_i}. Thus, setting
            intercept_0 = quantile_tau(y)
        provides a principled "best constant predictor" initialization for the pinball loss.

        Parameters
        ----------
        n_features : int
            Number of features (dimension of beta).
        y : ndarray of shape (n,)
            Target values (used to initialize the intercept via an empirical quantile).

        Returns
        -------
        beta : ndarray of shape (n_features,)
            Initial coefficient vector (all zeros).
        intercept : float
            Initial intercept, chosen as the empirical τ-quantile of y.
        '''
        beta = np.zeros(n_features, dtype=self.dtype)
        intercept = self.dtype(np.quantile(y, self.quantile))

        return beta, intercept

    def _pinball_loss_from_params(self, X, y_true, beta, intercept):
        '''
        Helper method for .fit(). Computes the mean pinball (check) loss for given model parameters.

        This method evaluates the quantile regression objective for fixed parameters
        (beta, intercept) on a dataset (X, y_true). For quantile level tau and residuals
            r_i = y_i - (x_i^T beta + intercept),
        the pinball loss is defined as
            pinball_loss(r_i) = tau * max(r_i, 0) + (1 - tau) * max(-r_i, 0).

        The returned value is the empirical mean of the pinball loss:
            (1/n) * sum_{i=1}^n pinball_loss(r_i).

        Unlike `_pinball_loss_and_subgrad`, this method does not compute subgradients and
        is used purely for objective evaluation, e.g., in diagnostics and model assessment.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        y_true : ndarray of shape (n_samples,)
            True target values.
        beta : ndarray of shape (n_features,)
            Coefficient vector at which to evaluate the loss.
        intercept : float
            Intercept term at which to evaluate the loss.

        Returns
        -------
        loss : float
            Mean pinball loss evaluated at (beta, intercept).
        '''
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = X @ beta + intercept
        r = y_true - y_pred
        per_sample = (
            self.quantile * np.maximum(r, 0.0)+ (1.0 - self.quantile) * np.maximum(-r, 0.0)
            )

        return float(np.mean(per_sample))

    def _kkt_stationarity_vector(self, X, y, beta, intercept):
        """
        Compute the unregularized stationarity proxy

            g = (1/n) X^T s,

        where s_i = tau - 1{r_i < 0} and
        r_i = y_i - (x_i^T beta + intercept).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)
        beta : ndarray of shape (n_features,)
        intercept : float

        Returns
        -------
        g : ndarray of shape (n_features,)
            Stationarity proxy vector for the empirical pinball loss.
        """
        X = np.asarray(X, dtype=self.dtype)
        y = np.asarray(y, dtype=self.dtype).reshape(-1)
        beta = np.asarray(beta, dtype=self.dtype).reshape(-1)
        intercept = self.dtype(intercept)

        r = y - (X @ beta + intercept)
        s = self.quantile - (r < 0).astype(self.dtype)
        g = (X.T @ s) / X.shape[0]

        return np.asarray(g, dtype=self.dtype)

    def _kkt_grad_inf(self, X, y, beta, intercept):
        """
        Compute ||g||_inf for the unregularized stationarity proxy

            g = (1/n) X^T s.
        """
        g = self._kkt_stationarity_vector(X, y, beta, intercept)
        return float(np.max(np.abs(g)))

    def fit(
        self,
        X,
        y,
        *,
        sample_weight=None,
        X_monitor=None,
        y_monitor=None,
        monitor_name="monitor",
        record_monitor_loss=True,
        record_train_kkt=False,
        record_monitor_kkt=False,
        store_param_path=False,
    ):
        '''
        Fits a linear quantile regression model using proximal stochastic gradient descent.

        This method solves the regularized quantile regression problem

            min_{beta, b}
                (1/n) * sum_{i=1}^n pinball_loss(y_i - x_i^T beta - b)
                + l1 * ||beta||_1 + (l2/2) * ||beta||_2^2,

        where pinball_loss is the pinball (check) loss at quantile level tau, beta is the coefficient
        vector, and b is the intercept. The intercept is unpenalized, while the coefficients
        are regularized via an l1 penalty (l2 term is included for numerical stability with default
        coefficient 10^{-6} so l1 regularization solution is minimally impacted).

        Optimization is performed using proximal stochastic subgradient descent with
        mini-batches. At each iteration:

        1) A mini-batch is sampled from the training data.
        2) A stochastic subgradient of the pinball loss is computed.
        3) A gradient step is taken on (beta, b).
        4) The elastic-net proximal operator is applied to beta.
        5) Running averages of beta and b are updated (Polyak/Ruppert averaging, if use_averaging=True).

        Two step-size regimes are supported:

        - Square-root decay SGD: step_size_t = base_lr / sqrt(t).
        - AdaGrad: per-coordinate adaptive step sizes based on accumulated squared gradients.

        All computations are carried out in the precision specified by `dtype`.

        By default, the final model parameters are taken as Polyak/Ruppert averages of the
        iterates rather than the last raw iterate, which typically improves stability for
        stochastic and nonsmooth optimization. If `use_averaging=False`, the last raw
        iterate is returned instead.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray of shape (n_samples,)
            Target values.
        sample_weight : ndarray of shape (n_samples,), optional
            Sample weights. For sklearn-compatible behavior, integer-valued weights are
            implemented via explicit data replication.
        X_monitor : ndarray of shape (n_monitor, n_features), optional
            External monitoring feature matrix. This can be a held-out test set used
            to record convergence diagnostics during training.
        y_monitor : ndarray of shape (n_monitor,), optional
            Targets corresponding to X_monitor.
        monitor_name : str, default="monitor"
            Descriptive label for the external monitoring set. Currently cosmetic.
        record_monitor_loss : bool, default=True
            If True and X_monitor/y_monitor are provided, record monitor pinball loss.
        record_train_kkt : bool, default=False
            If True, record the training-set KKT proxy ||g||_inf at evaluation times.
        record_monitor_kkt : bool, default=False
            If True and X_monitor/y_monitor are provided, record the monitor-set KKT proxy
            ||g||_inf at evaluation times.
        store_param_path : bool, default=False
            If True, store the effective parameter path at evaluation times.

        Returns
        -------
        self : object
            Fitted estimator. The learned parameters are stored in `coef_` and `intercept_`,
            and training diagnostics are available in `history_`.
        '''

        if not (0.0 < self.quantile < 1.0):
            raise ValueError('quantile must be in (0, 1)')
        if not isinstance(self.batch_size, (int, np.integer)) or self.batch_size <= 0:
            raise ValueError('batch_size must be a positive integer.')
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter <= 0:
            raise ValueError('max_iter must be a positive integer.')
        if not isinstance(self.eval_every, (int, np.integer)) or self.eval_every <= 0:
            raise ValueError('eval_every must be a positive integer.')
        if self.early_stopping:
            if not (0.0 < self.validation_fraction < 1.0):
                raise ValueError("validation_fraction must be in (0, 1).")
            if not isinstance(self.n_iter_no_change, (int, np.integer)) or self.n_iter_no_change <= 0:
                raise ValueError("n_iter_no_change must be a positive integer.")
            if self.tol < 0:
                raise ValueError("tol must be nonnegative.")

        # --- sklearn input validation ---
        X, y = validate_data(self, X, y, accept_sparse=False, y_numeric=True, reset=True)
        X = np.asarray(X, dtype=self.dtype)
        y = np.asarray(y, dtype=self.dtype).reshape(-1)

        # --- optional external monitor data validation ---
        if (X_monitor is None) != (y_monitor is None):
            raise ValueError("X_monitor and y_monitor must either both be provided or both be None.")

        if X_monitor is not None:
            X_monitor = np.asarray(X_monitor, dtype=self.dtype)
            y_monitor = np.asarray(y_monitor, dtype=self.dtype).reshape(-1)
            if X_monitor.ndim != 2:
                raise ValueError("X_monitor must be a 2D array.")
            if y_monitor.ndim != 1:
                raise ValueError("y_monitor must be a 1D array.")
            if X_monitor.shape[0] != y_monitor.shape[0]:
                raise ValueError(
                    f"X_monitor and y_monitor must have the same number of rows; "
                    f"got {X_monitor.shape[0]} and {y_monitor.shape[0]}."
                )
            if X_monitor.shape[1] != X.shape[1]:
                raise ValueError(
                    f"X_monitor must have {X.shape[1]} features, got {X_monitor.shape[1]}."
                )

        # --- sklearn-compatible sample_weight: expand to repeated/removed rows ---
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float).reshape(-1)

            if sw.shape[0] != X.shape[0]:
                raise ValueError(
                    f'sample_weight must have shape ({X.shape[0]},), got {sw.shape} instead'
                )
            if np.any(sw < 0):
                raise ValueError('sample_weight cannot contain negative values.')
            if not np.all(np.isfinite(sw)):
                raise ValueError('sample_weight must be finite.')

            if not np.all(sw == np.floor(sw)):
                raise ValueError(
                    'For sklearn-compatible sample_weight equivalence, sample_weight must be '
                    'nonnegative integers (0, 1, 2, ...).'
                )

            sw_int = sw.astype(int)
            total = int(sw_int.sum())
            if total <= 0:
                raise ValueError('sample_weight must sum to a positive value.')

            max_expanded = 5_000_000
            if total > max_expanded:
                raise ValueError(
                    f'Expanded dataset would have {total} rows, which is too large for exact '
                    f'sample_weight equivalence in memory.'
                )

            X = np.repeat(X, sw_int, axis=0)
            y = np.repeat(y, sw_int, axis=0)

        # --- RNG ---
        rng = check_random_state(self.random_state)

        # --- optional internal train/val split for early stopping ---
        X_val = y_val = None
        if self.early_stopping:
            X, y, X_val, y_val = self._train_val_split(X, y, rng, self.validation_fraction)

        # --- Dataset dimensions and effective batch size ---
        n_samples, n_features = X.shape
        bs = min(self.batch_size, n_samples)
        if bs <= 0:
            raise ValueError('batch_size must be a positive integer.')

        # --- Model parameter initialization ---
        beta, intercept = self._init_model_params(n_features, y)

        # --- Averaged parameters ---
        beta_avg = beta.copy()
        intercept_avg = intercept

        # --- AdaGrad state initialization ---
        G_beta = np.zeros_like(beta, dtype=self.dtype)
        G_intercept = self.dtype(0.0)
        eps = self.dtype(self.adagrad_eps)

        # --- Training diagnostics ---
        self.history_ = {
            'iter': [],
            'train_pinball_loss': [],
            'val_pinball_loss': [],
            'monitor_pinball_loss': [],
            'train_kkt_grad_inf': [],
            'monitor_kkt_grad_inf': [],
            'coef_path': [],
            'intercept_path': [],
            'monitor_name': monitor_name,
        }

        # --- "best" attributes ---
        self.best_val_loss_ = None
        self.best_iter_ = None
        self.best_train_loss_ = None

        # --- Early stopping state ---
        best_loss = None
        best_beta = None
        best_intercept = None
        no_improve_count = 0

        for t in range(1, self.max_iter + 1):
            idx = rng.choice(n_samples, size=bs, replace=False)
            X_batch, y_batch = X[idx], y[idx]

            if not self.use_adagrad:
                # --- Prox-SGD with diminishing step size step_size_t = base_lr / sqrt(t) ---
                step_size = self._sqrt_decay_lr(self.base_lr, t)

                beta, intercept = self._sgd_prox_step(
                    X_batch,
                    y_batch,
                    beta,
                    intercept,
                    step_size,
                    l1=self.l1,
                    l2=self.l2,
                )

            else:
                # --- AdaGrad update ---
                grad_beta, grad_intercept = self._batch_gradient(X_batch, y_batch, beta, intercept)

                G_beta += grad_beta**2
                G_intercept += self.dtype(grad_intercept**2)

                step_beta = self.base_lr / (np.sqrt(G_beta) + eps)
                step_intercept = self.base_lr / (np.sqrt(G_intercept) + eps)

                beta = beta - step_beta * grad_beta
                intercept = intercept - step_intercept * grad_intercept

                if self.l1 > 0 or self.l2 > 0:
                    beta = self._prox_elastic_net(beta, self.l1, self.l2, step_beta)

            # --- Parameter averaging (optional) ---
            if self.use_averaging:
                beta_avg = self._update_running_average(beta_avg, beta, t)
                intercept_avg = self._update_running_average(intercept_avg, intercept, t)
            else:
                beta_avg = beta
                intercept_avg = intercept

            # --- Periodic evaluation ---
            if t % self.eval_every == 0 or t == self.max_iter:
                eff_beta, eff_intercept = self._get_effective_params(
                    beta, intercept, beta_avg, intercept_avg
                )

                # Compute diagnostics first
                train_loss = self._pinball_loss_from_params(X, y, eff_beta, eff_intercept)

                val_loss = None
                if self.early_stopping:
                    val_loss = self._pinball_loss_from_params(X_val, y_val, eff_beta, eff_intercept)

                monitor_loss = None
                if record_monitor_loss and X_monitor is not None:
                    monitor_loss = self._pinball_loss_from_params(
                        X_monitor, y_monitor, eff_beta, eff_intercept
                    )

                train_kkt = None
                if record_train_kkt:
                    train_kkt = self._kkt_grad_inf(X, y, eff_beta, eff_intercept)

                monitor_kkt = None
                if record_monitor_kkt and X_monitor is not None:
                    monitor_kkt = self._kkt_grad_inf(
                        X_monitor, y_monitor, eff_beta, eff_intercept
                    )

                # Best-iteration bookkeeping for early stopping
                if self.early_stopping and val_loss is not None:
                    if best_loss is None or self._is_improvement(val_loss, best_loss):
                        self.best_iter_ = t
                        self.best_val_loss_ = val_loss
                        self.best_train_loss_ = train_loss

                # Append all diagnostics
                self.history_['iter'].append(t)
                self.history_['train_pinball_loss'].append(train_loss)
                self.history_['val_pinball_loss'].append(val_loss)
                self.history_['monitor_pinball_loss'].append(monitor_loss)
                self.history_['train_kkt_grad_inf'].append(train_kkt)
                self.history_['monitor_kkt_grad_inf'].append(monitor_kkt)

                if store_param_path:
                    self.history_['coef_path'].append(np.asarray(eff_beta, dtype=self.dtype).copy())
                    self.history_['intercept_path'].append(self.dtype(eff_intercept))
                else:
                    self.history_['coef_path'].append(None)
                    self.history_['intercept_path'].append(None)

                # Verbose printing
                if self.verbose:
                    msg = f"Iter {t}, train pinball loss = {train_loss:.6f}"
                    if val_loss is not None:
                        msg += f", val pinball loss = {val_loss:.6f}"
                    if monitor_loss is not None:
                        msg += f", {monitor_name} pinball loss = {monitor_loss:.6f}"
                    if train_kkt is not None:
                        msg += f", train KKT_inf = {train_kkt:.6e}"
                    if monitor_kkt is not None:
                        msg += f", {monitor_name} KKT_inf = {monitor_kkt:.6e}"
                    print(msg)

                # --- Early stopping update (still based on internal validation loss) ---
                if self.early_stopping:
                    best_loss, no_improve_count, best_beta, best_intercept, should_stop = (
                        self._early_stopping_step(
                            loss_value=val_loss,
                            best_loss=best_loss,
                            no_improve_count=no_improve_count,
                            best_beta=best_beta,
                            best_intercept=best_intercept,
                            current_beta=eff_beta,
                            current_intercept=eff_intercept,
                        )
                    )

                    if should_stop:
                        if self.verbose:
                            print(
                                f"Early stopping at iter {t}: no improvement "
                                f"for {self.n_iter_no_change} eval checks "
                                f"(best val loss={best_loss:.6f})."
                            )
                        break

        # --- Final parameter selection ---
        if self.early_stopping and self.restore_best_weights and best_beta is not None:
            beta_final, intercept_final = best_beta, best_intercept
        else:
            if self.use_averaging:
                beta_final, intercept_final = beta_avg, intercept_avg
            else:
                beta_final, intercept_final = beta, intercept

        self.coef_ = np.asarray(beta_final, dtype=self.dtype).reshape(-1)
        self.intercept_ = self.dtype(intercept_final)
        self.n_iter_ = t

        return self

    def predict(self, X):
        '''
        Predict targets using the fitted linear quantile regression model.

        Given a fitted model with learned parameters (coef_, intercept_), this returns
            y_pred = X @ coef_ + intercept_.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix for which to generate predictions.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values (estimated τ-quantile under the linear model).
        '''
        check_is_fitted(self, ['coef_', 'intercept_'])
        X = validate_data(self, X, accept_sparse=False, reset=False)
        X = np.asarray(X, dtype=self.dtype)
        y_pred = X @ self.coef_ + self.intercept_

        return np.asarray(y_pred).reshape(-1)

    def pinball_loss(self, X, y_true):
        '''
        Compute the mean pinball (check) loss of the fitted model on a dataset.

        For quantile level tau and residuals r_i = y_i - (x_i^T coef_ + intercept_),
        the pinball loss is
            tau * max(r_i, 0) + (1 - tau) * max(-r_i, 0).
        This method returns the empirical mean (1/n) * sum_i pinball_loss(r_i).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        y_true : ndarray of shape (n_samples,)
            True target values.

        Returns
        -------
        loss : float
            Mean pinball loss of the fitted model on (X, y_true).
        '''
        check_is_fitted(self, ['coef_', 'intercept_'])
        X = validate_data(self, X, accept_sparse=False, reset=False)
        X = np.asarray(X, dtype=self.dtype)
        y_true = np.asarray(y_true, dtype=self.dtype).reshape(-1)

        return self._pinball_loss_from_params(X, y_true, self.coef_, self.intercept_)
