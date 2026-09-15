"""
utils.py
--------
Shared, from-scratch numerical routines used across C1, C2, C3.

Nothing here calls sklearn's LinearRegression/Ridge/Lasso — those are only
used later in C2(b) where the assignment explicitly asks for them.
"""

import numpy as np


def normal_equation(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Closed-form OLS solution: theta_hat = (A^T A)^{-1} A^T y

    Parameters
    ----------
    A : (n, d) design matrix (include a column of ones for the intercept
        yourself before calling this).
    y : (n,) target vector.

    Returns
    -------
    theta_hat : (d,) parameter vector.
    """
    AtA = A.T @ A
    Aty = A.T @ y
    theta_hat = np.linalg.solve(AtA, Aty)  # more stable than explicit inverse
    return theta_hat


def ridge_normal_equation(A: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Regularized closed-form solution:
        theta_hat = (A^T A + lambda * I)^{-1} A^T y

    NOTE: by convention we do NOT penalize the intercept column (column 0,
    assumed to be all ones). This matches how sklearn's Ridge behaves
    (it centers the intercept out) and keeps comparisons apples-to-apples.
    """
    d = A.shape[1]
    I = np.eye(d)
    I[0, 0] = 0.0  # don't regularize the bias term
    AtA = A.T @ A
    Aty = A.T @ y
    theta_hat = np.linalg.solve(AtA + lam * I, Aty)
    return theta_hat


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def predict(A: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return A @ theta


def gradient_descent_linreg(A: np.ndarray, y: np.ndarray, lr: float,
                             n_iters: int, theta0: np.ndarray = None):
    """
    Full-batch gradient descent minimizing J(theta) = ||y - A theta||_2^2.

    Gradient of J w.r.t. theta:
        dJ/dtheta = -2 * A^T (y - A theta)

    Returns
    -------
    theta : (d,) final parameter vector
    history : list of J(theta_t) for t = 0 .. n_iters-1
    """
    n, d = A.shape
    theta = np.zeros(d) if theta0 is None else theta0.copy()
    history = []
    for _ in range(n_iters):
        residual = y - A @ theta
        cost = float(residual @ residual)  # ||y - A theta||_2^2
        history.append(cost)
        grad = -2.0 * (A.T @ residual)
        theta = theta - lr * grad
    return theta, history


def vandermonde(x: np.ndarray, p: int) -> np.ndarray:
    """
    Build a polynomial design matrix [1, x, x^2, ..., x^p] manually
    (columns ordered from lowest to highest degree, unlike numpy.vander's
    default which is highest-to-lowest -- we build it ourselves as the
    assignment allows either).
    """
    n = x.shape[0]
    A = np.ones((n, p + 1))
    for j in range(1, p + 1):
        A[:, j] = x ** j
    return A
