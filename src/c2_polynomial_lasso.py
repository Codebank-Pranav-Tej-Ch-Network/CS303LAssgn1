"""
C2: Polynomial Fitting and Lasso Regularization.

Run:
    python -m src.c2_polynomial_lasso

Sub-part (a) uses ONLY the from-scratch normal_equation() (no sklearn).
Sub-part (b)/(c) explicitly use sklearn's Ridge/Lasso as the assignment asks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso

from src.utils import normal_equation, rmse, predict, vandermonde

OUT = "outputs/c2"


def make_dataset():
    np.random.seed(1)
    n = 80
    x = np.sort(np.random.uniform(-1, 1, n))
    y_true = np.sin(2 * np.pi * x) + 0.5 * x ** 2
    y = y_true + np.random.normal(0, 0.25, n)
    return x, y


def part_a(x_train, y_train, x_test, y_test, log):
    log("\n--- (a) Polynomial fits via from-scratch normal equation ---")
    degrees = [1, 3, 9, 15]
    table_rows = []

    x_plot = np.linspace(-1, 1, 300)
    plt.figure(figsize=(8, 6))
    plt.scatter(x_train, y_train, s=15, color="black", alpha=0.6, label="training data")

    for p in degrees:
        A_train = vandermonde(x_train, p)
        A_test = vandermonde(x_test, p)
        theta = normal_equation(A_train, y_train)

        train_rmse = rmse(y_train, predict(A_train, theta))
        test_rmse = rmse(y_test, predict(A_test, theta))
        table_rows.append((p, train_rmse, test_rmse))

        A_plot = vandermonde(x_plot, p)
        plt.plot(x_plot, predict(A_plot, theta), label=f"p={p}")

    plt.ylim(-2.5, 2.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Polynomial fits (normal equation) for varying degree p")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/c2_poly_fits.png", dpi=150)
    plt.close()

    log(f"{'p':>4} | {'train RMSE':>12} | {'test RMSE':>12}")
    for p, tr, te in table_rows:
        log(f"{p:>4} | {tr:12.4f} | {te:12.4f}")

    log("p=1 underfits (a straight line cannot capture the sine shape, so both "
        "train and test RMSE are high). p=9 and p=15 drive training RMSE very "
        "low, but p=15's test RMSE is much worse than its training RMSE -- "
        "classic overfitting from an overly flexible polynomial with only 56 "
        "training points.")
    return table_rows


def part_b(x_train, y_train, x_test, y_test, log):
    log("\n--- (b) Ridge vs Lasso across lambda, for p=15 ---")
    p = 15
    A_train = vandermonde(x_train, p)
    A_test = vandermonde(x_test, p)
    # Drop the bias column for sklearn's models (they fit their own intercept);
    # use columns x^1..x^p as features.
    X_train, X_test = A_train[:, 1:], A_test[:, 1:]

    lambdas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    ridge_train, ridge_test, lasso_train, lasso_test = [], [], [], []
    ridge_models, lasso_models = {}, {}

    for lam in lambdas:
        ridge = Ridge(alpha=lam, max_iter=100000)
        ridge.fit(X_train, y_train)
        ridge_models[lam] = ridge
        ridge_train.append(rmse(y_train, ridge.predict(X_train)))
        ridge_test.append(rmse(y_test, ridge.predict(X_test)))

        lasso = Lasso(alpha=lam, max_iter=100000)
        lasso.fit(X_train, y_train)
        lasso_models[lam] = lasso
        lasso_train.append(rmse(y_train, lasso.predict(X_train)))
        lasso_test.append(rmse(y_test, lasso.predict(X_test)))

    plt.figure(figsize=(8, 6))
    plt.plot(lambdas, ridge_train, "o-", label="Ridge train RMSE")
    plt.plot(lambdas, ridge_test, "o--", label="Ridge test RMSE")
    plt.plot(lambdas, lasso_train, "s-", label="Lasso train RMSE")
    plt.plot(lambdas, lasso_test, "s--", label="Lasso test RMSE")
    plt.xscale("log")
    plt.xlabel("lambda (log scale)")
    plt.ylabel("RMSE")
    plt.title("Train/Test RMSE vs lambda for Ridge and Lasso (p=15)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/c2_ridge_lasso_rmse_vs_lambda.png", dpi=150)
    plt.close()

    best_ridge_lam = lambdas[int(np.argmin(ridge_test))]
    best_lasso_lam = lambdas[int(np.argmin(lasso_test))]
    log(f"lambdas tested: {lambdas}")
    log(f"Ridge test RMSE: {[round(v,4) for v in ridge_test]}")
    log(f"Lasso test RMSE: {[round(v,4) for v in lasso_test]}")
    log(f"best lambda (Ridge, min test RMSE) = {best_ridge_lam}")
    log(f"best lambda (Lasso, min test RMSE) = {best_lasso_lam}")

    return ridge_models, lasso_models, best_ridge_lam, best_lasso_lam


def part_c(ridge_models, lasso_models, best_ridge_lam, best_lasso_lam, log):
    log("\n--- (c) Coefficient profile: best Lasso vs best Ridge ---")
    best_lasso = lasso_models[best_lasso_lam]
    best_ridge = ridge_models[best_ridge_lam]

    degrees = np.arange(1, len(best_lasso.coef_) + 1)  # x^1 .. x^15
    threshold = 1e-3
    n_zero = int(np.sum(np.abs(best_lasso.coef_) < threshold))

    width = 0.4
    plt.figure(figsize=(9, 6))
    plt.bar(degrees - width / 2, np.abs(best_lasso.coef_), width=width, label=f"Lasso (lambda={best_lasso_lam})")
    plt.bar(degrees + width / 2, np.abs(best_ridge.coef_), width=width, label=f"Ridge (lambda={best_ridge_lam})")
    plt.xlabel("polynomial degree j")
    plt.ylabel("|theta_j|")
    plt.title("Coefficient magnitude vs degree: best Lasso vs best Ridge")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/c2_coef_bar_lasso_vs_ridge.png", dpi=150)
    plt.close()

    log(f"Best Lasso (lambda={best_lasso_lam}) has {n_zero} / {len(degrees)} "
        f"coefficients below {threshold} in magnitude (effectively zero).")
    log("Lasso produces a sparse coefficient profile -- it zeroes out most "
        "high-degree terms and keeps only a few informative ones -- while "
        "Ridge shrinks all coefficients smoothly toward zero without setting "
        "any of them exactly to zero, so its bar chart shows many small but "
        "nonzero bars across every degree.")


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []
    def log(msg):
        print(msg)
        lines.append(str(msg))

    x, y = make_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    part_a(x_train, y_train, x_test, y_test, log)
    ridge_models, lasso_models, best_ridge_lam, best_lasso_lam = part_b(
        x_train, y_train, x_test, y_test, log)
    part_c(ridge_models, lasso_models, best_ridge_lam, best_lasso_lam, log)

    with open(f"{OUT}/report.txt", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
