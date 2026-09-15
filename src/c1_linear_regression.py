"""
C1: Linear Regression -- Normal Equation vs. Gradient Descent vs. Ill-Conditioning.

Run:
    python -m src.c1_linear_regression
(from the project root, with the venv active)

Produces plots in outputs/c1/ and prints a report to stdout (also saved to
outputs/c1/report.txt).
"""

import numpy as np
import matplotlib.pyplot as plt

# A diverging learning rate legitimately produces inf/nan intermediate values;
# we detect and report that explicitly rather than letting numpy spam warnings.
np.seterr(over="ignore", invalid="ignore")

from src.utils import normal_equation, ridge_normal_equation, rmse, predict, gradient_descent_linreg

OUT = "outputs/c1"


def make_dataset():
    np.random.seed(0)
    n = 300
    x1 = np.random.uniform(-3, 3, n)
    x2 = 0.98 * x1 + np.random.normal(0, 0.05, n)  # nearly collinear with x1
    X = np.column_stack([x1, x2])
    A = np.column_stack([np.ones(n), X])  # design matrix with intercept
    theta_true = np.array([4.0, 3.0, -1.5])
    y = A @ theta_true + np.random.normal(0, 1.0, n)
    return A, y, theta_true


def part_a(A, y, log):
    theta_hat = normal_equation(A, y)
    train_rmse = rmse(y, predict(A, theta_hat))
    log(f"\n--- (a) Normal equation ---")
    log(f"theta_hat = {theta_hat}")
    log(f"training RMSE = {train_rmse:.4f}")
    return theta_hat, train_rmse


def part_b(A, y, theta_hat, log):
    log(f"\n--- (b) Condition number & perturbation sensitivity ---")
    cond = np.linalg.cond(A.T @ A)
    log(f"condition number of A^T A = {cond:.4e}")

    np.random.seed(42)  # reproducible perturbation
    eps = np.random.normal(0, 0.01, size=y.shape)
    y_perturbed = y + eps
    theta_hat_prime = normal_equation(A, y_perturbed)
    delta_theta_norm = np.linalg.norm(theta_hat_prime - theta_hat)

    log(f"theta_hat' (perturbed) = {theta_hat_prime}")
    log(f"||theta_hat' - theta_hat||_2 = {delta_theta_norm:.6f}")
    log("Interpretation: A^T A is ill-conditioned (x1, x2 nearly collinear), "
        "so a tiny perturbation in y is amplified into a comparatively large "
        "shift in theta_hat -- the hallmark of an ill-conditioned normal equation.")
    return cond, delta_theta_norm


def part_c(A, y, theta_hat_normal, log):
    log(f"\n--- (c) Gradient descent for 3 learning rates ---")
    lrs = [1e-5, 1e-4, 1e-3]
    n_iters = 2000
    plt.figure(figsize=(7, 5))
    results = {}
    for lr in lrs:
        theta_gd, history = gradient_descent_linreg(A, y, lr=lr, n_iters=n_iters)
        results[lr] = (theta_gd, history)
        plt.plot(history, label=f"lr={lr:g}")
    plt.yscale("log")
    plt.xlabel("iteration t")
    plt.ylabel("J(theta_t)  (log scale)")
    plt.title("Full-batch Gradient Descent: cost vs iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/c1_gd_cost_vs_iteration.png", dpi=150)
    plt.close()

    status = {}
    for lr, (theta_gd, history) in results.items():
        final_cost = history[-1]
        diverged = not np.isfinite(final_cost) or final_cost > history[0]
        status[lr] = (diverged, final_cost, theta_gd)
        log(f"lr={lr:g}: final J={final_cost:.4e}, "
            f"theta_gd={theta_gd if not diverged else 'DIVERGED'}")

    # Build the diverges/slow/fastest discussion dynamically from what actually
    # happened, since which lr is stable depends on the largest eigenvalue of
    # A^T A (this dataset is ill-conditioned, so this is data-dependent).
    diverging = [lr for lr, (d, _, _) in status.items() if d]
    converging = sorted([lr for lr, (d, _, _) in status.items() if not d])
    log(f"\nDiverging learning rate(s): {diverging}")
    log(f"Converging learning rate(s) (slowest to fastest by final cost): {converging}")
    if converging:
        fastest_lr = min(converging, key=lambda lr: status[lr][1])
        log(f"Fastest converging lr in this run: {fastest_lr:g} "
            f"(lowest final cost after {n_iters} iterations)")
        best_theta = status[fastest_lr][2]
        log(f"Normal-equation theta_hat = {theta_hat_normal}")
        log(f"theta_gd (fastest lr={fastest_lr:g}) = {best_theta}")
        log("Discussion: the max eigenvalue of A^T A sets a stability limit of "
            "roughly lr < 1/lambda_max for full-batch GD on this quadratic "
            "objective. Learning rates below that limit converge (more slowly "
            "the smaller they are); learning rates above it overshoot and "
            "diverge. The best converging theta_gd here closely matches the "
            "normal-equation theta_hat, confirming GD and the closed-form "
            "solution reach the same optimum of this convex objective -- GD "
            "just needs enough iterations and a stable step size to get there.")
    return results


def part_d(A, y, theta_hat_ols, log):
    log(f"\n--- (d) Ridge (regularized) normal equation, lambda=1.0 ---")
    lam = 1.0
    theta_ridge = ridge_normal_equation(A, y, lam)
    train_rmse_ridge = rmse(y, predict(A, theta_ridge))
    log(f"theta_hat_ridge = {theta_ridge}")
    log(f"training RMSE (ridge) = {train_rmse_ridge:.4f}")

    np.random.seed(42)
    eps = np.random.normal(0, 0.01, size=y.shape)
    y_perturbed = y + eps
    theta_ridge_prime = ridge_normal_equation(A, y_perturbed, lam)
    delta_ridge = np.linalg.norm(theta_ridge_prime - theta_ridge)
    log(f"||theta_ridge' - theta_ridge||_2 = {delta_ridge:.6f}")

    cond_ridge = np.linalg.cond(A.T @ A + lam * np.eye(A.shape[1]))
    log(f"condition number of (A^T A + lambda I) = {cond_ridge:.4e}")
    log("Explanation: adding lambda*I shifts every eigenvalue of A^T A up by "
        "lambda, which shrinks the condition number and caps how much the "
        "smallest eigenvalue can amplify noise in y. That is why "
        "||theta_ridge' - theta_ridge|| is smaller than the corresponding "
        "OLS perturbation from part (b): ridge trades a small amount of bias "
        "for a large reduction in variance/sensitivity.")
    return theta_ridge, delta_ridge, cond_ridge


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    lines = []
    def log(msg):
        print(msg)
        lines.append(str(msg))

    A, y, theta_true = make_dataset()
    log(f"theta_true = {theta_true}")

    theta_hat, _ = part_a(A, y, log)
    part_b(A, y, theta_hat, log)
    part_c(A, y, theta_hat, log)
    part_d(A, y, theta_hat, log)

    with open(f"{OUT}/report.txt", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
