# ML Assignment — Part C (Programming)

From-scratch implementations of linear regression (normal equation +
gradient descent), polynomial/Lasso/Ridge regression, and one-vs-rest
logistic regression, per the assignment spec. sklearn/PyTorch are used
**only** where the assignment explicitly calls for them (C2b/c Ridge & Lasso,
C3b PyTorch) — everything else is written from scratch with numpy.

## Directory structure

```
ml-assignment/
├── README.md              <- you are here
├── requirements.txt
├── run_all.py              <- runs C1, C2, C3 in one go
├── data/
│   └── iris.data           <- YOU put your iris.data file here
├── src/
│   ├── __init__.py
│   ├── utils.py             <- shared from-scratch routines (normal eqn, ridge, GD, etc.)
│   ├── c1_linear_regression.py
│   ├── c2_polynomial_lasso.py
│   └── c3_logistic_regression.py
└── outputs/
    ├── c1/   <- plots + report.txt for C1
    ├── c2/   <- plots + report.txt for C2
    └── c3/   <- report.txt for C3 (no plots required)
```

## Setup (WSL, inside your existing venv)

```bash
# 1. unzip/copy this folder somewhere, then cd into it
cd ml-assignment

# 2. activate your venv (adjust path to yours)
source ~/venvs/myenv/bin/activate      # or wherever your venv lives

# 3. install dependencies
pip install -r requirements.txt
```

If `pip install torch` is slow or gives a CUDA-related error on WSL, install
the CPU-only build instead — it's smaller and all you need here:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Put your data in place

You said you already have `iris.data` — just copy it in:

```bash
cp /path/to/your/iris.data ml-assignment/data/iris.data
```

It must be the raw UCI-format file: comma-separated, **no header**, 5 columns
(`sepal_length,sepal_width,petal_length,petal_width,class`), with class
labels `Iris-setosa`, `Iris-versicolor`, `Iris-virginica`. If your copy has a
header row or different labels, either strip the header or tell me and I'll
adjust the loader in `src/c3_logistic_regression.py::load_iris_data`.

## Running everything

From the `ml-assignment/` root (so the `src` package and relative
`data/`/`outputs/` paths resolve correctly):

```bash
python run_all.py
```

This runs all three parts back to back and prints results to the terminal
while also writing them to `outputs/c*/report.txt`. Plots land in
`outputs/c1/` and `outputs/c2/` as PNGs (C3 has no required plots).

### Or run each part individually

```bash
python -m src.c1_linear_regression
python -m src.c2_polynomial_lasso
python -m src.c3_logistic_regression
```

(Use `python -m src.xxx`, not `python src/xxx.py` directly — the `-m` form
is what lets `from src.utils import ...` resolve correctly.)

## What each file produces

| File | Sub-parts covered | Outputs |
|---|---|---|
| `c1_linear_regression.py` | C1 (a)-(d): normal equation, condition number + perturbation, gradient descent (3 learning rates), ridge normal equation | `outputs/c1/c1_gd_cost_vs_iteration.png`, `outputs/c1/report.txt` |
| `c2_polynomial_lasso.py` | C2 (a)-(c): polynomial fits (scratch), Ridge/Lasso vs lambda (sklearn), coefficient sparsity bar chart | `outputs/c2/c2_poly_fits.png`, `outputs/c2/c2_ridge_lasso_rmse_vs_lambda.png`, `outputs/c2/c2_coef_bar_lasso_vs_ridge.png`, `outputs/c2/report.txt` |
| `c3_logistic_regression.py` | C3 (a)-(b): one-vs-rest logistic regression from scratch (BCE loss + gradient + GD), then again with PyTorch's `nn.BCELoss` + `torch.optim.SGD` | `outputs/c3/report.txt` |

## Notes on design choices (useful for your write-up)

- **`src/utils.py`** centralizes `normal_equation`, `ridge_normal_equation`,
  `gradient_descent_linreg`, `rmse`, and `vandermonde` so C1 and C2(a) reuse
  the *same* from-scratch normal-equation solver, as the assignment's C2(a)
  explicitly requires ("solve for θ̂ using your normal-equation function
  from D1" — read as C1).
- Gradient descent uses `np.linalg.solve` instead of an explicit matrix
  inverse in the closed-form solver (more numerically stable, same
  mathematical result).
- The ridge normal equation does **not** regularize the intercept term,
  matching how `sklearn.linear_model.Ridge` behaves — this keeps the C1(d)
  vs. C2(b) comparison consistent.
- In C1(c), which learning rate diverges/converges fastest is **computed and
  reported dynamically** based on the actual run (it depends on the largest
  eigenvalue of AᵀA), rather than hardcoded — the printed discussion will
  correctly reflect whatever your machine actually produces (it should be
  deterministic given the fixed `np.random.seed(0)`, but this makes the
  script self-documenting either way).
- C3 standardizes features (`StandardScaler`) before both the scratch and
  PyTorch logistic regression — this is standard practice for gradient-based
  logistic regression and keeps the two implementations' convergence
  comparable; it doesn't count as "using a library for the model itself."

## Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`** — you're not running from
  the `ml-assignment/` root, or you ran `python src/c1_linear_regression.py`
  directly instead of `python -m src.c1_linear_regression`.
- **`FileNotFoundError: data/iris.data`** — copy your file in as shown above;
  the script looks for it relative to wherever you launch it from (the repo root).
- **PyTorch import errors on WSL** — reinstall with the CPU-only index URL
  shown above; you don't need CUDA for this assignment's dataset size.
