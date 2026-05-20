# Symbolic Regression Benchmarks

This folder contains benchmarks where the objective is **symbolic regression**: given a tabular dataset \((X, y)\), discover an interpretable equation that predicts \(y\) from \(X\).

SkyDiscover treats each candidate as a **Python program**. The evaluator loads the dataset, splits into train/validation, and calls the candidate program’s `fit_and_predict(...)` to produce predictions and (optionally) a human-readable equation.

**Structure vs parameters.** The intended workflow is to discover the **functional form** of a law (operators, which features appear, which basis transforms apply) while treating numeric **coefficients or scales** as a separate continuous problem: fit them on the training split only (for example with Gaussian-process Bayesian optimization), then report validation metrics. The reference `initial_program.py` demonstrates GP-BO (`scikit-optimize`) on linear coefficients over a fixed feature-transform library. Install evaluator dependencies with the `math` extra: `uv sync --extra math`.

## Contract (candidate program)

Your `initial_program.py` (and any evolved programs) must define:

- `fit_and_predict(X_train, y_train, X_test) -> dict`

The returned dict must include:

- `y_pred`: 1D array-like of predictions for `X_test`

Optional keys (strongly recommended):

- `equation_template`: SymPy expression or string describing the law **before** substituting fitted constants. Use dedicated parameter Symbols (e.g. `p0`, `p1`, …) for tunable quantities and feature symbols `x1`, `x2`, … in standardized or raw space—**document which you use**. When present, the evaluator’s **structural complexity** penalty is computed from this template (coefficient magnitudes do not inflate the penalty).
- `equation`: string representation of the fitted equation (numeric constants allowed).
- `equation_sympy`: a SymPy expression or string parseable by SymPy (fitted form).

## Metrics

The evaluator returns (among others):

- `r2_val`, `rmse_val` — on the held-out validation split using your `y_pred`.
- `structural_complexity` — SymPy tree size **excluding** numeric `Number` atoms (or from `equation_template` if supplied).
- `equation_complexity` — full preorder node count including numbers (diagnostic).
- `combined_score` — `clamp(r2_val, 0, 1) − λ·structural_complexity` (scalar objective for search).

With AdaEvolve you can instead enable Pareto mode on `r2_val` and `structural_complexity` (see `config_adaevolve.yaml`).

## Running the included example (EvoX)

From the repo root (`skydiscover` package):

```bash
uv sync --extra math
uv run skydiscover-run benchmarks/symbolic_regression/toy_friedman1/initial_program.py \
  benchmarks/symbolic_regression/toy_friedman1/evaluator.py \
  -c benchmarks/symbolic_regression/toy_friedman1/config_evox.yaml
```

This uses the EvoX search strategy to evolve the candidate program for better validation fit and simpler **structure**.

## Running with AdaEvolve (optional Pareto)

```bash
uv run skydiscover-run benchmarks/symbolic_regression/toy_friedman1/initial_program.py \
  benchmarks/symbolic_regression/toy_friedman1/evaluator.py \
  -c benchmarks/symbolic_regression/toy_friedman1/config_adaevolve.yaml --search adaevolve
```

## Adapting to your dataset (SRBench-style)

SRBench datasets are typically tabular with a single target column. To adapt:

- Put your dataset as `data.csv` in a new benchmark folder.
- Ensure the target column name matches `target` in the config (default: `y`).
- Update `feature_cols` if you want to restrict inputs (otherwise all non-target numeric columns are used).

The SRBench project is a good source of datasets and conventions: [cavalab/srbench](https://github.com/cavalab/srbench).
