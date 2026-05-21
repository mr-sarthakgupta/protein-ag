# PySR-Backed Symbolic Regression Benchmarks

Evolve **symbolic equation-discovery code** with SkyDiscover (EvoX / AdaEvolve), while using [PySR](https://github.com/MilesCranmer/PySR) utilities for expression export/evaluation and the harness for fitting constants.

## Design

- **Evolution algorithm (SkyDiscover):** AdaEvolve or EvoX evolves Python programs via LLM edits (`evaluate_symbolic_candidate()` in `initial_program.py`).
- **Candidate artifact:** `evaluate_symbolic_candidate()` builds a symbolic expression template with feature symbols and tunable constants.
- **PySR-backed utilities:** `pysr_harness` fits constants, simplifies/exports expressions, evaluates predictions, and scores validation NMSE.
- **Evaluator:** Runs the harness on a dataset (Friedman #1 is the first example task) and returns `combined_score = 1 / (1 + nmse_val)`.

```
skydiscover-run → LLM edits evaluate_symbolic_candidate() → fit/evaluate proposed equation → combined_score
```

## Setup

### 1. SkyDiscover + optional extra

```bash
cd /path/to/skydiscover
uv sync --extra pysr-symbolic
uv pip install -r benchmarks/pysr_symbolic/requirements.txt
```

### 2. Julia + PySR

PySR requires Julia. Install [juliaup](https://github.com/JuliaLang/juliaup), then:

```bash
# First import triggers Julia package install (can take several minutes)
python -c "import pysr; print('PySR OK')"
```

For local development, `requirements.txt` installs PySR from `../../../pysr` (editable).

**Note:** The benchmark does not call `PySRRegressor.fit()` in the evaluator. Importing the full PySR package may still initialize its environment depending on which helpers are used.

## Tasks

| Task | Description |
|------|-------------|
| [`friedman1/`](friedman1/) | sklearn Friedman #1 (5 features, sin interaction + polynomial terms) |
| [`alphasyn_gaspar2017_03um_seed_all/`](alphasyn_gaspar2017_03um_seed_all/) | Alpha-synuclein Gaspar 2017 0.3uM seed rescaled response curves from copied `fit.tsv` |

## Run

```bash
# OpenAI-compatible default
export OPENAI_API_KEY="..."

# Or AWS Bedrock
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-1"

# AdaEvolve (or use -s evox)
uv run skydiscover-run \
  benchmarks/pysr_symbolic/friedman1/initial_program.py \
  benchmarks/pysr_symbolic/friedman1/evaluator.py \
  -c benchmarks/pysr_symbolic/friedman1/config.yaml \
  -s adaevolve -i 50

uv run skydiscover-run \
  benchmarks/pysr_symbolic/friedman1/initial_program.py \
  benchmarks/pysr_symbolic/friedman1/evaluator.py \
  -c benchmarks/pysr_symbolic/friedman1/config.yaml \
  -s evox -i 50
```

Algorithm-specific search settings live in [`configs/adaevolve.yaml`](../../configs/adaevolve.yaml) and [`configs/evox.yaml`](../../configs/evox.yaml); override via CLI `-s` or by adding a `search:` block to `config.yaml` if needed.

### Smoke test (no LLM)

```bash
cd benchmarks/pysr_symbolic
python friedman1/evaluator.py friedman1/initial_program.py
```

## `pysr_harness` library

Fixed library imported by evolved programs (not rewritten by the LLM):

| Module | Role |
|--------|------|
| `equation_session.py` | Fit constants, export/evaluate expressions, and return benchmark metrics |
| `operators.py` | Full PySR operator vocabulary, constraints, and `operator_config()` helper |
| `metrics.py` | NMSE and `combined_score` helpers |
| `backend.py`, `gp_session.py` | Legacy full-PySR search helpers; not used by this benchmark contract |

### Operator vocabulary

`pysr_harness.operators` exposes the **general** PySR search space:

- **Unary:** `neg`, `square`, `cube`, `sqrt`, `cbrt`, `abs`, `sign`, trig/hyperbolic/inverse trig, `exp`/`log`/`log2`/`log10`/`log1p`, `floor`/`ceil`/`round`, `relu`, `erf`/`erfc`
- **Binary:** `+`, `-`, `*`, `/`, `pow`, `max`, `min`
- **Ternary:** `muladd`, `clamp`
- **Constraints:** default nesting limits for nonlinear ops (see `default_constraints()`)

Friedman #1 is only an **example benchmark task** — defaults are not tuned to that equation.

### PySR primitives used

- Feature symbol creation (`create_sympy_symbols`)
- PySR/SymPy parsing (`pysr2sympy`)
- NumPy callable export (`sympy2numpy`)
- PySR-compatible operator naming and mappings

## What the LLM should evolve

Inside `evaluate_symbolic_candidate()`:

- The symbolic expression template
- Which features appear
- Operators and compositions
- Tunable constants used for scales, offsets, and exponents

The LLM should **not** call `PySRRegressor.fit()` or run an inner PySR equation-search loop. EvoX/AdaEvolve are the algorithms proposing new candidate programs.

## Baseline (optional)

For a pure PySR baseline without SkyDiscover:

```python
from pysr import PySRRegressor
from sklearn.datasets import make_friedman1

X, y = make_friedman1(n_samples=400, n_features=5, noise=0.1, random_state=42)
model = PySRRegressor(niterations=40, populations=2, population_size=15)
model.fit(X, y)
print(model.equations_)
```
