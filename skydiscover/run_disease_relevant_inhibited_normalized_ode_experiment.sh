#!/usr/bin/env bash
set -euo pipefail

# ── Configuration (override via environment) ─────────────────────────
MAX_COST="${MAX_COST:-75}"
ITERATIONS="${ITERATIONS:-100}"
SEARCH="${SEARCH:-adaevolve}"  # supported: adaevolve, evox
ALLOW_SMOKE_FAIL="${ALLOW_SMOKE_FAIL:-0}"
SKYDISCOVER_INHIBITED_MAX_NFEV="${SKYDISCOVER_INHIBITED_MAX_NFEV:-300}"
SKYDISCOVER_ODE_CURVE_WORKERS="${SKYDISCOVER_ODE_CURVE_WORKERS:-96}"
SKYDISCOVER_ODE_MULTISTART_WORKERS="${SKYDISCOVER_ODE_MULTISTART_WORKERS:-3}"
SKYDISCOVER_SEED_INGESTION_CONCURRENCY="${SKYDISCOVER_SEED_INGESTION_CONCURRENCY:-8}"
SKYDISCOVER_SEED_CURVE_WORKERS="${SKYDISCOVER_SEED_CURVE_WORKERS:-1}"
SKYDISCOVER_SEED_MULTISTART_WORKERS="${SKYDISCOVER_SEED_MULTISTART_WORKERS:-1}"
SKYDISCOVER_SEED_FAST_MAX_NFEV="${SKYDISCOVER_SEED_FAST_MAX_NFEV:-120}"
SKYDISCOVER_SEED_TOP_K="${SKYDISCOVER_SEED_TOP_K:-all}"
SKYDISCOVER_SEED_DEDUPLICATE="${SKYDISCOVER_SEED_DEDUPLICATE:-1}"
# Exclude cd=0 controls by default for the temporary inhibitor-only run.
SKYDISCOVER_INCLUDE_UNINHIBITED="${SKYDISCOVER_INCLUDE_UNINHIBITED:-0}"

for arg in "$@"; do
    case "$arg" in
        --include-uninhibited)
            SKYDISCOVER_INCLUDE_UNINHIBITED=1
            ;;
        --exclude-uninhibited)
            SKYDISCOVER_INCLUDE_UNINHIBITED=0
            ;;
        -h|--help)
            echo "Usage: $0 [--include-uninhibited|--exclude-uninhibited]"
            echo "Default: --exclude-uninhibited"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $arg" >&2
            echo "Usage: $0 [--include-uninhibited|--exclude-uninhibited]" >&2
            exit 2
            ;;
    esac
done

case "$SKYDISCOVER_INCLUDE_UNINHIBITED" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        SKYDISCOVER_INCLUDE_UNINHIBITED=1
        ;;
    0|false|FALSE|False|no|NO|No|off|OFF|Off)
        SKYDISCOVER_INCLUDE_UNINHIBITED=0
        ;;
    *)
        echo "ERROR: SKYDISCOVER_INCLUDE_UNINHIBITED must be a boolean value" >&2
        exit 2
        ;;
esac

export SKYDISCOVER_INHIBITED_MAX_NFEV
export SKYDISCOVER_ODE_CURVE_WORKERS
export SKYDISCOVER_ODE_MULTISTART_WORKERS
export SKYDISCOVER_SEED_INGESTION_CONCURRENCY
export SKYDISCOVER_SEED_CURVE_WORKERS
export SKYDISCOVER_SEED_MULTISTART_WORKERS
export SKYDISCOVER_SEED_FAST_MAX_NFEV
export SKYDISCOVER_SEED_TOP_K
export SKYDISCOVER_SEED_DEDUPLICATE
export SKYDISCOVER_INCLUDE_UNINHIBITED
# Keep Bedrock calls pinned to the repo-standard region.
AWS_REGION="us-east-1"
export AWS_REGION
export AWS_DEFAULT_REGION="$AWS_REGION"
export BEDROCK_AWS_REGION="$AWS_REGION"
export BEDROCK_CONNECT_TIMEOUT="${BEDROCK_CONNECT_TIMEOUT:-30}"
export BEDROCK_READ_TIMEOUT="${BEDROCK_READ_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCHMARK_DIR="benchmarks/pysr_symbolic/disease_relevant_inhibited_all_normalized_ode"
BENCHMARK_ROOT="$SCRIPT_DIR/benchmarks/pysr_symbolic"
SKYDISCOVER_SEED_CHECKPOINT="${SKYDISCOVER_SEED_CHECKPOINT:-$SCRIPT_DIR/outputs_diff_inhibited_norm/$SEARCH/seed_checkpoints/disease_relevant_inhibited_all_normalized_ode_nfev${SKYDISCOVER_INHIBITED_MAX_NFEV}_curve${SKYDISCOVER_ODE_CURVE_WORKERS}_multistart${SKYDISCOVER_ODE_MULTISTART_WORKERS}_uninhibited${SKYDISCOVER_INCLUDE_UNINHIBITED}}"
export SKYDISCOVER_SEED_CHECKPOINT

# ── Preflight checks ────────────────────────────────────────────────
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Abeta42 Inhibitor — Normalized ODE Discovery                     ║"
if [ "$SKYDISCOVER_INCLUDE_UNINHIBITED" = "1" ]; then
    echo "║  1 dataset · 30 settings (including cd=0) · shared ODE            ║"
else
    echo "║  1 dataset · 15 inhibited settings (cd>0 only) · shared ODE       ║"
fi
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Search algorithm : $SEARCH"
echo "  Max iterations   : $ITERATIONS"
echo "  Cost budget      : \$$MAX_COST"
echo "  AWS region       : $AWS_REGION"
echo "  Agentic mode     : ENABLED (with internet tools)"
echo "  Benchmark root   : $BENCHMARK_ROOT"
echo "  Fit max nfev     : $SKYDISCOVER_INHIBITED_MAX_NFEV"
echo "  Curve workers    : $SKYDISCOVER_ODE_CURVE_WORKERS"
echo "  Multistart workers: $SKYDISCOVER_ODE_MULTISTART_WORKERS"
echo "  Seed concurrency : $SKYDISCOVER_SEED_INGESTION_CONCURRENCY"
echo "  Seed curve workers: $SKYDISCOVER_SEED_CURVE_WORKERS"
echo "  Seed multistart workers: $SKYDISCOVER_SEED_MULTISTART_WORKERS"
echo "  Seed fast nfev   : $SKYDISCOVER_SEED_FAST_MAX_NFEV"
echo "  Seed top-k       : $SKYDISCOVER_SEED_TOP_K"
echo "  Seed dedupe      : $SKYDISCOVER_SEED_DEDUPLICATE"
echo "  Include cd=0     : $SKYDISCOVER_INCLUDE_UNINHIBITED"
echo "  Seed checkpoint  : $SKYDISCOVER_SEED_CHECKPOINT"
echo ""

# Check Bedrock credentials
if [ -z "${AWS_BEARER_TOKEN_BEDROCK:-}" ] && [ -z "${AWS_ACCESS_KEY_ID:-}" ] && [ -z "${AWS_PROFILE:-}" ] && [ ! -f "${AWS_SHARED_CREDENTIALS_FILE:-$HOME/.aws/credentials}" ]; then
    echo "ERROR: Bedrock credentials not found."
    echo "  Set AWS_BEARER_TOKEN_BEDROCK, AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY,"
    echo "  AWS_PROFILE, or provide ~/.aws/credentials."
    exit 1
fi
echo "[OK] Bedrock credentials detected"

cd "$SCRIPT_DIR"

# ── Install dependencies ─────────────────────────────────────────────
echo ""
echo "Installing dependencies..."
uv sync --extra pysr-symbolic --extra bedrock --quiet 2>/dev/null || \
    uv sync --extra pysr-symbolic --extra bedrock
uv pip install -r benchmarks/pysr_symbolic/requirements.txt --quiet 2>/dev/null || \
    uv pip install -r benchmarks/pysr_symbolic/requirements.txt
echo "[OK] Dependencies installed"

# ── Verify PySR / Julia ──────────────────────────────────────────────
echo ""
echo "Checking PySR + Julia..."
uv run python -c "import pysr; print('[OK] PySR version:', pysr.__version__)"

# ── Verify dataset discovery ─────────────────────────────────────────
echo ""
echo "Checking dataset discovery..."
uv run python "$BENCHMARK_DIR/evaluator.py" --list-datasets | tail -1
echo "[OK] Datasets discovered"

# ── Smoke test (evaluator only, no LLM cost) ────────────────────────
echo ""
echo "Smoke test: evaluating seed ODE program on a bounded inhibitor subset..."
if uv run python - "$BENCHMARK_DIR/evaluator.py" "$BENCHMARK_DIR/initial_program.py" <<'PY'
import importlib.util
import sys

import numpy as np

evaluator_path, program_path = sys.argv[1:3]
spec = importlib.util.spec_from_file_location("benchmark_evaluator", evaluator_path)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)

import pysr_harness.equation_session as equation_session
equation_session.evaluate_expression = evaluator.evaluate_ode_expression
equation_session.nmse = evaluator._nmse_with_gaussian_smoothed_target
from pysr_harness.equation_session import (
    single_equation_evaluation,
    validate_single_equation_result,
)

program_spec = importlib.util.spec_from_file_location("seed_program", program_path)
program = importlib.util.module_from_spec(program_spec)
program_spec.loader.exec_module(program)


def subset_by_curve(X, y, *, max_curves=4, max_points_per_curve=60):
    curve_ids = evaluator._base._ARRAY_CURVE_IDS.get(id(X), np.zeros(X.shape[0], dtype=int))
    selected = []
    for curve_id in dict.fromkeys(np.asarray(curve_ids, dtype=int)):
        curve_idx = np.flatnonzero(curve_ids == curve_id)
        ordered = curve_idx[np.argsort(X[curve_idx, 0])]
        if ordered.size > max_points_per_curve:
            keep = np.linspace(0, ordered.size - 1, max_points_per_curve, dtype=int)
            ordered = ordered[keep]
        selected.append(ordered)
        if len(selected) >= max_curves:
            break
    subset_idx = np.concatenate(selected)
    X_subset = X[subset_idx].copy()
    y_subset = y[subset_idx].copy()
    evaluator._base._remember_curve_ids(X_subset, curve_ids[subset_idx])
    return X_subset, y_subset


datasets = evaluator.load_all_datasets()
if not datasets:
    raise RuntimeError("No inhibitor datasets discovered")
name, X_train, X_val, y_train, y_val = datasets[0]
X_train, y_train = subset_by_curve(X_train, y_train)
X_val, y_val = subset_by_curve(X_val, y_val)

with single_equation_evaluation():
    result = program.run_discovery(X_train, y_train, X_val, y_val)
    validate_single_equation_result(result)

if result.get("error"):
    print(f"ERROR: evaluator returned error: {result['error']}")
    sys.exit(1)
if float(result.get("combined_score", 0.0)) <= 0.0:
    print("ERROR: evaluator returned a non-positive combined_score")
    sys.exit(1)
print(f"[OK] Smoke subset {name}: combined_score={float(result['combined_score']):.6f}")
PY
then
    echo "[OK] Smoke test passed"
elif [ "$ALLOW_SMOKE_FAIL" = "1" ]; then
    echo "WARNING: Smoke test failed — continuing anyway (ALLOW_SMOKE_FAIL=1)"
else
    echo "ERROR: Smoke test failed. Fix evaluator/dependencies before launching a costed LLM run."
    echo "  Set ALLOW_SMOKE_FAIL=1 to override."
    exit 1
fi

# ── Prepare output dir; reference files are saved here directly ──────
OUTPUT_DIR="${OUTPUT_DIR:-$(uv run python -c "
from skydiscover.config import build_output_dir
import os
print(build_output_dir('${SEARCH}', os.path.abspath('${BENCHMARK_DIR}/initial_program.py'), base_dir='outputs_diff_inhibited_norm'))
")}"
mkdir -p "$OUTPUT_DIR/reference"

# Agentic path validation resolves symlinks, so use the real benchmark tree.
export BENCHMARK_CODEBASE_ROOT="$BENCHMARK_ROOT"

# ── Run the experiment ───────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Starting discovery run (agentic + internet tools)"
echo "  Finding ODE across normalized Abeta42 inhibitor data"
echo "  Output directory : $OUTPUT_DIR"
echo "  Reference files  : $OUTPUT_DIR/reference"
echo "════════════════════════════════════════════════════════════"
echo ""

uv run skydiscover-run \
    "$BENCHMARK_DIR/initial_program.py" \
    "$BENCHMARK_DIR/evaluator.py" \
    -c "$BENCHMARK_DIR/config.yaml" \
    -s "$SEARCH" \
    -i "$ITERATIONS" \
    --max-cost "$MAX_COST" \
    -o "$OUTPUT_DIR"

echo ""
echo "Done."
