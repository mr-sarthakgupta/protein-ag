#!/usr/bin/env bash
set -euo pipefail

# ── Configuration (override via environment) ─────────────────────────
MAX_COST="${MAX_COST:-45}"
ITERATIONS="${ITERATIONS:-100}"
SEARCH="${SEARCH:-adaevolve}"  # supported: adaevolve, evox
ALLOW_SMOKE_FAIL="${ALLOW_SMOKE_FAIL:-0}"
# Keep Bedrock calls pinned to the repo-standard region.
AWS_REGION="us-east-1"
export AWS_REGION
export AWS_DEFAULT_REGION="$AWS_REGION"
export BEDROCK_AWS_REGION="$AWS_REGION"
export BEDROCK_CONNECT_TIMEOUT="${BEDROCK_CONNECT_TIMEOUT:-30}"
export BEDROCK_READ_TIMEOUT="${BEDROCK_READ_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCHMARK_DIR="benchmarks/pysr_symbolic/disease_relevant_noninhibited_all_normalized_ode"
BENCHMARK_ROOT="$SCRIPT_DIR/benchmarks/pysr_symbolic"

# ── Preflight checks ────────────────────────────────────────────────
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Disease-Relevant Non-Inhibited — Normalized ODE Discovery        ║"
echo "║  60 datasets · 13 protein systems · shared differential equation  ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Search algorithm : $SEARCH"
echo "  Max iterations   : $ITERATIONS"
echo "  Cost budget      : \$$MAX_COST"
echo "  AWS region       : $AWS_REGION"
echo "  Agentic mode     : ENABLED (with internet tools)"
echo "  Benchmark root   : $BENCHMARK_ROOT"
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
echo "Smoke test: evaluating seed ODE program across all datasets..."
if uv run python "$BENCHMARK_DIR/evaluator.py" "$BENCHMARK_DIR/initial_program.py"; then
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
print(build_output_dir('${SEARCH}', os.path.abspath('${BENCHMARK_DIR}/initial_program.py'), base_dir='outputs_diff_norm'))
")}"
mkdir -p "$OUTPUT_DIR/reference"

# Agentic path validation resolves symlinks, so use the real benchmark tree.
export BENCHMARK_CODEBASE_ROOT="$BENCHMARK_ROOT"

# ── Run the experiment ───────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Starting discovery run (agentic + internet tools)"
echo "  Finding universal ODE across normalized amyloid datasets"
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
