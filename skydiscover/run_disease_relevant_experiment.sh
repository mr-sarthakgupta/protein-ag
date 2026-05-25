#!/usr/bin/env bash
set -euo pipefail

# ── Configuration (override via environment) ─────────────────────────
MAX_COST="${MAX_COST:-50}"
ITERATIONS="${ITERATIONS:-30}"
SEARCH="${SEARCH:-adaevolve}"  # supported: adaevolve, evox
ALLOW_SMOKE_FAIL="${ALLOW_SMOKE_FAIL:-0}"
export AWS_REGION="${AWS_REGION:-us-east-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCHMARK_DIR="benchmarks/pysr_symbolic/disease_relevant_noninhibited_all"

# Agentic mode: codebase root includes the harness library
export BENCHMARK_CODEBASE_ROOT="$SCRIPT_DIR/benchmarks/pysr_symbolic"

# ── Preflight checks ────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Disease-Relevant Non-Inhibited — Universal Symbolic Regression ║"
echo "║  60 datasets · 13 protein systems · shared equation structure   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Search algorithm : $SEARCH"
echo "  Max iterations   : $ITERATIONS"
echo "  Cost budget      : \$$MAX_COST"
echo "  AWS region       : $AWS_REGION"
echo "  Agentic mode     : ENABLED (with internet tools)"
echo "  Codebase root    : $BENCHMARK_CODEBASE_ROOT"
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
echo "Smoke test: evaluating seed program across all datasets..."
if uv run python "$BENCHMARK_DIR/evaluator.py" "$BENCHMARK_DIR/initial_program.py"; then
    echo "[OK] Smoke test passed"
elif [ "$ALLOW_SMOKE_FAIL" = "1" ]; then
    echo "WARNING: Smoke test failed — continuing anyway (ALLOW_SMOKE_FAIL=1)"
else
    echo "ERROR: Smoke test failed. Fix evaluator/dependencies before launching a costed LLM run."
    echo "  Set ALLOW_SMOKE_FAIL=1 to override."
    exit 1
fi

# ── Create reference directory for internet tool downloads ───────────
mkdir -p "$BENCHMARK_CODEBASE_ROOT/reference"

# ── Run the experiment ───────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Starting discovery run (agentic + internet tools)"
echo "  Finding universal equation across 60 amyloid datasets"
echo "════════════════════════════════════════════════════════════"
echo ""

uv run skydiscover-run \
    "$BENCHMARK_DIR/initial_program.py" \
    "$BENCHMARK_DIR/evaluator.py" \
    -c "$BENCHMARK_DIR/config.yaml" \
    -s "$SEARCH" \
    -i "$ITERATIONS" \
    --max-cost "$MAX_COST"

# ── Copy reference files to output for archival ──────────────────────
REF_DIR="$BENCHMARK_CODEBASE_ROOT/reference"
if [ -d "$REF_DIR" ] && [ "$(ls -A "$REF_DIR" 2>/dev/null)" ]; then
    LATEST_OUTPUT=$(find outputs/ -maxdepth 4 -name "best_program_info.json" -printf '%T@ %h\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -n "${LATEST_OUTPUT:-}" ]; then
        OUTPUT_ROOT="$(dirname "$LATEST_OUTPUT")"
        cp -r "$REF_DIR" "$OUTPUT_ROOT/reference_files" 2>/dev/null && \
            echo "Reference files copied to $OUTPUT_ROOT/reference_files" || true
    fi
fi

echo ""
echo "Done."
