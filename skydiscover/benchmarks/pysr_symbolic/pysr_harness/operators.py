"""Operator sets and constraints for general PySR symbolic regression."""

from __future__ import annotations

from typing import Any, TypedDict


class OperatorSet(TypedDict):
    unary_operators: list[str]
    binary_operators: list[str]


# Validated against SymbolicRegression.jl (via juliacall `jl.seval`).
GENERAL_UNARY_OPERATORS: list[str] = [
    "neg",
    "square",
    "cube",
    "sqrt",
    "cbrt",
    "abs",
    "sign",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "atanh_clip",
    "exp",
    "log",
    "log2",
    "log10",
    "log1p",
    "floor",
    "ceil",
    "round",
    "relu",
    "erf",
    "erfc",
]

GENERAL_BINARY_OPERATORS: list[str] = [
    "+",
    "-",
    "*",
    "/",
    "pow",
    "max",
    "min",
]

# Optional higher-arity operators supported by PySR's `operators` dict API.
GENERAL_TERNARY_OPERATORS: list[str] = [
    "muladd",
    "clamp",
]


def general_operators() -> dict[str, dict[int, list[str]]]:
    """
    Full multi-arity operator vocabulary for general equation discovery.

    Uses PySR's unified ``operators`` dict (arity -> operator names), which supports
    unary, binary, and ternary primitives in one search space.
    """
    return {
        "operators": {
            1: GENERAL_UNARY_OPERATORS.copy(),
            2: GENERAL_BINARY_OPERATORS.copy(),
            3: GENERAL_TERNARY_OPERATORS.copy(),
        }
    }


def default_operators() -> OperatorSet:
    """
    Unary/binary operator lists (legacy PySR API).

    Prefer ``general_operators()`` for the full search space; this remains for
    harnesses that only tune unary/binary lists.
    """
    return {
        "unary_operators": GENERAL_UNARY_OPERATORS.copy(),
        "binary_operators": GENERAL_BINARY_OPERATORS.copy(),
    }


def default_constraints() -> dict[str, int | tuple[int, ...]]:
    """
    Default complexity constraints for a broad operator set.

    Unary operators use integer constraints; multi-arity operators use per-argument
    tuples (PySR ``_process_constraints`` format). Operators omitted here receive
    PySR's automatic defaults (-1 for unary, ``(-1,) * arity`` for others).
    """
    return {
        "pow": (-1, 1),
        "max": (-1, -1),
        "min": (-1, -1),
        "muladd": (-1, -1, 1),
        "clamp": (-1, 1, 1),
    }


def operator_config(**overrides: Any) -> dict[str, Any]:
    """Build an operator + constraints config fragment with optional overrides."""
    config: dict[str, Any] = {
        **general_operators(),
        "constraints": default_constraints(),
    }
    config.update(overrides)
    return config
