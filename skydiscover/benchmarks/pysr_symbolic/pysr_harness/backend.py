"""Build PySR / SymbolicRegression.jl search objects from plain Python config."""

from __future__ import annotations

import inspect
from typing import Any

from pysr import PySRRegressor

from pysr_harness.operators import default_constraints, general_operators

# Harness-only keys (not passed to PySRRegressor).
_HARNESS_KEYS = frozenset(
    {
        "feature_names",
        "selection_strategy",
        "calculate_scores",
    }
)

_PYSR_INIT_PARAMS = frozenset(
    inspect.signature(PySRRegressor.__init__).parameters.keys()
) - {"self"}


def ensure_julia() -> None:
    """Import PySR, which initializes the Julia runtime on first use."""
    import pysr  # noqa: F401


def default_gp_config() -> dict[str, Any]:
    """
    General-purpose defaults aligned with PySRRegressor's built-in defaults.

    Uses the full operator vocabulary and PySR mutation/optimization settings.
    Search budgets are scaled down modestly so a single evaluator call finishes
    within typical benchmark timeouts; increase via evolved harness config.
    """
    return {
        **general_operators(),
        "constraints": default_constraints(),
        # Model selection (PySR name: model_selection)
        "model_selection": "best",
        # Search budget — moderate for benchmark evaluate(); LLM can raise these
        "niterations": 20,
        "populations": 8,
        "population_size": 27,
        "ncycles_per_iteration": 380,
        "maxsize": 30,
        "parsimony": 0.0,
        "crossover_probability": 0.0259,
        # PySR default mutation weights (SymbolicRegression regularized evolution)
        "weight_add_node": 2.47,
        "weight_insert_node": 0.0112,
        "weight_delete_node": 0.870,
        "weight_do_nothing": 0.273,
        "weight_mutate_constant": 0.0346,
        "weight_mutate_operator": 0.293,
        "weight_mutate_feature": 0.1,
        "weight_swap_operands": 0.198,
        "weight_rotate_tree": 4.26,
        "weight_randomize": 0.000502,
        "weight_simplify": 0.00209,
        "weight_optimize": 0.0,
        # Constant optimization and simplification
        "should_simplify": True,
        "should_optimize_constants": True,
        "optimizer_algorithm": "BFGS",
        "optimizer_nrestarts": 2,
        "optimize_probability": 0.14,
        "optimizer_iterations": 8,
        # Tournament selection (must satisfy tournament_selection_n < population_size)
        "tournament_selection_n": 10,
        "tournament_selection_p": 0.982,
        # Population dynamics
        "fraction_replaced": 0.00036,
        "fraction_replaced_hof": 0.0614,
        "fraction_replaced_guesses": 0.001,
        "migration": True,
        "hof_migration": True,
        "topn": 12,
        "use_frequency": True,
        "use_frequency_in_tournament": True,
        "adaptive_parsimony_scaling": 1040.0,
        "alpha": 3.17,
        "loss_scale": "log",
        "skip_mutation_failures": True,
        # Optional preprocessing (PySRRegressor kwargs)
        "denoise": False,
        "select_k_features": None,
        "random_state": 42,
        "deterministic": True,
        # Harness helper: compute Pareto scores before model selection
        "calculate_scores": True,
        "selection_strategy": "best",
    }


def build_pysr_regressor(
    config: dict[str, Any] | None = None,
) -> PySRRegressor:
    """Construct a configured PySRRegressor from a plain dict."""
    ensure_julia()
    merged = default_gp_config()
    if config:
        merged.update(config)

    # Map harness alias -> PySR parameter name
    if "selection_strategy" in merged and "model_selection" not in merged:
        merged["model_selection"] = merged["selection_strategy"]

    mutation_keys = {
        "weight_mutate_constant",
        "weight_mutate_operator",
        "weight_mutate_feature",
        "weight_swap_operands",
        "weight_rotate_tree",
        "weight_add_node",
        "weight_insert_node",
        "weight_delete_node",
        "weight_simplify",
        "weight_randomize",
        "weight_do_nothing",
        "weight_optimize",
    }

    regressor_kwargs = {
        k: v
        for k, v in merged.items()
        if k not in _HARNESS_KEYS
        and k not in mutation_keys
        and k in _PYSR_INIT_PARAMS
    }
    regressor_kwargs.update(
        {
            "temp_equation_file": True,
            "progress": False,
            "verbosity": 0,
            "parallelism": "serial",
            "procs": 0,
        }
    )
    for key in mutation_keys:
        if key in merged:
            regressor_kwargs[key] = merged[key]

    return PySRRegressor(**regressor_kwargs)
