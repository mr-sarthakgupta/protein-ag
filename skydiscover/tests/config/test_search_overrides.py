"""Regression tests for search-type runtime overrides."""

from skydiscover.config import (
    AdaEvolveDatabaseConfig,
    Config,
    apply_overrides,
)


def test_search_override_preserves_database_values_from_yaml_shape():
    config = Config.from_dict(
        {
            "search": {
                "database": {
                    "num_islands": 4,
                    "pareto_objectives": ["combined_score", "nmse_val"],
                    "higher_is_better": {
                        "combined_score": True,
                        "nmse_val": False,
                    },
                }
            }
        }
    )

    apply_overrides(config, search="adaevolve")

    assert isinstance(config.search.database, AdaEvolveDatabaseConfig)
    assert config.search.database.num_islands == 4
    assert config.search.database.pareto_objectives == [
        "combined_score",
        "nmse_val",
    ]
    assert config.search.database.higher_is_better["nmse_val"] is False
