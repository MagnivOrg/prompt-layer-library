"""Contains column scorer."""

from __future__ import annotations

from typing import Any, Dict, Optional

from promptlayer.evaluations.columns import column
from promptlayer.evaluations.scorers._helpers import (
    apply_scorecard_step_options,
    pop_scorecard_step_options,
    reject_legacy_parameters,
    require_exactly_one_of,
    require_non_empty_source,
    require_non_empty_title,
)
from promptlayer.types.table import EvalScorerColumn


def contains_scorer(
    title: str = "Contains",
    *,
    source_column: str = "Output",
    expected: Optional[str] = None,
    expected_column: Optional[str] = None,
    **settings: Any,
) -> EvalScorerColumn:
    """Build a CONTAINS scorer column."""
    reject_legacy_parameters(
        settings,
        names=("source", "value", "value_source"),
        scorer_name="contains_scorer",
    )
    require_non_empty_title(title, "contains_scorer")
    require_non_empty_source(source_column, "contains_scorer", field="source_column")
    require_exactly_one_of(
        expected is not None,
        expected_column is not None,
        names=("expected", "expected_column"),
        scorer_name="contains_scorer",
    )
    step_options, config_settings = pop_scorecard_step_options(settings)
    config: Dict[str, Any] = {"source": source_column, **config_settings}
    if expected is not None:
        config["value"] = expected
    if expected_column is not None:
        require_non_empty_source(expected_column, "contains_scorer", field="expected_column")
        config["value_source"] = expected_column
    return apply_scorecard_step_options(column(title, "CONTAINS", config), **step_options)
