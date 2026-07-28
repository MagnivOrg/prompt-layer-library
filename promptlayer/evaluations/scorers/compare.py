"""Compare column scorer."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from promptlayer.evaluations.columns import column
from promptlayer.evaluations.scorers._helpers import (
    apply_scorecard_step_options,
    pop_scorecard_step_options,
    require_non_empty_title,
)
from promptlayer.evaluations.validation import validation_error
from promptlayer.types.table import EvalScorerColumn


def compare_scorer(
    title: str = "Compare",
    *,
    source: str = "Output",
    value_source: str = "Expected",
    comparison_type: Optional[Union[Dict[str, Any], str]] = None,
    **settings: Any,
) -> EvalScorerColumn:
    """Build a COMPARE scorer column."""
    require_non_empty_title(title, "compare_scorer")
    for field, value in (("source", source), ("value_source", value_source)):
        if not isinstance(value, str) or not value.strip():
            raise validation_error(f"compare_scorer {field} must be a non-empty string.")
    if comparison_type is None:
        comparison: Union[Dict[str, Any], str] = {"type": "STRING"}
    elif isinstance(comparison_type, str):
        comparison = {"type": comparison_type}
    else:
        comparison = comparison_type
    step_options, config_settings = pop_scorecard_step_options(settings)
    config: Dict[str, Any] = {
        "sources": [source, value_source],
        "comparison_type": comparison,
        **config_settings,
    }
    return apply_scorecard_step_options(column(title, "COMPARE", config), **step_options)
