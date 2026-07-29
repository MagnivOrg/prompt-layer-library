"""Compare column scorer."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from promptlayer.evaluations.columns import column
from promptlayer.evaluations.scorers._helpers import (
    apply_scorecard_step_options,
    pop_scorecard_step_options,
    reject_legacy_parameters,
    require_non_empty_source,
    require_non_empty_title,
)
from promptlayer.evaluations.validation import validation_error
from promptlayer.types.table import EvalScorerColumn

_UNSET = object()


def compare_scorer(
    title: str = "Compare",
    *,
    source_column: str = "Output",
    expected: Any = _UNSET,
    expected_column: Optional[str] = None,
    comparison_type: Optional[Union[Dict[str, Any], str]] = None,
    **settings: Any,
) -> EvalScorerColumn:
    """Build a COMPARE scorer column."""
    reject_legacy_parameters(
        settings,
        names=("source", "value_source"),
        scorer_name="compare_scorer",
    )
    require_non_empty_title(title, "compare_scorer")
    require_non_empty_source(source_column, "compare_scorer", field="source_column")
    if expected is not _UNSET and expected_column is not None:
        raise validation_error("compare_scorer accepts only one of expected or expected_column.")
    if expected is _UNSET and expected_column is None:
        expected_column = "expected"
    if expected_column is not None:
        require_non_empty_source(expected_column, "compare_scorer", field="expected_column")
    if comparison_type is None:
        comparison: Union[Dict[str, Any], str] = {"type": "STRING"}
    elif isinstance(comparison_type, str):
        comparison = {"type": comparison_type}
    else:
        comparison = comparison_type
    step_options, config_settings = pop_scorecard_step_options(settings)
    config: Dict[str, Any] = {
        "sources": [source_column],
        "comparison_type": comparison,
        **config_settings,
    }
    if expected is not _UNSET:
        config["target"] = expected
    else:
        config["sources"].append(expected_column)
    return apply_scorecard_step_options(column(title, "COMPARE", config), **step_options)
