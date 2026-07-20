"""Compare column scorer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

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
    sources: Optional[List[str]] = None,
    comparison_type: Optional[Union[Dict[str, Any], str]] = None,
    **settings: Any,
) -> EvalScorerColumn:
    """Build a COMPARE scorer column."""
    require_non_empty_title(title, "compare_scorer")
    resolved_sources = ["Output", "Expected"] if sources is None else sources
    if not isinstance(resolved_sources, list) or len(resolved_sources) != 2:
        raise validation_error("compare_scorer requires exactly two sources.")
    for source in resolved_sources:
        if not isinstance(source, str) or not source.strip():
            raise validation_error("compare_scorer sources must be non-empty strings.")
    if comparison_type is None:
        comparison: Union[Dict[str, Any], str] = {"type": "STRING"}
    elif isinstance(comparison_type, str):
        comparison = {"type": comparison_type}
    else:
        comparison = comparison_type
    step_options, config_settings = pop_scorecard_step_options(settings)
    config: Dict[str, Any] = {
        "sources": list(resolved_sources),
        "comparison_type": comparison,
        **config_settings,
    }
    return apply_scorecard_step_options(column(title, "COMPARE", config), **step_options)
