"""Regex column scorer."""

from __future__ import annotations

from typing import Any, Dict

from promptlayer.evaluations.columns import column
from promptlayer.evaluations.scorers._helpers import (
    apply_scorecard_step_options,
    pop_scorecard_step_options,
    require_non_empty_source,
    require_non_empty_string,
    require_non_empty_title,
)
from promptlayer.types.table import EvalScorerColumn


def regex_scorer(
    title: str = "Regex",
    *,
    source: str = "Output",
    regex_pattern: str,
    **settings: Any,
) -> EvalScorerColumn:
    """Build a REGEX scorer column."""
    require_non_empty_title(title, "regex_scorer")
    require_non_empty_source(source, "regex_scorer")
    require_non_empty_string(regex_pattern, field="regex_pattern", scorer_name="regex_scorer")
    step_options, config_settings = pop_scorecard_step_options(settings)
    config: Dict[str, Any] = {
        "source": source,
        "regex_pattern": regex_pattern,
        **config_settings,
    }
    return apply_scorecard_step_options(column(title, "REGEX", config), **step_options)
