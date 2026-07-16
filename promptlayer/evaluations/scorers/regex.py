"""Regex column scorer."""

from __future__ import annotations

from typing import Any, Dict

from promptlayer.evaluations.columns import column
from promptlayer.evaluations.scorers._helpers import (
    require_non_empty_source,
    require_non_empty_string,
    require_non_empty_title,
)
from promptlayer.types.table import EvalScorerColumn


def regex_scorer(
    title: str = "Regex",
    *,
    source: str = "output",
    regex_pattern: str,
    **settings: Any,
) -> EvalScorerColumn:
    """Build a REGEX scorer column."""
    require_non_empty_title(title, "regex_scorer")
    require_non_empty_source(source, "regex_scorer")
    require_non_empty_string(regex_pattern, field="regex_pattern", scorer_name="regex_scorer")
    config: Dict[str, Any] = {
        "source": source,
        "regex_pattern": regex_pattern,
        **settings,
    }
    return column(title, "REGEX", config)
