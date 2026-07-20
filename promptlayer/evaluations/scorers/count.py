"""Count column scorer."""

from __future__ import annotations

from typing import Any, Dict, Optional

from promptlayer.evaluations.columns import column
from promptlayer.evaluations.scorers._helpers import (
    apply_scorecard_step_options,
    pop_scorecard_step_options,
    require_count_bounds,
    require_non_empty_source,
    require_non_empty_string,
    require_non_empty_title,
)
from promptlayer.types.table import EvalScorerColumn


def count_scorer(
    title: str = "Count",
    *,
    source: str = "Output",
    type: str = "chars",
    min_count: Optional[int] = None,
    max_count: Optional[int] = None,
    **settings: Any,
) -> EvalScorerColumn:
    """Build a COUNT scorer column."""
    require_non_empty_title(title, "count_scorer")
    require_non_empty_source(source, "count_scorer")
    require_non_empty_string(type, field="type", scorer_name="count_scorer")
    require_count_bounds(min_count=min_count, max_count=max_count)
    step_options, config_settings = pop_scorecard_step_options(settings)
    config: Dict[str, Any] = {"source": source, "type": type, **config_settings}
    if min_count is not None:
        config["min_count"] = min_count
    if max_count is not None:
        config["max_count"] = max_count
    return apply_scorecard_step_options(column(title, "COUNT", config), **step_options)
