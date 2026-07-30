"""Assert-valid column scorer."""

from __future__ import annotations

from typing import Any, Dict

from promptlayer.evaluations.columns import column
from promptlayer.evaluations.scorers._helpers import (
    apply_scorecard_step_options,
    pop_scorecard_step_options,
    reject_legacy_parameters,
    require_non_empty_source,
    require_non_empty_string,
    require_non_empty_title,
)
from promptlayer.types.table import EvalScorerColumn


def assert_valid_scorer(
    title: str = "Assert valid",
    *,
    source_column: str = "Output",
    type: str = "object",
    **settings: Any,
) -> EvalScorerColumn:
    """Build an ASSERT_VALID scorer column."""
    reject_legacy_parameters(settings, names=("source",), scorer_name="assert_valid_scorer")
    require_non_empty_title(title, "assert_valid_scorer")
    require_non_empty_source(source_column, "assert_valid_scorer", field="source_column")
    require_non_empty_string(type, field="type", scorer_name="assert_valid_scorer")
    step_options, config_settings = pop_scorecard_step_options(settings)
    config: Dict[str, Any] = {"source": source_column, "type": type, **config_settings}
    return apply_scorecard_step_options(column(title, "ASSERT_VALID", config), **step_options)
