"""Contains column scorer."""

from __future__ import annotations

from typing import Any, Dict, Optional

from promptlayer.evaluations.columns import column
from promptlayer.evaluations.scorers._helpers import require_non_empty_source, require_non_empty_title
from promptlayer.evaluations.validation import validation_error
from promptlayer.types.table import EvalScorerColumn


def contains_scorer(
    title: str = "Contains",
    *,
    source: str = "output",
    value: Optional[str] = None,
    value_source: Optional[str] = None,
    **settings: Any,
) -> EvalScorerColumn:
    """Build a CONTAINS scorer column."""
    require_non_empty_title(title, "contains_scorer")
    require_non_empty_source(source, "contains_scorer")
    if value is None and value_source is None:
        raise validation_error("contains_scorer requires either value or value_source.")
    config: Dict[str, Any] = {"source": source, **settings}
    if value is not None:
        config["value"] = value
    if value_source is not None:
        config["value_source"] = value_source
    return column(title, "CONTAINS", config)

