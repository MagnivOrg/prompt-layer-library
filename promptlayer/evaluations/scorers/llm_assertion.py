"""LLM assertion column scorer."""

from __future__ import annotations

from typing import Any, Dict, Optional

from promptlayer.evaluations.columns import column
from promptlayer.evaluations.scorers._helpers import require_non_empty_source, require_non_empty_title
from promptlayer.evaluations.validation import validation_error
from promptlayer.types.table import EvalScorerColumn


def llm_assertion_scorer(
    title: str = "LLM assertion",
    *,
    source: str = "output",
    prompt: Optional[str] = None,
    prompt_source: Optional[str] = None,
    variable_mappings: Optional[Dict[str, str]] = None,
    **settings: Any,
) -> EvalScorerColumn:
    """Build an LLM_ASSERTION scorer column."""
    require_non_empty_title(title, "llm_assertion_scorer")
    require_non_empty_source(source, "llm_assertion_scorer")
    if prompt is None and prompt_source is None:
        raise validation_error("llm_assertion_scorer requires either prompt or prompt_source.")
    config: Dict[str, Any] = {"source": source, **settings}
    if prompt is not None:
        config["prompt"] = prompt
    if prompt_source is not None:
        config["prompt_source"] = prompt_source
    if variable_mappings is not None:
        config["variable_mappings"] = variable_mappings
    return column(title, "LLM_ASSERTION", config)
