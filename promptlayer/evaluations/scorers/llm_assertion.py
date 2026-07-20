"""LLM assertion column scorer."""

from __future__ import annotations

from typing import Any, Dict, Optional

from promptlayer.evaluations.columns import column
from promptlayer.evaluations.scorers._helpers import (
    apply_scorecard_step_options,
    pop_scorecard_step_options,
    require_non_empty_source,
    require_non_empty_title,
)
from promptlayer.evaluations.validation import validation_error
from promptlayer.types.table import EvalScorerColumn


def llm_assertion_scorer(
    title: str = "LLM assertion",
    *,
    source: str = "Output",
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
    step_options, config_settings = pop_scorecard_step_options(settings)
    config: Dict[str, Any] = {"source": source, **config_settings}
    if prompt is not None:
        config["prompt"] = prompt
    if prompt_source is not None:
        config["prompt_source"] = prompt_source
    if variable_mappings is not None:
        config["variable_mappings"] = variable_mappings
    return apply_scorecard_step_options(column(title, "LLM_ASSERTION", config), **step_options)
