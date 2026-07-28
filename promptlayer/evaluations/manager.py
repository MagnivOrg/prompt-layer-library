from typing import Any, Dict, Optional

from opentelemetry.sdk.trace import TracerProvider

from promptlayer.evaluations.runner import arun_eval, run_eval
from promptlayer.types.table import EvalDefinition, EvalResult


def definition_to_run_kwargs(
    definition: EvalDefinition,
    *,
    api_key: str,
    base_url: str,
    throw_on_error: bool,
    tracer_provider: Optional[TracerProvider],
) -> Dict[str, Any]:
    """Map an ``EvalDefinition`` onto ``run_eval`` / ``arun_eval`` kwargs."""
    resolved_api_key = definition.get("api_key") or api_key
    resolved_base_url = definition.get("base_url") or base_url
    resolved_tracer = tracer_provider
    if resolved_tracer is None:
        from promptlayer.promptlayer_mixins import PromptLayerMixin

        resolved_tracer, _ = PromptLayerMixin._initialize_tracer(
            resolved_api_key,
            resolved_base_url,
            throw_on_error,
            enable_tracing=True,
        )

    max_concurrency = definition.get("max_concurrency", 1)
    if max_concurrency is None:
        max_concurrency = 1

    return {
        "name": definition["name"],
        "dataset": definition["dataset"],
        "runner": definition["runner"],
        "scorers": definition["scorers"],
        "columns": definition.get("columns"),
        "api_key": resolved_api_key,
        "base_url": resolved_base_url,
        "throw_on_error": throw_on_error,
        "tracer_provider": resolved_tracer,
        "table_id": definition.get("table_id"),
        "sheet_id": definition.get("sheet_id"),
        "folder_id": definition.get("folder_id"),
        "experiment_name": definition.get("experiment_name"),
        "max_concurrency": max_concurrency,
        "passing_score": definition.get("passing_score"),
        "include_failure_examples": bool(definition.get("include_failure_examples", False)),
    }


class EvalManager:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        throw_on_error: bool,
        tracer_provider: Optional[TracerProvider] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.throw_on_error = throw_on_error
        self.tracer_provider = tracer_provider

    def run(self, definition: EvalDefinition) -> EvalResult:
        return run_eval(
            **definition_to_run_kwargs(
                definition,
                api_key=self.api_key,
                base_url=self.base_url,
                throw_on_error=self.throw_on_error,
                tracer_provider=self.tracer_provider,
            )
        )


class AsyncEvalManager:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        throw_on_error: bool,
        tracer_provider: Optional[TracerProvider] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.throw_on_error = throw_on_error
        self.tracer_provider = tracer_provider

    async def run(self, definition: EvalDefinition) -> EvalResult:
        return await arun_eval(
            **definition_to_run_kwargs(
                definition,
                api_key=self.api_key,
                base_url=self.base_url,
                throw_on_error=self.throw_on_error,
                tracer_provider=self.tracer_provider,
            )
        )
