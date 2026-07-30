import asyncio
import json
import logging
import os
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

import nest_asyncio
from opentelemetry import context as otel_context, trace as otel_trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from promptlayer import exceptions as _exceptions
from promptlayer.evaluations import AsyncEvalManager, EvalManager
from promptlayer.groups import AsyncGroupManager, GroupManager
from promptlayer.promptlayer_base import PromptLayerBase
from promptlayer.promptlayer_mixins import PromptLayerMixin
from promptlayer.skills import AsyncSkillManager, SkillManager
from promptlayer.span_exporter import (
    _defer_prompt_span_attribute_activation,
    _mark_genai_request_span,
    _prompt_span_attributes,
    _scope_prompt_span_attributes,
)
from promptlayer.streaming import astream_response, stream_response
from promptlayer.tables import AsyncTableManager, TableManager
from promptlayer.template_cache import PromptTemplateCache
from promptlayer.templates import AsyncTemplateManager, TemplateManager
from promptlayer.tracing_context import (
    awrap_stream_with_span,
    format_otel_span_id,
    is_stream_result,
    wrap_stream_with_span,
)
from promptlayer.track import AsyncTrackManager, TrackManager
from promptlayer.track.error_tracking import categorize_error
from promptlayer.types.prompt_template import PromptTemplate
from promptlayer.utils import (
    RERAISE_ORIGINAL_EXCEPTION,
    SDK_VERSION,
    _get_workflow_workflow_id_or_name,
    arun_workflow_request,
    atrack_request,
    autil_log_request,
    track_request,
    util_log_request,
)

logger = logging.getLogger(__name__)

_AUTO_INSTRUMENTED_RUN_PROVIDERS = frozenset(
    {
        "openai",
        "openai.azure",
        "anthropic",
        "google",
        "vertexai",
    }
)

_OTEL_SCALAR_TYPES = (bool, int, float, str)


def _run_span_attributes(
    *,
    prompt_name: str,
    prompt_version: Union[int, None],
    prompt_release_label: Union[str, None],
    metadata: Union[Dict[str, str], None],
) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {
        "node_type": "CODE_EXECUTION",
        "prompt_name": prompt_name,
        "promptlayer.prompt.name": prompt_name,
        "promptlayer.telemetry.source": "promptlayer-python",
        "promptlayer.telemetry.source_version": SDK_VERSION,
    }
    if prompt_version is not None:
        attributes["promptlayer.prompt.version"] = str(prompt_version)
    if prompt_release_label is not None:
        attributes["promptlayer.prompt.label"] = prompt_release_label

    for key, value in (metadata or {}).items():
        if isinstance(key, str) and key and isinstance(value, _OTEL_SCALAR_TYPES):
            attributes[f"promptlayer.metadata.{key}"] = value

    return attributes


def _prompt_fetch_attributes(
    *,
    prompt_name: str,
    prompt_version: Union[int, None],
    prompt_release_label: Union[str, None],
) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {
        "node_type": "PROMPT_TEMPLATE",
        "prompt_name": prompt_name,
        "promptlayer.prompt.name": prompt_name,
        "promptlayer.prompt.requested.name": prompt_name,
    }
    if prompt_version is not None:
        attributes["promptlayer.prompt.requested.version"] = str(prompt_version)
    if prompt_release_label is not None:
        attributes["promptlayer.prompt.requested.label"] = prompt_release_label
    return attributes


def _record_span_error(span, error: Exception) -> None:
    try:
        error_type = type(error).__qualname__
        module = type(error).__module__
        if module and module != "builtins":
            error_type = f"{module}.{error_type}"

        span.set_attribute("error.type", error_type)
        span.record_exception(error)
        span.set_status(Status(status_code=StatusCode.ERROR, description=f"{error_type}: {error}"))
    except Exception:
        logger.debug("PromptLayer could not record a tracing error", exc_info=True)


def _set_span_attributes(span, attributes: Dict[str, str]) -> None:
    try:
        span.set_attributes(attributes)
    except Exception:
        logger.debug("PromptLayer could not set tracing attributes", exc_info=True)


@contextmanager
def _prompt_fetch_span(
    tracer,
    run_span,
    *,
    prompt_name: str,
    prompt_version: Union[int, None],
    prompt_release_label: Union[str, None],
):
    if tracer is None or run_span is None:
        yield None
        return

    with tracer.start_as_current_span(
        "Prompt template fetch",
        kind=SpanKind.INTERNAL,
        attributes=_prompt_fetch_attributes(
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            prompt_release_label=prompt_release_label,
        ),
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except Exception as exc:
            _record_span_error(span, exc)
            raise


def get_base_url(base_url: Union[str, None]):
    return base_url or os.environ.get("PROMPTLAYER_BASE_URL", "https://api.promptlayer.com")


def is_workflow_results_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False

    required_keys = {
        "status",
        "value",
        "error_message",
        "raw_error_message",
        "is_output_node",
    }

    for val in obj.values():
        if not isinstance(val, dict):
            return False
        if not required_keys.issubset(val.keys()):
            return False

    return True


class PromptLayer(PromptLayerMixin):
    def __init__(
        self,
        api_key: Union[str, None] = None,
        enable_tracing: bool = False,
        base_url: Union[str, None] = None,
        throw_on_error: bool = True,
        cache_ttl_seconds: int = 0,
        tracer_provider=None,
        tracing_providers: Optional[Iterable[str]] = None,
    ):
        if api_key is None:
            api_key = os.environ.get("PROMPTLAYER_API_KEY")

        if api_key is None:
            raise ValueError(
                "PromptLayer API key not provided. "
                "Please set the PROMPTLAYER_API_KEY environment variable or pass the api_key parameter."
            )

        self.base_url = get_base_url(base_url)
        self.api_key = api_key
        self.throw_on_error = throw_on_error
        cache = PromptTemplateCache(cache_ttl_seconds) if cache_ttl_seconds else None
        self.templates = TemplateManager(api_key, self.base_url, self.throw_on_error, cache=cache)
        self.skills = SkillManager(api_key, self.base_url, self.throw_on_error)
        self.tables = TableManager(api_key, self.base_url, self.throw_on_error)
        self.tracer_provider, self.tracer = self._initialize_tracer(
            api_key,
            self.base_url,
            self.throw_on_error,
            enable_tracing,
            tracer_provider,
            tracing_providers,
        )
        self.evals = EvalManager(api_key, self.base_url, self.throw_on_error, self.tracer_provider)
        self.group = GroupManager(api_key, self.base_url, self.throw_on_error)
        self.track = TrackManager(api_key, self.base_url, self.throw_on_error)

    def invalidate(self, prompt_name: Optional[str] = None) -> None:
        """Invalidate SDK template cache for a prompt or entirely."""
        self.templates.invalidate(prompt_name)

    def __getattr__(
        self,
        name: Union[Literal["openai"], Literal["anthropic"], Literal["prompts"]],
    ):
        if name == "openai":
            import openai as openai_module

            return PromptLayerBase(
                self.api_key, self.base_url, openai_module, function_name="openai", tracer=self._tracer
            )
        elif name == "anthropic":
            import anthropic as anthropic_module

            return PromptLayerBase(
                self.api_key,
                self.base_url,
                anthropic_module,
                function_name="anthropic",
                provider_type="anthropic",
                tracer=self._tracer,
            )
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")

    def _create_track_request_callable(
        self,
        *,
        request_params,
        tags,
        input_variables,
        group_id,
        pl_run_span_id: Union[str, None] = None,
        request_start_time: Union[float, None] = None,
    ):
        def _track_request(**body):
            track_request_kwargs = self._prepare_track_request_kwargs(
                self.api_key,
                request_params,
                tags,
                input_variables,
                group_id,
                pl_run_span_id,
                request_start_time=request_start_time,
                **body,
            )
            return track_request(self.base_url, self.throw_on_error, **track_request_kwargs)

        return _track_request

    def _run_internal(
        self,
        *,
        prompt_name: str,
        prompt_version: Union[int, None] = None,
        prompt_release_label: Union[str, None] = None,
        input_variables: Union[Dict[str, Any], None] = None,
        model_parameter_overrides: Union[Dict[str, Any], None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, str], None] = None,
        group_id: Union[int, None] = None,
        stream: bool = False,
        pl_run_span_id: Union[str, None] = None,
        pl_run_span=None,
        provider: Union[str, None] = None,
        model: Union[str, None] = None,
    ) -> Dict[str, Any]:
        import datetime

        get_prompt_template_params = self._prepare_get_prompt_template_params(
            prompt_version=prompt_version,
            prompt_release_label=prompt_release_label,
            input_variables=input_variables,
            metadata=metadata,
            provider=provider,
            model=model,
            model_parameter_overrides=model_parameter_overrides,
        )
        tracer = self.tracer
        with _prompt_fetch_span(
            tracer,
            pl_run_span,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            prompt_release_label=prompt_release_label,
        ) as fetch_span:
            fetch_context = _defer_prompt_span_attribute_activation() if fetch_span is not None else nullcontext()
            with fetch_context:
                prompt_blueprint = self.templates.get(prompt_name, get_prompt_template_params)
            if not prompt_blueprint:
                raise _exceptions.PromptLayerNotFoundError(
                    f"Prompt template '{prompt_name}' not found.",
                    response=None,
                    body=None,
                )
            if fetch_span is not None:
                resolved_prompt_attributes = _prompt_span_attributes(
                    prompt_blueprint,
                    prompt_name,
                    label=prompt_release_label,
                )
                _set_span_attributes(fetch_span, resolved_prompt_attributes)
        if fetch_span is not None:
            _set_span_attributes(pl_run_span, resolved_prompt_attributes)
        prompt_blueprint_model = self._validate_and_extract_model_from_prompt_blueprint(
            prompt_blueprint=prompt_blueprint, prompt_name=prompt_name
        )
        llm_data = self._prepare_llm_data(
            prompt_blueprint=prompt_blueprint,
            prompt_template=prompt_blueprint["prompt_template"],
            prompt_blueprint_model=prompt_blueprint_model,
            stream=stream,
        )

        # Capture start time before making the LLM request
        request_start_time = datetime.datetime.now(datetime.timezone.utc).timestamp()

        # response is just whatever the LLM call returns
        # streaming=False > Pydantic model instance
        # streaming=True > generator that yields ChatCompletionChunk pieces as they arrive
        traced_run = tracer is not None and pl_run_span is not None
        prompt_context = _scope_prompt_span_attributes(resolved_prompt_attributes) if traced_run else nullcontext()
        try:
            with (
                prompt_context,
                _mark_genai_request_span(
                    enabled=llm_data["provider"] in _AUTO_INSTRUMENTED_RUN_PROVIDERS,
                    request_log_span_id=pl_run_span_id,
                ),
            ):
                response = llm_data["request_function"](
                    prompt_blueprint=llm_data["prompt_blueprint"],
                    client_kwargs=llm_data["client_kwargs"],
                    function_kwargs=llm_data["function_kwargs"],
                )
        except Exception as e:
            request_end_time = datetime.datetime.now(datetime.timezone.utc).timestamp()
            try:
                self._track_request_log(
                    llm_data,
                    tags,
                    input_variables,
                    group_id,
                    pl_run_span_id,
                    metadata=metadata,
                    request_response={},
                    request_start_time=request_start_time,
                    request_end_time=request_end_time,
                    status="ERROR",
                    error_type=categorize_error(e),
                    error_message=str(e)[:1024],
                )
            except Exception:
                logger.debug("Failed to log error to PromptLayer", exc_info=True)
            raise

        # Capture end time after the LLM request completes
        request_end_time = datetime.datetime.now(datetime.timezone.utc).timestamp()

        if stream:
            return stream_response(
                generator=response,
                after_stream=self._create_track_request_callable(
                    request_params=llm_data,
                    tags=tags,
                    input_variables=input_variables,
                    group_id=group_id,
                    pl_run_span_id=pl_run_span_id,
                    request_start_time=request_start_time,
                ),
                map_results=llm_data["stream_function"],
                metadata=llm_data["prompt_blueprint"]["metadata"],
            )

        if isinstance(response, dict):
            request_response = response
        else:
            request_response = response.model_dump(mode="json")

        request_log = self._track_request_log(
            llm_data,
            tags,
            input_variables,
            group_id,
            pl_run_span_id,
            metadata=metadata,
            request_response=request_response,
            request_start_time=request_start_time,
            request_end_time=request_end_time,
        )

        return {
            "request_id": request_log.get("request_id", None),
            "raw_response": response,
            "prompt_blueprint": request_log.get("prompt_blueprint", None),
        }

    def _track_request_log(
        self,
        request_params,
        tags,
        input_variables,
        group_id,
        pl_run_span_id: Union[str, None] = None,
        metadata: Union[Dict[str, str], None] = None,
        **body,
    ):
        track_request_kwargs = self._prepare_track_request_kwargs(
            self.api_key,
            request_params,
            tags,
            input_variables,
            group_id,
            pl_run_span_id,
            metadata=metadata,
            **body,
        )
        return track_request(self.base_url, self.throw_on_error, **track_request_kwargs)

    def run(
        self,
        prompt_name: str,
        prompt_version: Union[int, None] = None,
        prompt_release_label: Union[str, None] = None,
        input_variables: Union[Dict[str, Any], None] = None,
        model_parameter_overrides: Union[Dict[str, Any], None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, str], None] = None,
        group_id: Union[int, None] = None,
        stream: bool = False,
        provider: Union[str, None] = None,
        model: Union[str, None] = None,
    ) -> Dict[str, Any]:
        _run_internal_kwargs = {
            "prompt_name": prompt_name,
            "prompt_version": prompt_version,
            "prompt_release_label": prompt_release_label,
            "input_variables": input_variables or {},
            "model_parameter_overrides": model_parameter_overrides,
            "tags": tags,
            "metadata": metadata,
            "group_id": group_id,
            "stream": stream,
            "provider": provider,
            "model": model,
        }

        tracer = self.tracer
        if tracer:
            with _scope_prompt_span_attributes(None):
                span = tracer.start_span(
                    "PromptLayer Run",
                    kind=SpanKind.INTERNAL,
                    attributes=_run_span_attributes(
                        prompt_name=prompt_name,
                        prompt_version=prompt_version,
                        prompt_release_label=prompt_release_label,
                        metadata=metadata,
                    ),
                )
                token = otel_context.attach(otel_trace.set_span_in_context(span))
                result = None
                try:
                    span_ctx = span.get_span_context()
                    pl_run_span_id = format_otel_span_id(span_ctx.span_id) if span_ctx and span_ctx.is_valid else None
                    result = self._run_internal(
                        **_run_internal_kwargs,
                        pl_run_span_id=pl_run_span_id,
                        pl_run_span=span,
                    )
                    if is_stream_result(result):
                        otel_context.detach(token)
                        return wrap_stream_with_span(result, span)
                    return result
                except Exception as exc:
                    _record_span_error(span, exc)
                    raise
                finally:
                    if not is_stream_result(result):
                        otel_context.detach(token)
                        span.end()
        else:
            return self._run_internal(**_run_internal_kwargs)

    def run_workflow(
        self,
        workflow_id_or_name: Optional[Union[int, str]] = None,
        input_variables: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        workflow_label_name: Optional[str] = None,
        workflow_version: Optional[int] = None,
        return_all_outputs: Optional[bool] = False,
        # `workflow_name` deprecated, kept for backward compatibility only.
        # Allows `workflow_name` to be passed both as keyword and positional argument
        # (virtually identical to `workflow_id_or_name`)
        workflow_name: Optional[str] = None,
        timeout: Optional[Union[int, float]] = None,
    ) -> Union[Dict[str, Any], Any]:
        try:
            try:
                loop = asyncio.get_running_loop()  # Check if we're inside a running event loop
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                nest_asyncio.apply()

            coro = arun_workflow_request(
                api_key=self.api_key,
                base_url=self.base_url,
                throw_on_error=self.throw_on_error,
                workflow_id_or_name=_get_workflow_workflow_id_or_name(workflow_id_or_name, workflow_name),
                input_variables=input_variables or {},
                metadata=metadata,
                workflow_label_name=workflow_label_name,
                workflow_version_number=workflow_version,
                return_all_outputs=return_all_outputs,
            )
            results = asyncio.run(asyncio.wait_for(coro, timeout=timeout))

            if not return_all_outputs and is_workflow_results_dict(results):
                output_nodes = [node_data for node_data in results.values() if node_data.get("is_output_node")]
                if not output_nodes:
                    raise _exceptions.PromptLayerNotFoundError(
                        f"Output nodes not found: {json.dumps(results, indent=4)}", response=None, body=results
                    )

                if not any(node.get("status") == "SUCCESS" for node in output_nodes):
                    raise _exceptions.PromptLayerAPIError(
                        f"None of the output nodes have succeeded: {json.dumps(results, indent=4)}",
                        response=None,
                        body=results,
                    )

            return results
        except (asyncio.TimeoutError, TimeoutError) as ex:
            raise _exceptions.PromptLayerAPITimeoutError(
                "Workflow execution timed out", response=None, body=None
            ) from ex
        except _exceptions.PromptLayerAPITimeoutError:
            raise  # Re-raise as-is (from inner arun_workflow_request)
        except Exception as ex:
            logger.exception("Error running workflow")
            if RERAISE_ORIGINAL_EXCEPTION:
                raise
            else:
                raise _exceptions.PromptLayerAPIError(
                    f"Error running workflow: {str(ex)}", response=None, body=None
                ) from ex

    def log_request(
        self,
        *,
        provider: str,
        model: str,
        input: PromptTemplate,
        output: PromptTemplate,
        request_start_time: float,
        request_end_time: float,
        # TODO(dmu) MEDIUM: Avoid using mutable defaults
        # TODO(dmu) MEDIUM: Deprecate and remove this wrapper function?
        parameters: Dict[str, Any] = {},
        tags: List[str] = [],
        metadata: Dict[str, str] = {},
        prompt_name: Union[str, None] = None,
        prompt_version_number: Union[int, None] = None,
        prompt_input_variables: Dict[str, Any] = {},
        input_tokens: int = 0,
        output_tokens: int = 0,
        price: float = 0.0,
        function_name: str = "",
        score: int = 0,
        prompt_id: Union[int, None] = None,
        score_name: Union[str, None] = None,
        api_type: Union[str, None] = None,
        status: str = "SUCCESS",
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        return util_log_request(
            self.api_key,
            self.base_url,
            throw_on_error=self.throw_on_error,
            provider=provider,
            model=model,
            input=input,
            output=output,
            request_start_time=request_start_time,
            request_end_time=request_end_time,
            parameters=parameters,
            tags=tags,
            metadata=metadata,
            prompt_name=prompt_name,
            prompt_version_number=prompt_version_number,
            prompt_input_variables=prompt_input_variables,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            price=price,
            function_name=function_name,
            score=score,
            prompt_id=prompt_id,
            score_name=score_name,
            api_type=api_type,
            status=status,
            error_type=error_type,
            error_message=error_message,
        )


class AsyncPromptLayer(PromptLayerMixin):
    def __init__(
        self,
        api_key: Union[str, None] = None,
        enable_tracing: bool = False,
        base_url: Union[str, None] = None,
        throw_on_error: bool = True,
        cache_ttl_seconds: int = 0,
        tracer_provider=None,
        tracing_providers: Optional[Iterable[str]] = None,
    ):
        if api_key is None:
            api_key = os.environ.get("PROMPTLAYER_API_KEY")

        if api_key is None:
            raise ValueError(
                "PromptLayer API key not provided. "
                "Please set the PROMPTLAYER_API_KEY environment variable or pass the api_key parameter."
            )

        self.base_url = get_base_url(base_url)
        self.api_key = api_key
        self.throw_on_error = throw_on_error
        cache = PromptTemplateCache(cache_ttl_seconds) if cache_ttl_seconds else None
        self.templates = AsyncTemplateManager(api_key, self.base_url, self.throw_on_error, cache=cache)
        self.skills = AsyncSkillManager(api_key, self.base_url, self.throw_on_error)
        self.tables = AsyncTableManager(api_key, self.base_url, self.throw_on_error)
        self.tracer_provider, self.tracer = self._initialize_tracer(
            api_key,
            self.base_url,
            self.throw_on_error,
            enable_tracing,
            tracer_provider,
            tracing_providers,
        )
        self.evals = AsyncEvalManager(api_key, self.base_url, self.throw_on_error, self.tracer_provider)
        self.group = AsyncGroupManager(api_key, self.base_url, self.throw_on_error)
        self.track = AsyncTrackManager(api_key, self.base_url, self.throw_on_error)

    def invalidate(self, prompt_name: Optional[str] = None) -> None:
        """Invalidate SDK template cache for a prompt or entirely."""
        self.templates.invalidate(prompt_name)

    def __getattr__(self, name: Union[Literal["openai"], Literal["anthropic"], Literal["prompts"]]):
        if name == "openai":
            import openai as openai_module

            openai = PromptLayerBase(
                self.api_key, self.base_url, openai_module, function_name="openai", tracer=self._tracer
            )
            return openai
        elif name == "anthropic":
            import anthropic as anthropic_module

            anthropic = PromptLayerBase(
                self.api_key,
                self.base_url,
                anthropic_module,
                function_name="anthropic",
                provider_type="anthropic",
                tracer=self._tracer,
            )
            return anthropic
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")

    async def run_workflow(
        self,
        workflow_id_or_name: Optional[Union[int, str]] = None,
        input_variables: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        workflow_label_name: Optional[str] = None,
        workflow_version: Optional[int] = None,  # This is the version number, not the version ID
        return_all_outputs: Optional[bool] = False,
        # `workflow_name` deprecated, kept for backward compatibility only.
        # Allows `workflow_name` to be passed both as keyword and positional argument
        # (virtually identical to `workflow_id_or_name`)
        workflow_name: Optional[str] = None,
        timeout: Optional[Union[int, float]] = None,
    ) -> Union[Dict[str, Any], Any]:
        try:
            coro = arun_workflow_request(
                api_key=self.api_key,
                base_url=self.base_url,
                throw_on_error=self.throw_on_error,
                workflow_id_or_name=_get_workflow_workflow_id_or_name(workflow_id_or_name, workflow_name),
                input_variables=input_variables or {},
                metadata=metadata,
                workflow_label_name=workflow_label_name,
                workflow_version_number=workflow_version,
                return_all_outputs=return_all_outputs,
            )
            return await asyncio.wait_for(coro, timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError) as ex:
            raise _exceptions.PromptLayerAPITimeoutError(
                "Workflow execution timed out", response=None, body=None
            ) from ex
        except _exceptions.PromptLayerAPITimeoutError:
            raise  # Re-raise as-is (from inner arun_workflow_request)
        except Exception as ex:
            logger.exception("Error running workflow")
            if RERAISE_ORIGINAL_EXCEPTION:
                raise
            else:
                raise _exceptions.PromptLayerAPIError(
                    f"Error running workflow: {str(ex)}", response=None, body=None
                ) from ex

    async def run(
        self,
        prompt_name: str,
        prompt_version: Union[int, None] = None,
        prompt_release_label: Union[str, None] = None,
        input_variables: Union[Dict[str, Any], None] = None,
        model_parameter_overrides: Union[Dict[str, Any], None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, str], None] = None,
        group_id: Union[int, None] = None,
        stream: bool = False,
        provider: Union[str, None] = None,
        model: Union[str, None] = None,
    ) -> Dict[str, Any]:
        _run_internal_kwargs = {
            "prompt_name": prompt_name,
            "prompt_version": prompt_version,
            "prompt_release_label": prompt_release_label,
            "input_variables": input_variables,
            "model_parameter_overrides": model_parameter_overrides,
            "tags": tags,
            "metadata": metadata,
            "group_id": group_id,
            "stream": stream,
            "provider": provider,
            "model": model,
        }

        tracer = self.tracer
        if tracer:
            with _scope_prompt_span_attributes(None):
                span = tracer.start_span(
                    "PromptLayer Run",
                    kind=SpanKind.INTERNAL,
                    attributes=_run_span_attributes(
                        prompt_name=prompt_name,
                        prompt_version=prompt_version,
                        prompt_release_label=prompt_release_label,
                        metadata=metadata,
                    ),
                )
                token = otel_context.attach(otel_trace.set_span_in_context(span))
                result = None
                try:
                    span_ctx = span.get_span_context()
                    pl_run_span_id = format_otel_span_id(span_ctx.span_id) if span_ctx and span_ctx.is_valid else None
                    result = await self._run_internal(
                        **_run_internal_kwargs,
                        pl_run_span_id=pl_run_span_id,
                        pl_run_span=span,
                    )
                    if is_stream_result(result):
                        otel_context.detach(token)
                        return awrap_stream_with_span(result, span)
                    return result
                except Exception as exc:
                    _record_span_error(span, exc)
                    raise
                finally:
                    if not is_stream_result(result):
                        otel_context.detach(token)
                        span.end()
        else:
            return await self._run_internal(**_run_internal_kwargs)

    async def log_request(
        self,
        *,
        provider: str,
        model: str,
        input: PromptTemplate,
        output: PromptTemplate,
        request_start_time: float,
        request_end_time: float,
        parameters: Dict[str, Any] = {},
        tags: List[str] = [],
        metadata: Dict[str, str] = {},
        prompt_name: Union[str, None] = None,
        prompt_version_number: Union[int, None] = None,
        prompt_input_variables: Dict[str, Any] = {},
        input_tokens: int = 0,
        output_tokens: int = 0,
        price: float = 0.0,
        function_name: str = "",
        score: int = 0,
        prompt_id: Union[int, None] = None,
        status: str = "SUCCESS",
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        return await autil_log_request(
            self.api_key,
            self.base_url,
            throw_on_error=self.throw_on_error,
            provider=provider,
            model=model,
            input=input,
            output=output,
            request_start_time=request_start_time,
            request_end_time=request_end_time,
            parameters=parameters,
            tags=tags,
            metadata=metadata,
            prompt_name=prompt_name,
            prompt_version_number=prompt_version_number,
            prompt_input_variables=prompt_input_variables,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            price=price,
            function_name=function_name,
            score=score,
            prompt_id=prompt_id,
            status=status,
            error_type=error_type,
            error_message=error_message,
        )

    async def _create_track_request_callable(
        self,
        *,
        request_params,
        tags,
        input_variables,
        group_id,
        pl_run_span_id: Union[str, None] = None,
        request_start_time: Union[float, None] = None,
    ):
        async def _track_request(**body):
            track_request_kwargs = self._prepare_track_request_kwargs(
                self.api_key,
                request_params,
                tags,
                input_variables,
                group_id,
                pl_run_span_id,
                request_start_time=request_start_time,
                **body,
            )
            return await atrack_request(self.base_url, self.throw_on_error, **track_request_kwargs)

        return _track_request

    async def _track_request_log(
        self,
        request_params,
        tags,
        input_variables,
        group_id,
        pl_run_span_id: Union[str, None] = None,
        metadata: Union[Dict[str, str], None] = None,
        **body,
    ):
        track_request_kwargs = self._prepare_track_request_kwargs(
            self.api_key,
            request_params,
            tags,
            input_variables,
            group_id,
            pl_run_span_id,
            metadata=metadata,
            **body,
        )
        return await atrack_request(self.base_url, self.throw_on_error, **track_request_kwargs)

    async def _run_internal(
        self,
        *,
        prompt_name: str,
        prompt_version: Union[int, None] = None,
        prompt_release_label: Union[str, None] = None,
        input_variables: Union[Dict[str, Any], None] = None,
        model_parameter_overrides: Union[Dict[str, Any], None] = None,
        tags: Union[List[str], None] = None,
        metadata: Union[Dict[str, str], None] = None,
        group_id: Union[int, None] = None,
        stream: bool = False,
        pl_run_span_id: Union[str, None] = None,
        pl_run_span=None,
        provider: Union[str, None] = None,
        model: Union[str, None] = None,
    ) -> Dict[str, Any]:
        import datetime

        get_prompt_template_params = self._prepare_get_prompt_template_params(
            prompt_version=prompt_version,
            prompt_release_label=prompt_release_label,
            input_variables=input_variables,
            metadata=metadata,
            provider=provider,
            model=model,
            model_parameter_overrides=model_parameter_overrides,
        )
        tracer = self.tracer
        with _prompt_fetch_span(
            tracer,
            pl_run_span,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            prompt_release_label=prompt_release_label,
        ) as fetch_span:
            fetch_context = _defer_prompt_span_attribute_activation() if fetch_span is not None else nullcontext()
            with fetch_context:
                prompt_blueprint = await self.templates.get(prompt_name, get_prompt_template_params)
            if not prompt_blueprint:
                raise _exceptions.PromptLayerNotFoundError(
                    f"Prompt template '{prompt_name}' not found.",
                    response=None,
                    body=None,
                )
            if fetch_span is not None:
                resolved_prompt_attributes = _prompt_span_attributes(
                    prompt_blueprint,
                    prompt_name,
                    label=prompt_release_label,
                )
                _set_span_attributes(fetch_span, resolved_prompt_attributes)
        if fetch_span is not None:
            _set_span_attributes(pl_run_span, resolved_prompt_attributes)
        prompt_blueprint_model = self._validate_and_extract_model_from_prompt_blueprint(
            prompt_blueprint=prompt_blueprint, prompt_name=prompt_name
        )
        llm_data = self._prepare_llm_data(
            prompt_blueprint=prompt_blueprint,
            prompt_template=prompt_blueprint["prompt_template"],
            prompt_blueprint_model=prompt_blueprint_model,
            stream=stream,
            is_async=True,
        )

        # Capture start time before making the LLM request
        request_start_time = datetime.datetime.now(datetime.timezone.utc).timestamp()

        traced_run = tracer is not None and pl_run_span is not None
        prompt_context = _scope_prompt_span_attributes(resolved_prompt_attributes) if traced_run else nullcontext()
        try:
            with (
                prompt_context,
                _mark_genai_request_span(
                    enabled=llm_data["provider"] in _AUTO_INSTRUMENTED_RUN_PROVIDERS,
                    request_log_span_id=pl_run_span_id,
                ),
            ):
                response = await llm_data["request_function"](
                    prompt_blueprint=llm_data["prompt_blueprint"],
                    client_kwargs=llm_data["client_kwargs"],
                    function_kwargs=llm_data["function_kwargs"],
                )
        except Exception as e:
            request_end_time = datetime.datetime.now(datetime.timezone.utc).timestamp()
            try:
                await self._track_request_log(
                    llm_data,
                    tags,
                    input_variables,
                    group_id,
                    pl_run_span_id,
                    metadata=metadata,
                    request_response={},
                    request_start_time=request_start_time,
                    request_end_time=request_end_time,
                    status="ERROR",
                    error_type=categorize_error(e),
                    error_message=str(e)[:1024],
                )
            except Exception:
                logger.debug("Failed to log error to PromptLayer", exc_info=True)
            raise

        # Capture end time after the LLM request completes
        request_end_time = datetime.datetime.now(datetime.timezone.utc).timestamp()

        if stream:
            track_request_callable = await self._create_track_request_callable(
                request_params=llm_data,
                tags=tags,
                input_variables=input_variables,
                group_id=group_id,
                pl_run_span_id=pl_run_span_id,
                request_start_time=request_start_time,
            )
            return astream_response(
                response,
                track_request_callable,
                llm_data["stream_function"],
                llm_data["prompt_blueprint"]["metadata"],
            )

        if hasattr(response, "model_dump"):
            request_response = response.model_dump(mode="json")
        else:
            request_response = response

        request_log = await self._track_request_log(
            llm_data,
            tags,
            input_variables,
            group_id,
            pl_run_span_id,
            metadata=metadata,
            request_response=request_response,
            request_start_time=request_start_time,
            request_end_time=request_end_time,
        )

        return {
            "request_id": request_log.get("request_id", None),
            "raw_response": response,
            "prompt_blueprint": request_log.get("prompt_blueprint", None),
        }
