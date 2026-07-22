from __future__ import annotations

import os

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider

from promptlayer.otlp import create_promptlayer_tracer_provider, resolve_otlp_traces_endpoint

from .processor import PromptLayerOpenAIAgentsProcessor


class OpenAIAgentsTracingProviderError(RuntimeError):
    pass


def instrument_openai_agents(
    *,
    tracer_provider: TracerProvider | None = None,
    api_key: str | None = None,
    endpoint: str | None = None,
    base_url: str | None = None,
    exclusive: bool = True,
    include_raw_payloads: bool = True,
) -> PromptLayerOpenAIAgentsProcessor:
    provider = tracer_provider
    if provider is None:
        resolved_api_key = api_key or os.environ.get("PROMPTLAYER_API_KEY")
        if not resolved_api_key:
            provider = trace_api.get_tracer_provider()
        else:
            provider = create_openai_agents_tracer_provider(
                api_key=resolved_api_key,
                endpoint=endpoint,
                base_url=base_url,
            )

    processor = PromptLayerOpenAIAgentsProcessor(
        tracer_provider=_validate_tracer_provider(provider),
        include_raw_payloads=include_raw_payloads,
    )

    try:
        from agents import tracing as agents_tracing
    except ImportError as exc:
        raise ImportError(
            "openai-agents is required for PromptLayer OpenAI Agents instrumentation. "
            "Install the 'openai-agents' Poetry extra."
        ) from exc

    if exclusive:
        agents_tracing.set_trace_processors([processor])
    else:
        agents_tracing.add_trace_processor(processor)

    return processor


def create_openai_agents_tracer_provider(
    *,
    api_key: str,
    endpoint: str | None = None,
    base_url: str | None = None,
) -> TracerProvider:
    return create_promptlayer_tracer_provider(
        api_key=api_key,
        endpoint=endpoint,
        base_url=base_url,
        service_name="promptlayer-openai-agents",
    )


def _validate_tracer_provider(provider) -> TracerProvider:
    if not isinstance(provider, TracerProvider):
        raise OpenAIAgentsTracingProviderError(
            "instrument_openai_agents requires an opentelemetry.sdk.trace.TracerProvider."
        )

    span_processors = getattr(getattr(provider, "_active_span_processor", None), "_span_processors", ())
    if not span_processors:
        raise OpenAIAgentsTracingProviderError(
            "instrument_openai_agents requires a TracerProvider with at least one span processor."
        )

    return provider


def _resolve_endpoint(*, endpoint: str | None, base_url: str | None) -> str:
    return resolve_otlp_traces_endpoint(endpoint=endpoint, base_url=base_url)
