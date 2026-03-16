import os

from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

from promptlayer.utils import _PROMPTLAYER_USER_AGENT, SDK_VERSION

from .processor import PromptLayerOpenAIAgentsProcessor

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except ImportError:
    OTLPSpanExporter = None


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
    if OTLPSpanExporter is None:
        raise ImportError(
            "opentelemetry-exporter-otlp-proto-http is required to create a PromptLayer OTLP "
            "tracer provider for OpenAI Agents."
        )

    endpoint = _resolve_endpoint(endpoint=endpoint, base_url=base_url)
    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers={
            "X-Api-Key": api_key,
            "User-Agent": _PROMPTLAYER_USER_AGENT,
            "X-SDK-Version": SDK_VERSION,
        },
    )
    provider = TracerProvider(
        resource=Resource(attributes={ResourceAttributes.SERVICE_NAME: "promptlayer-openai-agents"})
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


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
    if endpoint:
        return endpoint

    env_endpoint = os.environ.get("PROMPTLAYER_OTLP_TRACES_ENDPOINT")
    if env_endpoint:
        return env_endpoint

    normalized_base_url = (base_url or os.environ.get("PROMPTLAYER_BASE_URL") or "https://api.promptlayer.com").rstrip(
        "/"
    )
    return f"{normalized_base_url}/v1/traces"
