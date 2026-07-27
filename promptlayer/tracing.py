from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import threading
import weakref
from typing import Any, Iterable, Optional, Tuple

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

from promptlayer.span_exporter import OpenAIPromptTemplateSpanProcessor
from promptlayer.utils import _PROMPTLAYER_USER_AGENT, SDK_VERSION

logger = logging.getLogger(__name__)

NATIVE_OTEL_PROVIDERS = ("openai",)

_PROVIDER_ALIASES = {
    "openai.azure": "openai",
}
_exporter_settings: weakref.WeakKeyDictionary[Any, Tuple[str, str]] = weakref.WeakKeyDictionary()
_prompt_processor_providers: weakref.WeakKeyDictionary[Any, OpenAIPromptTemplateSpanProcessor] = (
    weakref.WeakKeyDictionary()
)
_configuration_lock = threading.RLock()
_openai_instrumented_provider: Optional[Any] = None


def instrument_openai(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    endpoint: Optional[str] = None,
    tracer_provider: Optional[Any] = None,
) -> Any:
    """Export direct OpenAI SDK spans to PromptLayer.

    This is the provider-specific convenience API for applications that do not
    create a ``PromptLayer`` client. ``configure_tracing`` remains available for
    advanced OpenTelemetry configuration.
    """

    global _openai_instrumented_provider

    with _configuration_lock:
        resolved_api_key = _resolve_api_key(api_key)
        instrumentor = _load_openai_instrumentor(explicit=True)
        provider = tracer_provider if tracer_provider is not None else _get_or_create_tracer_provider()
        _validate_tracer_provider(provider)
        if (
            instrumentor is not None
            and instrumentor.is_instrumented_by_opentelemetry
            and _openai_instrumented_provider is not None
            and _openai_instrumented_provider is not provider
        ):
            raise RuntimeError(
                "The OpenAI SDK is already instrumented with a different tracer provider. "
                "Reuse that tracer provider when calling instrument_openai()."
            )

        configured_provider = configure_tracing(
            api_key=resolved_api_key,
            base_url=base_url,
            endpoint=endpoint,
            tracer_provider=provider,
            providers=("openai",),
        )
        if (
            instrumentor is not None
            and instrumentor.is_instrumented_by_opentelemetry
            and _openai_instrumented_provider is None
        ):
            _openai_instrumented_provider = provider
        return configured_provider


def configure_tracing(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    endpoint: Optional[str] = None,
    tracer_provider: Optional[Any] = None,
    providers: Optional[Iterable[str]] = None,
) -> Any:
    """Export spans to PromptLayer and auto-instrument the OpenAI SDK."""

    resolved_api_key = _resolve_api_key(api_key)

    provider = tracer_provider if tracer_provider is not None else _get_or_create_tracer_provider()
    _validate_tracer_provider(provider)
    selected_providers = _normalize_providers(providers)
    if "openai" in selected_providers:
        _add_openai_prompt_processor(provider)
    _add_exporter(provider, resolved_api_key, _resolve_endpoint(endpoint, base_url))
    _enable_latest_genai_semantic_conventions()
    if "openai" in selected_providers:
        _instrument_openai(provider, explicit=providers is not None)
    return provider


def _resolve_api_key(api_key: Optional[str]) -> str:
    resolved_api_key = api_key or os.environ.get("PROMPTLAYER_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "PromptLayer API key not provided. Please set PROMPTLAYER_API_KEY or pass the api_key parameter."
        )
    return resolved_api_key


def _get_or_create_tracer_provider() -> Any:
    provider = trace.get_tracer_provider()
    if callable(getattr(provider, "add_span_processor", None)):
        return provider

    provider = TracerProvider(resource=Resource.create({ResourceAttributes.SERVICE_NAME: "promptlayer-python"}))
    trace.set_tracer_provider(provider)
    return provider


def _validate_tracer_provider(provider: Any) -> None:
    if not callable(getattr(provider, "add_span_processor", None)) or not callable(
        getattr(provider, "get_tracer", None)
    ):
        raise TypeError("tracer_provider must be an OpenTelemetry SDK tracer provider")


def _resolve_endpoint(endpoint: Optional[str], base_url: Optional[str]) -> str:
    if endpoint:
        return endpoint
    if os.environ.get("PROMPTLAYER_OTLP_TRACES_ENDPOINT"):
        return os.environ["PROMPTLAYER_OTLP_TRACES_ENDPOINT"]

    root = (base_url or os.environ.get("PROMPTLAYER_BASE_URL") or "https://api.promptlayer.com").rstrip("/")
    return f"{root}/v1/traces"


def _add_openai_prompt_processor(provider: Any) -> None:
    if provider in _prompt_processor_providers:
        return

    processor = OpenAIPromptTemplateSpanProcessor()
    provider.add_span_processor(processor)
    _prompt_processor_providers[provider] = processor


def _add_exporter(provider: Any, api_key: str, endpoint: str) -> None:
    settings = (endpoint, api_key)
    previous_settings = _exporter_settings.get(provider)
    if previous_settings == settings:
        return
    if previous_settings is not None:
        raise RuntimeError("A different PromptLayer exporter is already configured on this tracer provider")

    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers={
            "X-Api-Key": api_key,
            "User-Agent": _PROMPTLAYER_USER_AGENT,
            "X-SDK-Version": SDK_VERSION,
        },
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    _exporter_settings[provider] = settings


def _instrument_openai(tracer_provider: Any, *, explicit: bool) -> None:
    global _openai_instrumented_provider

    instrumentor = _load_openai_instrumentor(explicit=explicit)
    if instrumentor is None:
        return

    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument(tracer_provider=tracer_provider)
        if instrumentor.is_instrumented_by_opentelemetry:
            _openai_instrumented_provider = tracer_provider
        elif explicit:
            raise RuntimeError(
                "OpenTelemetry OpenAI instrumentation could not be enabled. "
                "Check that the installed OpenAI SDK version is supported."
            )
    elif _openai_instrumented_provider is not None and _openai_instrumented_provider is not tracer_provider:
        logger.warning(
            "The OpenAI SDK is already instrumented with a different tracer provider; "
            "the existing provider remains active"
        )


def _load_openai_instrumentor(*, explicit: bool) -> Optional[Any]:
    sdk_installed = _module_available("openai")
    instrumentor_class = None
    if sdk_installed:
        try:
            module = importlib.import_module("opentelemetry.instrumentation.openai_v2")
            instrumentor_class = getattr(module, "OpenAIInstrumentor")
        except (AttributeError, ImportError):
            pass

    if instrumentor_class is None:
        if explicit:
            raise ImportError(
                "OpenTelemetry OpenAI instrumentation is unavailable. "
                'Install the "promptlayer[otel-genai-instrumentation]" extra and the OpenAI SDK.'
            )
        if sdk_installed:
            logger.warning(
                "OpenTelemetry instrumentation is unavailable for OpenAI; "
                "install promptlayer[otel-genai-instrumentation]"
            )
        return None
    return instrumentor_class()


def _configure_openai_sdk_instrumentation(tracer_provider: Any) -> None:
    """Add OpenAI auto-instrumentation to an already configured provider."""

    _validate_tracer_provider(tracer_provider)
    _add_openai_prompt_processor(tracer_provider)
    _enable_latest_genai_semantic_conventions()
    _instrument_openai(tracer_provider, explicit=False)


def _normalize_providers(providers: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if providers is None:
        return NATIVE_OTEL_PROVIDERS

    normalized = []
    for provider in providers:
        provider_name = _PROVIDER_ALIASES.get(provider, provider)
        if provider_name != "openai":
            raise ValueError(f"Unknown tracing provider: {provider}")
        if provider_name not in normalized:
            normalized.append(provider_name)
    return tuple(normalized)


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _enable_latest_genai_semantic_conventions() -> None:
    env_name = "OTEL_SEMCONV_STABILITY_OPT_IN"
    value = "gen_ai_latest_experimental"
    configured = [item.strip() for item in os.environ.get(env_name, "").split(",") if item.strip()]
    if value not in configured:
        os.environ[env_name] = ",".join((*configured, value))
