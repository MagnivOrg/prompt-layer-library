from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import threading
import weakref
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, Iterable, Optional, Tuple

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

from promptlayer.integrations.bedrock import bedrock_request_hook, bedrock_response_hook
from promptlayer.span_exporter import (
    _PROMPTLAYER_API_TYPE,
    GenAIPromptTemplateSpanProcessor,
)
from promptlayer.utils import _PROMPTLAYER_USER_AGENT, SDK_VERSION

logger = logging.getLogger(__name__)

_GENAI_CAPTURE_MESSAGE_CONTENT_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
_DEFAULT_GENAI_CAPTURE_MESSAGE_CONTENT = "SPAN_ONLY"


@dataclass(frozen=True)
class _ProviderInstrumentation:
    sdk_module: str
    instrumentation_module: str
    instrumentor_class: str
    display_name: str


_PROVIDER_INSTRUMENTATIONS = {
    "openai": _ProviderInstrumentation(
        sdk_module="openai",
        instrumentation_module="opentelemetry.instrumentation.genai.openai",
        instrumentor_class="OpenAIInstrumentor",
        display_name="OpenAI",
    ),
    "anthropic": _ProviderInstrumentation(
        sdk_module="anthropic",
        instrumentation_module="opentelemetry.instrumentation.genai.anthropic",
        instrumentor_class="AnthropicInstrumentor",
        display_name="Anthropic",
    ),
    "google": _ProviderInstrumentation(
        sdk_module="google.genai",
        instrumentation_module="opentelemetry.instrumentation.google_genai",
        instrumentor_class="GoogleGenAiSdkInstrumentor",
        display_name="Google GenAI",
    ),
    "bedrock": _ProviderInstrumentation(
        sdk_module="boto3",
        instrumentation_module="opentelemetry.instrumentation.botocore",
        instrumentor_class="BotocoreInstrumentor",
        display_name="AWS Bedrock",
    ),
}

NATIVE_OTEL_PROVIDERS = tuple(_PROVIDER_INSTRUMENTATIONS)

_PROVIDER_ALIASES = {
    "openai.azure": "openai",
    "amazon.bedrock": "bedrock",
    "aws.bedrock": "bedrock",
}
_exporter_settings: weakref.WeakKeyDictionary[Any, Tuple[str, str]] = weakref.WeakKeyDictionary()
_prompt_processor_providers: weakref.WeakKeyDictionary[Any, GenAIPromptTemplateSpanProcessor] = (
    weakref.WeakKeyDictionary()
)
_configuration_lock = threading.RLock()
_instrumented_provider_owners: Dict[str, Any] = {}


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

    with _configuration_lock:
        resolved_api_key = _resolve_api_key(api_key)
        instrumentor = _load_provider_instrumentor("openai", explicit=True)
        provider = tracer_provider if tracer_provider is not None else _get_or_create_tracer_provider()
        _validate_tracer_provider(provider)
        if (
            instrumentor is not None
            and instrumentor.is_instrumented_by_opentelemetry
            and _instrumented_provider_owners.get("openai") is not None
            and _instrumented_provider_owners["openai"] is not provider
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
            and _instrumented_provider_owners.get("openai") is None
        ):
            _instrumented_provider_owners["openai"] = provider
        return configured_provider


def configure_tracing(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    endpoint: Optional[str] = None,
    tracer_provider: Optional[Any] = None,
    providers: Optional[Iterable[str]] = None,
) -> Any:
    """Export spans to PromptLayer and auto-instrument selected GenAI SDKs."""

    with _configuration_lock:
        resolved_api_key = _resolve_api_key(api_key)
        selected_providers = _normalize_providers(providers)
        explicit_instrumentors = {}
        if providers is not None:
            explicit_instrumentors = {
                provider_name: _load_provider_instrumentor(provider_name, explicit=True)
                for provider_name in selected_providers
            }

        provider = tracer_provider if tracer_provider is not None else _get_or_create_tracer_provider()
        _validate_tracer_provider(provider)
        if selected_providers:
            _add_genai_prompt_processor(provider)
        _add_exporter(provider, resolved_api_key, _resolve_endpoint(endpoint, base_url))
        _enable_latest_genai_semantic_conventions()
        for provider_name in selected_providers:
            _instrument_provider(
                provider_name,
                provider,
                explicit=providers is not None,
                instrumentor=explicit_instrumentors.get(provider_name),
            )
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


def _add_genai_prompt_processor(provider: Any) -> None:
    if provider in _prompt_processor_providers:
        return

    processor = GenAIPromptTemplateSpanProcessor()
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


def _instrument_provider(
    provider_name: str,
    tracer_provider: Any,
    *,
    explicit: bool,
    instrumentor: Optional[Any] = None,
) -> None:
    instrumentor = instrumentor or _load_provider_instrumentor(provider_name, explicit=explicit)
    if instrumentor is None:
        return

    _set_default_genai_message_content_capture()

    if provider_name == "openai":
        _configure_openai_response_api_enrichment()

    config = _PROVIDER_INSTRUMENTATIONS[provider_name]
    if not instrumentor.is_instrumented_by_opentelemetry:
        try:
            instrument_kwargs = {"tracer_provider": tracer_provider}
            if provider_name == "bedrock":
                instrument_kwargs.update(
                    request_hook=bedrock_request_hook,
                    response_hook=bedrock_response_hook,
                )
            instrumentor.instrument(**instrument_kwargs)
        except Exception:
            if explicit:
                raise
            logger.warning(
                "PromptLayer could not enable %s auto-instrumentation; provider calls will continue uninstrumented",
                config.display_name,
                exc_info=True,
            )
            return
        if instrumentor.is_instrumented_by_opentelemetry:
            _instrumented_provider_owners[provider_name] = tracer_provider
        elif explicit:
            raise RuntimeError(
                f"OpenTelemetry {config.display_name} instrumentation could not be enabled. "
                f"Check that the installed {config.display_name} SDK version is supported."
            )
    elif (
        _instrumented_provider_owners.get(provider_name) is not None
        and _instrumented_provider_owners[provider_name] is not tracer_provider
    ):
        logger.warning(
            "%s SDK is already instrumented with a different tracer provider; the existing provider remains active",
            config.display_name,
        )


def _configure_openai_response_api_enrichment() -> None:
    """Tag official OpenAI Responses spans without changing their OTel operation."""

    try:
        patch_responses = importlib.import_module("opentelemetry.instrumentation.genai.openai.patch_responses")
        current = getattr(patch_responses, "apply_request_attributes", None)
        if current is None or getattr(current, "_promptlayer_enriched", False):
            return

        @wraps(current)
        def apply_request_attributes(invocation, *args, **kwargs):
            try:
                current(invocation, *args, **kwargs)
                invocation.attributes[_PROMPTLAYER_API_TYPE] = "responses"
            except Exception:
                logger.debug("PromptLayer could not classify an OpenAI Responses span", exc_info=True)

        apply_request_attributes._promptlayer_enriched = True
        patch_responses.apply_request_attributes = apply_request_attributes
    except Exception:
        logger.debug("OpenAI Responses enrichment is unavailable", exc_info=True)


def _load_provider_instrumentor(provider_name: str, *, explicit: bool) -> Optional[Any]:
    config = _PROVIDER_INSTRUMENTATIONS[provider_name]
    sdk_installed = _module_available(config.sdk_module)
    instrumentor_class = None
    if sdk_installed:
        try:
            module = importlib.import_module(config.instrumentation_module)
            instrumentor_class = getattr(module, config.instrumentor_class)
        except (AttributeError, ImportError):
            pass

    if instrumentor_class is None:
        if explicit:
            raise ImportError(
                f"OpenTelemetry {config.display_name} instrumentation is unavailable. "
                f'Install the "promptlayer[otel-genai-instrumentation]" extra and the {config.display_name} SDK.'
            )
        if sdk_installed:
            logger.warning(
                "OpenTelemetry instrumentation is unavailable for %s; install promptlayer[otel-genai-instrumentation]",
                config.display_name,
            )
        return None
    return instrumentor_class()


def _load_openai_instrumentor(*, explicit: bool) -> Optional[Any]:
    """Backward-compatible internal wrapper for the OpenAI loader."""

    return _load_provider_instrumentor("openai", explicit=explicit)


def _configure_sdk_instrumentation(
    tracer_provider: Any,
    providers: Optional[Iterable[str]] = None,
) -> None:
    """Add supported GenAI SDK auto-instrumentation to a configured provider."""

    _validate_tracer_provider(tracer_provider)
    selected_providers = _normalize_providers(providers)
    if selected_providers:
        _add_genai_prompt_processor(tracer_provider)
    _enable_latest_genai_semantic_conventions()
    for provider_name in selected_providers:
        _instrument_provider(provider_name, tracer_provider, explicit=False)


def _configure_openai_sdk_instrumentation(tracer_provider: Any) -> None:
    """Backward-compatible internal helper for OpenAI-only instrumentation."""

    _validate_tracer_provider(tracer_provider)
    _add_genai_prompt_processor(tracer_provider)
    _enable_latest_genai_semantic_conventions()
    _instrument_provider("openai", tracer_provider, explicit=False)


def _normalize_providers(providers: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if providers is None:
        return NATIVE_OTEL_PROVIDERS

    normalized = []
    for provider in providers:
        provider_name = _PROVIDER_ALIASES.get(provider, provider)
        if provider_name not in _PROVIDER_INSTRUMENTATIONS:
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


def _set_default_genai_message_content_capture() -> None:
    os.environ.setdefault(
        _GENAI_CAPTURE_MESSAGE_CONTENT_ENV,
        _DEFAULT_GENAI_CAPTURE_MESSAGE_CONTENT,
    )
