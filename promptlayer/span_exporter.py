import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

from opentelemetry import baggage, context, trace
from opentelemetry.sdk.trace import SpanProcessor

logger = logging.getLogger(__name__)

_BAGGAGE_KEYS = (
    "promptlayer.prompt.name",
    "promptlayer.prompt.id",
    "promptlayer.prompt.version",
    "promptlayer.prompt.label",
)
_LEGACY_OPENAI_INSTRUMENTATION_SCOPE = "opentelemetry.instrumentation.openai_v2"
_GENAI_HANDLER_SCOPE = "opentelemetry.util.genai.handler"
_BOTOCORE_BEDROCK_SCOPE = "opentelemetry.instrumentation.botocore.bedrock-runtime"
_SUPPORTED_GENAI_PROVIDER_NAMES = frozenset(
    {
        "openai",
        "anthropic",
        "aws.bedrock",
        "gemini",
        "vertex_ai",
        "gcp.gemini",
        "gcp.vertex_ai",
    }
)
_PROMPTLAYER_REQUEST_LOG_MANAGED = "promptlayer.request_log.managed"
_PROMPTLAYER_REQUEST_LOG_SPAN_ID = "promptlayer.request_log.span_id"
_PROMPTLAYER_PROVIDER_TYPE = "promptlayer.provider.type"
_PROMPTLAYER_API_TYPE = "promptlayer.api.type"
_active_prompt_template: ContextVar[Optional[Dict[str, str]]] = ContextVar(
    "promptlayer_active_prompt_template",
    default=None,
)
_defer_prompt_span_attributes: ContextVar[bool] = ContextVar(
    "promptlayer_defer_prompt_span_attributes",
    default=False,
)


@dataclass(frozen=True)
class _GenAIRequestSpanState:
    request_log_span_id: Optional[str] = None


_GENAI_REQUEST_SPAN_STATE_KEY = context.create_key("promptlayer.genai_request_span")


@contextmanager
def _mark_genai_request_span(
    *,
    enabled: bool,
    request_log_span_id: Optional[str],
) -> Iterator[None]:
    """Link a native GenAI span to the request log managed by PromptLayer.run."""

    if not enabled:
        yield
        return

    token = None
    try:
        state = _GenAIRequestSpanState(request_log_span_id=request_log_span_id)
        token = context.attach(context.set_value(_GENAI_REQUEST_SPAN_STATE_KEY, state))
    except Exception:
        logger.debug("PromptLayer could not attach GenAI request context", exc_info=True)
    try:
        yield
    finally:
        if token is not None:
            try:
                context.detach(token)
            except Exception:
                logger.debug("PromptLayer could not detach GenAI request context", exc_info=True)


class GenAIPromptTemplateSpanProcessor(SpanProcessor):
    """Enrich supported GenAI spans and correlate PromptLayer-managed requests."""

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        try:
            self._on_start(span, parent_context)
        except Exception:
            # Span processors run inline with provider SDK calls. PromptLayer
            # enrichment must never affect the underlying request.
            logger.debug("PromptLayer could not enrich a GenAI span", exc_info=True)

    @staticmethod
    def _on_start(span: Any, parent_context: Any = None) -> None:
        instrumentation_scope = getattr(span, "instrumentation_scope", None)
        scope_name = getattr(instrumentation_scope, "name", "")
        attributes = getattr(span, "attributes", {}) or {}
        is_legacy_openai_span = scope_name == _LEGACY_OPENAI_INSTRUMENTATION_SCOPE
        is_botocore_bedrock_span = scope_name == _BOTOCORE_BEDROCK_SCOPE and (
            attributes.get("gen_ai.system") == "aws.bedrock"
        )
        is_supported_genai_span = (
            scope_name == _GENAI_HANDLER_SCOPE
            and attributes.get("gen_ai.provider.name") in _SUPPORTED_GENAI_PROVIDER_NAMES
        )
        if not is_legacy_openai_span and not is_botocore_bedrock_span and not is_supported_genai_span:
            return

        if is_legacy_openai_span:
            span.set_attribute("gen_ai.provider.name", "openai")
        else:
            if is_botocore_bedrock_span:
                span.set_attribute("gen_ai.provider.name", "aws.bedrock")
                attributes = getattr(span, "attributes", {}) or {}
            canonical_provider = _canonical_provider_type(attributes)
            if canonical_provider:
                span.set_attribute(_PROMPTLAYER_PROVIDER_TYPE, canonical_provider)
                api_type = _canonical_api_type(attributes, canonical_provider)
                if api_type:
                    span.set_attribute(_PROMPTLAYER_API_TYPE, api_type)
        span.set_attribute("node_type", "LLM_CALL")
        prompt_attributes = _active_prompt_template.get()
        if prompt_attributes:
            span.set_attributes(prompt_attributes)

        request_span_state = context.get_value(_GENAI_REQUEST_SPAN_STATE_KEY, parent_context)
        if request_span_state is None:
            return

        # PromptLayer.run creates the request log; OTLP ingestion must not create another.
        span.set_attribute(_PROMPTLAYER_REQUEST_LOG_MANAGED, True)
        if request_span_state.request_log_span_id:
            span.set_attribute(
                _PROMPTLAYER_REQUEST_LOG_SPAN_ID,
                request_span_state.request_log_span_id,
            )


def _canonical_provider_type(attributes: Dict[str, Any]) -> Optional[str]:
    provider = str(attributes.get("gen_ai.provider.name") or "").strip().lower()
    server_address = str(attributes.get("server.address") or "").strip().lower()

    if provider == "openai":
        if server_address.endswith(".openai.azure.com") or server_address.endswith(".cognitiveservices.azure.com"):
            return "openai.azure"
        return "openai"
    if provider == "anthropic":
        if re.fullmatch(
            r"(?:[a-z0-9]+(?:-[a-z0-9]+)*-)?aiplatform\.googleapis\.com",
            server_address,
        ):
            return "vertexai"
        return "anthropic"
    if provider in {"gemini", "gcp.gemini"}:
        return "google"
    if provider in {"vertex_ai", "gcp.vertex_ai"}:
        return "vertexai"
    if provider == "aws.bedrock":
        return "amazon.bedrock"
    return None


def _canonical_api_type(attributes: Dict[str, Any], canonical_provider: str) -> Optional[str]:
    operation = str(attributes.get("gen_ai.operation.name") or "").strip().lower()
    provider = str(attributes.get("gen_ai.provider.name") or "").strip().lower()

    if operation == "embeddings":
        return "embeddings"
    if canonical_provider in {"openai", "openai.azure"}:
        return "chat-completions"
    if provider == "anthropic":
        return "messages"
    if canonical_provider == "amazon.bedrock":
        rpc_method = str(attributes.get("rpc.method") or "").strip().lower()
        if rpc_method in {"converse", "conversestream"}:
            return "converse"
        if rpc_method in {"invokemodel", "invokemodelwithresponsestream"}:
            return "invoke-model"
    if operation == "generate_content":
        return "generate-content"
    if operation == "interactions.create":
        return "interactions"
    return None


# Preserve the existing import while the processor now supports all configured
# GenAI providers.
OpenAIPromptTemplateSpanProcessor = GenAIPromptTemplateSpanProcessor

# Preserve the internal context manager for callers that have not migrated to
# the provider-neutral name.
_mark_openai_request_span = _mark_genai_request_span


def _prompt_span_attributes(
    prompt_blueprint: Dict[str, Any],
    prompt_name: str,
    *,
    label: Optional[str] = None,
) -> Dict[str, str]:
    entries: Dict[str, str] = {"promptlayer.prompt.name": prompt_name}

    prompt_id = prompt_blueprint.get("id")
    if prompt_id is not None:
        entries["promptlayer.prompt.id"] = str(prompt_id)

    version = prompt_blueprint.get("version")
    if version is not None:
        entries["promptlayer.prompt.version"] = str(version)

    if label is not None:
        entries["promptlayer.prompt.label"] = label

    return entries


@contextmanager
def _defer_prompt_span_attribute_activation() -> Iterator[None]:
    """Let a logical template-fetch span finish before activating prompt baggage."""

    token = _defer_prompt_span_attributes.set(True)
    try:
        yield
    finally:
        _defer_prompt_span_attributes.reset(token)


@contextmanager
def _scope_prompt_span_attributes(
    entries: Optional[Dict[str, str]],
) -> Iterator[Optional[Dict[str, str]]]:
    """Temporarily activate or clear prompt identity for child spans."""

    active_prompt_token = None
    baggage_token = None
    try:
        try:
            active_prompt_token = _active_prompt_template.set(entries)
            ctx = context.get_current()
            for key in _BAGGAGE_KEYS:
                if entries is not None and key in entries:
                    ctx = baggage.set_baggage(key, entries[key], ctx)
                else:
                    ctx = baggage.remove_baggage(key, ctx)
            baggage_token = context.attach(ctx)
        except Exception:
            logger.debug("PromptLayer could not scope prompt tracing attributes", exc_info=True)
        yield entries
    finally:
        if baggage_token is not None:
            try:
                context.detach(baggage_token)
            except Exception:
                logger.debug("PromptLayer could not detach prompt tracing attributes", exc_info=True)
        if active_prompt_token is not None:
            _active_prompt_template.reset(active_prompt_token)


def _set_prompt_fetch_cache_hit(cache_hit: bool) -> None:
    """Annotate the active logical prompt fetch when SDK caching is in use."""

    try:
        if not _defer_prompt_span_attributes.get():
            return
        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("promptlayer.prompt.cache_hit", cache_hit)
    except Exception:
        logger.debug("PromptLayer could not annotate prompt cache status", exc_info=True)


def set_prompt_span_attributes(
    prompt_blueprint: Dict[str, Any],
    prompt_name: str,
    *,
    label: Optional[str] = None,
) -> None:
    """Set ``promptlayer.prompt.*`` as OTEL baggage so child spans inherit them.

    When used with a ``BaggageSpanProcessor``, these baggage entries are
    automatically copied as attributes onto every child span — including
    auto-instrumented GenAI spans.

    Also sets the attributes on the current span for direct visibility.
    """
    try:
        if _defer_prompt_span_attributes.get():
            return

        entries = _prompt_span_attributes(prompt_blueprint, prompt_name, label=label)

        _active_prompt_template.set(entries)

        span = trace.get_current_span()
        if span.is_recording():
            for key, value in entries.items():
                span.set_attribute(key, value)

        ctx = context.get_current()
        for key in _BAGGAGE_KEYS:
            if key in entries:
                ctx = baggage.set_baggage(key, entries[key], ctx)
            else:
                ctx = baggage.remove_baggage(key, ctx)
        context.attach(ctx)
    except Exception:
        logger.debug("PromptLayer could not attach prompt tracing attributes", exc_info=True)
