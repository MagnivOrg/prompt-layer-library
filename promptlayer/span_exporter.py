from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

from opentelemetry import baggage, context, trace
from opentelemetry.sdk.trace import SpanProcessor

_BAGGAGE_KEYS = (
    "promptlayer.prompt.name",
    "promptlayer.prompt.id",
    "promptlayer.prompt.version",
    "promptlayer.prompt.label",
)
_OPENAI_INSTRUMENTATION_SCOPE = "opentelemetry.instrumentation.openai_v2"
_GENAI_HANDLER_SCOPE = "opentelemetry.util.genai.handler"
_PROMPTLAYER_REQUEST_LOG_MANAGED = "promptlayer.request_log.managed"
_PROMPTLAYER_REQUEST_LOG_SPAN_ID = "promptlayer.request_log.span_id"
_active_prompt_template: ContextVar[Optional[Dict[str, str]]] = ContextVar(
    "promptlayer_active_prompt_template",
    default=None,
)


@dataclass(frozen=True)
class _OpenAIRequestSpanState:
    request_log_span_id: Optional[str] = None


_OPENAI_REQUEST_SPAN_STATE_KEY = context.create_key("promptlayer.openai_request_span")


@contextmanager
def _mark_openai_request_span(
    *,
    enabled: bool,
    request_log_span_id: Optional[str],
) -> Iterator[None]:
    """Link a native OpenAI span to the request log managed by PromptLayer.run."""

    if not enabled:
        yield
        return

    state = _OpenAIRequestSpanState(request_log_span_id=request_log_span_id)
    token = context.attach(context.set_value(_OPENAI_REQUEST_SPAN_STATE_KEY, state))
    try:
        yield
    finally:
        context.detach(token)


class OpenAIPromptTemplateSpanProcessor(SpanProcessor):
    """Enrich native OpenAI spans and correlate PromptLayer-managed requests."""

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        instrumentation_scope = getattr(span, "instrumentation_scope", None)
        scope_name = getattr(instrumentation_scope, "name", "")
        attributes = getattr(span, "attributes", {}) or {}
        is_openai_span = scope_name == _OPENAI_INSTRUMENTATION_SCOPE or (
            scope_name == _GENAI_HANDLER_SCOPE
            and attributes.get("gen_ai.provider.name") == "openai"
        )
        if not is_openai_span:
            return

        span.set_attribute("gen_ai.provider.name", "openai")
        prompt_attributes = _active_prompt_template.get()
        if prompt_attributes:
            span.set_attributes(prompt_attributes)

        request_span_state = context.get_value(_OPENAI_REQUEST_SPAN_STATE_KEY, parent_context)
        if request_span_state is None:
            return

        # PromptLayer.run creates the request log; OTLP ingestion must not create another.
        span.set_attribute(_PROMPTLAYER_REQUEST_LOG_MANAGED, True)
        if request_span_state.request_log_span_id:
            span.set_attribute(
                _PROMPTLAYER_REQUEST_LOG_SPAN_ID,
                request_span_state.request_log_span_id,
            )


def set_prompt_span_attributes(
    prompt_blueprint: Dict[str, Any],
    prompt_name: str,
    *,
    label: Optional[str] = None,
) -> None:
    """Set ``promptlayer.prompt.*`` as OTEL baggage so child spans inherit them.

    When used with a ``BaggageSpanProcessor``, these baggage entries are
    automatically copied as attributes onto every child span — including
    auto-instrumented LLM spans (e.g. from ``OpenAIInstrumentor``).

    Also sets the attributes on the current span for direct visibility.
    """
    entries: Dict[str, str] = {"promptlayer.prompt.name": prompt_name}

    prompt_id = prompt_blueprint.get("id")
    if prompt_id is not None:
        entries["promptlayer.prompt.id"] = str(prompt_id)

    version = prompt_blueprint.get("version")
    if version is not None:
        entries["promptlayer.prompt.version"] = str(version)

    if label is not None:
        entries["promptlayer.prompt.label"] = label

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
