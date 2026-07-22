from typing import Any, Dict, Optional

from opentelemetry import baggage, context, trace

_BAGGAGE_KEYS = (
    "promptlayer.prompt.name",
    "promptlayer.prompt.id",
    "promptlayer.prompt.version",
    "promptlayer.prompt.label",
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
    span = trace.get_current_span()
    if not span.is_recording():
        return

    # Build the set of entries to propagate
    entries: Dict[str, str] = {"promptlayer.prompt.name": prompt_name}

    prompt_id = prompt_blueprint.get("id")
    if prompt_id is not None:
        entries["promptlayer.prompt.id"] = str(prompt_id)

    version = prompt_blueprint.get("version")
    if version is not None:
        entries["promptlayer.prompt.version"] = str(version)

    if label is not None:
        entries["promptlayer.prompt.label"] = label

    # Set on current span
    for key, value in entries.items():
        span.set_attribute(key, value)

    # Set as baggage so BaggageSpanProcessor copies them onto child spans.
    # Clear any stale keys from a previous prompt fetch first.
    ctx = context.get_current()
    for key in _BAGGAGE_KEYS:
        if key in entries:
            ctx = baggage.set_baggage(key, entries[key], ctx)
        else:
            ctx = baggage.remove_baggage(key, ctx)
    context.attach(ctx)
