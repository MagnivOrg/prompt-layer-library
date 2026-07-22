"""Context for nesting PromptLayer client spans under an active Eval case."""

from __future__ import annotations

import inspect
from contextvars import ContextVar
from typing import Any, AsyncGenerator, Generator, Optional

from opentelemetry import trace
from opentelemetry.trace import Tracer

# Set by Eval while a case runner executes. PromptLayer clients resolve this
# dynamically so spans nest under the Eval case without customer wiring.
active_eval_tracer: ContextVar[Optional[Tracer]] = ContextVar("active_eval_tracer", default=None)


def resolve_tracer(fallback: Optional[Tracer] = None) -> Optional[Tracer]:
    """Prefer the active Eval tracer; otherwise the caller's own tracer."""
    return active_eval_tracer.get() or fallback


def format_otel_trace_id(trace_id: int) -> str:
    return format(trace_id, "032x")


def format_otel_span_id(span_id: int) -> str:
    return format(span_id, "016x")


def current_traceparent() -> Optional[str]:
    span_context = trace.get_current_span().get_span_context()
    if span_context is None or not span_context.is_valid:
        return None

    flags = int(span_context.trace_flags) & 0xFF
    return (
        f"00-{format_otel_trace_id(span_context.trace_id)}-"
        f"{format_otel_span_id(span_context.span_id)}-{flags:02x}"
    )


def is_stream_result(result: Any) -> bool:
    return inspect.isgenerator(result) or inspect.isasyncgen(result)


def format_run_output(result: Any) -> str:
    if not isinstance(result, dict):
        return str(result)
    prompt_blueprint = result.get("prompt_blueprint")
    if prompt_blueprint is not None:
        return str(prompt_blueprint)
    raw_response = result.get("raw_response")
    if raw_response is not None:
        return str(raw_response)
    return str(result)


def wrap_stream_with_span(stream: Generator, span) -> Generator:
    final_chunk = None
    try:
        for chunk in stream:
            final_chunk = chunk
            yield chunk
        if final_chunk is not None:
            span.set_attribute("function_output", format_run_output(final_chunk))
    except Exception as exc:
        span.record_exception(exc)
        raise
    finally:
        span.end()


async def awrap_stream_with_span(stream: AsyncGenerator, span) -> AsyncGenerator:
    final_chunk = None
    try:
        async for chunk in stream:
            final_chunk = chunk
            yield chunk
        if final_chunk is not None:
            span.set_attribute("function_output", format_run_output(final_chunk))
    except Exception as exc:
        span.record_exception(exc)
        raise
    finally:
        span.end()
