from contextlib import contextmanager
from typing import Any, Iterator, Optional, Tuple

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Status, StatusCode

from promptlayer.evaluations.utils import serialize_cell_value
from promptlayer.evaluations.validation import validation_error
from promptlayer.tracing_context import active_eval_tracer, format_otel_span_id, format_otel_trace_id
from promptlayer.utils import logger


def flush_traces(tracer_provider: Optional[TracerProvider], *, throw_on_error: bool = False) -> None:
    if tracer_provider is None:
        return
    try:
        if tracer_provider.force_flush() is False:
            raise RuntimeError("Tracer provider did not flush before the deadline.")
    except Exception:
        logger.warning("Failed to flush eval traces before Table import.", exc_info=True)
        if throw_on_error:
            raise


def _set_eval_span_metadata(
    span: Any,
    *,
    name: str,
    input_value: Any,
    table_id: Any = None,
    sheet_id: Any = None,
) -> None:
    span.set_attribute("node_type", "EVAL")
    span.set_attribute("eval.name", name)
    span.set_attribute("eval.input", serialize_cell_value(input_value))
    if table_id is not None:
        span.set_attribute("table_id", str(table_id))
    if sheet_id is not None:
        span.set_attribute("sheet_id", str(sheet_id))


@contextmanager
def eval_case_span(
    name: str,
    input_value: Any,
    tracer_provider: Optional[TracerProvider] = None,
    *,
    table_id: Any = None,
    sheet_id: Any = None,
) -> Iterator[Tuple[Any, list]]:
    """Yield (span, ids_holder) where ids_holder becomes [trace_id, span_id]."""
    tracer = (
        tracer_provider.get_tracer("promptlayer.evals")
        if tracer_provider is not None
        else trace.get_tracer("promptlayer.evals")
    )
    ids = ["", ""]
    with tracer.start_as_current_span(f"Eval: {name}") as span:
        _set_eval_span_metadata(
            span,
            name=name,
            input_value=input_value,
            table_id=table_id,
            sheet_id=sheet_id,
        )
        span_context = span.get_span_context()
        if span_context and span_context.is_valid:
            ids[0] = format_otel_trace_id(span_context.trace_id)
            ids[1] = format_otel_span_id(span_context.span_id)
        # Publish the Eval tracer so PromptLayer clients constructed earlier
        # nest under this case span.
        token = active_eval_tracer.set(tracer)
        try:
            yield span, ids
        finally:
            active_eval_tracer.reset(token)


def run_case_in_span(
    name: str,
    runner: Any,
    input_value: Any,
    tracer_provider: Optional[TracerProvider] = None,
    *,
    table_id: Any = None,
    sheet_id: Any = None,
) -> tuple:
    """Run the case runner inside an OTel span; returns (output, trace_id, span_id)."""
    with eval_case_span(
        name,
        input_value,
        tracer_provider,
        table_id=table_id,
        sheet_id=sheet_id,
    ) as (span, ids):
        try:
            output_value = maybe_await(runner(input_value))
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        span.set_attribute("eval.output", serialize_cell_value(output_value))
    return output_value, ids[0], ids[1]


async def arun_case_in_span(
    name: str,
    runner: Any,
    input_value: Any,
    tracer_provider: Optional[TracerProvider] = None,
    *,
    table_id: Any = None,
    sheet_id: Any = None,
) -> tuple:
    with eval_case_span(
        name,
        input_value,
        tracer_provider,
        table_id=table_id,
        sheet_id=sheet_id,
    ) as (span, ids):
        try:
            output_value = await maybe_await_async(runner(input_value))
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        span.set_attribute("eval.output", serialize_cell_value(output_value))
    return output_value, ids[0], ids[1]


def _reject_stream(value: Any) -> Any:
    import inspect

    if inspect.isgenerator(value) or inspect.isasyncgen(value):
        raise validation_error("Eval runner returned a stream. Consume the stream and return its final value.")
    return value


def maybe_await(value: Any) -> Any:
    import inspect

    if inspect.isawaitable(value):
        raise validation_error("Eval runner returned an awaitable. Use aevaluate(...) for async runners.")
    return _reject_stream(value)


async def maybe_await_async(value: Any) -> Any:
    import inspect

    if inspect.isawaitable(value):
        value = await value
    return _reject_stream(value)
