from __future__ import annotations

import threading

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.id_generator import IdGenerator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import Status, StatusCode

from .ids import hex_id_to_int, map_span_id, map_trace_id, synthetic_root_span_id
from .mapping import base_span_attributes, base_trace_attributes, span_data_attributes, span_kind_for, span_name_for
from .time import iso_to_unix_nano


class _FixedIdGenerator(IdGenerator):
    def __init__(self, *, trace_id: int | None = None, span_id: int):
        self._trace_id = trace_id
        self._span_id = span_id

    def generate_trace_id(self) -> int:
        if self._trace_id is None:
            raise RuntimeError("Fixed trace ID was not provided")
        return self._trace_id

    def generate_span_id(self) -> int:
        return self._span_id


class PromptLayerOpenAIAgentsProcessor:
    def __init__(
        self,
        *,
        tracer_provider: TracerProvider,
        include_raw_payloads: bool = True,
    ) -> None:
        self._tracer_provider = tracer_provider
        self._include_raw_payloads = include_raw_payloads
        self._lock = threading.RLock()
        self._root_spans: dict[str, trace_api.Span] = {}
        self._agent_spans: dict[str, trace_api.Span] = {}
        self._trace_members: dict[str, set[str]] = {}

    def on_trace_start(self, trace) -> None:
        parent_context = self._resolve_upstream_context(trace)
        mapped_trace_id = None if parent_context is not None else map_trace_id(trace.trace_id)
        root_span_id = synthetic_root_span_id(trace.trace_id)
        attributes = base_trace_attributes(trace, include_raw_payloads=self._include_raw_payloads)

        root_span = self._start_span_with_fixed_ids(
            name=trace.name,
            kind=SpanKind.INTERNAL,
            context=parent_context,
            attributes=attributes,
            trace_id_hex=mapped_trace_id,
            span_id_hex=root_span_id,
        )

        with self._lock:
            self._root_spans[trace.trace_id] = root_span
            self._trace_members.setdefault(trace.trace_id, set())

    def on_trace_end(self, trace) -> None:
        with self._lock:
            root_span = self._root_spans.pop(trace.trace_id, None)
            member_ids = self._trace_members.pop(trace.trace_id, set())
            for member_id in member_ids:
                self._agent_spans.pop(member_id, None)

        if root_span is not None:
            root_span.end()

    def on_span_start(self, span) -> None:
        root_span = self._ensure_root_span(span)
        parent_span = self._resolve_parent_span(span, root_span)
        parent_context = trace_api.set_span_in_context(parent_span) if parent_span is not None else None

        start_time = iso_to_unix_nano(span.started_at)
        attributes = base_span_attributes(span)

        otel_span = self._start_span_with_fixed_ids(
            name=span_name_for(span),
            kind=span_kind_for(span),
            context=parent_context,
            start_time=start_time,
            attributes=attributes,
            span_id_hex=map_span_id(span.span_id),
        )

        with self._lock:
            self._agent_spans[span.span_id] = otel_span
            self._trace_members.setdefault(span.trace_id, set()).add(span.span_id)

    def on_span_end(self, span) -> None:
        with self._lock:
            otel_span = self._agent_spans.get(span.span_id)

        if otel_span is None:
            return

        for key, value in span_data_attributes(span.span_data, include_raw_payloads=self._include_raw_payloads).items():
            otel_span.set_attribute(key, value)

        if span.error:
            message = span.error.get("message", "OpenAI Agents span error")
            otel_span.set_status(Status(StatusCode.ERROR, message))
            error_payload = span.error
            otel_span.add_event(
                "exception",
                {
                    "exception.type": "OpenAIAgentsError",
                    "exception.message": message,
                    "openai_agents.error_json": _error_json(error_payload),
                },
                timestamp=iso_to_unix_nano(span.ended_at),
            )

        otel_span.end(end_time=iso_to_unix_nano(span.ended_at))

    def shutdown(self) -> None:
        self.force_flush()

    def force_flush(self) -> None:
        self._tracer_provider.force_flush()

    def _ensure_root_span(self, span):
        with self._lock:
            root_span = self._root_spans.get(span.trace_id)
        if root_span is not None:
            return root_span

        trace_metadata = span.trace_metadata or {}

        class _TraceShim:
            def __init__(self, span_obj):
                self.trace_id = span_obj.trace_id
                self.name = str(trace_metadata.get("workflow_name") or "OpenAI Agents Trace")
                self.group_id = trace_metadata.get("group_id")
                self.metadata = trace_metadata.get("metadata")

        shim = _TraceShim(span)
        self.on_trace_start(shim)
        with self._lock:
            return self._root_spans[span.trace_id]

    def _resolve_parent_span(self, span, root_span):
        with self._lock:
            if span.parent_id:
                parent = self._agent_spans.get(span.parent_id)
                if parent is not None:
                    return parent
        return root_span

    def _resolve_upstream_context(self, trace):
        metadata = getattr(trace, "metadata", None)
        if isinstance(metadata, dict):
            traceparent = metadata.get("traceparent")
            if isinstance(traceparent, str) and traceparent.strip():
                carrier = {"traceparent": traceparent.strip()}
                tracestate = metadata.get("tracestate")
                if isinstance(tracestate, str) and tracestate.strip():
                    carrier["tracestate"] = tracestate.strip()
                extracted = TraceContextTextMapPropagator().extract(carrier=carrier)
                if self._context_has_valid_span(extracted):
                    return extracted

        return None

    @staticmethod
    def _context_has_valid_span(context) -> bool:
        span_context = trace_api.get_current_span(context).get_span_context()
        return span_context is not None and span_context.is_valid

    def _start_span_with_fixed_ids(
        self,
        *,
        name: str,
        kind: SpanKind,
        context=None,
        start_time: int | None = None,
        attributes=None,
        trace_id_hex: str | None = None,
        span_id_hex: str,
    ):
        tracer = self._tracer_provider.get_tracer("promptlayer.integrations.openai_agents")
        tracer.id_generator = _FixedIdGenerator(
            trace_id=hex_id_to_int(trace_id_hex) if trace_id_hex is not None else None,
            span_id=hex_id_to_int(span_id_hex),
        )
        if context is None and trace_id_hex is not None:
            context = context_api.Context()
        return tracer.start_span(
            name=name,
            context=context,
            kind=kind,
            start_time=start_time,
            attributes=attributes,
        )


def _error_json(payload) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
