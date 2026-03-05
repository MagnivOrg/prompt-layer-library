from typing import Any, Dict, Optional, Sequence

import requests
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from promptlayer.utils import _get_requests_session, raise_on_bad_response, retry_on_api_error


def set_prompt_span_attributes(
    prompt_blueprint: Dict[str, Any],
    prompt_name: str,
    *,
    label: Optional[str] = None,
) -> None:
    """Set ``promptlayer.prompt.*`` attributes on the current OTEL span.

    This allows any OTEL exporter (including PromptLayer's own OTLP endpoint)
    to automatically link traces back to the prompt template that was used.
    """
    span = trace.get_current_span()
    if not span.is_recording():
        return

    span.set_attribute("promptlayer.prompt.name", prompt_name)

    prompt_id = prompt_blueprint.get("id")
    if prompt_id is not None:
        span.set_attribute("promptlayer.prompt.id", prompt_id)

    version = prompt_blueprint.get("version")
    if version is not None:
        span.set_attribute("promptlayer.prompt.version", version)

    if label is not None:
        span.set_attribute("promptlayer.prompt.label", label)


class PromptLayerSpanExporter(SpanExporter):
    def __init__(self, api_key: str, base_url: str, throw_on_error: bool):
        self.api_key = api_key
        self.url = f"{base_url}/spans-bulk"
        self.throw_on_error = throw_on_error

    @retry_on_api_error
    def _post_spans(self, request_data):
        response = _get_requests_session().post(
            self.url,
            headers={"X-Api-Key": self.api_key, "Content-Type": "application/json"},
            json={"spans": request_data},
        )
        if response.status_code not in (200, 201):
            raise_on_bad_response(response, "PromptLayer had the following error while exporting spans")
        return response

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        request_data = []

        for span in spans:
            span_info = {
                "name": span.name,
                "context": {
                    "trace_id": hex(span.context.trace_id)[2:].zfill(32),  # Ensure 32 characters
                    "span_id": hex(span.context.span_id)[2:].zfill(16),  # Ensure 16 characters
                    "trace_state": str(span.context.trace_state),
                },
                "kind": str(span.kind),
                "parent_id": hex(span.parent.span_id)[2:] if span.parent else None,
                "start_time": span.start_time,
                "end_time": span.end_time,
                "status": {
                    "status_code": str(span.status.status_code),
                    "description": span.status.description,
                },
                "attributes": dict(span.attributes),
                "events": [
                    {
                        "name": event.name,
                        "timestamp": event.timestamp,
                        "attributes": dict(event.attributes),
                    }
                    for event in span.events
                ],
                "links": [{"context": link.context, "attributes": dict(link.attributes)} for link in span.links],
                "resource": {
                    "attributes": dict(span.resource.attributes),
                    "schema_url": span.resource.schema_url,
                },
            }
            request_data.append(span_info)

        try:
            self._post_spans(request_data)
            return SpanExportResult.SUCCESS
        except requests.RequestException:
            return SpanExportResult.FAILURE

    def shutdown(self):
        pass
