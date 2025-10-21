from typing import Sequence

import requests
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class PromptLayerSpanExporter(SpanExporter):
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.url = f"{base_url}/spans-bulk"

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
            response = requests.post(
                self.url,
                headers={"X-Api-Key": self.api_key, "Content-Type": "application/json"},
                json={"spans": request_data},
            )
            response.raise_for_status()
            return SpanExportResult.SUCCESS
        except requests.RequestException:
            return SpanExportResult.FAILURE

    def shutdown(self):
        pass
