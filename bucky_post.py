import json
from typing import Sequence

import requests
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.semconv.resource import ResourceAttributes

from promptlayer import PromptLayer


class PromptLayerSpanExporter(SpanExporter):
    def __init__(self, url="http://localhost:8000/spans-bulk", api_key=None):
        self.url = url
        self.api_key = (
            api_key
            or ""
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        request_data = []

        for span in spans:
            span_info = {
                "name": span.name,
                "context": {
                    "trace_id": hex(span.context.trace_id)[2:].zfill(
                        32
                    ),  # Ensure 32 characters
                    "span_id": hex(span.context.span_id)[2:].zfill(
                        16
                    ),  # Ensure 16 characters
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
                "links": [
                    {"context": link.context, "attributes": dict(link.attributes)}
                    for link in span.links
                ],
                "resource": {
                    "attributes": dict(span.resource.attributes),
                    "schema_url": span.resource.schema_url,
                },
            }
            request_data.append(span_info)

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            response = requests.post(
                self.url,
                headers=headers,
                json={
                    "spans": request_data,
                    "workspace_id": 1,
                },
            )

            print(f"Response Status Code: {response.status_code}")
            print("Response Headers:")
            for header, value in response.headers.items():
                print(f"  {header}: {value}")

            print("Response Content:")
            try:
                json_response = response.json()
                print(json.dumps(json_response, indent=2))
            except json.JSONDecodeError:
                print(response.text)

            response.raise_for_status()
            return SpanExportResult.SUCCESS
        except requests.RequestException:
            return SpanExportResult.FAILURE

    def shutdown(self):
        pass


# Set up the resource
resource = Resource(
    attributes={ResourceAttributes.SERVICE_NAME: "prompt-layer-library"}
)

# Set up the tracer provider
tracer_provider = TracerProvider(resource=resource)

# Create and add the custom PromptLayerSpanExporter
promptlayer_exporter = PromptLayerSpanExporter()
span_processor = BatchSpanProcessor(promptlayer_exporter)
tracer_provider.add_span_processor(span_processor)

# Set the global trace provider
trace.set_tracer_provider(tracer_provider)

promptlayer_client = PromptLayer(api_key="pl_7b93c14d8a188035bcaa52e61d9b8c58")
OpenAI = promptlayer_client.openai.OpenAI
client = OpenAI()


def get_response(input_variables):
    response = promptlayer_client.run(
        prompt_name="ai-poet",
        prompt_release_label="prod",
        input_variables=input_variables,
        metadata={
            "user_id": "123",
        },
    )

    return response


if __name__ == "__main__":
    _input_variables = {"topic": "watermelon"}
    get_response(_input_variables)
