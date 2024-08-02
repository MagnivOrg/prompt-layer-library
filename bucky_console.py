from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.semconv.resource import ResourceAttributes

from promptlayer import PromptLayer

resource = Resource(
    attributes={ResourceAttributes.SERVICE_NAME: "prompt-layer-library"}
)
tracer_provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(ConsoleSpanExporter())
tracer_provider.add_span_processor(processor)
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
