"""Configure PromptLayer auto-instrumentation for raw AWS Bedrock calls."""

import os

import boto3

from promptlayer import configure_tracing

os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY")
tracer_provider = configure_tracing(providers=("bedrock",))
client = boto3.client(
    "bedrock-runtime",
    region_name=os.environ.get("AWS_REGION", "us-east-1"),
)
model = os.environ.get("AWS_BEDROCK_MODEL", "global.anthropic.claude-sonnet-5")


try:
    response = client.converse(
        modelId=model,
        messages=[
            {
                "role": "user",
                "content": [{"text": "Explain distributed tracing in one sentence."}],
            }
        ],
        inferenceConfig={"maxTokens": 128},
    )
    print(response["output"]["message"]["content"][0]["text"])
finally:
    tracer_provider.force_flush()
