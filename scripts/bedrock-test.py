import boto3

from promptlayer import configure_tracing

tracer_provider = configure_tracing(providers=("bedrock",))
client = boto3.client("bedrock-runtime", region_name="us-east-1")

models = [
    # "anthropic.claude-sonnet-4-6",
    "global.anthropic.claude-sonnet-5",
    # "global.openai.gpt-oss-20b-1:0",
    # "global.openai.gpt-oss-120b-1:0",
]

try:
    for model_id in models:
        try:
            response = client.converse(
                modelId=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": f"Reply exactly: otel-ok-{model_id}"}],
                    }
                ],
                inferenceConfig={"maxTokens": 32},
            )
            print(model_id, response["usage"], response["output"]["message"])
        except Exception as exc:
            print(model_id, type(exc).__name__, exc)
finally:
    print("promptlayer traces flushed:", tracer_provider.force_flush(timeout_millis=10_000))
