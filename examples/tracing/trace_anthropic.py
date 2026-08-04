"""Configure PromptLayer auto-instrumentation for the Anthropic SDK."""

import os

from anthropic import Anthropic

from promptlayer import configure_tracing

tracer_provider = configure_tracing(providers=("anthropic",))
client = Anthropic()
model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")


try:
    response = client.messages.create(
        model=model,
        max_tokens=128,
        messages=[{"role": "user", "content": "Explain distributed tracing in one sentence."}],
    )
    print(response.content[0].text)
finally:
    client.close()
    tracer_provider.force_flush()
