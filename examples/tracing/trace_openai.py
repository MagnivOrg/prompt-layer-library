"""Configure PromptLayer auto-instrumentation for the OpenAI SDK."""

import os

from openai import OpenAI

from promptlayer import configure_tracing

tracer_provider = configure_tracing(providers=("openai",))
client = OpenAI()
model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")


try:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Explain distributed tracing in one sentence."}],
    )
    print(response.choices[0].message.content)
finally:
    client.close()
    tracer_provider.force_flush()
