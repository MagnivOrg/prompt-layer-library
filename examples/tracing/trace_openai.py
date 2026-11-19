"""Configure PromptLayer auto-instrumentation for the OpenAI SDK."""

import os

from openai import OpenAI

from promptlayer import configure_tracing

tracer_provider = configure_tracing(providers=("openai",))
client = OpenAI()
default_model = "gpt-6-luna"
model = os.environ.get("OPENAI_MODEL", default_model)
model_options = {"reasoning_effort": "none"} if model == default_model else {}


try:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Explain distributed tracing in one sentence."}],
        **model_options,
    )
    print(response.choices[0].message.content)
finally:
    client.close()
    tracer_provider.force_flush()
