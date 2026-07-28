"""Trace OpenAI Chat Completions and Responses calls and export them to PromptLayer."""

import os

from openai import OpenAI

from promptlayer import configure_tracing

tracer_provider = configure_tracing(providers=("openai",))
client = OpenAI()

try:
    chat_completion = client.chat.completions.create(
        model=os.environ["OPENAI_MODEL"],
        messages=[{"role": "user", "content": "Explain distributed tracing in one sentence."}],
    )
    print("Chat Completions:", chat_completion.choices[0].message.content)

    response = client.responses.create(
        model=os.environ["OPENAI_MODEL"],
        input="Explain distributed tracing in one sentence.",
    )
    print("Responses:", response.output_text)
finally:
    tracer_provider.force_flush()
