"""Configure PromptLayer auto-instrumentation for the Google GenAI SDK."""

import os

from google import genai

from promptlayer import configure_tracing

tracer_provider = configure_tracing(providers=("google",))
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
model = os.environ.get("GOOGLE_GENAI_MODEL", "gemini-2.5-flash")


try:
    response = client.models.generate_content(
        model=model,
        contents="Explain distributed tracing in one sentence.",
    )
    print(response.text)
finally:
    client.close()
    tracer_provider.force_flush()
