# Provider auto-instrumentation examples

These examples show the minimum setup for tracing direct provider SDK calls
with PromptLayer.

Install the tracing extra and the provider SDKs you use:

```bash
pip install "promptlayer[otel-genai-instrumentation]" openai anthropic google-genai
```

Set `PROMPTLAYER_API_KEY` and the provider's standard API-key environment
variable, then run its example:

```bash
python examples/tracing/trace_openai.py
python examples/tracing/trace_anthropic.py
python examples/tracing/trace_google_genai.py
```

Each example configures its provider before making a normal SDK request:

```python
from promptlayer import configure_tracing

tracer_provider = configure_tracing(providers=("openai",))
```

Supported selectors are `openai`, `anthropic`, and `google`. The
`openai.azure` alias uses the OpenAI instrumentor. Anthropic clients configured
for Vertex AI use `anthropic`, and Google GenAI clients created with
`vertexai=True` use `google`.

Omit `providers` to auto-instrument every supported provider SDK that is
installed. Pass an empty iterable to configure trace export without provider
SDK auto-instrumentation.

## Tracing `PromptLayer.run()`

Set the registry prompt name, adjust the example input variables if needed,
then run:

```bash
export PROMPTLAYER_PROMPT_NAME="support-answer"
python examples/tracing/trace_promptlayer_run.py
```

The trace contains a `PromptLayer Run` parent with sibling
`Prompt template fetch` and provider LLM-call children. Provider
auto-instrumentation requires Python 3.10 or newer.
