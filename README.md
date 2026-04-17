# PromptLayer Python SDK

The PromptLayer Python SDK lets you fetch and run PromptLayer-managed prompts, execute workflows, log and annotate LLM requests, and instrument supported providers and agent runtimes with PromptLayer tracing.

## Installation

Install the base SDK:

```bash
pip install promptlayer
```

Optional extras:

```bash
pip install "promptlayer[openai-agents]"
pip install "promptlayer[claude-agents]"
```

- `promptlayer[openai-agents]` installs the dependencies required to instrument `openai-agents` runs and export their traces to PromptLayer.
- `promptlayer[claude-agents]` installs the Claude Agents SDK dependency used by PromptLayer's Claude Agents integration and vendored plugin config helper.

## Quick Start

Create a client and fetch a prompt template from PromptLayer:

```python
from promptlayer import PromptLayer

pl = PromptLayer(api_key="pl_xxxxx")

prompt = pl.templates.get(
    "support-reply",
    {
        "input_variables": {
            "customer_name": "Ada",
            "question": "How do I reset my password?",
        }
    },
)

print(prompt["prompt_template"])
```

Async version:

```python
import asyncio

from promptlayer import AsyncPromptLayer


async def main():
    pl = AsyncPromptLayer(api_key="pl_xxxxx")

    prompt = await pl.templates.get(
        "support-reply",
        {
            "input_variables": {
                "customer_name": "Ada",
                "question": "How do I reset my password?",
            }
        },
    )

    print(prompt["prompt_template"])


asyncio.run(main())
```

Every method has an async version.

You can also use the client as a proxy around supported provider SDKs:

```python
from promptlayer import PromptLayer

pl = PromptLayer(api_key="pl_xxxxx")
openai = pl.openai

response = openai.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
    pl_tags=["proxy-example"],
)
```

## Environment Variables

The SDK relies on the following environment variables:

| Variable | Required | Description |
| --- | --- | --- |
| `PROMPTLAYER_API_KEY` | Yes, unless passed as `api_key=` | API key used to authenticate requests to PromptLayer. |
| `PROMPTLAYER_BASE_URL` | No | Overrides the PromptLayer API base URL. Defaults to `https://api.promptlayer.com`. |
| `PROMPTLAYER_OTLP_TRACES_ENDPOINT` | No | Overrides the OTLP trace endpoint used by the OpenAI Agents tracing integration. |
| `PROMPTLAYER_TRACEPARENT` | No | Optional trace context passed through the Claude Agents integration. |

## SDK Resources

The main resources surfaced by `PromptLayer` and related modules are:

| Resource | What it does |
| --- | --- |
| `pl.openai` | Proxy around the OpenAI Python SDK that logs requests to PromptLayer. |
| `pl.anthropic` | Proxy around the Anthropic Python SDK that logs requests to PromptLayer. |
| `pl.templates` | Fetches prompt templates, lists templates, publishes templates, and manages local cache invalidation. |
| `pl.run()` | Fetches a PromptLayer-managed prompt, executes it with the configured provider/model, and logs the resulting request. |
| `pl.run_workflow()` | Runs a PromptLayer workflow by ID or name and returns its outputs. |
| `pl.log_request()` | Manually logs a request/response pair to PromptLayer. |
| `pl.track` | Attaches metadata, prompt linkage, score, and group information to an existing PromptLayer request. |
| `pl.group` | Creates PromptLayer groups for organizing related requests. |
| `pl.skills` | Pulls, creates, publishes, and updates PromptLayer skill collections. |
| `pl.invalidate()` | Clears the SDK prompt-template cache for one prompt or all prompts. |
| `pl.traceable()` | Decorator for wrapping your own functions in PromptLayer-exported tracing spans. |
| `instrument_openai_agents()` | Instruments `openai-agents` runs and exports traces to PromptLayer. |
| `create_openai_agents_tracer_provider()` | Creates an OTLP-backed tracer provider configured for PromptLayer ingestion. |
| `get_claude_config()` | Returns the vendored PromptLayer Claude Agents plugin config and required environment settings. |

## Notes

- `PromptLayer` and `AsyncPromptLayer` accept `api_key`, `base_url`, `enable_tracing`, `throw_on_error`, and `cache_ttl_seconds`.
- When tracing is enabled, spans are exported to PromptLayer using OpenTelemetry.
- Prompt execution supports PromptLayer-managed configurations for multiple providers, including OpenAI, Anthropic, Azure OpenAI, Google, Mistral, and Bedrock-backed runtimes.
