<div align="center">

# 🍰 PromptLayer

**Version, test, and monitor every prompt and agent with robust evals, tracing, and regression sets.**

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://docs.promptlayer.com"><img alt="Docs" src="https://custom-icon-badges.herokuapp.com/badge/docs-PL-green.svg?logo=cake&style=for-the-badge"></a>
<a href="https://www.loom.com/share/196c42e43acd4a369d75e9a7374a0850"><img alt="Demo with Loom" src="https://img.shields.io/badge/Demo-loom-552586.svg?logo=loom&style=for-the-badge&labelColor=gray"></a>

---

<div align="left">

This library provides convenient access to the PromptLayer API from applications written in python. 

## Installation

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

To follow along, you need a [PromptLayer](https://www.promptlayer.com/) API key. Once logged in, go to Settings to generate a key.

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

Async client:

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

## Configuration

### Client Options

`PromptLayer(...)` and `AsyncPromptLayer(...)` accept these parameters:

- `api_key: str | None = None`: Your PromptLayer API key. If omitted, the SDK looks for `PROMPTLAYER_API_KEY`.
- `enable_tracing: bool = False`: Enables OpenTelemetry tracing export to PromptLayer.
- `base_url: str | None = None`: Overrides the PromptLayer API base URL. If omitted, the SDK uses `PROMPTLAYER_BASE_URL` or the default API URL.
- `throw_on_error: bool = True`: Controls whether SDK methods raise PromptLayer exceptions or return `None` for many API errors.
- `cache_ttl_seconds: int = 0`: Enables in-memory prompt-template caching when greater than `0`.

### Environment Variables

The SDK relies on the following environment variables:

| Variable | Required | Description |
| --- | --- | --- |
| `PROMPTLAYER_API_KEY` | Yes, unless passed as `api_key=` | API key used to authenticate requests to PromptLayer. |
| `PROMPTLAYER_BASE_URL` | No | Overrides the PromptLayer API base URL. Defaults to `https://api.promptlayer.com`. |
| `PROMPTLAYER_OTLP_TRACES_ENDPOINT` | No | Overrides the OTLP trace endpoint used by the OpenAI Agents tracing integration. |
| `PROMPTLAYER_TRACEPARENT` | No | Optional trace context passed through the Claude Agents integration. |

## Client Resources

The main resources surfaced by `PromptLayer` and `AsyncPromptLayer` are:

| Resource | Description |
| --- | --- |
| `client.templates` | Prompt template retrieval, listing, publishing, and cache invalidation. |
| `client.run()` and `client.run_workflow()` | Helpers for running prompts and workflows. |
| `client.log_request()` | Manual request logging. |
| `client.track` | Request annotation utilities for metadata, prompt linkage, scores, and groups. |
| `client.group` | Group creation for organizing related requests. |
| `client.traceable()` | Decorator for tracing your own functions and sending those spans to PromptLayer when tracing is enabled. |
| `client.skills` | Skill collection pull, create, publish, and update operations. |
| `client.openai` and `client.anthropic` | Provider proxies that wrap those SDKs and log requests to PromptLayer. |

Note: When tracing is enabled, spans are exported to PromptLayer using OpenTelemetry.

## Integration Modules

The SDK also exposes integration-specific modules that are imported directly rather than accessed through the client:

| Module | Description |
| --- | --- |
| `promptlayer.integrations.openai_agents` | OpenAI Agents tracing utilities for instrumenting `openai-agents` runs and exporting their traces to PromptLayer. |
| `promptlayer.integrations.claude_agents` | Claude Agents configuration utilities for loading the PromptLayer plugin and required environment settings. |

## Error Handling

The SDK raises `PromptLayerError` as the base exception for SDK failures, with more specific subclasses for common API and validation cases.

| Error type | Description |
| --- | --- |
| `PromptLayerValidationError` | Invalid input passed to the SDK before or during a request. |
| `PromptLayerAPIConnectionError` | The SDK could not connect to PromptLayer. |
| `PromptLayerAPITimeoutError` | A PromptLayer request or workflow run timed out. |
| `PromptLayerAuthenticationError` | Authentication failed, usually because the API key is missing or invalid. |
| `PromptLayerPermissionDeniedError` | The API key does not have permission for the requested operation. |
| `PromptLayerNotFoundError` | The requested resource, such as a prompt or workflow, was not found. |
| `PromptLayerBadRequestError` | The request was malformed or used invalid parameters. |
| `PromptLayerConflictError` | The request conflicts with the current state of a resource. |
| `PromptLayerUnprocessableEntityError` | The request was well-formed but semantically invalid. |
| `PromptLayerRateLimitError` | PromptLayer rejected the request because of rate limiting. |
| `PromptLayerInternalServerError` | PromptLayer returned a 5xx server error. |
| `PromptLayerAPIStatusError` | Other non-success API responses that do not map to a more specific error type. |

By default, the clients raise these exceptions. If you initialize `PromptLayer` or `AsyncPromptLayer` with `throw_on_error=False`, many resource methods return `None` instead of raising on PromptLayer API errors.

## Caching

When enabled, the SDK caches fetched prompt templates in memory for faster repeat reads, locally re-renders them with new variables, and falls back to stale cache on temporary API failures.
- Caching is disabled by default and is enabled by setting `cache_ttl_seconds` when creating `PromptLayer` or `AsyncPromptLayer`.
- The cache applies to prompt templates fetched through `client.templates.get(...)`.
- Cached entries are stored in memory and keyed by prompt name, version, label, provider, and model.
- Requests that include `metadata_filters` or `model_parameter_overrides` bypass the cache.
- Templates that require server-side rendering behavior, such as placeholder messages or tool-variable expansion, are not cached for local rendering.
- If a cached template is stale and PromptLayer returns a transient error, the SDK can serve the stale cached version as a fallback.
- You can clear cached entries with `client.invalidate(...)` or `client.templates.invalidate(...)`.
