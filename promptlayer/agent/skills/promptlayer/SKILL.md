---
name: promptlayer
description: >
  Guides coding agents through PromptLayer Python SDK integration: authentication,
  prompt templates, run/run_workflow, tracing, logging, evals, provider proxies, and
  agent integrations. Use when installing or configuring PromptLayer, calling
  PromptLayer or AsyncPromptLayer, fetching templates, running prompts or workflows,
  enabling tracing or auto-instrumentation, writing evals, or integrating openai-agents
  or claude-agent-sdk with PromptLayer.
license: Apache-2.0
metadata:
  author: promptlayer
  docs_index: https://docs.promptlayer.com/llms.txt
---

# PromptLayer Python SDK

You are integrating the installed `promptlayer` package. Prefer the public API,
package metadata, docstrings, and PromptLayer documentation over reading
implementation source under `site-packages`.

Documentation index: https://docs.promptlayer.com/llms.txt
Python SDK docs: https://docs.promptlayer.com/sdks/python.md

## Before you start

1. Confirm `promptlayer` is installed (`pip show promptlayer`).
2. Require `PROMPTLAYER_API_KEY` or an `api_key=` argument.
3. Use high-level client methods first (`templates`, `run`, `run_workflow`,
   `track`, proxies, tracing helpers).
4. Prefer the PromptLayer Docs MCP server (`https://docs.promptlayer.com/mcp`)
   or the docs index for API details. Install local agent config with
   `promptlayer setup`.
5. Do not reverse-engineer private modules unless the user explicitly asks.

## Canonical entry points

```python
from promptlayer import PromptLayer, AsyncPromptLayer

pl = PromptLayer()  # or PromptLayer(api_key="pl_xxxxx")
prompt = pl.templates.get("my-prompt", {"input_variables": {"name": "Ada"}})
```

| Task | Prefer | Docs |
| --- | --- | --- |
| Sync / async client | `PromptLayer`, `AsyncPromptLayer` | https://docs.promptlayer.com/sdks/python.md |
| Prompt templates | `client.templates` | https://docs.promptlayer.com/features/prompt-registry/overview.md |
| Run a prompt | `client.run()` | https://docs.promptlayer.com/sdks/python.md |
| Run a workflow | `client.run_workflow()` | https://docs.promptlayer.com/why-promptlayer/workflows.md |
| Manual logging | `client.log_request()` | https://docs.promptlayer.com/features/observability/request-logs/custom-logging.md |
| Metadata / scores | `client.track` | https://docs.promptlayer.com/features/observability/request-logs/metadata.md |
| Groups / tags | `client.group` | https://docs.promptlayer.com/features/observability/request-logs/tags.md |
| Manual tracing | `client.traceable()` | https://docs.promptlayer.com/features/observability/traces/manual-tracing.md |
| Tracing setup | `enable_tracing=True`, `configure_tracing()` | https://docs.promptlayer.com/features/observability/traces/overview.md |
| Auto-instrumentation | `instrument_openai()`, GenAI instrumentors | https://docs.promptlayer.com/features/observability/traces/auto-instrumentation/overview.md |
| Skill collections | `client.skills` | https://docs.promptlayer.com/features/skill-collections/overview.md |
| Tables / scorecards | `client.tables` | https://docs.promptlayer.com/features/tables/overview.md |
| Provider proxies | `client.openai`, `client.anthropic` | https://docs.promptlayer.com/sdks/python.md |
| Evals | `evaluate()`, `aevaluate()`, `promptlayer eval run` | https://docs.promptlayer.com/sdks/evals/overview.md |
| OpenAI Agents | `promptlayer.integrations.openai_agents` | https://docs.promptlayer.com/sdks/evals/agent-tracing.md |
| Claude Agents | `promptlayer.integrations.claude_agents` | https://docs.promptlayer.com/agents/overview.md |

## Common workflows

### Fetch and use a prompt template

```python
from promptlayer import PromptLayer

pl = PromptLayer()
template = pl.templates.get(
    "support-reply",
    {"input_variables": {"customer_name": "Ada", "question": "Reset password?"}},
)
```

### Proxy a provider SDK

```python
from promptlayer import PromptLayer

pl = PromptLayer()
openai = pl.openai
response = openai.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Say hello."}],
    pl_tags=["agent-example"],
)
```

### Enable tracing

```python
from promptlayer import PromptLayer

pl = PromptLayer(enable_tracing=True)
```

Install optional extras only when needed:

```bash
pip install "promptlayer[otel-genai-instrumentation]"
pip install "promptlayer[openai-agents]"
pip install "promptlayer[claude-agents]"
```

## Docs MCP

Use PromptLayer Docs MCP for documentation search and retrieval:

- URL: https://docs.promptlayer.com/mcp
- Install into agent config: `promptlayer setup` or `promptlayer setup mcp`

Prefer Docs MCP / docs pages over reading installed package source.

## Mistakes to avoid

- Do not invent private APIs by scanning `promptlayer/*.py` internals.
- Do not hardcode undocumented request payloads when a client method exists.
- Do not skip authentication; missing API keys fail immediately.
- Prefer async APIs (`AsyncPromptLayer`, `aevaluate`) in async code.
- Treat optional extras as opt-in; import integration modules only after install.

## More detail

See [references/sdk-reference.md](references/sdk-reference.md) for configuration,
resource notes, and troubleshooting.
