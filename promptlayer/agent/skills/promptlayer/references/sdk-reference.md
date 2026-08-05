# PromptLayer SDK reference for coding agents

Use this file when you need more detail than `SKILL.md`. Prefer online docs for
the latest product behavior: https://docs.promptlayer.com/llms.txt

## Authentication and configuration

- API key: pass `api_key=` or set `PROMPTLAYER_API_KEY`.
- Base URL override: `base_url=` / `PROMPTLAYER_BASE_URL`.
- Tracing endpoint override: `PROMPTLAYER_OTLP_TRACES_ENDPOINT`.
- Error behavior: `throw_on_error=True` (default) raises; `False` may return `None`.
- Template cache: `cache_ttl_seconds > 0` enables in-memory caching for
  `client.templates.get(...)`.

## Public resources

| Resource | Notes |
| --- | --- |
| `client.templates` | Get, list, publish, invalidate prompt templates. |
| `client.run()` / `client.run_workflow()` | Execute managed prompts and workflows. |
| `client.log_request()` | Manually log requests when not using proxies. |
| `client.track` | Attach metadata, prompt linkage, scores, and groups. |
| `client.group` | Create request groups / tags. |
| `client.traceable()` | Trace application functions when tracing is enabled. |
| `client.skills` | PromptLayer skill collections API (product feature). |
| `client.tables` | Tables and scorecards. |
| `client.openai` / `client.anthropic` | Provider proxies that log to PromptLayer. |

## Optional integration modules

Import these directly; they are not nested under the client:

```python
from promptlayer.integrations.openai_agents import instrument_openai_agents
from promptlayer.integrations.claude_agents import get_claude_config
```

Require the matching extras:

- `promptlayer[openai-agents]`
- `promptlayer[claude-agents]`
- `promptlayer[otel-genai-instrumentation]` for GenAI auto-instrumentation

## Evals CLI

```bash
promptlayer eval run path/to/suite_or_file.eval.py
```

Docs: https://docs.promptlayer.com/sdks/evals/cli-and-ci.md

## Agent skill and Docs MCP installation

Consumers install skills and Docs MCP config with:

```bash
promptlayer setup
promptlayer setup skills
promptlayer setup mcp
promptlayer setup --agent cursor --agent claude
promptlayer setup --force
```

`promptlayer setup` downloads the PromptLayer docs skill, hosted SDK eval skills
(`sdk-eval-builder`), and configures Docs MCP for Cursor and Claude Code by
default.

`pip install promptlayer` alone does not register the skill or MCP config with
coding agents.

Docs MCP: https://docs.promptlayer.com/mcp

## Troubleshooting

1. `ImportError` for an integration: install the optional extra.
2. Auth failures: verify `PROMPTLAYER_API_KEY` and workspace permissions.
3. Unknown method/attribute: check the Python SDK docs before inspecting source.
4. Tracing spans missing: confirm `enable_tracing=True` and provider SDKs are
   installed when using auto-instrumentation.
