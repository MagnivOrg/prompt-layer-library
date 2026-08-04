# PromptLayer Python SDK

You are working on the official PromptLayer Python SDK (`prompt-layer-library`). Before changing SDK behavior, read the relevant product documentation for the feature you are touching.

Documentation index: https://docs.promptlayer.com/llms.txt

## SDK features

| Feature | SDK entry point | Documentation |
| --- | --- | --- |
| Python SDK | `PromptLayer`, `AsyncPromptLayer` | https://docs.promptlayer.com/sdks/python.md |
| Prompt templates | `client.templates` | https://docs.promptlayer.com/features/prompt-registry/overview.md |
| Run prompts | `client.run()` | https://docs.promptlayer.com/sdks/python.md |
| Run workflows | `client.run_workflow()` | https://docs.promptlayer.com/why-promptlayer/workflows.md |
| Request logging | `client.log_request()` | https://docs.promptlayer.com/features/observability/request-logs/custom-logging.md |
| Track metadata, prompts, and scores | `client.track` | https://docs.promptlayer.com/features/observability/request-logs/metadata.md |
| Request groups | `client.group` | https://docs.promptlayer.com/features/observability/request-logs/tags.md |
| Manual tracing | `client.traceable()` | https://docs.promptlayer.com/features/observability/traces/manual-tracing.md |
| Tracing overview | `configure_tracing()`, `enable_tracing=True` | https://docs.promptlayer.com/features/observability/traces/overview.md |
| Provider auto-instrumentation | `instrument_openai()`, GenAI instrumentors | https://docs.promptlayer.com/features/observability/traces/auto-instrumentation/overview.md |
| Skill collections | `client.skills` | https://docs.promptlayer.com/features/skill-collections/overview.md |
| Tables | `client.tables` | https://docs.promptlayer.com/features/tables/overview.md |
| Table scorecards | `client.tables.sheets.scorecards` | https://docs.promptlayer.com/features/tables/scorecards.md |
| Provider proxies | `client.openai`, `client.anthropic` | https://docs.promptlayer.com/sdks/python.md |
| SDK evals | `evaluate()`, `aevaluate()` | https://docs.promptlayer.com/sdks/evals/overview.md |
| Eval CLI and CI | `promptlayer eval run` | https://docs.promptlayer.com/sdks/evals/cli-and-ci.md |
| OpenAI Agents integration | `promptlayer.integrations.openai_agents` | https://docs.promptlayer.com/sdks/evals/agent-tracing.md |
| Claude Agents integration | `promptlayer.integrations.claude_agents` | https://docs.promptlayer.com/agents/overview.md |

## Working in this repo

- Match existing module layout under `promptlayer/`.
- Reuse helpers in `promptlayer/utils.py` and existing managers instead of duplicating API logic.
- Add or update tests under `tests/` for behavior changes.
- See `README.md` for installation, configuration, and client resource overview.
