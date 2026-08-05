DOCS_URL = "https://docs.promptlayer.com"
DOCS_MCP_URL = "https://docs.promptlayer.com/mcp"
LLMS_TXT_URL = "https://docs.promptlayer.com/llms.txt"
PYTHON_SDK_DOCS_URL = "https://docs.promptlayer.com/sdks/python"
SDK_EVALS_SKILLS_ZIP_URL = "https://share.promptlayer.com/api/sessions/sdk-evals/skills?format=zip"

SKILL_NAME = "promptlayer"
DOCS_MCP_SERVER_NAME = "promptlayer-docs"

SKILL_SOURCE_URLS = (
    f"{DOCS_URL}/.well-known/skills/{SKILL_NAME}/SKILL.md",
    f"{DOCS_URL}/skill.md",
)

PYTHON_SDK_SKILL_APPENDIX = f"""

## Python SDK guidance (from `promptlayer setup`)

When implementing the PromptLayer Python SDK in this project:

- Prefer the published docs and type declarations over reverse-engineering installed package internals.
- Python SDK guide: {PYTHON_SDK_DOCS_URL}
- Curated docs index: {LLMS_TXT_URL}
- Prefer `client.run()`, `client.templates`, and other public APIs instead of scanning `site-packages/promptlayer`.
- Use the PromptLayer Docs MCP server (`{DOCS_MCP_SERVER_NAME}`) to look up current documentation before guessing APIs.
"""
