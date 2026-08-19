from copy import deepcopy
from unittest.mock import MagicMock, patch

from promptlayer.template_cache import is_locally_renderable, render_response
from promptlayer.types import (
    BuiltInTool,
    LegacyOpenAINativeMcpTool,
    McpTool,
    OpenRouterWebSearchToolConfig,
    PromptTool,
    ToolVariable,
)
from promptlayer.types.prompt_template import PublishPromptTemplate
from promptlayer.utils import get_prompt_template, publish_prompt_template


def _prompt_tools() -> list[PromptTool]:
    managed_mcp: McpTool = {"type": "mcp", "mcp_server_id": 42}
    native_mcp: BuiltInTool = {
        "id": "openai_mcp",
        "name": "MCP",
        "description": "OpenAI-native MCP",
        "provider": "openai",
        "type": "openai_mcp",
        "config": {
            "type": "mcp",
            "server_label": "docs",
            "server_url": "https://docs.example.com/mcp",
            "headers": {"Authorization": "Bearer token"},
            "require_approval": "never",
        },
    }
    tool_variable: ToolVariable = {"type": "variable", "name": "additional_tools"}
    openrouter_config: OpenRouterWebSearchToolConfig = {
        "id": "web",
        "engine": "exa",
        "max_results": 3,
    }
    openrouter_tool: BuiltInTool = {
        "id": "openrouter_web",
        "name": "Web Search",
        "description": "OpenRouter web search",
        "provider": "openrouter",
        "type": "web_search",
        "config": openrouter_config,
    }
    return [managed_mcp, native_mcp, tool_variable, openrouter_tool]


def test_publish_prompt_template_preserves_new_tool_shapes():
    tools = _prompt_tools()
    body: PublishPromptTemplate = {
        "prompt_name": "mcp-prompt",
        "prompt_template": {
            "type": "chat",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Search"}]}],
            "tools": tools,
        },
        "metadata": {
            "model": {
                "provider": "openai",
                "name": "gpt-5",
                "parameters": {},
            }
        },
        "parent_version_id": 7,
    }
    mock_response = MagicMock(status_code=201)
    mock_response.json.return_value = {"success": True}

    with patch("promptlayer.utils._get_requests_session") as mock_session:
        mock_session.return_value.post.return_value = mock_response
        publish_prompt_template("test-key", "https://api.promptlayer.com", True, body)

    payload = mock_session.return_value.post.call_args.kwargs["json"]
    assert payload["prompt_version"]["prompt_template"]["tools"] == tools
    assert payload["prompt_version"]["parent_version_id"] == 7


def test_get_prompt_template_preserves_managed_mcp_and_resolved_kwargs():
    response_body = {
        "prompt_template": {
            "type": "chat",
            "messages": [],
            "tools": [{"type": "mcp", "mcp_server_id": 42}],
        },
        "llm_kwargs": {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_docs",
                        "description": "Search documentation",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        },
    }
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = response_body

    with patch("promptlayer.utils._get_requests_session") as mock_session:
        mock_session.return_value.post.return_value = mock_response
        result = get_prompt_template("test-key", "https://api.promptlayer.com", True, "mcp-prompt")

    assert result == response_body


def test_mcp_prompt_cache_rendering_preserves_resolved_kwargs():
    response = {
        "prompt_template": {
            "type": "chat",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Search"}]}],
            "tools": [{"type": "mcp", "mcp_server_id": 42}],
        },
        "llm_kwargs": {"tools": [{"type": "function", "function": {"name": "search_docs"}}]},
    }
    expected_kwargs = deepcopy(response["llm_kwargs"])

    assert is_locally_renderable(response)
    assert render_response(response)["llm_kwargs"] == expected_kwargs


def test_legacy_openai_native_mcp_type_remains_available():
    legacy_tool: LegacyOpenAINativeMcpTool = {
        "id": "openai_mcp",
        "name": "MCP",
        "description": "Legacy OpenAI-native MCP",
        "provider": "openai",
        "type": "mcp",
        "config": {
            "type": "mcp",
            "server_label": "docs",
            "server_url": "https://docs.example.com/mcp",
            "execution_mode": "provider",
        },
    }

    assert legacy_tool["config"]["execution_mode"] == "provider"
