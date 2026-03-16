"""Tests for built-in tools streaming support across providers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from promptlayer.streaming.blueprint_builder import (
    build_prompt_blueprint_from_anthropic_event,
    build_prompt_blueprint_from_google_event,
    build_prompt_blueprint_from_openai_images_event,
    build_prompt_blueprint_from_openai_responses_event,
)
from promptlayer.streaming.response_handlers import (
    _process_openai_response_event,
    openai_images_stream,
)
from promptlayer.types.prompt_template import (
    ApplyPatchCallContent,
    ApplyPatchCallOutputContent,
    BashCodeExecutionToolResultContent,
    BuiltInTool,
    CodeExecutionResultContent,
    ContainerFileAnnotation,
    CodeContent,
    MapAnnotation,
    McpApprovalRequestContent,
    McpApprovalResponseContent,
    McpCallContent,
    McpListToolsContent,
    OutputMediaContent,
    ServerToolUseContent,
    ShellCallContent,
    ShellCallOutputContent,
    TextEditorCodeExecutionToolResultContent,
    WebSearchResult,
    WebSearchToolResultContent,
)

METADATA = {"model": {"provider": "test", "name": "test-model", "parameters": {}}}


# ── Phase 1: Type smoke tests ───────────────────────────────────────


class TestTypeInstantiation:
    """Verify new TypedDicts can be instantiated."""

    def test_map_annotation(self):
        a: MapAnnotation = {"type": "map_citation", "title": "Place", "url": "https://maps.google.com", "start_index": 0, "end_index": 5}
        assert a["type"] == "map_citation"

    def test_container_file_annotation(self):
        a: ContainerFileAnnotation = {"type": "container_file_citation", "container_id": "ctr_1"}
        assert a["type"] == "container_file_citation"

    def test_server_tool_use_content(self):
        c: ServerToolUseContent = {"type": "server_tool_use", "id": "stu_1", "name": "web_search", "input": {"query": "test"}}
        assert c["name"] == "web_search"

    def test_web_search_result(self):
        r: WebSearchResult = {"type": "web_search_result", "url": "https://example.com", "title": "Example"}
        assert r["url"] == "https://example.com"

    def test_web_search_tool_result_content(self):
        c: WebSearchToolResultContent = {"type": "web_search_tool_result", "tool_use_id": "stu_1", "content": []}
        assert c["type"] == "web_search_tool_result"

    def test_bash_code_execution_tool_result(self):
        c: BashCodeExecutionToolResultContent = {"type": "bash_code_execution_tool_result", "tool_use_id": "stu_2", "content": {}}
        assert c["type"] == "bash_code_execution_tool_result"

    def test_text_editor_code_execution_tool_result(self):
        c: TextEditorCodeExecutionToolResultContent = {"type": "text_editor_code_execution_tool_result", "tool_use_id": "stu_3", "content": {}}
        assert c["type"] == "text_editor_code_execution_tool_result"

    def test_code_execution_result_content(self):
        c: CodeExecutionResultContent = {"type": "code_execution_result", "output": "42", "outcome": "OUTCOME_OK"}
        assert c["output"] == "42"

    def test_code_content_with_language(self):
        c: CodeContent = {"type": "code", "code": "print(1)", "language": "PYTHON"}
        assert c["code"] == "print(1)"
        assert c["language"] == "PYTHON"

    def test_shell_call_content(self):
        c: ShellCallContent = {"type": "shell_call", "id": "sc_1", "call_id": "call_1", "action": {"commands": ["ls"]}, "status": "completed"}
        assert c["action"]["commands"] == ["ls"]

    def test_shell_call_output_content(self):
        c: ShellCallOutputContent = {"type": "shell_call_output", "id": "sco_1", "output": [{"type": "stdout", "text": "hello"}]}
        assert c["output"][0]["text"] == "hello"

    def test_apply_patch_call_content(self):
        c: ApplyPatchCallContent = {"type": "apply_patch_call", "id": "ap_1", "operation": {"type": "create", "path": "/tmp/f"}}
        assert c["operation"]["type"] == "create"

    def test_apply_patch_call_output_content(self):
        c: ApplyPatchCallOutputContent = {"type": "apply_patch_call_output", "id": "apo_1", "output": "ok"}
        assert c["output"] == "ok"

    def test_mcp_list_tools_content(self):
        c: McpListToolsContent = {"type": "mcp_list_tools", "server_label": "srv", "tools": [{"name": "tool1"}]}
        assert c["tools"][0]["name"] == "tool1"

    def test_mcp_call_content(self):
        c: McpCallContent = {"type": "mcp_call", "name": "tool1", "server_label": "srv", "arguments": "{}"}
        assert c["name"] == "tool1"

    def test_mcp_approval_request_content(self):
        c: McpApprovalRequestContent = {"type": "mcp_approval_request", "name": "tool1", "arguments": "{}", "server_label": "srv"}
        assert c["type"] == "mcp_approval_request"

    def test_mcp_approval_response_content(self):
        c: McpApprovalResponseContent = {"type": "mcp_approval_response", "approval_request_id": "req_1", "approve": True}
        assert c["approve"] is True

    def test_output_media_content(self):
        c: OutputMediaContent = {"type": "output_media", "url": "https://img.png", "mime_type": "image/png", "media_type": "image"}
        assert c["url"] == "https://img.png"

    def test_builtin_tool(self):
        t: BuiltInTool = {"type": "web_search", "name": "web_search"}
        assert t["type"] == "web_search"


# ── Phase 2 & 3: Anthropic blueprint builder + content block builder ─


def _make_anthropic_event(event_type, **kwargs):
    """Create a mock Anthropic streaming event."""
    event = SimpleNamespace(type=event_type)
    for k, v in kwargs.items():
        setattr(event, k, v)
    return event


class TestAnthropicBlueprintBuilder:
    """Test Anthropic blueprint builder with new block types."""

    def test_server_tool_use_start(self):
        content_block = SimpleNamespace(type="server_tool_use", id="stu_1", name="web_search", input={})
        event = _make_anthropic_event("content_block_start", content_block=content_block)
        blueprint = build_prompt_blueprint_from_anthropic_event(event, METADATA)
        msg = blueprint["prompt_template"]["messages"][0]
        assert any(c["type"] == "server_tool_use" for c in msg["content"])
        stu = [c for c in msg["content"] if c["type"] == "server_tool_use"][0]
        assert stu["name"] == "web_search"
        assert stu["id"] == "stu_1"

    def test_web_search_tool_result_start(self):
        content_block = SimpleNamespace(
            type="web_search_tool_result",
            tool_use_id="stu_1",
            content=[
                SimpleNamespace(
                    type="web_search_result",
                    url="https://example.com",
                    title="Example",
                    encrypted_content="enc",
                    page_age="2d",
                    model_dump=lambda: {
                        "type": "web_search_result",
                        "url": "https://example.com",
                        "title": "Example",
                        "encrypted_content": "enc",
                        "page_age": "2d",
                    },
                )
            ],
        )
        event = _make_anthropic_event("content_block_start", content_block=content_block)
        blueprint = build_prompt_blueprint_from_anthropic_event(event, METADATA)
        msg = blueprint["prompt_template"]["messages"][0]
        assert any(c["type"] == "web_search_tool_result" for c in msg["content"])

    def test_bash_code_execution_result_start(self):
        content_block = SimpleNamespace(
            type="bash_code_execution_tool_result",
            tool_use_id="stu_2",
            content={"stdout": "hello", "stderr": ""},
        )
        event = _make_anthropic_event("content_block_start", content_block=content_block)
        blueprint = build_prompt_blueprint_from_anthropic_event(event, METADATA)
        msg = blueprint["prompt_template"]["messages"][0]
        assert any(c["type"] == "bash_code_execution_tool_result" for c in msg["content"])

    def test_text_editor_result_start(self):
        content_block = SimpleNamespace(
            type="text_editor_code_execution_tool_result",
            tool_use_id="stu_3",
            content={"output": "file contents"},
        )
        event = _make_anthropic_event("content_block_start", content_block=content_block)
        blueprint = build_prompt_blueprint_from_anthropic_event(event, METADATA)
        msg = blueprint["prompt_template"]["messages"][0]
        assert any(c["type"] == "text_editor_code_execution_tool_result" for c in msg["content"])

    def test_text_and_tool_use_still_work(self):
        # text
        event = _make_anthropic_event(
            "content_block_start", content_block=SimpleNamespace(type="text", text="")
        )
        bp = build_prompt_blueprint_from_anthropic_event(event, METADATA)
        assert bp["prompt_template"]["messages"][0]["content"][0]["type"] == "text"

        # tool_use
        event = _make_anthropic_event(
            "content_block_start",
            content_block=SimpleNamespace(type="tool_use", id="tu_1", name="calculator", input={}),
        )
        bp = build_prompt_blueprint_from_anthropic_event(event, METADATA)
        assert bp["prompt_template"]["messages"][0]["tool_calls"] is not None


class TestAnthropicContentBlockBuilder:
    """Test build_anthropic_content_blocks with new event types."""

    def test_server_tool_use_accumulation(self):
        from promptlayer.utils import build_anthropic_content_blocks

        events = [
            _make_anthropic_event(
                "content_block_start",
                content_block=SimpleNamespace(type="server_tool_use", id="stu_1", name="web_search", input={}),
            ),
            _make_anthropic_event(
                "content_block_delta",
                delta=SimpleNamespace(type="input_json_delta", partial_json='{"query":'),
            ),
            _make_anthropic_event(
                "content_block_delta",
                delta=SimpleNamespace(type="input_json_delta", partial_json='"test"}'),
            ),
            _make_anthropic_event("content_block_stop"),
        ]
        blocks, usage, stop_reason = build_anthropic_content_blocks(events)
        assert len(blocks) == 1
        assert blocks[0].type == "server_tool_use"
        assert blocks[0].input == {"query": "test"}

    def test_web_search_tool_result_passthrough(self):
        from promptlayer.utils import build_anthropic_content_blocks

        content_block = SimpleNamespace(
            type="web_search_tool_result",
            tool_use_id="stu_1",
            content=[],
        )
        events = [
            _make_anthropic_event("content_block_start", content_block=content_block),
            _make_anthropic_event("content_block_stop"),
        ]
        blocks, _, _ = build_anthropic_content_blocks(events)
        assert len(blocks) == 1
        assert blocks[0].type == "web_search_tool_result"


# ── Phase 4: OpenAI Responses blueprint builder ─────────────────────


def _make_openai_response_event(event_type, **kwargs):
    """Create a mock OpenAI response event dict."""
    event = {"type": event_type}
    event.update(kwargs)
    mock = MagicMock()
    mock.model_dump.return_value = event
    return mock


class TestOpenAIResponsesBlueprintBuilder:
    """Test OpenAI Responses blueprint builder with new event types."""

    def test_web_search_call_added(self):
        event = _make_openai_response_event(
            "response.output_item.added",
            item={"type": "web_search_call", "id": "ws_1", "status": "in_progress"},
        )
        bp = build_prompt_blueprint_from_openai_responses_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        # web_search_call maps to text with empty annotation list
        assert any(c["type"] == "text" for c in msg["content"])

    def test_shell_call_added(self):
        event = _make_openai_response_event(
            "response.output_item.added",
            item={
                "type": "shell_call",
                "id": "sh_1",
                "call_id": "call_1",
                "action": {"commands": ["ls -la"]},
                "status": "in_progress",
            },
        )
        bp = build_prompt_blueprint_from_openai_responses_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        shell_items = [c for c in msg["content"] if c["type"] == "shell_call"]
        assert len(shell_items) == 1
        assert shell_items[0]["action"] == {"commands": ["ls -la"]}

    def test_shell_call_output_added(self):
        event = _make_openai_response_event(
            "response.output_item.added",
            item={
                "type": "shell_call_output",
                "id": "sho_1",
                "call_id": "call_1",
                "output": [{"type": "stdout", "text": "file.txt"}],
            },
        )
        bp = build_prompt_blueprint_from_openai_responses_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "shell_call_output"]
        assert len(items) == 1

    def test_apply_patch_call_added(self):
        event = _make_openai_response_event(
            "response.output_item.added",
            item={
                "type": "apply_patch_call",
                "id": "ap_1",
                "call_id": "call_2",
                "operation": {"type": "create", "path": "/tmp/test.py"},
                "status": "in_progress",
            },
        )
        bp = build_prompt_blueprint_from_openai_responses_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "apply_patch_call"]
        assert len(items) == 1
        assert items[0]["operation"]["path"] == "/tmp/test.py"

    def test_mcp_call_added(self):
        event = _make_openai_response_event(
            "response.output_item.added",
            item={
                "type": "mcp_call",
                "id": "mcp_1",
                "call_id": "call_3",
                "name": "get_weather",
                "server_label": "weather_srv",
                "arguments": '{"city": "SF"}',
            },
        )
        bp = build_prompt_blueprint_from_openai_responses_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "mcp_call"]
        assert len(items) == 1
        assert items[0]["name"] == "get_weather"

    def test_mcp_list_tools_added(self):
        event = _make_openai_response_event(
            "response.output_item.added",
            item={
                "type": "mcp_list_tools",
                "id": "mlt_1",
                "server_label": "srv",
                "tools": [{"name": "tool1"}],
            },
        )
        bp = build_prompt_blueprint_from_openai_responses_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "mcp_list_tools"]
        assert len(items) == 1

    def test_image_generation_call_done(self):
        event = _make_openai_response_event(
            "response.output_item.done",
            item={
                "type": "image_generation_call",
                "id": "ig_1",
                "result": "https://img.example.com/gen.png",
                "status": "completed",
            },
        )
        bp = build_prompt_blueprint_from_openai_responses_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "output_media"]
        assert len(items) == 1
        assert items[0]["url"] == "https://img.example.com/gen.png"

    def test_shell_call_done(self):
        event = _make_openai_response_event(
            "response.output_item.done",
            item={
                "type": "shell_call",
                "id": "sh_1",
                "call_id": "call_1",
                "action": {"commands": ["pwd"]},
                "status": "completed",
            },
        )
        bp = build_prompt_blueprint_from_openai_responses_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "shell_call"]
        assert len(items) == 1
        assert items[0]["status"] == "completed"

    def test_mcp_approval_request_added(self):
        event = _make_openai_response_event(
            "response.output_item.added",
            item={
                "type": "mcp_approval_request",
                "id": "mar_1",
                "name": "dangerous_tool",
                "arguments": '{"force": true}',
                "server_label": "srv",
            },
        )
        bp = build_prompt_blueprint_from_openai_responses_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "mcp_approval_request"]
        assert len(items) == 1

    def test_mcp_approval_response_added(self):
        event = _make_openai_response_event(
            "response.output_item.added",
            item={
                "type": "mcp_approval_response",
                "id": "mars_1",
                "approval_request_id": "mar_1",
                "approve": True,
            },
        )
        bp = build_prompt_blueprint_from_openai_responses_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "mcp_approval_response"]
        assert len(items) == 1


class TestOpenAIResponseHandler:
    """Test _process_openai_response_event with new event types."""

    def test_web_search_call_added(self):
        response_data = {"output": []}
        current_items = {}
        chunk = {
            "type": "response.output_item.added",
            "item": {"type": "web_search_call", "id": "ws_1", "status": "searching"},
        }
        _process_openai_response_event(chunk, response_data, current_items)
        assert "ws_1" in current_items
        assert current_items["ws_1"]["type"] == "web_search_call"

    def test_shell_call_added(self):
        response_data = {"output": []}
        current_items = {}
        chunk = {
            "type": "response.output_item.added",
            "item": {
                "type": "shell_call",
                "id": "sh_1",
                "call_id": "call_1",
                "action": {"commands": ["echo hi"]},
                "status": "in_progress",
            },
        }
        _process_openai_response_event(chunk, response_data, current_items)
        assert "sh_1" in current_items
        assert current_items["sh_1"]["action"] == {"commands": ["echo hi"]}

    def test_apply_patch_call_added(self):
        response_data = {"output": []}
        current_items = {}
        chunk = {
            "type": "response.output_item.added",
            "item": {
                "type": "apply_patch_call",
                "id": "ap_1",
                "call_id": "call_2",
                "operation": {"type": "create", "path": "/f.py"},
            },
        }
        _process_openai_response_event(chunk, response_data, current_items)
        assert "ap_1" in current_items
        assert current_items["ap_1"]["operation"]["path"] == "/f.py"

    def test_mcp_call_added(self):
        response_data = {"output": []}
        current_items = {}
        chunk = {
            "type": "response.output_item.added",
            "item": {
                "type": "mcp_call",
                "id": "mcp_1",
                "call_id": "call_3",
                "name": "get_weather",
                "server_label": "srv",
                "arguments": "{}",
            },
        }
        _process_openai_response_event(chunk, response_data, current_items)
        assert "mcp_1" in current_items
        assert current_items["mcp_1"]["name"] == "get_weather"

    def test_image_generation_call_added(self):
        response_data = {"output": []}
        current_items = {}
        chunk = {
            "type": "response.output_item.added",
            "item": {"type": "image_generation_call", "id": "ig_1", "result": "", "status": "in_progress"},
        }
        _process_openai_response_event(chunk, response_data, current_items)
        assert "ig_1" in current_items
        assert current_items["ig_1"]["type"] == "image_generation_call"

    def test_shell_call_done_updates(self):
        response_data = {"output": []}
        current_items = {
            "sh_1": {
                "type": "shell_call",
                "id": "sh_1",
                "call_id": "call_1",
                "action": {"commands": ["ls"]},
                "status": "in_progress",
            }
        }
        chunk = {
            "type": "response.output_item.done",
            "item": {
                "type": "shell_call",
                "id": "sh_1",
                "call_id": "call_1",
                "action": {"commands": ["ls"]},
                "status": "completed",
            },
        }
        _process_openai_response_event(chunk, response_data, current_items)
        assert current_items["sh_1"]["status"] == "completed"
        assert len(response_data["output"]) == 1

    def test_mcp_call_done_with_output(self):
        response_data = {"output": []}
        current_items = {
            "mcp_1": {
                "type": "mcp_call",
                "id": "mcp_1",
                "call_id": "call_3",
                "name": "get_weather",
                "server_label": "srv",
                "arguments": "{}",
                "output": None,
                "status": "in_progress",
            }
        }
        chunk = {
            "type": "response.output_item.done",
            "item": {
                "type": "mcp_call",
                "id": "mcp_1",
                "call_id": "call_3",
                "name": "get_weather",
                "server_label": "srv",
                "arguments": "{}",
                "output": '{"temp": 72}',
                "status": "completed",
            },
        }
        _process_openai_response_event(chunk, response_data, current_items)
        assert current_items["mcp_1"]["output"] == '{"temp": 72}'


# ── Phase 5: Google blueprint builder ───────────────────────────────


def _make_google_event(parts, grounding_metadata=None, url_context_metadata=None):
    """Create a mock Google streaming event with given parts."""
    content = SimpleNamespace(parts=parts)
    candidate = SimpleNamespace(content=content, grounding_metadata=grounding_metadata, url_context_metadata=url_context_metadata)
    event = SimpleNamespace(candidates=[candidate])
    return event


class TestGoogleBlueprintBuilder:
    """Test Google blueprint builder with new part types."""

    def test_executable_code_part(self):
        exec_code = SimpleNamespace(code="print('hello')", language="PYTHON")
        part = SimpleNamespace(
            thought=False,
            executable_code=exec_code,
            code_execution_result=None,
            inline_data=None,
            text=None,
            function_call=None,
        )
        event = _make_google_event([part])
        bp = build_prompt_blueprint_from_google_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "code"]
        assert len(items) == 1
        assert items[0]["code"] == "print('hello')"
        assert items[0]["language"] == "PYTHON"

    def test_code_execution_result_part(self):
        exec_result = SimpleNamespace(output="hello", outcome="OUTCOME_OK")
        part = SimpleNamespace(
            thought=False,
            executable_code=None,
            code_execution_result=exec_result,
            inline_data=None,
            text=None,
            function_call=None,
        )
        event = _make_google_event([part])
        bp = build_prompt_blueprint_from_google_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "code_execution_result"]
        assert len(items) == 1
        assert items[0]["output"] == "hello"
        assert items[0]["outcome"] == "OUTCOME_OK"

    def test_inline_data_part(self):
        inline = SimpleNamespace(data="base64data==", mime_type="image/png")
        part = SimpleNamespace(
            thought=False,
            executable_code=None,
            code_execution_result=None,
            inline_data=inline,
            text=None,
            function_call=None,
        )
        event = _make_google_event([part])
        bp = build_prompt_blueprint_from_google_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "output_media"]
        assert len(items) == 1
        assert items[0]["url"] == "base64data=="
        assert items[0]["mime_type"] == "image/png"

    def test_mixed_parts(self):
        """Test event with text, executable_code, and result parts together."""
        text_part = SimpleNamespace(
            thought=False,
            executable_code=None,
            code_execution_result=None,
            inline_data=None,
            text="Let me calculate that.",
            function_call=None,
        )
        exec_code = SimpleNamespace(code="2+2", language="PYTHON")
        code_part = SimpleNamespace(
            thought=False,
            executable_code=exec_code,
            code_execution_result=None,
            inline_data=None,
            text=None,
            function_call=None,
        )
        exec_result = SimpleNamespace(output="4", outcome="OUTCOME_OK")
        result_part = SimpleNamespace(
            thought=False,
            executable_code=None,
            code_execution_result=exec_result,
            inline_data=None,
            text=None,
            function_call=None,
        )
        event = _make_google_event([text_part, code_part, result_part])
        bp = build_prompt_blueprint_from_google_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        types = [c["type"] for c in msg["content"]]
        assert "text" in types
        assert "code" in types
        assert "code_execution_result" in types

    def test_thought_part_still_works(self):
        part = SimpleNamespace(
            thought=True,
            executable_code=None,
            code_execution_result=None,
            inline_data=None,
            text="thinking...",
            function_call=None,
        )
        event = _make_google_event([part])
        bp = build_prompt_blueprint_from_google_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        assert msg["content"][0]["type"] == "thinking"
        assert msg["content"][0]["thinking"] == "thinking..."

    def test_web_search_grounding_annotations(self):
        """Grounding metadata with web chunks produces url_citation annotations on text."""
        text_part = SimpleNamespace(
            thought=False, executable_code=None, code_execution_result=None,
            inline_data=None, text="Paris is the capital.", function_call=None,
        )
        grounding = {
            "grounding_chunks": [
                {"web": {"uri": "https://example.com/paris", "title": "Paris Info"}},
            ],
            "grounding_supports": [
                {
                    "segment": {"start_index": 0, "end_index": 21, "text": "Paris is the capital."},
                    "grounding_chunk_indices": [0],
                }
            ],
        }
        event = _make_google_event([text_part], grounding_metadata=grounding)
        bp = build_prompt_blueprint_from_google_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        text_items = [c for c in msg["content"] if c["type"] == "text"]
        assert len(text_items) == 1
        assert "annotations" in text_items[0]
        ann = text_items[0]["annotations"]
        assert len(ann) == 1
        assert ann[0]["type"] == "url_citation"
        assert ann[0]["url"] == "https://example.com/paris"
        assert ann[0]["start_index"] == 0
        assert ann[0]["end_index"] == 21
        assert ann[0]["cited_text"] == "Paris is the capital."

    def test_maps_grounding_annotations(self):
        """Grounding metadata with maps chunks produces map_citation annotations."""
        text_part = SimpleNamespace(
            thought=False, executable_code=None, code_execution_result=None,
            inline_data=None, text="The Eiffel Tower is in Paris.", function_call=None,
        )
        grounding = {
            "grounding_chunks": [
                {"maps": {"uri": "https://maps.google.com/eiffel", "title": "Eiffel Tower", "placeId": "ChIJ..."}},
            ],
            "grounding_supports": [
                {
                    "segment": {"start_index": 0, "end_index": 28},
                    "grounding_chunk_indices": [0],
                }
            ],
        }
        event = _make_google_event([text_part], grounding_metadata=grounding)
        bp = build_prompt_blueprint_from_google_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        text_items = [c for c in msg["content"] if c["type"] == "text"]
        ann = text_items[0]["annotations"]
        assert len(ann) == 1
        assert ann[0]["type"] == "map_citation"
        assert ann[0]["url"] == "https://maps.google.com/eiffel"

    def test_file_search_grounding_annotations(self):
        """Grounding metadata with retrieved_context chunks produces file_citation annotations."""
        text_part = SimpleNamespace(
            thought=False, executable_code=None, code_execution_result=None,
            inline_data=None, text="According to the doc...", function_call=None,
        )
        grounding = {
            "grounding_chunks": [
                {"retrieved_context": {"title": "my_doc.pdf"}},
            ],
            "grounding_supports": [
                {
                    "segment": {"start_index": 0, "end_index": 22},
                    "grounding_chunk_indices": [0],
                }
            ],
        }
        event = _make_google_event([text_part], grounding_metadata=grounding)
        bp = build_prompt_blueprint_from_google_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        text_items = [c for c in msg["content"] if c["type"] == "text"]
        ann = text_items[0]["annotations"]
        assert len(ann) == 1
        assert ann[0]["type"] == "file_citation"
        assert ann[0]["file_id"] == "my_doc.pdf"

    def test_grounding_chunks_only_fallback(self):
        """When grounding_supports is absent, annotations are built from chunks alone."""
        text_part = SimpleNamespace(
            thought=False, executable_code=None, code_execution_result=None,
            inline_data=None, text="Some text.", function_call=None,
        )
        grounding = {
            "grounding_chunks": [
                {"web": {"uri": "https://example.com", "title": "Example"}},
                {"web": {"uri": "https://other.com", "title": "Other"}},
            ],
        }
        event = _make_google_event([text_part], grounding_metadata=grounding)
        bp = build_prompt_blueprint_from_google_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        text_items = [c for c in msg["content"] if c["type"] == "text"]
        ann = text_items[0]["annotations"]
        assert len(ann) == 2
        assert ann[0]["start_index"] == 0  # fallback has no segment info
        assert ann[1]["url"] == "https://other.com"

    def test_no_grounding_metadata(self):
        """Without grounding_metadata, no annotations are added."""
        text_part = SimpleNamespace(
            thought=False, executable_code=None, code_execution_result=None,
            inline_data=None, text="Hello", function_call=None,
        )
        event = _make_google_event([text_part])
        bp = build_prompt_blueprint_from_google_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        text_items = [c for c in msg["content"] if c["type"] == "text"]
        assert text_items[0].get("annotations") is None

    def test_mixed_grounding_chunk_types(self):
        """Grounding with web + maps + file chunks in one response."""
        text_part = SimpleNamespace(
            thought=False, executable_code=None, code_execution_result=None,
            inline_data=None, text="Mixed results.", function_call=None,
        )
        grounding = {
            "grounding_chunks": [
                {"web": {"uri": "https://web.com", "title": "Web"}},
                {"maps": {"uri": "https://maps.com", "title": "Map Place"}},
                {"retrieved_context": {"title": "doc.pdf"}},
            ],
            "grounding_supports": [
                {
                    "segment": {"start_index": 0, "end_index": 14},
                    "grounding_chunk_indices": [0, 1, 2],
                }
            ],
        }
        event = _make_google_event([text_part], grounding_metadata=grounding)
        bp = build_prompt_blueprint_from_google_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        ann = msg["content"][0]["annotations"]
        types = [a["type"] for a in ann]
        assert "url_citation" in types
        assert "map_citation" in types
        assert "file_citation" in types

    def test_url_context_metadata_annotations(self):
        """url_context_metadata should produce url_citation annotations on text content."""
        text_part = SimpleNamespace(
            thought=False, executable_code=None, code_execution_result=None,
            inline_data=None, text="I was unable to access the URL.", function_call=None,
        )
        url_meta = SimpleNamespace(
            retrieved_url="https://www.example.com/recipe",
        )
        url_context = SimpleNamespace(url_metadata=[url_meta])
        event = _make_google_event([text_part], url_context_metadata=url_context)
        bp = build_prompt_blueprint_from_google_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        ann = msg["content"][0].get("annotations", [])
        assert len(ann) == 1
        assert ann[0]["type"] == "url_citation"
        assert ann[0]["url"] == "https://www.example.com/recipe"
        assert ann[0]["title"] is None
        assert ann[0]["start_index"] is None
        assert ann[0]["end_index"] is None

    def test_url_context_metadata_no_url(self):
        """url_context_metadata with no retrieved_url should not produce annotations."""
        text_part = SimpleNamespace(
            thought=False, executable_code=None, code_execution_result=None,
            inline_data=None, text="Some text.", function_call=None,
        )
        url_meta = SimpleNamespace(retrieved_url=None)
        url_context = SimpleNamespace(url_metadata=[url_meta])
        event = _make_google_event([text_part], url_context_metadata=url_context)
        bp = build_prompt_blueprint_from_google_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        assert "annotations" not in msg["content"][0]


# ── Phase 6: Image generation support ───────────────────────────────


class TestImageGenerationCallEnriched:
    """Test enriched image_generation_call fields in response handler and blueprint builder."""

    def test_output_item_added_captures_all_fields(self):
        response_data = {"output": []}
        current_items = {}
        chunk = {
            "type": "response.output_item.added",
            "item": {
                "type": "image_generation_call",
                "id": "ig_1",
                "result": "",
                "status": "in_progress",
                "revised_prompt": "a cute cat in space",
                "background": "transparent",
                "size": "1024x1024",
                "quality": "high",
                "output_format": "webp",
            },
        }
        _process_openai_response_event(chunk, response_data, current_items)
        assert "ig_1" in current_items
        item = current_items["ig_1"]
        assert item["revised_prompt"] == "a cute cat in space"
        assert item["background"] == "transparent"
        assert item["size"] == "1024x1024"
        assert item["quality"] == "high"
        assert item["output_format"] == "webp"

    def test_output_item_added_defaults_when_fields_missing(self):
        response_data = {"output": []}
        current_items = {}
        chunk = {
            "type": "response.output_item.added",
            "item": {
                "type": "image_generation_call",
                "id": "ig_2",
                "result": "",
                "status": "in_progress",
            },
        }
        _process_openai_response_event(chunk, response_data, current_items)
        item = current_items["ig_2"]
        assert item["revised_prompt"] == ""
        assert item["background"] is None
        assert item["size"] is None
        assert item["quality"] is None
        assert item["output_format"] is None

    def test_output_item_done_updates_all_fields(self):
        response_data = {"output": []}
        current_items = {
            "ig_1": {
                "type": "image_generation_call",
                "id": "ig_1",
                "result": "",
                "status": "in_progress",
                "revised_prompt": "",
                "background": None,
                "size": None,
                "quality": None,
                "output_format": None,
            }
        }
        chunk = {
            "type": "response.output_item.done",
            "item": {
                "type": "image_generation_call",
                "id": "ig_1",
                "result": "https://img.example.com/gen.png",
                "status": "completed",
                "revised_prompt": "a photorealistic cute cat floating in outer space",
                "background": "auto",
                "size": "1024x1024",
                "quality": "high",
                "output_format": "png",
            },
        }
        _process_openai_response_event(chunk, response_data, current_items)
        item = current_items["ig_1"]
        assert item["result"] == "https://img.example.com/gen.png"
        assert item["revised_prompt"] == "a photorealistic cute cat floating in outer space"
        assert item["background"] == "auto"
        assert item["size"] == "1024x1024"
        assert item["quality"] == "high"
        assert item["output_format"] == "png"
        assert item["status"] == "completed"
        assert len(response_data["output"]) == 1

    def test_blueprint_image_generation_call_with_output_format(self):
        """output_item.done mime_type should be derived from metadata parameters, not from event item."""
        webp_metadata = {"model": {"provider": "openai", "name": "gpt-image-1", "parameters": {"output_format": "webp"}}}
        event = _make_openai_response_event(
            "response.output_item.done",
            item={
                "type": "image_generation_call",
                "id": "ig_1",
                "result": "https://img.example.com/gen.webp",
                "status": "completed",
            },
        )
        bp = build_prompt_blueprint_from_openai_responses_event(event, webp_metadata)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "output_media"]
        assert len(items) == 1
        assert items[0]["mime_type"] == "image/webp"
        assert "provider_metadata" not in items[0]

    def test_blueprint_image_generation_call_with_provider_metadata(self):
        """When image_generation_call item has provider fields, they should appear in provider_metadata."""
        event = _make_openai_response_event(
            "response.output_item.done",
            item={
                "type": "image_generation_call",
                "id": "ig_pm",
                "result": "base64data",
                "status": "completed",
                "revised_prompt": "A revised prompt",
                "size": "1024x1024",
                "quality": "high",
                "background": "transparent",
                "output_format": "webp",
            },
        )
        bp = build_prompt_blueprint_from_openai_responses_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "output_media"]
        assert len(items) == 1
        assert items[0]["mime_type"] == "image/webp"
        pm = items[0]["provider_metadata"]
        assert pm["revised_prompt"] == "A revised prompt"
        assert pm["size"] == "1024x1024"
        assert pm["quality"] == "high"
        assert pm["background"] == "transparent"
        assert pm["output_format"] == "webp"

    def test_blueprint_image_generation_call_no_provider_metadata(self):
        """Streaming blueprint should never include provider_metadata for image_generation_call."""
        event = _make_openai_response_event(
            "response.output_item.done",
            item={
                "type": "image_generation_call",
                "id": "ig_2",
                "result": "https://img.example.com/gen.png",
                "status": "completed",
            },
        )
        bp = build_prompt_blueprint_from_openai_responses_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "output_media"]
        assert len(items) == 1
        assert items[0]["mime_type"] == "image/png"
        assert "provider_metadata" not in items[0]

    def test_blueprint_response_completed_with_image_generation_call(self):
        """response.completed with image_generation_call in output should map correctly."""
        jpeg_metadata = {"model": {"provider": "openai", "name": "gpt-image-1", "parameters": {"output_format": "jpeg"}}}
        event = _make_openai_response_event(
            "response.completed",
            response={
                "output": [
                    {
                        "type": "image_generation_call",
                        "id": "ig_1",
                        "result": "https://img.example.com/final.png",
                        "status": "completed",
                    }
                ],
                "status": "completed",
            },
        )
        bp = build_prompt_blueprint_from_openai_responses_event(event, jpeg_metadata)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "output_media"]
        assert len(items) == 1
        assert items[0]["mime_type"] == "image/jpeg"
        assert "provider_metadata" not in items[0]

    def test_blueprint_output_item_added_with_result(self):
        """output_item.added with a result already present should emit output_media without provider_metadata."""
        webp_metadata = {"model": {"provider": "openai", "name": "gpt-image-1", "parameters": {"output_format": "webp"}}}
        event = _make_openai_response_event(
            "response.output_item.added",
            item={
                "type": "image_generation_call",
                "id": "ig_1",
                "result": "https://img.example.com/gen.webp",
                "status": "completed",
            },
        )
        bp = build_prompt_blueprint_from_openai_responses_event(event, webp_metadata)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "output_media"]
        assert len(items) == 1
        assert items[0]["mime_type"] == "image/webp"
        assert "provider_metadata" not in items[0]


class TestImageGenerationCallStreamingEvents:
    """Test intermediate image_generation_call streaming events."""

    def test_partial_image_blueprint(self):
        event = _make_openai_response_event(
            "response.image_generation_call.partial_image",
            item_id="ig_1",
            partial_image_b64="iVBORw0KGgoAAAANSUhEUgAABAAAA",
        )
        bp = build_prompt_blueprint_from_openai_responses_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "output_media"]
        assert len(items) == 1
        assert items[0]["url"] == "iVBORw0KGgoAAAANSUhEUgAABAAAA"
        assert items[0]["mime_type"] == "image/png"
        assert items[0]["id"] == "ig_1"

    def test_partial_image_empty_b64_blueprint(self):
        event = _make_openai_response_event(
            "response.image_generation_call.partial_image",
            item_id="ig_1",
            partial_image_b64="",
        )
        bp = build_prompt_blueprint_from_openai_responses_event(event, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        assert len(msg["content"]) == 0

    def test_in_progress_handler(self):
        response_data = {"output": []}
        current_items = {
            "ig_1": {
                "type": "image_generation_call",
                "id": "ig_1",
                "result": None,
                "status": "generating",
                "revised_prompt": "",
                "background": None,
                "size": None,
                "quality": None,
                "output_format": None,
            }
        }
        chunk = {
            "type": "response.image_generation_call.in_progress",
            "item_id": "ig_1",
        }
        _process_openai_response_event(chunk, response_data, current_items)
        assert current_items["ig_1"]["status"] == "in_progress"

    def test_partial_image_handler(self):
        response_data = {"output": []}
        current_items = {
            "ig_1": {
                "type": "image_generation_call",
                "id": "ig_1",
                "result": None,
                "status": "generating",
                "revised_prompt": "",
                "background": None,
                "size": None,
                "quality": None,
                "output_format": None,
            }
        }
        chunk = {
            "type": "response.image_generation_call.partial_image",
            "item_id": "ig_1",
            "partial_image_b64": "iVBORw0KGgoAAAA",
        }
        _process_openai_response_event(chunk, response_data, current_items)
        assert current_items["ig_1"]["result"] == "iVBORw0KGgoAAAA"
        assert current_items["ig_1"]["status"] == "generating"

    def test_partial_image_handler_empty_b64_no_update(self):
        response_data = {"output": []}
        current_items = {
            "ig_1": {
                "type": "image_generation_call",
                "id": "ig_1",
                "result": None,
                "status": "generating",
                "revised_prompt": "",
                "background": None,
                "size": None,
                "quality": None,
                "output_format": None,
            }
        }
        chunk = {
            "type": "response.image_generation_call.partial_image",
            "item_id": "ig_1",
            "partial_image_b64": "",
        }
        _process_openai_response_event(chunk, response_data, current_items)
        assert current_items["ig_1"]["result"] is None  # unchanged


class TestOpenAIImagesAPIBlueprint:
    """Test build_prompt_blueprint_from_openai_images_event."""

    def test_partial_image_event(self):
        event_dict = {
            "type": "image_generation.partial_image",
            "b64_json": "partial_base64_data==",
        }
        bp = build_prompt_blueprint_from_openai_images_event(event_dict, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "output_media"]
        assert len(items) == 1
        assert items[0]["url"] == "partial_base64_data=="
        assert items[0]["mime_type"] == "image/png"

    def test_partial_image_empty_b64(self):
        event_dict = {
            "type": "image_generation.partial_image",
            "b64_json": "",
        }
        bp = build_prompt_blueprint_from_openai_images_event(event_dict, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        assert len(msg["content"]) == 0

    def test_completed_event(self):
        event_dict = {
            "type": "image_generation.completed",
            "b64_json": "final_base64_data==",
            "output_format": "webp",
            "revised_prompt": "a beautiful landscape",
            "size": "1024x1024",
            "quality": "high",
        }
        bp = build_prompt_blueprint_from_openai_images_event(event_dict, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "output_media"]
        assert len(items) == 1
        assert items[0]["url"] == "final_base64_data=="
        assert items[0]["mime_type"] == "image/png"
        pm = items[0]["provider_metadata"]
        assert pm["revised_prompt"] == "a beautiful landscape"
        assert pm["size"] == "1024x1024"
        assert pm["quality"] == "high"
        assert pm["output_format"] == "webp"

    def test_completed_event_default_format(self):
        event_dict = {
            "type": "image_generation.completed",
            "b64_json": "final_base64_data==",
        }
        bp = build_prompt_blueprint_from_openai_images_event(event_dict, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "output_media"]
        assert len(items) == 1
        assert items[0]["mime_type"] == "image/png"
        assert items[0].get("provider_metadata") is None

    def test_completed_event_with_background(self):
        event_dict = {
            "type": "image_generation.completed",
            "b64_json": "data==",
            "output_format": "png",
            "background": "transparent",
        }
        bp = build_prompt_blueprint_from_openai_images_event(event_dict, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "output_media"]
        assert items[0]["url"] == "data=="
        pm = items[0]["provider_metadata"]
        assert pm["background"] == "transparent"
        assert pm["output_format"] == "png"

    def test_unknown_event_type(self):
        event_dict = {"type": "image_generation.unknown_event"}
        bp = build_prompt_blueprint_from_openai_images_event(event_dict, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        assert len(msg["content"]) == 0

    def test_model_dump_input(self):
        """Accepts objects with model_dump()."""
        mock = MagicMock()
        mock.model_dump.return_value = {
            "type": "image_generation.completed",
            "b64_json": "data==",
            "output_format": "jpeg",
        }
        bp = build_prompt_blueprint_from_openai_images_event(mock, METADATA)
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "output_media"]
        assert len(items) == 1
        assert items[0]["mime_type"] == "image/png"


class TestOpenAIImagesStreamHandler:
    """Test openai_images_stream response accumulator."""

    def test_completed_event(self):
        chunks = [
            {"type": "image_generation.partial_image", "b64_json": "partial=="},
            {
                "type": "image_generation.completed",
                "b64_json": "final==",
                "created_at": 1234567890,
                "usage": {"input_tokens": 10, "output_tokens": 20},
                "revised_prompt": "a cat",
                "size": "1024x1024",
                "quality": "high",
                "output_format": "png",
            },
        ]
        result = openai_images_stream(chunks)
        assert result["created"] == 1234567890
        assert result["usage"] == {"input_tokens": 10, "output_tokens": 20}
        assert len(result["data"]) == 1
        assert result["data"][0]["b64_json"] == "final=="
        assert result["data"][0]["revised_prompt"] == "a cat"
        assert result["size"] == "1024x1024"
        assert result["quality"] == "high"

    def test_no_completed_event(self):
        chunks = [
            {"type": "image_generation.partial_image", "b64_json": "partial=="},
        ]
        result = openai_images_stream(chunks)
        assert result["created"] is None
        assert result["data"] == []

    def test_model_dump_chunks(self):
        """Chunks with model_dump() are handled."""
        mock = MagicMock()
        mock.model_dump.return_value = {
            "type": "image_generation.completed",
            "b64_json": "final==",
            "created_at": 999,
            "usage": None,
        }
        result = openai_images_stream([mock])
        assert result["created"] == 999
        assert len(result["data"]) == 1

    def test_empty_chunks(self):
        result = openai_images_stream([])
        assert result["created"] is None
        assert result["data"] == []


class TestStreamProcessorImagesRouting:
    """Test that stream_processor routes images api_type correctly."""

    def test_images_api_type_routing(self):
        from promptlayer.streaming.stream_processor import _build_stream_blueprint

        metadata = {
            "model": {
                "provider": "openai",
                "name": "gpt-image-1",
                "parameters": {},
                "api_type": "images",
            }
        }
        event_dict = {
            "type": "image_generation.completed",
            "b64_json": "data==",
            "output_format": "png",
        }
        bp = _build_stream_blueprint(event_dict, metadata)
        assert bp is not None
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "output_media"]
        assert len(items) == 1

    def test_images_api_type_partial(self):
        from promptlayer.streaming.stream_processor import _build_stream_blueprint

        metadata = {
            "model": {
                "provider": "openai",
                "name": "gpt-image-1",
                "parameters": {},
                "api_type": "images",
            }
        }
        event_dict = {
            "type": "image_generation.partial_image",
            "b64_json": "partial==",
        }
        bp = _build_stream_blueprint(event_dict, metadata)
        assert bp is not None
        msg = bp["prompt_template"]["messages"][0]
        items = [c for c in msg["content"] if c["type"] == "output_media"]
        assert len(items) == 1
